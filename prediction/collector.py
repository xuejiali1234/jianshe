from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import PredictionConfig


CSV_FIELDS = [
    "run_id",
    "scenario_id",
    "seed",
    "demand_scale",
    "base_demand_factor",
    "signal_variant",
    "incident_type",
    "incident_start_s",
    "incident_end_s",
    "affected_edges",
    "timestamp",
    "step",
    "edge_id",
    "flow",
    "speed_mps",
    "speed_kmh",
    "queue",
    "incident_flag",
    "phase_id",
    "phase_elapsed_s",
    "green_remaining_s",
]

SIGNAL_DEFAULTS = {
    "phase_id": -1,
    "phase_elapsed_s": 0.0,
    "green_remaining_s": 0.0,
}


class EdgeRealtimeCollector:
    def __init__(
        self,
        config: PredictionConfig,
        csv_path: str | Path,
        run_id: str | None = None,
        scenario_id: str = "",
        seed: int | str = "",
        demand_scale: float | str = "",
        base_demand_factor: float | str = "",
        signal_variant: str = "webster_base",
        incident_type: str = "",
        incident_start_s: int | str = "",
        incident_end_s: int | str = "",
        affected_edges: list[str] | tuple[str, ...] | None = None,
    ):
        self.config = config
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.scenario_id = scenario_id
        self.seed = seed
        self.demand_scale = demand_scale
        self.base_demand_factor = base_demand_factor
        self.signal_variant = signal_variant
        self.incident_type = incident_type
        self.incident_start_s = incident_start_s
        self.incident_end_s = incident_end_s
        self.affected_edges = list(affected_edges or [])
        self.interval_start_time = 0.0
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self.accumulators = {
            edge_id: {"vehicle_ids": set(), "speeds": [], "queues": []}
            for edge_id in self.config.observed_edges
        }

    def record_step(
        self,
        traci_module: Any,
        step: int,
        sim_time_s: float,
        incident_edges: set[str] | None = None,
    ) -> dict[str, Any] | None:
        incident_edges = incident_edges or set()

        for edge_id in self.config.observed_edges:
            acc = self.accumulators[edge_id]
            try:
                for vehicle_id in traci_module.edge.getLastStepVehicleIDs(edge_id):
                    acc["vehicle_ids"].add(vehicle_id)
            except Exception:
                pass

            try:
                speed = float(traci_module.edge.getLastStepMeanSpeed(edge_id))
                if speed >= 0.0:
                    acc["speeds"].append(speed)
            except Exception:
                pass

            try:
                queue = int(traci_module.edge.getLastStepHaltingNumber(edge_id))
                acc["queues"].append(queue)
            except Exception:
                pass

        if sim_time_s - self.interval_start_time < self.config.sample_interval_s:
            return None

        snapshot = self._build_snapshot(step, sim_time_s, incident_edges, traci_module)
        self._append_csv(snapshot)
        self.interval_start_time = sim_time_s
        self._reset_accumulators()
        return snapshot

    def _build_snapshot(
        self,
        step: int,
        sim_time_s: float,
        incident_edges: set[str],
        traci_module: Any,
    ) -> dict[str, Any]:
        start_time = datetime.fromisoformat(self.config.simulation_start_iso)
        timestamp = (start_time + timedelta(seconds=int(sim_time_s))).isoformat()
        nodes = []
        signal_by_edge = self._collect_signal_features(traci_module, sim_time_s)

        for edge_id, acc in self.accumulators.items():
            speeds = acc["speeds"]
            speed_mps = sum(speeds) / len(speeds) if speeds else 0.0
            queue = max(acc["queues"]) if acc["queues"] else 0
            flow = len(acc["vehicle_ids"])
            signal_features = signal_by_edge.get(edge_id, SIGNAL_DEFAULTS)
            nodes.append(
                {
                    "edge_id": edge_id,
                    "flow": flow,
                    "speed": speed_mps,
                    "speed_mps": speed_mps,
                    "speed_kmh": speed_mps * 3.6,
                    "queue": queue,
                    "incident_flag": 1 if edge_id in incident_edges else 0,
                    "phase_id": int(signal_features["phase_id"]),
                    "phase_elapsed_s": float(signal_features["phase_elapsed_s"]),
                    "green_remaining_s": float(signal_features["green_remaining_s"]),
                }
            )

        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "demand_scale": self.demand_scale,
            "base_demand_factor": self.base_demand_factor,
            "signal_variant": self.signal_variant,
            "incident_type": self.incident_type,
            "incident_start_s": self.incident_start_s,
            "incident_end_s": self.incident_end_s,
            "affected_edges": list(self.affected_edges),
            "timestamp": timestamp,
            "step": step,
            "nodes": nodes,
        }

    def _collect_signal_features(
        self,
        traci_module: Any,
        sim_time_s: float,
    ) -> dict[str, dict[str, float]]:
        signal_by_edge: dict[str, dict[str, float]] = {}
        try:
            tls_ids = list(traci_module.trafficlight.getIDList())
        except Exception:
            return signal_by_edge

        observed_edges = set(self.config.observed_edges)
        for tls_id in tls_ids:
            try:
                state = str(traci_module.trafficlight.getRedYellowGreenState(tls_id))
                phase_id = int(traci_module.trafficlight.getPhase(tls_id))
                next_switch = float(traci_module.trafficlight.getNextSwitch(tls_id))
                controlled_links = traci_module.trafficlight.getControlledLinks(tls_id)
            except Exception:
                continue

            time_to_switch = max(0.0, next_switch - float(sim_time_s))
            phase_elapsed_s = 0.0
            try:
                phase_duration = float(traci_module.trafficlight.getPhaseDuration(tls_id))
                phase_elapsed_s = max(0.0, phase_duration - time_to_switch)
            except Exception:
                pass

            for link_index, link_group in enumerate(controlled_links):
                signal_state = state[link_index] if link_index < len(state) else "r"
                is_green = 1 if signal_state in {"G", "g"} else 0
                green_remaining_s = time_to_switch if is_green else 0.0
                features = {
                    "phase_id": float(phase_id),
                    "phase_elapsed_s": float(phase_elapsed_s),
                    "green_remaining_s": float(green_remaining_s),
                    "_is_green": float(is_green),
                }
                for link in link_group:
                    if not link:
                        continue
                    incoming_lane = str(link[0] or "")
                    edge_id = self._lane_to_edge_id(incoming_lane)
                    if edge_id not in observed_edges:
                        continue
                    existing = signal_by_edge.get(edge_id)
                    if existing is None or features["_is_green"] > existing.get("_is_green", 0.0):
                        signal_by_edge[edge_id] = dict(features)

        for features in signal_by_edge.values():
            features.pop("_is_green", None)
        return signal_by_edge

    @staticmethod
    def _lane_to_edge_id(lane_id: str) -> str:
        if "_" not in lane_id:
            return lane_id
        edge_id, lane_index = lane_id.rsplit("_", 1)
        return edge_id if lane_index.isdigit() else lane_id

    def _append_csv(self, snapshot: dict[str, Any]) -> None:
        file_exists = self._prepare_csv_for_append()
        with self.csv_path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
            if not file_exists:
                writer.writeheader()

            for node in snapshot["nodes"]:
                writer.writerow(
                    {
                        "run_id": snapshot["run_id"],
                        "scenario_id": snapshot.get("scenario_id", ""),
                        "seed": snapshot.get("seed", ""),
                        "demand_scale": snapshot.get("demand_scale", ""),
                        "base_demand_factor": snapshot.get("base_demand_factor", ""),
                        "signal_variant": snapshot.get("signal_variant", "webster_base"),
                        "incident_type": snapshot.get("incident_type", ""),
                        "incident_start_s": snapshot.get("incident_start_s", ""),
                        "incident_end_s": snapshot.get("incident_end_s", ""),
                        "affected_edges": "|".join(snapshot.get("affected_edges", [])),
                        "timestamp": snapshot["timestamp"],
                        "step": snapshot["step"],
                        "edge_id": node["edge_id"],
                        "flow": node["flow"],
                        "speed_mps": round(node["speed_mps"], 6),
                        "speed_kmh": round(node["speed_kmh"], 6),
                        "queue": node["queue"],
                        "incident_flag": node["incident_flag"],
                        "phase_id": node["phase_id"],
                        "phase_elapsed_s": round(node["phase_elapsed_s"], 6),
                        "green_remaining_s": round(node["green_remaining_s"], 6),
                    }
                )

    def _prepare_csv_for_append(self) -> bool:
        if not self.csv_path.exists():
            return False

        with self.csv_path.open("r", encoding="utf-8", errors="ignore") as fp:
            first_line = fp.readline().strip()
        if not first_line:
            return False

        existing_fields = [field.strip() for field in first_line.split(",")]
        if existing_fields == CSV_FIELDS:
            return True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        legacy_path = self.csv_path.with_name(
            f"{self.csv_path.stem}_legacy_{timestamp}{self.csv_path.suffix}"
        )
        self.csv_path.replace(legacy_path)
        print(f"Archived incompatible collector CSV header to {legacy_path}")
        return False
