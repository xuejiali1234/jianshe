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
]


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

        snapshot = self._build_snapshot(step, sim_time_s, incident_edges)
        self._append_csv(snapshot)
        self.interval_start_time = sim_time_s
        self._reset_accumulators()
        return snapshot

    def _build_snapshot(
        self,
        step: int,
        sim_time_s: float,
        incident_edges: set[str],
    ) -> dict[str, Any]:
        start_time = datetime.fromisoformat(self.config.simulation_start_iso)
        timestamp = (start_time + timedelta(seconds=int(sim_time_s))).isoformat()
        nodes = []

        for edge_id, acc in self.accumulators.items():
            speeds = acc["speeds"]
            speed_mps = sum(speeds) / len(speeds) if speeds else 0.0
            queue = max(acc["queues"]) if acc["queues"] else 0
            flow = len(acc["vehicle_ids"])
            nodes.append(
                {
                    "edge_id": edge_id,
                    "flow": flow,
                    "speed": speed_mps,
                    "speed_mps": speed_mps,
                    "speed_kmh": speed_mps * 3.6,
                    "queue": queue,
                    "incident_flag": 1 if edge_id in incident_edges else 0,
                }
            )

        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "demand_scale": self.demand_scale,
            "base_demand_factor": self.base_demand_factor,
            "incident_type": self.incident_type,
            "incident_start_s": self.incident_start_s,
            "incident_end_s": self.incident_end_s,
            "affected_edges": list(self.affected_edges),
            "timestamp": timestamp,
            "step": step,
            "nodes": nodes,
        }

    def _append_csv(self, snapshot: dict[str, Any]) -> None:
        file_exists = self.csv_path.exists()
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
                    }
                )
