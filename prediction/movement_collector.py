from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sim import configure_sumo_python_path, resolve_runtime_net_file
from sim.movement_tools import build_movement_config, load_movement_config

from .config import PredictionConfig


MOVEMENT_CSV_FIELDS = [
    "run_id",
    "scenario_id",
    "seed",
    "demand_scale",
    "base_demand_factor",
    "signal_variant",
    "event_type",
    "event_policy",
    "speed_factor",
    "incident_type",
    "incident_start_s",
    "incident_end_s",
    "affected_edges",
    "timestamp",
    "step",
    "movement_id",
    "tls_id",
    "incoming_edge",
    "outgoing_edge",
    "turn_type",
    "lane_ids",
    "arrival_flow",
    "discharge_flow",
    "mean_speed_mps",
    "speed_kmh",
    "occupancy",
    "queue_veh",
    "queue_meter",
    "incident_flag",
    "phase_id",
    "phase_elapsed_s",
    "green_remaining_s",
    "signal_state",
    "zone_quality",
]


SIGNAL_DEFAULTS = {
    "phase_id": -1,
    "phase_elapsed_s": 0.0,
    "green_remaining_s": 0.0,
    "signal_state": "",
}


class MovementRealtimeCollector:
    """Virtual movement-level detector for SUMO TraCI snapshots.

    The final short SUMO incoming edge is used as a stopbar/discharge detector.
    Arrival and queue states are measured over the functional upstream approach
    zone, extending across predecessor edges when the incoming edge is too short.
    """

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
        event_type: str = "",
        event_policy: str = "",
        speed_factor: float | str = "",
        incident_type: str = "",
        incident_start_s: int | str = "",
        incident_end_s: int | str = "",
        affected_edges: list[str] | tuple[str, ...] | None = None,
        project_root: str | Path | None = None,
        net_file: str | Path | None = None,
        movement_config_path: str | Path | None = None,
    ):
        configure_sumo_python_path()
        import sumolib

        self.config = config
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.scenario_id = scenario_id
        self.seed = seed
        self.demand_scale = demand_scale
        self.base_demand_factor = base_demand_factor
        self.signal_variant = signal_variant
        self.event_type = event_type
        self.event_policy = event_policy
        self.speed_factor = speed_factor
        self.incident_type = incident_type
        self.incident_start_s = incident_start_s
        self.incident_end_s = incident_end_s
        self.affected_edges = list(affected_edges or [])
        self.interval_start_time = 0.0

        root = Path(project_root or Path.cwd())
        self.movement_config_path = Path(
            movement_config_path or getattr(config, "movement_config_file", "configs/movement_config.json")
        )
        if not self.movement_config_path.is_absolute():
            self.movement_config_path = root / self.movement_config_path
        self.net_file = Path(net_file or resolve_runtime_net_file(root, config.sumo_net_file))
        if not self.movement_config_path.exists():
            build_movement_config(
                self.net_file,
                config.observed_edges,
                self.movement_config_path,
                root / "data" / "processed" / "movement_map.csv",
            )

        movement_payload = load_movement_config(self.movement_config_path)
        self.movements = list(movement_payload.get("movements", []))
        self.movement_ids = [movement["movement_id"] for movement in self.movements]
        self.movement_by_id = {movement["movement_id"]: movement for movement in self.movements}
        self.edge_to_movements: dict[str, list[dict[str, Any]]] = {}
        for movement in self.movements:
            for edge_id in movement.get("zone_edges", []):
                self.edge_to_movements.setdefault(edge_id, []).append(movement)

        self.net = sumolib.net.readNet(str(self.net_file))
        self.edge_lengths = {
            edge.getID(): float(edge.getLength())
            for edge in self.net.getEdges()
        }
        self._last_route_edge_by_vehicle: dict[str, str] = {}
        self._last_distance_by_vehicle_movement: dict[tuple[str, str], float] = {}
        self._arrival_ready: set[tuple[str, str]] = set()
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self.accumulators = {
            movement_id: {
                "arrival_ids": set(),
                "discharge_ids": set(),
                "speeds": [],
                "queue_counts": [],
                "queue_meters": [],
                "occupancy_samples": [],
            }
            for movement_id in self.movement_ids
        }

    def record_step(
        self,
        traci_module: Any,
        step: int,
        sim_time_s: float,
        incident_edges: set[str] | None = None,
    ) -> dict[str, Any] | None:
        incident_edges = incident_edges or set()
        vehicles_in_zone_by_movement: dict[str, int] = {movement_id: 0 for movement_id in self.movement_ids}
        queue_by_movement: dict[str, int] = {movement_id: 0 for movement_id in self.movement_ids}
        queue_meter_by_movement: dict[str, float] = {movement_id: 0.0 for movement_id in self.movement_ids}

        try:
            vehicle_ids = list(traci_module.vehicle.getIDList())
        except Exception:
            vehicle_ids = []

        current_seen: set[str] = set()
        for vehicle_id in vehicle_ids:
            current_seen.add(vehicle_id)
            try:
                current_edge = str(traci_module.vehicle.getRoadID(vehicle_id))
            except Exception:
                continue
            if not current_edge or current_edge.startswith(":"):
                self._last_route_edge_by_vehicle.setdefault(vehicle_id, current_edge)
                continue

            try:
                route = list(traci_module.vehicle.getRoute(vehicle_id))
                route_index = int(traci_module.vehicle.getRouteIndex(vehicle_id))
            except Exception:
                route = []
                route_index = -1

            current_route_edge = self._route_edge(route, route_index, current_edge)
            last_route_edge = self._last_route_edge_by_vehicle.get(vehicle_id)
            if current_route_edge and last_route_edge and current_route_edge != last_route_edge:
                self._record_discharge(vehicle_id, last_route_edge, current_route_edge)
            if current_route_edge:
                self._last_route_edge_by_vehicle[vehicle_id] = current_route_edge

            candidate_movements = self.edge_to_movements.get(current_edge, [])
            if not candidate_movements:
                continue

            try:
                lane_pos = float(traci_module.vehicle.getLanePosition(vehicle_id))
                speed = float(traci_module.vehicle.getSpeed(vehicle_id))
            except Exception:
                continue

            for movement in candidate_movements:
                if not self._vehicle_matches_movement(route, route_index, current_edge, movement):
                    continue
                distance = self._distance_to_stopline(current_edge, lane_pos, movement)
                if distance is None:
                    continue

                movement_id = movement["movement_id"]
                if 0.0 <= distance <= float(movement.get("queue_end_m", 150.0)):
                    self.accumulators[movement_id]["speeds"].append(speed)
                    vehicles_in_zone_by_movement[movement_id] += 1
                    if speed < 0.1:
                        queue_by_movement[movement_id] += 1
                        queue_meter_by_movement[movement_id] = max(
                            queue_meter_by_movement[movement_id],
                            float(distance),
                        )
                self._record_arrival_crossing(vehicle_id, movement_id, distance, movement)

        for movement in self.movements:
            movement_id = movement["movement_id"]
            zone_lane_m = max(1.0, float(movement.get("zone_lane_m", 1.0)))
            occupancy = min(1.0, vehicles_in_zone_by_movement[movement_id] * 7.5 / zone_lane_m)
            self.accumulators[movement_id]["occupancy_samples"].append(occupancy)
            self.accumulators[movement_id]["queue_counts"].append(queue_by_movement[movement_id])
            self.accumulators[movement_id]["queue_meters"].append(queue_meter_by_movement[movement_id])

        for vehicle_id in list(self._last_route_edge_by_vehicle.keys()):
            if vehicle_id not in current_seen:
                self._last_route_edge_by_vehicle.pop(vehicle_id, None)
        self._drop_departed_vehicle_state(current_seen)

        if sim_time_s - self.interval_start_time < self.config.sample_interval_s:
            return None

        snapshot = self._build_snapshot(step, sim_time_s, incident_edges, traci_module)
        self._append_csv(snapshot)
        self.interval_start_time = sim_time_s
        self._reset_accumulators()
        return snapshot

    def _record_discharge(self, vehicle_id: str, from_edge: str, to_edge: str) -> None:
        for movement in self.movements:
            if movement["incoming_edge"] == from_edge and movement["outgoing_edge"] == to_edge:
                self.accumulators[movement["movement_id"]]["discharge_ids"].add(vehicle_id)
                return

    def _record_arrival_crossing(
        self,
        vehicle_id: str,
        movement_id: str,
        distance_to_stopline: float,
        movement: dict[str, Any],
    ) -> None:
        detector_m = float(movement.get("arrival_start_m", 50.0))
        detector_end_m = float(movement.get("arrival_end_m", 150.0))
        key = (vehicle_id, movement_id)
        previous_distance = self._last_distance_by_vehicle_movement.get(key)

        if detector_m < distance_to_stopline <= detector_end_m:
            self._arrival_ready.add(key)
        crossed_detector = (
            previous_distance is not None
            and previous_distance > detector_m
            and distance_to_stopline <= detector_m
        )
        if crossed_detector or (key in self._arrival_ready and distance_to_stopline <= detector_m):
            self.accumulators[movement_id]["arrival_ids"].add(vehicle_id)
            self._arrival_ready.discard(key)

        self._last_distance_by_vehicle_movement[key] = distance_to_stopline

    def _drop_departed_vehicle_state(self, current_seen: set[str]) -> None:
        for key in list(self._last_distance_by_vehicle_movement.keys()):
            if key[0] not in current_seen:
                self._last_distance_by_vehicle_movement.pop(key, None)
                self._arrival_ready.discard(key)

    def _build_snapshot(
        self,
        step: int,
        sim_time_s: float,
        incident_edges: set[str],
        traci_module: Any,
    ) -> dict[str, Any]:
        start_time = datetime.fromisoformat(self.config.simulation_start_iso)
        timestamp = (start_time + timedelta(seconds=int(sim_time_s))).isoformat()
        signal_by_movement = self._collect_signal_features(traci_module, sim_time_s)

        movement_rows: list[dict[str, Any]] = []
        for movement in self.movements:
            movement_id = movement["movement_id"]
            acc = self.accumulators[movement_id]
            speeds = acc["speeds"]
            mean_speed_mps = sum(speeds) / len(speeds) if speeds else 0.0
            queue_veh = max(acc["queue_counts"]) if acc["queue_counts"] else 0
            queue_meter = max(acc["queue_meters"]) if acc["queue_meters"] else 0.0
            occupancy = (
                sum(acc["occupancy_samples"]) / len(acc["occupancy_samples"])
                if acc["occupancy_samples"]
                else 0.0
            )
            signal_features = signal_by_movement.get(movement_id, SIGNAL_DEFAULTS)
            incoming_edge = movement["incoming_edge"]
            movement_rows.append(
                {
                    "movement_id": movement_id,
                    "tls_id": movement["tls_id"],
                    "incoming_edge": incoming_edge,
                    "outgoing_edge": movement["outgoing_edge"],
                    "turn_type": movement["turn_type"],
                    "lane_ids": list(movement.get("lane_ids", [])),
                    "arrival_flow": len(acc["arrival_ids"]),
                    "discharge_flow": len(acc["discharge_ids"]),
                    "mean_speed": mean_speed_mps,
                    "mean_speed_mps": mean_speed_mps,
                    "speed_kmh": mean_speed_mps * 3.6,
                    "occupancy": occupancy,
                    "queue_veh": queue_veh,
                    "queue_meter": queue_meter,
                    "incident_flag": 1 if incoming_edge in incident_edges else 0,
                    "phase_id": int(signal_features["phase_id"]),
                    "phase_elapsed_s": float(signal_features["phase_elapsed_s"]),
                    "green_remaining_s": float(signal_features["green_remaining_s"]),
                    "signal_state": str(signal_features.get("signal_state", "")),
                    "zone_quality": movement.get("zone_quality", ""),
                }
            )

        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "demand_scale": self.demand_scale,
            "base_demand_factor": self.base_demand_factor,
            "signal_variant": self.signal_variant,
            "event_type": self.event_type,
            "event_policy": self.event_policy,
            "speed_factor": self.speed_factor,
            "incident_type": self.incident_type,
            "incident_start_s": self.incident_start_s,
            "incident_end_s": self.incident_end_s,
            "affected_edges": list(self.affected_edges),
            "timestamp": timestamp,
            "step": step,
            "movements": movement_rows,
            "nodes": movement_rows_to_legacy_nodes(movement_rows),
        }

    def _collect_signal_features(
        self,
        traci_module: Any,
        sim_time_s: float,
    ) -> dict[str, dict[str, Any]]:
        signal_by_movement: dict[str, dict[str, Any]] = {}
        try:
            tls_ids = set(traci_module.trafficlight.getIDList())
        except Exception:
            return signal_by_movement

        movements_by_tls: dict[str, list[dict[str, Any]]] = {}
        for movement in self.movements:
            movements_by_tls.setdefault(movement["tls_id"], []).append(movement)

        for tls_id, movements in movements_by_tls.items():
            if tls_id not in tls_ids:
                continue
            try:
                state = str(traci_module.trafficlight.getRedYellowGreenState(tls_id))
                phase_id = int(traci_module.trafficlight.getPhase(tls_id))
                next_switch = float(traci_module.trafficlight.getNextSwitch(tls_id))
            except Exception:
                continue

            time_to_switch = max(0.0, next_switch - float(sim_time_s))
            phase_elapsed_s = 0.0
            try:
                phase_duration = float(traci_module.trafficlight.getPhaseDuration(tls_id))
                phase_elapsed_s = max(0.0, phase_duration - time_to_switch)
            except Exception:
                pass

            for movement in movements:
                states = []
                is_green = False
                for link_index in movement.get("link_indexes", []):
                    link_state = state[link_index] if 0 <= int(link_index) < len(state) else "r"
                    states.append(link_state)
                    is_green = is_green or link_state in {"G", "g"}
                signal_by_movement[movement["movement_id"]] = {
                    "phase_id": phase_id,
                    "phase_elapsed_s": phase_elapsed_s,
                    "green_remaining_s": time_to_switch if is_green else 0.0,
                    "signal_state": "".join(states),
                }
        return signal_by_movement

    def _vehicle_matches_movement(
        self,
        route: list[str],
        route_index: int,
        current_edge: str,
        movement: dict[str, Any],
    ) -> bool:
        if not route:
            return current_edge == movement["incoming_edge"]
        search_start = max(0, route_index)
        if current_edge in route[search_start:]:
            search_start = search_start + route[search_start:].index(current_edge)
        future = route[search_start:]
        try:
            incoming_pos = future.index(movement["incoming_edge"])
        except ValueError:
            return False
        try:
            outgoing_pos = future.index(movement["outgoing_edge"], incoming_pos + 1)
        except ValueError:
            return False
        return outgoing_pos > incoming_pos

    def _distance_to_stopline(
        self,
        edge_id: str,
        lane_pos: float,
        movement: dict[str, Any],
    ) -> float | None:
        offsets = movement.get("zone_edge_offsets", {})
        if edge_id not in offsets:
            return None
        edge_length = self.edge_lengths.get(edge_id)
        if edge_length is None:
            return None
        return max(0.0, float(offsets[edge_id]) + edge_length - lane_pos)

    @staticmethod
    def _route_edge(route: list[str], route_index: int, current_edge: str) -> str:
        if 0 <= route_index < len(route):
            return str(route[route_index])
        return current_edge

    def _append_csv(self, snapshot: dict[str, Any]) -> None:
        file_exists = self._prepare_csv_for_append()
        with self.csv_path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=MOVEMENT_CSV_FIELDS)
            if not file_exists:
                writer.writeheader()

            for movement in snapshot["movements"]:
                writer.writerow(
                    {
                        "run_id": snapshot["run_id"],
                        "scenario_id": snapshot.get("scenario_id", ""),
                        "seed": snapshot.get("seed", ""),
                        "demand_scale": snapshot.get("demand_scale", ""),
                        "base_demand_factor": snapshot.get("base_demand_factor", ""),
                        "signal_variant": snapshot.get("signal_variant", "webster_base"),
                        "event_type": snapshot.get("event_type", ""),
                        "event_policy": snapshot.get("event_policy", ""),
                        "speed_factor": snapshot.get("speed_factor", ""),
                        "incident_type": snapshot.get("incident_type", ""),
                        "incident_start_s": snapshot.get("incident_start_s", ""),
                        "incident_end_s": snapshot.get("incident_end_s", ""),
                        "affected_edges": "|".join(snapshot.get("affected_edges", [])),
                        "timestamp": snapshot["timestamp"],
                        "step": snapshot["step"],
                        "movement_id": movement["movement_id"],
                        "tls_id": movement["tls_id"],
                        "incoming_edge": movement["incoming_edge"],
                        "outgoing_edge": movement["outgoing_edge"],
                        "turn_type": movement["turn_type"],
                        "lane_ids": "|".join(movement.get("lane_ids", [])),
                        "arrival_flow": movement["arrival_flow"],
                        "discharge_flow": movement["discharge_flow"],
                        "mean_speed_mps": round(movement["mean_speed_mps"], 6),
                        "speed_kmh": round(movement["speed_kmh"], 6),
                        "occupancy": round(movement["occupancy"], 6),
                        "queue_veh": movement["queue_veh"],
                        "queue_meter": round(movement["queue_meter"], 6),
                        "incident_flag": movement["incident_flag"],
                        "phase_id": movement["phase_id"],
                        "phase_elapsed_s": round(movement["phase_elapsed_s"], 6),
                        "green_remaining_s": round(movement["green_remaining_s"], 6),
                        "signal_state": movement.get("signal_state", ""),
                        "zone_quality": movement.get("zone_quality", ""),
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
        if existing_fields == MOVEMENT_CSV_FIELDS:
            return True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        legacy_path = self.csv_path.with_name(
            f"{self.csv_path.stem}_legacy_{timestamp}{self.csv_path.suffix}"
        )
        self.csv_path.replace(legacy_path)
        print(f"Archived incompatible movement collector CSV header to {legacy_path}")
        return False


def movement_rows_to_legacy_nodes(movements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for movement in movements:
        edge_id = movement["incoming_edge"]
        item = grouped.setdefault(
            edge_id,
            {
                "edge_id": edge_id,
                "flow": 0.0,
                "speed_values": [],
                "queue": 0.0,
                "incident_flag": 0.0,
                "phase_id": movement.get("phase_id", -1),
                "phase_elapsed_s": movement.get("phase_elapsed_s", 0.0),
                "green_remaining_s": movement.get("green_remaining_s", 0.0),
            },
        )
        item["flow"] += float(movement.get("arrival_flow", 0.0))
        speed = float(movement.get("mean_speed_mps", movement.get("mean_speed", 0.0)) or 0.0)
        if speed > 0:
            item["speed_values"].append(speed)
        item["queue"] += float(movement.get("queue_veh", 0.0))
        item["incident_flag"] = max(float(item["incident_flag"]), float(movement.get("incident_flag", 0.0)))
        if float(movement.get("green_remaining_s", 0.0) or 0.0) > float(item["green_remaining_s"] or 0.0):
            item["phase_id"] = movement.get("phase_id", -1)
            item["phase_elapsed_s"] = movement.get("phase_elapsed_s", 0.0)
            item["green_remaining_s"] = movement.get("green_remaining_s", 0.0)

    nodes: list[dict[str, Any]] = []
    for item in grouped.values():
        speed_mps = sum(item["speed_values"]) / len(item["speed_values"]) if item["speed_values"] else 0.0
        nodes.append(
            {
                "edge_id": item["edge_id"],
                "flow": item["flow"],
                "speed": speed_mps,
                "speed_mps": speed_mps,
                "speed_kmh": speed_mps * 3.6,
                "queue": item["queue"],
                "incident_flag": item["incident_flag"],
                "phase_id": item["phase_id"],
                "phase_elapsed_s": item["phase_elapsed_s"],
                "green_remaining_s": item["green_remaining_s"],
            }
        )
    return nodes
