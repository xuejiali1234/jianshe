from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from sim.movement_tools import load_movement_config


QUEUE_SCALE = 20.0
ARRIVAL_SCALE = 20.0
DISCHARGE_SCALE = 20.0
SPEED_SCALE = 15.0
FEATURE_CLIP = 3.0
NEIGHBOR_QUEUE_SCALE = 40.0
NEIGHBOR_ARRIVAL_SCALE = 40.0
NEIGHBOR_PRESSURE_SCALE = 40.0


class MultiTLSStateBuilder:
    def __init__(
        self,
        movement_config_path: str | Path,
        coordination_graph_path: str | Path,
        cluster_tls_ids: list[str],
        prediction_horizons: list[int],
    ) -> None:
        payload = load_movement_config(movement_config_path)
        coordination_payload = json.loads(Path(coordination_graph_path).read_text(encoding="utf-8"))
        self.cluster_tls_ids = [str(tls_id) for tls_id in cluster_tls_ids]
        self.prediction_horizons = list(prediction_horizons)
        self.movements_by_tls: dict[str, list[dict[str, Any]]] = {}
        self.base_legal_green_phases: dict[str, list[int]] = {}
        self.legal_green_phases: dict[str, list[int]] = {}
        self.phase_to_movements: dict[str, dict[int, list[dict[str, Any]]]] = {}
        self.neighbors = coordination_payload.get("neighbors", {})

        for tls_id in self.cluster_tls_ids:
            movements = [
                movement
                for movement in payload.get("movements", [])
                if str(movement.get("tls_id", "")) == tls_id
            ]
            self.movements_by_tls[tls_id] = movements
            phases = sorted(
                {
                    int(phase_id)
                    for movement in movements
                    for phase_id in _phase_ids(movement)
                    if int(phase_id) >= 0
                }
            )
            self.base_legal_green_phases[tls_id] = phases
            self.legal_green_phases[tls_id] = list(phases)
            phase_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for movement in movements:
                for phase_id in _phase_ids(movement):
                    if phase_id >= 0:
                        phase_map[phase_id].append(movement)
            self.phase_to_movements[tls_id] = phase_map

        self.max_phase_slots = max((len(phases) for phases in self.legal_green_phases.values()), default=0)
        self.feature_names = self._build_feature_names()
        self.local_observation_size = len(self.feature_names)

    def restrict_legal_green_phases(self, tls_id: str, allowed_phase_ids: set[int]) -> None:
        tls_id = str(tls_id)
        self.legal_green_phases[tls_id] = [
            int(phase_id)
            for phase_id in self.base_legal_green_phases.get(tls_id, [])
            if int(phase_id) in allowed_phase_ids
        ]

    def build(
        self,
        traci_module: Any,
        phase_state_by_tls: dict[str, dict[str, float | int]],
        max_green_s: float,
        prediction_phase_payload_by_tls: dict[str, dict[int, dict[str, Any]]] | None = None,
        include_prediction_features: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        prediction_phase_payload_by_tls = prediction_phase_payload_by_tls or {}
        per_tls_info: dict[str, dict[str, Any]] = {}
        for tls_id in self.cluster_tls_ids:
            current_phase = int(phase_state_by_tls.get(tls_id, {}).get("current_phase", -1))
            phase_elapsed_s = float(phase_state_by_tls.get(tls_id, {}).get("phase_elapsed_s", 0.0))
            prediction_by_phase = prediction_phase_payload_by_tls.get(tls_id, {})
            prediction_available = bool(include_prediction_features and prediction_by_phase)
            phase_stats = []
            for phase_id in self.legal_green_phases.get(tls_id, []):
                movements = self.phase_to_movements.get(tls_id, {}).get(phase_id, [])
                edges = sorted(
                    {
                        str(movement.get("incoming_edge", ""))
                        for movement in movements
                        if movement.get("incoming_edge")
                    }
                )
                stats = _edge_stats(traci_module, edges)
                pred = prediction_by_phase.get(phase_id, {})
                phase_stats.append(
                    {
                        "phase_id": int(phase_id),
                        "incoming_edges": edges,
                        "queue_sum": stats["queue_sum"],
                        "arrival_flow_sum": stats["vehicle_sum"],
                        "discharge_flow_sum": 0.0,
                        "mean_speed_mean": stats["mean_speed"],
                        "predicted": _prediction_features(pred, self.prediction_horizons),
                    }
                )

            per_tls_info[tls_id] = {
                "target_tls_id": tls_id,
                "legal_green_phases": list(self.legal_green_phases.get(tls_id, [])),
                "current_phase": current_phase,
                "phase_elapsed_s": phase_elapsed_s,
                "prediction_available": prediction_available,
                "phase_stats": phase_stats,
            }

        observations = []
        action_masks: dict[str, list[int]] = {}
        for tls_id in self.cluster_tls_ids:
            neighbors = self.neighbors.get(tls_id, {})
            upstream_tls = (neighbors.get("upstream") or [None])[0]
            downstream_tls = (neighbors.get("downstream") or [None])[0]
            vector = self._vectorize_tls_info(
                tls_id,
                per_tls_info[tls_id],
                per_tls_info.get(str(upstream_tls)) if upstream_tls else None,
                per_tls_info.get(str(downstream_tls)) if downstream_tls else None,
                max_green_s,
                include_prediction_features,
            )
            observations.append(vector)
            valid_actions = 1 + len(self.legal_green_phases.get(tls_id, []))
            action_masks[tls_id] = [1 if index < valid_actions else 0 for index in range(1 + self.max_phase_slots)]

        cluster_info = {
            "tls_ids": list(self.cluster_tls_ids),
            "per_tls": per_tls_info,
            "action_masks": action_masks,
            "feature_names": list(self.feature_names),
            "local_observation_size": int(self.local_observation_size),
            "prediction_available_tls_count": sum(
                1 for item in per_tls_info.values() if bool(item.get("prediction_available"))
            ),
        }
        return np.asarray(observations, dtype=np.float32), cluster_info

    def _vectorize_tls_info(
        self,
        tls_id: str,
        tls_info: dict[str, Any],
        upstream_info: dict[str, Any] | None,
        downstream_info: dict[str, Any] | None,
        max_green_s: float,
        include_prediction_features: bool,
    ) -> list[float]:
        legal_phases = list(self.legal_green_phases.get(tls_id, []))
        phase_slots = {
            int(phase_id): index
            for index, phase_id in enumerate(legal_phases)
        }
        current_phase = int(tls_info.get("current_phase", -1))
        phase_onehot = [0.0] * self.max_phase_slots
        if current_phase in phase_slots:
            phase_onehot[phase_slots[current_phase]] = 1.0

        vector: list[float] = phase_onehot + [
            float(tls_info.get("phase_elapsed_s", 0.0)) / max(float(max_green_s), 1.0),
            1.0 if include_prediction_features and bool(tls_info.get("prediction_available")) else 0.0,
        ]
        stats_by_phase = {
            int(row.get("phase_id", -1)): row
            for row in tls_info.get("phase_stats", [])
            if int(row.get("phase_id", -1)) >= 0
        }
        for slot_index in range(self.max_phase_slots):
            if slot_index < len(legal_phases):
                phase_id = legal_phases[slot_index]
                row = stats_by_phase.get(int(phase_id), {})
                vector.extend(
                    [
                        _normalize_positive(row.get("queue_sum", 0.0), QUEUE_SCALE),
                        _normalize_positive(row.get("arrival_flow_sum", 0.0), ARRIVAL_SCALE),
                        _normalize_positive(row.get("discharge_flow_sum", 0.0), DISCHARGE_SCALE),
                        _normalize_positive(row.get("mean_speed_mean", 0.0), SPEED_SCALE, clip_value=2.0),
                    ]
                )
                for horizon in self.prediction_horizons:
                    horizon_key = f"h{horizon}"
                    predicted = row.get("predicted", {}).get(horizon_key, {})
                    if include_prediction_features:
                        vector.extend(
                            [
                                _normalize_positive(predicted.get("arrival_per_step", 0.0), ARRIVAL_SCALE),
                                _normalize_positive(predicted.get("queue_mean", 0.0), QUEUE_SCALE),
                            ]
                        )
                    else:
                        vector.extend([0.0, 0.0])
            else:
                vector.extend([0.0, 0.0, 0.0, 0.0])
                vector.extend([0.0, 0.0] * len(self.prediction_horizons))

        vector.extend(self._neighbor_summary(upstream_info, include_prediction_features))
        vector.extend(self._neighbor_summary(downstream_info, include_prediction_features))
        return vector

    def _neighbor_summary(
        self,
        neighbor_info: dict[str, Any] | None,
        include_prediction_features: bool,
    ) -> list[float]:
        if not neighbor_info:
            return [0.0] * (8)
        phase_stats = list(neighbor_info.get("phase_stats", []))
        queue_total = sum(float(item.get("queue_sum", 0.0)) for item in phase_stats)
        arrival_total = sum(float(item.get("arrival_flow_sum", 0.0)) for item in phase_stats)
        mean_speed = _mean(float(item.get("mean_speed_mean", 0.0)) for item in phase_stats)
        current_phase = int(neighbor_info.get("current_phase", -1))
        current_phase_row = next(
            (item for item in phase_stats if int(item.get("phase_id", -1)) == current_phase),
            None,
        )
        active_phase_pressure = 0.0
        pred_h_values = {f"h{horizon}": 0.0 for horizon in self.prediction_horizons}
        if current_phase_row:
            active_phase_pressure = float(current_phase_row.get("queue_sum", 0.0)) + float(
                current_phase_row.get("arrival_flow_sum", 0.0)
            )
            if include_prediction_features:
                for horizon in self.prediction_horizons:
                    horizon_key = f"h{horizon}"
                    predicted = current_phase_row.get("predicted", {}).get(horizon_key, {})
                    pred_h_values[horizon_key] = float(predicted.get("arrival_per_step", 0.0)) + float(
                        predicted.get("queue_mean", 0.0)
                    )
        return [
            1.0,
            _normalize_positive(queue_total, NEIGHBOR_QUEUE_SCALE),
            _normalize_positive(arrival_total, NEIGHBOR_ARRIVAL_SCALE),
            _normalize_positive(mean_speed, SPEED_SCALE, clip_value=2.0),
            _normalize_positive(active_phase_pressure, NEIGHBOR_PRESSURE_SCALE),
            _normalize_positive(pred_h_values.get("h5", 0.0), NEIGHBOR_PRESSURE_SCALE) if include_prediction_features else 0.0,
            _normalize_positive(pred_h_values.get("h10", 0.0), NEIGHBOR_PRESSURE_SCALE) if include_prediction_features else 0.0,
            _normalize_positive(pred_h_values.get("h15", 0.0), NEIGHBOR_PRESSURE_SCALE) if include_prediction_features else 0.0,
        ]

    def _build_feature_names(self) -> list[str]:
        names = [f"phase_slot_{slot}_active" for slot in range(self.max_phase_slots)]
        names.extend(["phase_elapsed_ratio", "prediction_available"])
        for slot in range(self.max_phase_slots):
            names.extend(
                [
                    f"phase_slot_{slot}_queue_sum",
                    f"phase_slot_{slot}_arrival_flow_sum",
                    f"phase_slot_{slot}_discharge_flow_sum",
                    f"phase_slot_{slot}_mean_speed_mean",
                ]
            )
            for horizon in self.prediction_horizons:
                names.extend(
                    [
                        f"phase_slot_{slot}_pred_arrival_per_step_h{horizon}",
                        f"phase_slot_{slot}_pred_queue_mean_h{horizon}",
                    ]
                )
        for role in ("upstream", "downstream"):
            names.extend(
                [
                    f"{role}_neighbor_present",
                    f"{role}_queue_total",
                    f"{role}_arrival_total",
                    f"{role}_mean_speed",
                    f"{role}_active_phase_pressure",
                    f"{role}_pred_pressure_h5",
                    f"{role}_pred_pressure_h10",
                    f"{role}_pred_pressure_h15",
                ]
            )
        return names


def _phase_ids(movement: dict[str, Any]) -> list[int]:
    raw = movement.get("green_phase_ids", [])
    if not raw and movement.get("phase_id") is not None:
        raw = [movement.get("phase_id")]
    result = []
    for value in raw:
        try:
            phase_id = int(value)
        except (TypeError, ValueError):
            continue
        if phase_id >= 0:
            result.append(phase_id)
    return result


def _edge_stats(traci_module: Any, edge_ids: list[str]) -> dict[str, float]:
    queue_sum = 0.0
    vehicle_sum = 0.0
    speed_values = []
    for edge_id in edge_ids:
        try:
            vehicle_sum += float(traci_module.edge.getLastStepVehicleNumber(edge_id))
            queue_sum += float(traci_module.edge.getLastStepHaltingNumber(edge_id))
            speed = float(traci_module.edge.getLastStepMeanSpeed(edge_id))
            if speed >= 0:
                speed_values.append(speed)
        except Exception:
            continue
    return {
        "queue_sum": queue_sum,
        "vehicle_sum": vehicle_sum,
        "mean_speed": float(sum(speed_values) / len(speed_values)) if speed_values else 0.0,
    }


def _prediction_features(phase_payload: dict[str, Any], horizons: list[int]) -> dict[str, dict[str, float]]:
    summary = phase_payload.get("horizon_summary", {}) if phase_payload else {}
    result = {}
    for horizon in horizons:
        bucket = summary.get(f"h{horizon}", {})
        arrival_sum = float(bucket.get("arrival_sum", 0.0) or 0.0)
        queue_mean = float(bucket.get("queue_mean", bucket.get("queue_sum", 0.0) or 0.0) or 0.0)
        result[f"h{horizon}"] = {
            "arrival_per_step": arrival_sum / max(int(horizon), 1),
            "queue_mean": queue_mean,
        }
    return result


def _normalize_positive(value: float, scale: float, clip_value: float = FEATURE_CLIP) -> float:
    safe_scale = max(float(scale), 1e-6)
    return float(min(max(float(value), 0.0) / safe_scale, clip_value))


def _mean(values: Any) -> float:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else 0.0
