from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from sim.movement_tools import load_movement_config


class PhaseStateBuilder:
    def __init__(self, movement_config_path: str | Path, target_tls_id: str, prediction_horizons: list[int]):
        payload = load_movement_config(movement_config_path)
        self.target_tls_id = str(target_tls_id)
        self.prediction_horizons = list(prediction_horizons)
        self.movements = [
            movement
            for movement in payload.get("movements", [])
            if str(movement.get("tls_id", "")) == self.target_tls_id
        ]
        self.legal_green_phases = sorted(
            {
                int(phase_id)
                for movement in self.movements
                for phase_id in _phase_ids(movement)
                if int(phase_id) >= 0
            }
        )
        self.phase_to_movements: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for movement in self.movements:
            for phase_id in _phase_ids(movement):
                if phase_id >= 0:
                    self.phase_to_movements[phase_id].append(movement)

    def build(
        self,
        traci_module: Any,
        current_phase: int,
        phase_elapsed_s: float,
        max_green_s: float,
        prediction_phase_payload: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        phase_stats = []
        prediction_available = bool(prediction_phase_payload)
        prediction_by_phase = _prediction_by_phase(prediction_phase_payload, self.target_tls_id)
        for phase_id in self.legal_green_phases:
            movements = self.phase_to_movements.get(phase_id, [])
            edges = sorted({str(movement.get("incoming_edge", "")) for movement in movements if movement.get("incoming_edge")})
            stats = _edge_stats(traci_module, edges)
            pred = prediction_by_phase.get(phase_id, {})
            row = {
                "phase_id": phase_id,
                "incoming_edges": edges,
                "queue_sum": stats["queue_sum"],
                "arrival_flow_sum": stats["vehicle_sum"],
                "discharge_flow_sum": 0.0,
                "mean_speed_mean": stats["mean_speed"],
                "predicted": _prediction_features(pred, self.prediction_horizons),
            }
            phase_stats.append(row)

        phase_onehot = [1.0 if phase_id == current_phase else 0.0 for phase_id in self.legal_green_phases]
        vector: list[float] = phase_onehot + [
            float(phase_elapsed_s) / max(float(max_green_s), 1.0),
            1.0 if prediction_available else 0.0,
        ]
        for row in phase_stats:
            vector.extend(
                [
                    row["queue_sum"],
                    row["arrival_flow_sum"],
                    row["discharge_flow_sum"],
                    row["mean_speed_mean"],
                ]
            )
            for horizon in self.prediction_horizons:
                horizon_key = f"h{horizon}"
                vector.extend(
                    [
                        row["predicted"].get(horizon_key, {}).get("arrival_pressure", 0.0),
                        row["predicted"].get(horizon_key, {}).get("queue_pressure", 0.0),
                    ]
                )

        info = {
            "target_tls_id": self.target_tls_id,
            "legal_green_phases": self.legal_green_phases,
            "current_phase": current_phase,
            "phase_elapsed_s": float(phase_elapsed_s),
            "prediction_available": prediction_available,
            "phase_stats": phase_stats,
        }
        return np.asarray(vector, dtype=np.float32), info


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


def _prediction_by_phase(payload: dict[str, Any] | None, target_tls_id: str) -> dict[int, dict[str, Any]]:
    if not payload:
        return {}
    for tls in payload.get("tls", []):
        if str(tls.get("tls_id", "")) != str(target_tls_id):
            continue
        return {
            int(phase.get("phase_id")): phase
            for phase in tls.get("phases", [])
            if str(phase.get("phase_id", "")).lstrip("-").isdigit()
        }
    return {}


def _prediction_features(phase_payload: dict[str, Any], horizons: list[int]) -> dict[str, dict[str, float]]:
    summary = phase_payload.get("horizon_summary", {}) if phase_payload else {}
    return {
        f"h{horizon}": {
            "arrival_pressure": float(summary.get(f"h{horizon}", {}).get("arrival_sum", 0.0)),
            "queue_pressure": float(summary.get(f"h{horizon}", {}).get("queue_sum", 0.0)),
        }
        for horizon in horizons
    }
