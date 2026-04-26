from __future__ import annotations

from typing import Any


def aggregate_predictions_by_phase(
    prediction_payload: dict[str, Any],
    movement_config: dict[str, Any] | None,
) -> dict[str, Any]:
    if not movement_config:
        return {"status": "ok", "tls": [], "message": "movement config is not available"}

    movements = prediction_payload.get("movements") or []
    if not movements:
        return {
            "status": "ok",
            "tls": [],
            "message": "latest prediction does not contain movement-level outputs",
        }

    config_by_id = {
        str(movement.get("movement_id")): movement
        for movement in movement_config.get("movements", [])
    }
    horizon = prediction_payload.get("horizon") or []
    horizon_len = len(horizon)
    grouped: dict[str, dict[int, dict[str, Any]]] = {}

    for movement in movements:
        movement_id = str(movement.get("movement_id", ""))
        meta = config_by_id.get(movement_id, {})
        tls_id = str(movement.get("tls_id") or meta.get("tls_id") or "")
        if not tls_id:
            continue

        phase_ids = [
            int(phase_id)
            for phase_id in meta.get("green_phase_ids", [])
            if _is_valid_phase_id(phase_id)
        ]
        if not phase_ids and _is_valid_phase_id(meta.get("phase_id")):
            phase_ids = [int(meta["phase_id"])]
        if not phase_ids:
            phase_ids = [-1]

        arrival = _prediction_values(movement, "pred_arrival_flow", "pred_flow", horizon_len)
        queue = _prediction_values(movement, "pred_queue_veh", "pred_queue", horizon_len)
        for phase_id in phase_ids:
            tls_group = grouped.setdefault(tls_id, {})
            phase_group = tls_group.setdefault(
                phase_id,
                {
                    "phase_id": phase_id,
                    "movement_ids": [],
                    "incoming_edges": set(),
                    "turn_type_counts": {},
                    "pred_arrival_flow_sum": [0.0] * horizon_len,
                    "pred_queue_veh_sum": [0.0] * horizon_len,
                },
            )
            phase_group["movement_ids"].append(movement_id)
            incoming_edge = movement.get("incoming_edge") or meta.get("incoming_edge")
            if incoming_edge:
                phase_group["incoming_edges"].add(str(incoming_edge))
            turn_type = str(movement.get("turn_type") or meta.get("turn_type") or "")
            if turn_type:
                phase_group["turn_type_counts"][turn_type] = (
                    phase_group["turn_type_counts"].get(turn_type, 0) + 1
                )
            for idx, value in enumerate(arrival):
                phase_group["pred_arrival_flow_sum"][idx] += float(value)
            for idx, value in enumerate(queue):
                phase_group["pred_queue_veh_sum"][idx] += float(value)

    tls_payload = []
    for tls_id, phases in sorted(grouped.items()):
        phase_payloads = []
        for phase_id, phase in sorted(phases.items()):
            phase["incoming_edges"] = sorted(phase["incoming_edges"])
            phase["movement_count"] = len(phase["movement_ids"])
            phase["horizon_summary"] = {
                f"h{steps}": _pressure_summary(
                    phase["pred_arrival_flow_sum"],
                    phase["pred_queue_veh_sum"],
                    steps,
                )
                for steps in (5, 10, 15)
            }
            phase_payloads.append(phase)
        tls_payload.append(
            {
                "tls_id": tls_id,
                "phase_count": len(phase_payloads),
                "phases": phase_payloads,
            }
        )

    return {
        "status": "ok",
        "model": prediction_payload.get("model"),
        "active_model": prediction_payload.get("active_model"),
        "history_size": prediction_payload.get("history_size"),
        "history_required": prediction_payload.get("history_required"),
        "horizon": horizon,
        "tls": tls_payload,
    }


def _prediction_values(
    movement: dict[str, Any],
    preferred_key: str,
    fallback_key: str,
    horizon_len: int,
) -> list[float]:
    raw = movement.get(preferred_key)
    if raw is None:
        raw = movement.get(fallback_key)
    values = [float(value) for value in (raw or [])]
    if len(values) < horizon_len:
        values.extend([0.0] * (horizon_len - len(values)))
    return values[:horizon_len]


def _pressure_summary(arrival: list[float], queue: list[float], steps: int) -> dict[str, float]:
    n = min(steps, len(arrival), len(queue))
    if n <= 0:
        return {"arrival_sum": 0.0, "queue_mean": 0.0, "pressure": 0.0}
    arrival_sum = float(sum(arrival[:n]))
    queue_mean = float(sum(queue[:n]) / n)
    return {
        "arrival_sum": arrival_sum,
        "queue_mean": queue_mean,
        "pressure": arrival_sum + queue_mean,
    }


def _is_valid_phase_id(value: Any) -> bool:
    try:
        return int(value) >= 0
    except (TypeError, ValueError):
        return False
