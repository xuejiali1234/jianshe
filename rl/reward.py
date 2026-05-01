from __future__ import annotations

from typing import Any

FUTURE_PHASE_WEIGHTS = {
    "h5": 0.6,
    "h10": 0.3,
    "h15": 0.1,
}
FUTURE_PHASE_WEIGHTS_SHORT = {
    "h5": 0.75,
    "h10": 0.25,
    "h15": 0.0,
}
FUTURE_PRESSURE_SCALE = 20.0


def compute_reward(
    info: dict[str, Any],
    switch_applied: bool,
    weights: dict[str, float],
    reward_mode: str = "current_pressure_v1",
    use_prediction_reward: bool = False,
    prediction_by_phase: dict[int, dict[str, Any]] | None = None,
) -> tuple[float, dict[str, float | int | str]]:
    phase_stats = list(info.get("phase_stats", []))
    queue = sum(float(item.get("queue_sum", 0.0)) for item in phase_stats)
    pressure = max(
        [float(item.get("queue_sum", 0.0)) + float(item.get("arrival_flow_sum", 0.0)) for item in phase_stats]
        or [0.0]
    )
    mean_speed = _mean(float(item.get("mean_speed_mean", 0.0)) for item in phase_stats)
    throughput_proxy = max(mean_speed, 0.0)
    base_reward = float(
        -weights.get("queue", 1.0) * queue / 50.0
        -weights.get("waiting", 0.5) * queue / 80.0
        -weights.get("pressure", 0.4) * pressure / 60.0
        +weights.get("throughput", 0.3) * throughput_proxy / 15.0
        -weights.get("switch", 0.05) * (1.0 if switch_applied else 0.0)
    )
    reward = base_reward
    current_phase_future = 0.0
    future_peak = 0.0
    anticipation_gap = 0.0
    future_bonus = 0.0
    prediction_reward_used = 0

    if reward_mode == "anticipatory_phase_pressure_v1" and use_prediction_reward:
        current_phase = int(info.get("current_phase", -1))
        future_by_phase = _future_pressure_by_phase(phase_stats, prediction_by_phase)
        if current_phase >= 0 and future_by_phase:
            current_phase_future = float(future_by_phase.get(current_phase, 0.0))
            future_peak = max(future_by_phase.values()) if future_by_phase else 0.0
            anticipation_gap = current_phase_future - future_peak
            future_bonus = float(
                weights.get("future_align", 0.25) * current_phase_future / FUTURE_PRESSURE_SCALE
                - weights.get("future_peak", 0.15) * future_peak / FUTURE_PRESSURE_SCALE
                + weights.get("future_gap", 0.20) * anticipation_gap / FUTURE_PRESSURE_SCALE
            )
            reward = base_reward + future_bonus
            prediction_reward_used = 1
    elif reward_mode == "anticipatory_delta_pressure_v2" and use_prediction_reward:
        current_phase = int(info.get("current_phase", -1))
        current_pressure_by_phase = _current_pressure_by_phase(phase_stats)
        future_abs_by_phase = _future_pressure_by_phase(phase_stats, prediction_by_phase)
        future_by_phase = {
            phase_id: max(float(future_abs_by_phase.get(phase_id, 0.0)) - float(current_pressure_by_phase.get(phase_id, 0.0)), 0.0)
            for phase_id in future_abs_by_phase
        }
        if current_phase >= 0 and future_by_phase:
            current_phase_future = float(future_by_phase.get(current_phase, 0.0))
            future_peak = max(future_by_phase.values()) if future_by_phase else 0.0
            anticipation_gap = current_phase_future - future_peak
            future_bonus = float(
                weights.get("future_align", 0.12) * current_phase_future / FUTURE_PRESSURE_SCALE
                - weights.get("future_peak", 0.06) * future_peak / FUTURE_PRESSURE_SCALE
                + weights.get("future_gap", 0.08) * anticipation_gap / FUTURE_PRESSURE_SCALE
            )
            reward = base_reward + future_bonus
            prediction_reward_used = 1
    elif reward_mode == "anticipatory_short_peak_v4" and use_prediction_reward:
        current_phase = int(info.get("current_phase", -1))
        future_by_phase = _future_pressure_by_phase(
            phase_stats,
            prediction_by_phase,
            horizon_weights=FUTURE_PHASE_WEIGHTS_SHORT,
        )
        if current_phase >= 0 and future_by_phase:
            current_phase_future = float(future_by_phase.get(current_phase, 0.0))
            future_peak = max(future_by_phase.values()) if future_by_phase else 0.0
            anticipation_gap = max(future_peak - current_phase_future, 0.0)
            peak_ratio = current_phase_future / max(future_peak, 1e-6) if future_peak > 0 else 0.0
            future_bonus = float(
                weights.get("future_align", 0.18) * peak_ratio
                - weights.get("future_gap", 0.12) * anticipation_gap / FUTURE_PRESSURE_SCALE
            )
            reward = base_reward + future_bonus
            prediction_reward_used = 1

    meta: dict[str, float | int | str] = {
        "reward_mode": str(reward_mode),
        "prediction_reward_enabled": int(bool(use_prediction_reward)),
        "prediction_reward_used": int(prediction_reward_used),
        "reward_base": float(base_reward),
        "reward_future_bonus": float(future_bonus),
        "future_peak": float(future_peak),
        "current_phase_future": float(current_phase_future),
        "anticipation_gap": float(anticipation_gap),
    }
    return float(reward), meta


def _mean(values: Any) -> float:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else 0.0


def _future_pressure_by_phase(
    phase_stats: list[dict[str, Any]],
    prediction_by_phase: dict[int, dict[str, Any]] | None,
    horizon_weights: dict[str, float] | None = None,
) -> dict[int, float]:
    future_by_phase: dict[int, float] = {}
    if not prediction_by_phase:
        return future_by_phase
    weights = horizon_weights or FUTURE_PHASE_WEIGHTS
    for row in phase_stats:
        phase_id = int(row.get("phase_id", -1))
        if phase_id < 0:
            continue
        phase_payload = prediction_by_phase.get(phase_id, {})
        horizon_summary = dict(phase_payload.get("horizon_summary", {}))
        score = 0.0
        for horizon_key, horizon_weight in weights.items():
            if horizon_weight <= 0:
                continue
            bucket = dict(horizon_summary.get(horizon_key, {}))
            arrival = float(bucket.get("arrival_sum", 0.0) or 0.0)
            horizon = max(int(horizon_key.lstrip("h") or 0), 1)
            arrival_per_step = arrival / horizon
            queue_mean = float(bucket.get("queue_mean", bucket.get("queue_sum", 0.0) or 0.0) or 0.0)
            score += horizon_weight * (arrival_per_step + queue_mean)
        future_by_phase[phase_id] = float(score)
    return future_by_phase


def _current_pressure_by_phase(phase_stats: list[dict[str, Any]]) -> dict[int, float]:
    current_by_phase: dict[int, float] = {}
    for row in phase_stats:
        phase_id = int(row.get("phase_id", -1))
        if phase_id < 0:
            continue
        current_by_phase[phase_id] = float(row.get("queue_sum", 0.0)) + float(row.get("arrival_flow_sum", 0.0))
    return current_by_phase
