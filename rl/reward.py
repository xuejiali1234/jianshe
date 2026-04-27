from __future__ import annotations

from typing import Any


def compute_reward(info: dict[str, Any], switch_applied: bool, weights: dict[str, float]) -> float:
    phase_stats = list(info.get("phase_stats", []))
    queue = sum(float(item.get("queue_sum", 0.0)) for item in phase_stats)
    arrival = sum(float(item.get("arrival_flow_sum", 0.0)) for item in phase_stats)
    pressure = max(
        [float(item.get("queue_sum", 0.0)) + float(item.get("arrival_flow_sum", 0.0)) for item in phase_stats]
        or [0.0]
    )
    mean_speed = _mean(float(item.get("mean_speed_mean", 0.0)) for item in phase_stats)
    throughput_proxy = max(mean_speed, 0.0)
    return float(
        -weights.get("queue", 1.0) * queue / 50.0
        -weights.get("waiting", 0.5) * queue / 80.0
        -weights.get("pressure", 0.4) * pressure / 60.0
        +weights.get("throughput", 0.3) * throughput_proxy / 15.0
        -weights.get("switch", 0.05) * (1.0 if switch_applied else 0.0)
    )


def _mean(values: Any) -> float:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else 0.0
