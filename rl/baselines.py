from __future__ import annotations

from typing import Any


class WebsterPolicy:
    name = "webster"

    def act(self, observation: Any, info: dict[str, Any]) -> int:
        return 0


class MaxPressurePolicy:
    name = "max_pressure"

    def act(self, observation: Any, info: dict[str, Any]) -> int:
        phases = list(info.get("legal_green_phases", []))
        stats = list(info.get("phase_stats", []))
        if not phases or not stats:
            return 0
        pressure_by_phase = {
            int(item.get("phase_id")): float(item.get("queue_sum", 0.0)) + float(item.get("arrival_flow_sum", 0.0))
            for item in stats
        }
        best_phase = max(phases, key=lambda phase: pressure_by_phase.get(int(phase), 0.0))
        try:
            return phases.index(best_phase) + 1
        except ValueError:
            return 0


def make_policy(name: str):
    normalized = (name or "").strip().lower()
    if normalized == "webster":
        return WebsterPolicy()
    if normalized == "max_pressure":
        return MaxPressurePolicy()
    raise ValueError(f"Unknown RL baseline policy: {name}")
