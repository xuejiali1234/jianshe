from __future__ import annotations

from collections.abc import Callable
from typing import Any


class PhaseController:
    def __init__(
        self,
        tls_id: str,
        legal_green_phases: list[int],
        min_green_s: float,
        max_green_s: float,
        yellow_s: float = 3.0,
        all_red_s: float = 1.0,
    ):
        self.tls_id = str(tls_id)
        self.legal_green_phases = list(legal_green_phases)
        self.min_green_s = float(min_green_s)
        self.max_green_s = float(max_green_s)
        self.yellow_s = float(yellow_s)
        self.all_red_s = float(all_red_s)
        self.last_phase = -1
        self.phase_started_s = 0.0

    def reset(self, traci_module: Any, sim_time_s: float) -> None:
        self.last_phase = int(traci_module.trafficlight.getPhase(self.tls_id))
        self.phase_started_s = float(sim_time_s)

    def phase_elapsed(self, traci_module: Any, sim_time_s: float) -> float:
        current = int(traci_module.trafficlight.getPhase(self.tls_id))
        if current != self.last_phase:
            self.last_phase = current
            self.phase_started_s = float(sim_time_s)
        return max(0.0, float(sim_time_s) - self.phase_started_s)

    def apply_action(
        self,
        traci_module: Any,
        action: int,
        sim_time_s: float,
        pressure_by_phase: dict[int, float],
        step_callback: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        current = int(traci_module.trafficlight.getPhase(self.tls_id))
        elapsed = self.phase_elapsed(traci_module, sim_time_s)
        target = self._action_to_phase(action, current)
        info = {
            "requested_phase": int(target),
            "current_phase_before": int(current),
            "switch_applied": False,
            "forced_switch": False,
            "transition_fallback": False,
            "transition_phases": [],
            "transition_steps": 0,
        }
        forced = False
        if elapsed >= self.max_green_s:
            candidates = [phase for phase in self.legal_green_phases if phase != current]
            if candidates:
                target = max(candidates, key=lambda phase: pressure_by_phase.get(phase, 0.0))
                forced = True
                info["requested_phase"] = int(target)
                info["forced_switch"] = True
        if target == current:
            return info
        if not forced and elapsed < self.min_green_s:
            return info
        transition = self._run_clearance_transition(traci_module, current, target, step_callback)
        info.update(transition)
        traci_module.trafficlight.setPhase(self.tls_id, int(target))
        self.last_phase = int(target)
        self.phase_started_s = float(traci_module.simulation.getTime())
        info["switch_applied"] = True
        info["current_phase_after"] = int(target)
        return info

    def _action_to_phase(self, action: int, current_phase: int) -> int:
        if int(action) <= 0:
            return int(current_phase)
        index = int(action) - 1
        if 0 <= index < len(self.legal_green_phases):
            return int(self.legal_green_phases[index])
        return int(current_phase)

    def _run_clearance_transition(
        self,
        traci_module: Any,
        current_phase: int,
        target_phase: int,
        step_callback: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        phases = _program_phases(traci_module, self.tls_id)
        transition_phases = _clearance_phases_after(phases, current_phase, self.legal_green_phases)
        info = {
            "transition_fallback": False,
            "transition_phases": [],
            "transition_steps": 0,
        }
        if not transition_phases:
            info["transition_fallback"] = True
            return info

        for phase_id, state, default_duration in transition_phases:
            duration = self.yellow_s if _is_yellow_state(state) else self.all_red_s
            if duration <= 0:
                duration = max(1.0, float(default_duration))
            traci_module.trafficlight.setPhase(self.tls_id, int(phase_id))
            steps = max(1, int(round(duration)))
            for _ in range(steps):
                traci_module.simulationStep()
                if step_callback:
                    step_callback()
            info["transition_phases"].append(int(phase_id))
            info["transition_steps"] += steps

        if target_phase not in _next_green_targets(phases, current_phase, self.legal_green_phases):
            info["transition_fallback"] = True
        return info


def _program_phases(traci_module: Any, tls_id: str) -> list[tuple[int, str, float]]:
    try:
        logics = traci_module.trafficlight.getAllProgramLogics(tls_id)
    except Exception:
        return []
    if not logics:
        return []
    phases = []
    for index, phase in enumerate(getattr(logics[0], "phases", []) or []):
        phases.append((index, str(getattr(phase, "state", "")), float(getattr(phase, "duration", 0.0))))
    return phases


def _clearance_phases_after(
    phases: list[tuple[int, str, float]],
    current_phase: int,
    legal_green_phases: list[int],
) -> list[tuple[int, str, float]]:
    if not phases:
        return []
    legal = set(int(phase) for phase in legal_green_phases)
    result = []
    phase_count = len(phases)
    for offset in range(1, phase_count + 1):
        phase_id, state, duration = phases[(int(current_phase) + offset) % phase_count]
        if phase_id in legal:
            break
        if _is_yellow_state(state) or _is_all_red_state(state) or result:
            result.append((phase_id, state, duration))
        if _is_all_red_state(state):
            break
    return result


def _next_green_targets(
    phases: list[tuple[int, str, float]],
    current_phase: int,
    legal_green_phases: list[int],
) -> set[int]:
    legal = set(int(phase) for phase in legal_green_phases)
    targets = set()
    phase_count = len(phases)
    for offset in range(1, phase_count + 1):
        phase_id = phases[(int(current_phase) + offset) % phase_count][0]
        if phase_id in legal:
            targets.add(int(phase_id))
            break
    return targets


def _is_yellow_state(state: str) -> bool:
    return "y" in state


def _is_all_red_state(state: str) -> bool:
    chars = set(state)
    return bool(chars) and chars.issubset({"r", "R"})
