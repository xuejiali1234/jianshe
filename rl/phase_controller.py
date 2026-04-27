from __future__ import annotations

from typing import Any


class PhaseController:
    def __init__(
        self,
        tls_id: str,
        legal_green_phases: list[int],
        min_green_s: float,
        max_green_s: float,
    ):
        self.tls_id = str(tls_id)
        self.legal_green_phases = list(legal_green_phases)
        self.min_green_s = float(min_green_s)
        self.max_green_s = float(max_green_s)
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

    def apply_action(self, traci_module: Any, action: int, sim_time_s: float, pressure_by_phase: dict[int, float]) -> bool:
        current = int(traci_module.trafficlight.getPhase(self.tls_id))
        elapsed = self.phase_elapsed(traci_module, sim_time_s)
        target = self._action_to_phase(action, current)
        forced = False
        if elapsed >= self.max_green_s:
            candidates = [phase for phase in self.legal_green_phases if phase != current]
            if candidates:
                target = max(candidates, key=lambda phase: pressure_by_phase.get(phase, 0.0))
                forced = True
        if target == current:
            return False
        if not forced and elapsed < self.min_green_s:
            return False
        traci_module.trafficlight.setPhase(self.tls_id, int(target))
        self.last_phase = int(target)
        self.phase_started_s = float(sim_time_s)
        return True

    def _action_to_phase(self, action: int, current_phase: int) -> int:
        if int(action) <= 0:
            return int(current_phase)
        index = int(action) - 1
        if 0 <= index < len(self.legal_green_phases):
            return int(self.legal_green_phases[index])
        return int(current_phase)
