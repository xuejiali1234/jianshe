from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from prediction.config import load_prediction_config
from sim import configure_sumo_python_path, prepare_runtime_route_file, resolve_runtime_net_file

from .phase_controller import PhaseController
from .reward import compute_reward
from .state_builder import PhaseStateBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SignalControlEnv:
    def __init__(self, config_path: str | Path):
        configure_sumo_python_path()
        import sumolib
        import traci

        self.traci = traci
        self.sumolib = sumolib
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = PROJECT_ROOT / self.config_path
        self.raw_config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.prediction_config = load_prediction_config(PROJECT_ROOT / self.raw_config["prediction_config"])
        self.target_tls_id = str(self.raw_config["target_tls_id"])
        self.control_interval_s = int(self.raw_config.get("control_interval_s", 10))
        self.warmup_s = int(self.raw_config.get("warmup_s", 300))
        self.episode_s = int(self.raw_config.get("episode_s", 3600))
        self.reward_weights = dict(self.raw_config.get("reward_weights", {}))
        self.net_file = _project_path(self.raw_config["net_file"])
        self.route_file = prepare_runtime_route_file(
            _project_path(self.raw_config["route_file"]),
            PROJECT_ROOT / "data" / "raw" / "runtime_routes",
            scale_factor=float(self.prediction_config.base_demand_factor),
            output_name="rl_runtime.rou.xml",
        )
        self.state_builder = PhaseStateBuilder(
            _project_path(self.raw_config["movement_config"]),
            self.target_tls_id,
            list(self.raw_config.get("prediction_horizons", [5, 10, 15])),
        )
        self.controller = PhaseController(
            self.target_tls_id,
            self.state_builder.legal_green_phases,
            float(self.raw_config.get("min_green_s", 10)),
            float(self.raw_config.get("max_green_s", 60)),
        )
        self.sumo_binary = self.sumolib.checkBinary("sumo")
        self.started = False
        self.last_info: dict[str, Any] = {}

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.close()
        cmd = [
            self.sumo_binary,
            "-n",
            str(self.net_file),
            "-r",
            str(self.route_file),
            "--begin",
            "0",
            "--end",
            str(self.episode_s),
            "--no-warnings",
            "--ignore-route-errors",
            "--time-to-teleport",
            "15",
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
        ]
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        self.traci.start(cmd)
        self.started = True
        for _ in range(max(0, self.warmup_s)):
            self.traci.simulationStep()
        self.controller.reset(self.traci, self.traci.simulation.getTime())
        observation, info = self._observe()
        self.last_info = info
        return observation

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        sim_time = float(self.traci.simulation.getTime())
        pressure_by_phase = {
            int(item.get("phase_id")): float(item.get("queue_sum", 0.0)) + float(item.get("arrival_flow_sum", 0.0))
            for item in self.last_info.get("phase_stats", [])
        }
        switched = self.controller.apply_action(self.traci, int(action), sim_time, pressure_by_phase)
        for _ in range(max(1, self.control_interval_s)):
            if self.traci.simulation.getTime() >= self.episode_s:
                break
            self.traci.simulationStep()
        observation, info = self._observe()
        info["switch_applied"] = switched
        reward = compute_reward(info, switched, self.reward_weights)
        done = float(self.traci.simulation.getTime()) >= float(self.episode_s)
        self.last_info = info
        return observation, reward, done, info

    def close(self) -> None:
        if not self.started:
            return
        try:
            self.traci.close()
        except Exception:
            pass
        self.started = False

    def _observe(self) -> tuple[np.ndarray, dict[str, Any]]:
        sim_time = float(self.traci.simulation.getTime())
        current_phase = int(self.traci.trafficlight.getPhase(self.target_tls_id))
        elapsed = self.controller.phase_elapsed(self.traci, sim_time)
        observation, info = self.state_builder.build(
            self.traci,
            current_phase,
            elapsed,
            float(self.raw_config.get("max_green_s", 60)),
            prediction_phase_payload=None,
        )
        info["sim_time_s"] = sim_time
        info["vehicle_count"] = int(self.traci.vehicle.getIDCount())
        info["mean_speed_mps"] = _mean_speed(self.traci)
        return observation, info


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _mean_speed(traci_module: Any) -> float:
    vehicle_ids = list(traci_module.vehicle.getIDList())
    if not vehicle_ids:
        return 0.0
    speeds = []
    for vehicle_id in vehicle_ids:
        try:
            speeds.append(float(traci_module.vehicle.getSpeed(vehicle_id)))
        except Exception:
            continue
    return float(sum(speeds) / len(speeds)) if speeds else 0.0
