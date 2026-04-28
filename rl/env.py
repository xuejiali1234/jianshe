from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from prediction.config import load_prediction_config
from prediction.movement_collector import MovementRealtimeCollector
from prediction.service import PredictionService
from sim import configure_sumo_python_path, prepare_runtime_route_file, resolve_runtime_net_file

from .phase_controller import PhaseController
from .reward import compute_reward
from .state_builder import PhaseStateBuilder

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - reported clearly by SumoSignalGymEnv.
    gym = None
    spaces = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SignalControlEnv:
    def __init__(self, config_path: str | Path, use_prediction_features: bool | None = None):
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
        self.use_prediction_features = (
            bool(self.raw_config.get("use_prediction_features", False))
            if use_prediction_features is None
            else bool(use_prediction_features)
        )
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
        self.net_green_phases = _green_phase_ids_from_net(
            self.net_file,
            self.target_tls_id,
            float(self.raw_config.get("min_action_green_duration_s", 5.0)),
        )
        if self.net_green_phases:
            self.state_builder.restrict_legal_green_phases(self.net_green_phases)
        self.observation_size = self.state_builder.observation_size
        self.action_count = 1 + len(self.state_builder.legal_green_phases)
        self.controller = PhaseController(
            self.target_tls_id,
            self.state_builder.legal_green_phases,
            float(self.raw_config.get("min_green_s", 10)),
            float(self.raw_config.get("max_green_s", 60)),
            float(self.raw_config.get("yellow_s", 3)),
            float(self.raw_config.get("all_red_s", 1)),
        )
        self.sumo_binary = self.sumolib.checkBinary("sumo")
        self.started = False
        self.last_info: dict[str, Any] = {}
        self.prediction_service: PredictionService | None = None
        self.prediction_collector: MovementRealtimeCollector | None = None
        self._prediction_snapshots = 0
        if self.use_prediction_features:
            self._init_prediction_bridge()

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
        self._reset_prediction_bridge()
        for _ in range(max(0, self.warmup_s)):
            self.traci.simulationStep()
            self._record_prediction_step()
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
        action_info = self.controller.apply_action(
            self.traci,
            int(action),
            sim_time,
            pressure_by_phase,
            step_callback=self._record_prediction_step,
        )
        remaining_steps = max(0, self.control_interval_s - int(action_info.get("transition_steps", 0)))
        for _ in range(remaining_steps):
            if self.traci.simulation.getTime() >= self.episode_s:
                break
            self.traci.simulationStep()
            self._record_prediction_step()
        observation, info = self._observe()
        info["action"] = int(action)
        info.update(action_info)
        reward = compute_reward(info, bool(action_info.get("switch_applied")), self.reward_weights)
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
            prediction_phase_payload=self._prediction_phase_payload(),
        )
        info["sim_time_s"] = sim_time
        info["vehicle_count"] = int(self.traci.vehicle.getIDCount())
        info["mean_speed_mps"] = _mean_speed(self.traci)
        info["action_count"] = self.action_count
        info["observation_size"] = self.observation_size
        info["use_prediction_features"] = self.use_prediction_features
        info["prediction_snapshots"] = self._prediction_snapshots
        info["prediction_fallback_used"] = False
        info["prediction_latency_ms"] = 0.0
        if self.prediction_service is not None:
            latest = self.prediction_service.latest_prediction or {}
            info["prediction_fallback_used"] = bool(latest.get("fallback_used"))
            info["prediction_latency_ms"] = float(latest.get("prediction_latency_ms", 0.0) or 0.0)
        info["feature_names"] = self.state_builder.feature_names
        return observation, info

    def _init_prediction_bridge(self) -> None:
        self.prediction_service = PredictionService(
            self.prediction_config,
            PROJECT_ROOT / self.prediction_config.artifact_dir,
            PROJECT_ROOT / self.prediction_config.metrics_file,
            PROJECT_ROOT / self.prediction_config.batch_csv_file,
            PROJECT_ROOT / self.prediction_config.scenario_manifest_file,
        )
        self.prediction_collector = MovementRealtimeCollector(
            self.prediction_config,
            PROJECT_ROOT / "data" / "tmp_rl" / "rl_runtime_movement_aggregates.csv",
            run_id="rl_runtime",
            scenario_id="rl_control",
            base_demand_factor=self.prediction_config.base_demand_factor,
            project_root=PROJECT_ROOT,
            net_file=self.net_file,
            movement_config_path=_project_path(self.raw_config["movement_config"]),
        )

    def _reset_prediction_bridge(self) -> None:
        self._prediction_snapshots = 0
        if self.prediction_service is not None:
            self.prediction_service.history.clear()
            self.prediction_service.latest_observation = None
            self.prediction_service.latest_prediction = self.prediction_service._attach_prediction_meta(
                self.prediction_service.fallback_predictor.predict([], self.prediction_config.horizon_steps),
                0,
            )
        if self.prediction_collector is not None:
            self.prediction_collector.interval_start_time = float(self.traci.simulation.getTime())
            self.prediction_collector._last_route_edge_by_vehicle.clear()
            self.prediction_collector._last_distance_by_vehicle_movement.clear()
            self.prediction_collector._arrival_ready.clear()
            self.prediction_collector._reset_accumulators()

    def _record_prediction_step(self) -> None:
        if self.prediction_collector is None or self.prediction_service is None:
            return
        sim_time = float(self.traci.simulation.getTime())
        snapshot = self.prediction_collector.record_step(
            self.traci,
            int(round(sim_time)),
            sim_time,
            incident_edges=set(),
        )
        if snapshot:
            self._prediction_snapshots += 1
            self.prediction_service.update_observation(snapshot)

    def _prediction_phase_payload(self) -> dict[str, Any] | None:
        if not self.use_prediction_features or self.prediction_service is None:
            return None
        payload = self.prediction_service.phase_aggregate_payload()
        tls_items = [
            item for item in payload.get("tls", [])
            if str(item.get("tls_id")) == self.target_tls_id
        ]
        if not tls_items:
            return None
        return payload


class SumoSignalGymEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(self, config_path: str | Path, sim_end: int | None = None, use_prediction_features: bool = False):
        if gym is None or spaces is None:
            raise RuntimeError(
                "Gymnasium is required for SB3 DQN training. Install it with: pip install gymnasium"
            )
        self.core_env = SignalControlEnv(config_path, use_prediction_features=use_prediction_features)
        if sim_end is not None:
            self.core_env.episode_s = int(sim_end)
        self.action_space = spaces.Discrete(self.core_env.action_count)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.core_env.observation_size,),
            dtype=np.float32,
        )
        self._last_info: dict[str, Any] = {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        observation = self.core_env.reset(seed=seed)
        self._last_info = dict(self.core_env.last_info)
        return observation.astype(np.float32), self._last_info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, done, info = self.core_env.step(int(action))
        self._last_info = dict(info)
        return observation.astype(np.float32), float(reward), bool(done), False, info

    def close(self) -> None:
        self.core_env.close()


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


def _green_phase_ids_from_net(net_file: Path, tls_id: str, min_duration_s: float = 5.0) -> set[int]:
    try:
        root = ET.parse(net_file).getroot()
    except Exception:
        return set()
    logic = root.find(f".//tlLogic[@id='{tls_id}']")
    if logic is None:
        return set()
    result = set()
    for index, phase in enumerate(logic.findall("phase")):
        state = str(phase.attrib.get("state", ""))
        duration = float(phase.attrib.get("duration", 0.0))
        if duration >= float(min_duration_s) and ("g" in state or "G" in state) and "y" not in state:
            result.add(index)
    return result


def _queue_sum(info: dict[str, Any]) -> float:
    return float(sum(float(item.get("queue_sum", 0.0)) for item in info.get("phase_stats", [])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the single-intersection SUMO RL environment.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rl_signal_config.json"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sim-end", type=int, default=None)
    parser.add_argument("--use-prediction", default="false")
    args = parser.parse_args()
    if not args.smoke_test:
        parser.error("Only --smoke-test is supported for rl.env CLI.")
    env = SumoSignalGymEnv(
        args.config,
        sim_end=args.sim_end,
        use_prediction_features=_parse_bool(args.use_prediction),
    )
    observation, info = env.reset(seed=0)
    print(f"obs_shape={observation.shape}")
    print(f"action_count={env.action_space.n}")
    print(f"legal_green_phases={info.get('legal_green_phases')}")
    try:
        for step in range(max(0, int(args.steps))):
            action = step % max(1, env.action_space.n)
            observation, reward, done, _, info = env.step(action)
            print(
                "step={step} action={action} reward={reward:.4f} queue={queue:.2f} "
                "switch={switch} phase={phase} transition_fallback={fallback} "
                "pred={pred} pred_snapshots={snapshots} pred_fallback={pred_fallback}".format(
                    step=step + 1,
                    action=action,
                    reward=reward,
                    queue=_queue_sum(info),
                    switch=int(bool(info.get("switch_applied"))),
                    phase=info.get("current_phase"),
                    fallback=int(bool(info.get("transition_fallback"))),
                    pred=int(bool(info.get("prediction_available"))),
                    snapshots=int(info.get("prediction_snapshots", 0)),
                    pred_fallback=int(bool(info.get("prediction_fallback_used"))),
                )
            )
            if done:
                break
    finally:
        env.close()


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    main()
