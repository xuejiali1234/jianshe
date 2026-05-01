from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from prediction.config import load_prediction_config
from prediction.movement_collector import MovementRealtimeCollector
from prediction.service import PredictionService
from sim import configure_sumo_python_path, prepare_runtime_route_file
from sim.scripts.build_tls_coordination_graph import build_tls_coordination_graph

from .env import (
    DEFAULT_VSL_SPEED_FACTOR,
    EVENT_VCLASS_BLOCKLIST,
    PROJECT_ROOT,
    _green_phase_ids_from_net,
    _mean_speed,
    _project_path,
    _safe_float,
)
from .multi_reward import compute_multi_tls_rewards
from .multi_state_builder import MultiTLSStateBuilder
from .phase_controller import PhaseController


class MultiSignalControlEnv:
    def __init__(
        self,
        config_path: str | Path,
        use_prediction_features: bool | None = None,
        use_prediction_reward: bool | None = None,
        reward_mode: str | None = None,
        scenario_run_id: str | None = None,
    ) -> None:
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
        self.cluster_tls_ids = [str(tls_id) for tls_id in self.raw_config["cluster_tls_ids"]]
        self.control_interval_s = int(self.raw_config.get("control_interval_s", 10))
        self.warmup_s = int(self.raw_config.get("warmup_s", 300))
        self.episode_s = int(self.raw_config.get("episode_s", 3600))
        self.reward_weights = dict(self.raw_config.get("reward_weights", {}))
        self.reward_mode = str(reward_mode or self.raw_config.get("reward_mode", "current_pressure_v1"))
        self.use_prediction_features = (
            bool(self.raw_config.get("use_prediction_features", False))
            if use_prediction_features is None
            else bool(use_prediction_features)
        )
        self.use_prediction_reward = (
            bool(self.raw_config.get("use_prediction_reward", False))
            if use_prediction_reward is None
            else bool(use_prediction_reward)
        )
        self.reference_single_tls_id = str(self.raw_config.get("reference_single_tls_id", ""))
        self.scenario_run_id = (scenario_run_id or str(self.raw_config.get("scenario_run_id", ""))).strip()
        self.scenario_manifest_path = _project_path(self.prediction_config.scenario_manifest_file)
        self.scenario_meta: dict[str, str] | None = None

        self.runtime_net_file = _project_path(self.raw_config["net_file"])
        self.runtime_route_template = _project_path(self.raw_config["route_file"])
        self.net_file = self.runtime_net_file
        self.route_file = self.runtime_route_template

        self.movement_config_path = _project_path(self.raw_config["movement_config"])
        self.coordination_graph_path = _project_path(self.raw_config["coordination_graph"])
        if not self.coordination_graph_path.exists():
            build_tls_coordination_graph(
                movement_config_path=self.movement_config_path,
                net_file=self.runtime_net_file,
                out_path=self.coordination_graph_path,
                tls_ids=self.cluster_tls_ids,
            )

        self.state_builder = MultiTLSStateBuilder(
            self.movement_config_path,
            self.coordination_graph_path,
            self.cluster_tls_ids,
            list(self.raw_config.get("prediction_horizons", [5, 10, 15])),
        )
        self.reference_net_file = _project_path(self.raw_config["net_file"])
        self.reference_green_phases = {
            tls_id: _green_phase_ids_from_net(
                self.reference_net_file,
                tls_id,
                float(self.raw_config.get("min_action_green_duration_s", 5.0)),
            )
            for tls_id in self.cluster_tls_ids
        }
        self.net_green_phases = dict(self.reference_green_phases)
        self.controllers: dict[str, PhaseController] = {}
        self.max_action_count = 1 + self.state_builder.max_phase_slots
        self.local_observation_size = self.state_builder.local_observation_size
        self.sumo_binary = self.sumolib.checkBinary("sumo")
        self.started = False
        self.last_info: dict[str, Any] = {}
        self.prediction_service: PredictionService | None = None
        self.prediction_collector: MovementRealtimeCollector | None = None
        self._prediction_snapshots = 0
        self._scenario_event_applied = False
        self._scenario_event_edges: set[str] = set()
        self._scenario_baseline_speeds: dict[str, float] = {}
        self._scenario_baseline_lane_disallowed: dict[str, tuple[str, ...]] = {}
        self._configure_active_scenario(self.scenario_run_id)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.close()
        self._configure_active_scenario(self.scenario_run_id)
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
        self._init_scenario_event_state()
        self._reset_prediction_bridge()
        for _ in range(max(0, self.warmup_s)):
            self.traci.simulationStep()
            incident_edges = self._apply_scenario_event_controls(float(self.traci.simulation.getTime()))
            self._record_prediction_step(incident_edges)
        current_time = float(self.traci.simulation.getTime())
        for tls_id, controller in self.controllers.items():
            controller.reset(self.traci, current_time)
        observation, info = self._observe()
        self.last_info = info
        return observation

    def step(self, actions: dict[str, int] | list[int] | np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action_by_tls = self._normalize_actions(actions)
        sim_time = float(self.traci.simulation.getTime())
        decisions: dict[str, dict[str, Any]] = {}
        transition_schedules: dict[str, list[int]] = {}
        switch_executed: dict[str, bool] = {}

        for tls_id in self.cluster_tls_ids:
            controller = self.controllers[tls_id]
            current_phase = int(self.traci.trafficlight.getPhase(tls_id))
            elapsed = controller.phase_elapsed(self.traci, sim_time)
            last_tls_info = dict(self.last_info.get("per_tls", {}).get(tls_id, {}))
            pressure_by_phase = {
                int(item.get("phase_id")): float(item.get("queue_sum", 0.0)) + float(item.get("arrival_flow_sum", 0.0))
                for item in last_tls_info.get("phase_stats", [])
            }
            decision = controller.decide_action(
                int(action_by_tls.get(tls_id, 0)),
                current_phase,
                elapsed,
                pressure_by_phase,
            )
            decisions[tls_id] = decision
            if bool(decision.get("switch_requested")):
                transition = controller.build_transition_schedule(
                    self.traci,
                    current_phase,
                    int(decision.get("requested_phase", current_phase)),
                )
                decisions[tls_id].update(
                    {
                        "transition_fallback": bool(transition.get("transition_fallback")),
                        "transition_program_mismatch": bool(transition.get("transition_program_mismatch")),
                        "transition_phases": list(transition.get("transition_phases", [])),
                        "transition_steps": int(transition.get("transition_steps", 0)),
                    }
                )
                transition_schedules[tls_id] = list(transition.get("transition_schedule", []))
                switch_executed[tls_id] = not bool(transition.get("transition_fallback"))
            else:
                switch_executed[tls_id] = False

        max_transition_steps = max((len(schedule) for schedule in transition_schedules.values()), default=0)
        for substep in range(max_transition_steps):
            for tls_id, schedule in transition_schedules.items():
                if substep < len(schedule):
                    self.traci.trafficlight.setPhase(tls_id, int(schedule[substep]))
            self.traci.simulationStep()
            incident_edges = self._apply_scenario_event_controls(float(self.traci.simulation.getTime()))
            self._record_prediction_step(incident_edges)

        current_time = float(self.traci.simulation.getTime())
        for tls_id, executed in switch_executed.items():
            if not executed:
                continue
            target_phase = int(decisions[tls_id]["requested_phase"])
            self.traci.trafficlight.setPhase(tls_id, target_phase)
            controller = self.controllers[tls_id]
            controller.last_phase = target_phase
            controller.phase_started_s = current_time
            decisions[tls_id]["switch_applied"] = True
            decisions[tls_id]["current_phase_after"] = target_phase

        remaining_steps = max(0, self.control_interval_s - max_transition_steps)
        for _ in range(remaining_steps):
            if self.traci.simulation.getTime() >= self.episode_s:
                break
            self.traci.simulationStep()
            incident_edges = self._apply_scenario_event_controls(float(self.traci.simulation.getTime()))
            self._record_prediction_step(incident_edges)

        observation, info = self._observe()
        prediction_by_tls = self._reward_prediction_by_tls()
        rewards_by_tls, reward_meta_by_tls, cluster_reward_meta = compute_multi_tls_rewards(
            info["per_tls"],
            self.state_builder.neighbors,
            switch_executed,
            self.reward_weights,
            reward_mode=self.reward_mode,
            use_prediction_reward=self.use_prediction_reward,
            prediction_by_tls=prediction_by_tls,
        )
        for tls_id in self.cluster_tls_ids:
            tls_info = info["per_tls"][tls_id]
            tls_info["action"] = int(action_by_tls[tls_id])
            tls_info.update(decisions.get(tls_id, {}))
            tls_info["reward"] = float(rewards_by_tls.get(tls_id, 0.0))
            tls_info.update(reward_meta_by_tls.get(tls_id, {}))
        info["cluster_reward_mean"] = float(cluster_reward_meta["mean_reward"])
        info["mean_coordination_penalty"] = float(cluster_reward_meta["mean_coordination_penalty"])
        info["reward_mode"] = self.reward_mode
        info["prediction_reward_enabled"] = bool(self.use_prediction_reward)
        info["action_by_tls"] = {tls_id: int(action_by_tls[tls_id]) for tls_id in self.cluster_tls_ids}
        info["switch_count"] = sum(1 for value in switch_executed.values() if value)
        info["per_tls_reward"] = {tls_id: float(rewards_by_tls[tls_id]) for tls_id in self.cluster_tls_ids}
        done = float(self.traci.simulation.getTime()) >= float(self.episode_s)
        self.last_info = info
        return observation, float(cluster_reward_meta["mean_reward"]), done, info

    def close(self) -> None:
        if not self.started:
            return
        try:
            self._restore_scenario_event_controls()
            self.traci.close()
        except Exception:
            pass
        self.started = False

    def _observe(self) -> tuple[np.ndarray, dict[str, Any]]:
        sim_time = float(self.traci.simulation.getTime())
        phase_state_by_tls = {}
        for tls_id, controller in self.controllers.items():
            current_phase = int(self.traci.trafficlight.getPhase(tls_id))
            elapsed = controller.phase_elapsed(self.traci, sim_time)
            phase_state_by_tls[tls_id] = {
                "current_phase": current_phase,
                "phase_elapsed_s": elapsed,
            }
        prediction_payload_by_tls = self._prediction_phase_payload_by_tls(include_prediction_features=self.use_prediction_features)
        observation, cluster_info = self.state_builder.build(
            self.traci,
            phase_state_by_tls,
            float(self.raw_config.get("max_green_s", 60)),
            prediction_phase_payload_by_tls=prediction_payload_by_tls,
            include_prediction_features=self.use_prediction_features,
        )
        info: dict[str, Any] = {
            "tls_ids": list(self.cluster_tls_ids),
            "per_tls": cluster_info["per_tls"],
            "action_masks": cluster_info["action_masks"],
            "feature_names": cluster_info["feature_names"],
            "local_observation_size": cluster_info["local_observation_size"],
            "sim_time_s": sim_time,
            "vehicle_count": int(self.traci.vehicle.getIDCount()),
            "mean_speed_mps": _mean_speed(self.traci),
            "scenario_run_id": self.scenario_run_id,
            "scenario_id": self.scenario_meta.get("scenario_id", "") if self.scenario_meta else "",
            "event_type": self.scenario_meta.get("event_type", "") if self.scenario_meta else "",
            "signal_variant": self.scenario_meta.get("signal_variant", "") if self.scenario_meta else "",
            "prediction_snapshots": int(self._prediction_snapshots),
            "prediction_latency_ms": 0.0,
            "prediction_fallback_used": False,
            "prediction_ready": False,
            "prediction_available_tls_count": cluster_info["prediction_available_tls_count"],
            "max_action_count": int(self.max_action_count),
        }
        if self.prediction_service is not None:
            latest = self.prediction_service.latest_prediction or {}
            info["prediction_fallback_used"] = bool(latest.get("fallback_used"))
            info["prediction_latency_ms"] = float(latest.get("prediction_latency_ms", 0.0) or 0.0)
        info["prediction_ready"] = bool(info["prediction_available_tls_count"]) and not bool(info["prediction_fallback_used"])
        info["cluster_queue_sum"] = float(
            sum(
                sum(float(item.get("queue_sum", 0.0)) for item in tls_info.get("phase_stats", []))
                for tls_info in info["per_tls"].values()
            )
        )
        return observation, info

    def _normalize_actions(self, actions: dict[str, int] | list[int] | np.ndarray) -> dict[str, int]:
        if isinstance(actions, dict):
            return {tls_id: int(actions.get(tls_id, 0)) for tls_id in self.cluster_tls_ids}
        action_list = list(actions)
        return {
            tls_id: int(action_list[index]) if index < len(action_list) else 0
            for index, tls_id in enumerate(self.cluster_tls_ids)
        }

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
            PROJECT_ROOT / "data" / "tmp_rl_multi" / "rl_multi_runtime_movement_aggregates.csv",
            run_id="rl_multi_runtime",
            scenario_id="rl_multi_control",
            base_demand_factor=self.prediction_config.base_demand_factor,
            project_root=PROJECT_ROOT,
            net_file=self.net_file,
            movement_config_path=self.movement_config_path,
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

    def _record_prediction_step(self, incident_edges: set[str] | None = None) -> None:
        if self.prediction_collector is None or self.prediction_service is None:
            return
        sim_time = float(self.traci.simulation.getTime())
        snapshot = self.prediction_collector.record_step(
            self.traci,
            int(round(sim_time)),
            sim_time,
            incident_edges=set(incident_edges or set()),
        )
        if snapshot:
            self._prediction_snapshots += 1
            self.prediction_service.update_observation(snapshot)

    def _prediction_phase_payload_by_tls(self, include_prediction_features: bool) -> dict[str, dict[int, dict[str, Any]]]:
        if not include_prediction_features or self.prediction_service is None:
            return {}
        latest = self.prediction_service.latest_prediction or {}
        if bool(latest.get("fallback_used")):
            return {}
        payload = self.prediction_service.phase_aggregate_payload()
        result: dict[str, dict[int, dict[str, Any]]] = {}
        for tls in payload.get("tls", []):
            tls_id = str(tls.get("tls_id", ""))
            if tls_id not in self.cluster_tls_ids:
                continue
            result[tls_id] = {
                int(phase.get("phase_id")): phase
                for phase in tls.get("phases", [])
                if str(phase.get("phase_id", "")).lstrip("-").isdigit()
            }
        return result

    def _reward_prediction_by_tls(self) -> dict[str, dict[int, dict[str, Any]]]:
        if not self.use_prediction_reward or self.prediction_service is None:
            return {}
        latest = self.prediction_service.latest_prediction or {}
        if bool(latest.get("fallback_used")):
            return {}
        payload = self.prediction_service.phase_aggregate_payload()
        result: dict[str, dict[int, dict[str, Any]]] = {}
        for tls in payload.get("tls", []):
            tls_id = str(tls.get("tls_id", ""))
            if tls_id not in self.cluster_tls_ids:
                continue
            result[tls_id] = {
                int(phase.get("phase_id")): phase
                for phase in tls.get("phases", [])
                if str(phase.get("phase_id", "")).lstrip("-").isdigit()
            }
        return result

    def _load_scenario_meta(self, run_id: str) -> dict[str, str]:
        if not run_id:
            return {}
        if not self.scenario_manifest_path.exists():
            raise FileNotFoundError(f"scenario manifest not found: {self.scenario_manifest_path}")
        with self.scenario_manifest_path.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                if str(row.get("run_id", "")).strip() == str(run_id).strip():
                    return {key: (value or "") for key, value in row.items()}
        raise ValueError(f"scenario run_id not found in manifest: {run_id}")

    def _configure_active_scenario(self, run_id: str) -> None:
        self.scenario_run_id = (run_id or "").strip()
        self.scenario_meta = self._load_scenario_meta(self.scenario_run_id) if self.scenario_run_id else None
        if self.scenario_meta:
            self.net_file = _project_path(self.scenario_meta.get("net_file", self.raw_config["net_file"]))
            self.route_file = _project_path(self.scenario_meta.get("route_file", self.raw_config["route_file"]))
        else:
            self.net_file = self.runtime_net_file
            self.route_file = prepare_runtime_route_file(
                self.runtime_route_template,
                PROJECT_ROOT / "data" / "raw" / "runtime_routes",
                scale_factor=float(self.prediction_config.base_demand_factor),
                output_name="rl_multi_runtime.rou.xml",
            )

        self.net_green_phases = {
            tls_id: _green_phase_ids_from_net(
                self.net_file,
                tls_id,
                float(self.raw_config.get("min_action_green_duration_s", 5.0)),
            )
            for tls_id in self.cluster_tls_ids
        }
        for tls_id in self.cluster_tls_ids:
            allowed = set(self.net_green_phases.get(tls_id, set()))
            if self.scenario_meta and self.reference_green_phases.get(tls_id):
                allowed = allowed & set(self.reference_green_phases[tls_id])
            if allowed:
                self.state_builder.restrict_legal_green_phases(tls_id, allowed)
            else:
                self.state_builder.restrict_legal_green_phases(
                    tls_id,
                    set(self.reference_green_phases.get(tls_id, []))
                    or set(self.state_builder.base_legal_green_phases.get(tls_id, [])),
                )
        self.controllers = {
            tls_id: PhaseController(
                tls_id,
                self.state_builder.legal_green_phases.get(tls_id, []),
                float(self.raw_config.get("min_green_s", 10)),
                float(self.raw_config.get("max_green_s", 60)),
                float(self.raw_config.get("yellow_s", 3)),
                float(self.raw_config.get("all_red_s", 1)),
            )
            for tls_id in self.cluster_tls_ids
        }
        self.max_action_count = 1 + self.state_builder.max_phase_slots
        self.local_observation_size = self.state_builder.local_observation_size
        if self.use_prediction_features or self.use_prediction_reward:
            self._init_prediction_bridge()
        else:
            self.prediction_service = None
            self.prediction_collector = None

    def _init_scenario_event_state(self) -> None:
        self._scenario_event_applied = False
        self._scenario_event_edges = set()
        self._scenario_baseline_speeds = {}
        self._scenario_baseline_lane_disallowed = {}
        if not self.scenario_meta:
            return
        event_type = str(self.scenario_meta.get("event_type", "")).strip()
        if not event_type:
            return
        net_obj = self.sumolib.net.readNet(str(self.net_file))
        affected_edges = self._scenario_affected_edges()
        self._scenario_baseline_speeds = {
            edge_id: float(net_obj.getEdge(edge_id).getSpeed())
            for edge_id in affected_edges
            if self._net_has_edge(net_obj, edge_id)
        }
        if event_type == "incident_closure":
            self._scenario_baseline_lane_disallowed = self._collect_lane_disallowed_baseline(net_obj, affected_edges)

    def _apply_scenario_event_controls(self, sim_time_s: float) -> set[str]:
        if not self.scenario_meta:
            return set()
        event_type = str(self.scenario_meta.get("event_type", "")).strip()
        if not event_type:
            return set()
        start_s = _safe_float(self.scenario_meta.get("incident_start_s"), 0.0)
        end_s = _safe_float(self.scenario_meta.get("incident_end_s"), 0.0)
        affected_edges = self._scenario_affected_edges()
        event_active = bool(start_s <= sim_time_s <= end_s)
        if event_active and not self._scenario_event_applied:
            if event_type == "vsl_speed_drop":
                speed_factor = _safe_float(self.scenario_meta.get("speed_factor"), DEFAULT_VSL_SPEED_FACTOR)
                for edge_id, base_speed in self._scenario_baseline_speeds.items():
                    self.traci.edge.setMaxSpeed(edge_id, max(0.1, base_speed * max(speed_factor, 0.0)))
            elif event_type == "incident_closure":
                for lane_id, disallowed in self._scenario_baseline_lane_disallowed.items():
                    merged = sorted(set(disallowed) | set(EVENT_VCLASS_BLOCKLIST))
                    self.traci.lane.setDisallowed(lane_id, merged)
            self._scenario_event_applied = True
        elif not event_active and self._scenario_event_applied:
            self._restore_scenario_event_controls()
        self._scenario_event_edges = set(affected_edges) if event_active else set()
        return set(self._scenario_event_edges)

    def _restore_scenario_event_controls(self) -> None:
        if not self._scenario_event_applied:
            return
        event_type = str(self.scenario_meta.get("event_type", "")).strip() if self.scenario_meta else ""
        try:
            if event_type == "vsl_speed_drop":
                for edge_id, base_speed in self._scenario_baseline_speeds.items():
                    self.traci.edge.setMaxSpeed(edge_id, base_speed)
            elif event_type == "incident_closure":
                for lane_id, disallowed in self._scenario_baseline_lane_disallowed.items():
                    self.traci.lane.setDisallowed(lane_id, list(disallowed))
        finally:
            self._scenario_event_applied = False
            self._scenario_event_edges = set()

    def _scenario_affected_edges(self) -> list[str]:
        if not self.scenario_meta:
            return []
        return [
            edge_id
            for edge_id in str(self.scenario_meta.get("affected_edges", "")).split("|")
            if edge_id
        ]

    def _collect_lane_disallowed_baseline(
        self,
        net_obj: Any,
        affected_edges: list[str],
    ) -> dict[str, tuple[str, ...]]:
        lane_disallowed: dict[str, tuple[str, ...]] = {}
        for edge_id in affected_edges:
            if not self._net_has_edge(net_obj, edge_id):
                continue
            edge = net_obj.getEdge(edge_id)
            for lane in edge.getLanes():
                lane_id = lane.getID()
                try:
                    lane_disallowed[lane_id] = tuple(self.traci.lane.getDisallowed(lane_id))
                except Exception:
                    lane_disallowed[lane_id] = tuple()
        return lane_disallowed

    @staticmethod
    def _net_has_edge(net_obj: Any, edge_id: str) -> bool:
        try:
            net_obj.getEdge(edge_id)
            return True
        except Exception:
            return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the multi-intersection SUMO RL environment.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rl_multi_signal_config_v1.json"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sim-end", type=int, default=None)
    parser.add_argument("--use-prediction", default="false")
    parser.add_argument("--use-prediction-reward", default="")
    parser.add_argument("--reward-mode", default="")
    parser.add_argument("--scenario-run-id", default="")
    args = parser.parse_args()
    if not args.smoke_test:
        parser.error("Only --smoke-test is supported for rl.multi_env CLI.")
    env = MultiSignalControlEnv(
        args.config,
        use_prediction_features=_parse_bool(args.use_prediction),
        use_prediction_reward=(
            _parse_bool(args.use_prediction_reward)
            if str(args.use_prediction_reward).strip() != ""
            else None
        ),
        reward_mode=(args.reward_mode or "").strip() or None,
        scenario_run_id=(args.scenario_run_id or "").strip() or None,
    )
    if args.sim_end is not None:
        env.episode_s = int(args.sim_end)
    observation = env.reset(seed=42)
    print(f"obs_shape={observation.shape} tls_ids={env.cluster_tls_ids} local_obs={env.local_observation_size} max_actions={env.max_action_count}")
    try:
        for step in range(max(1, args.steps)):
            action_masks = env.last_info.get("action_masks", {})
            actions = {
                tls_id: 0 if not any(mask[1:] for mask in [action_masks.get(tls_id, [])]) else 0
                for tls_id in env.cluster_tls_ids
            }
            observation, reward, done, info = env.step(actions)
            print(
                f"step={step} reward={reward:.4f} queue={info.get('cluster_queue_sum', 0.0):.2f} "
                f"switches={info.get('switch_count', 0)} pred_ready={int(info.get('prediction_ready', False))}"
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
