from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .baselines import make_policy
from .env import SignalControlEnv, PROJECT_ROOT


def evaluate_policy(
    config_path: str | Path,
    policy_name: str,
    sim_end: int | None,
    out_path: str | Path,
    seed: int | None = None,
    model_path: str | Path | None = None,
    use_prediction: bool = False,
    use_prediction_reward: bool | None = None,
    reward_mode: str | None = None,
    scenario_run_id: str | None = None,
) -> dict[str, float | int | str]:
    env = SignalControlEnv(
        config_path,
        use_prediction_features=use_prediction,
        use_prediction_reward=use_prediction_reward,
        reward_mode=reward_mode,
        scenario_run_id=scenario_run_id if scenario_run_id is not None else "",
    )
    if sim_end is not None:
        env.episode_s = int(sim_end)
    policy = _load_policy(policy_name, model_path)
    rows = []
    total_reward = 0.0
    switch_count = 0
    observation = env.reset(seed=seed)
    done = False
    try:
        while not done:
            action = policy.act(observation, env.last_info)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            switch_count += 1 if info.get("switch_applied") else 0
            queue_sum = sum(float(item.get("queue_sum", 0.0)) for item in info.get("phase_stats", []))
            arrival_sum = sum(float(item.get("arrival_flow_sum", 0.0)) for item in info.get("phase_stats", []))
            throughput_proxy = max(float(info.get("mean_speed_mps", 0.0)), 0.0) * max(int(info.get("vehicle_count", 0)), 0)
            prediction_available = bool(info.get("prediction_available"))
            prediction_fallback_used = bool(info.get("prediction_fallback_used"))
            rows.append(
                {
                    "policy": policy.name,
                    "sim_time_s": round(float(info.get("sim_time_s", 0.0)), 3),
                    "reward": round(float(reward), 6),
                    "queue_sum": round(queue_sum, 4),
                    "arrival_flow_sum": round(arrival_sum, 4),
                    "throughput_proxy": round(float(throughput_proxy), 4),
                    "vehicle_count": int(info.get("vehicle_count", 0)),
                    "mean_speed_mps": round(float(info.get("mean_speed_mps", 0.0)), 4),
                    "switch_applied": int(bool(info.get("switch_applied"))),
                    "current_phase": int(info.get("current_phase", -1)),
                    "transition_fallback": int(bool(info.get("transition_fallback"))),
                    "transition_program_mismatch": int(bool(info.get("transition_program_mismatch"))),
                    "prediction_available": int(prediction_available),
                    "prediction_ready": int(prediction_available and not prediction_fallback_used),
                    "prediction_snapshots": int(info.get("prediction_snapshots", 0)),
                    "prediction_fallback_used": int(prediction_fallback_used),
                    "prediction_latency_ms": round(float(info.get("prediction_latency_ms", 0.0)), 3),
                }
            )
    finally:
        env.close()

    destination = Path(out_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else ["policy"])
        writer.writeheader()
        writer.writerows(rows)

    ready_rows = [row for row in rows if int(row.get("prediction_ready", 0)) == 1]
    event_start_s = _float(env.scenario_meta.get("incident_start_s")) if env.scenario_meta else 0.0
    pre_event_rows = [row for row in rows if _float(row.get("sim_time_s")) < event_start_s] if event_start_s > 0 else rows
    post_event_rows = [row for row in rows if _float(row.get("sim_time_s")) >= event_start_s] if event_start_s > 0 else rows
    post_event_300_rows = [
        row for row in rows
        if event_start_s <= _float(row.get("sim_time_s")) < (event_start_s + 300.0)
    ] if event_start_s > 0 else rows
    first_switch_after_event_s = ""
    if event_start_s > 0:
        for row in rows:
            if _float(row.get("sim_time_s")) >= event_start_s and int(_float(row.get("switch_applied"))) == 1:
                first_switch_after_event_s = str(round(_float(row.get("sim_time_s")), 3))
                break
    summary = {
        "policy": policy.name,
        "steps": len(rows),
        "total_reward": round(float(total_reward), 6),
        "mean_reward": round(float(total_reward / max(len(rows), 1)), 6),
        "switch_count": switch_count,
        "mean_queue": round(sum(row["queue_sum"] for row in rows) / max(len(rows), 1), 4) if rows else 0.0,
        "mean_speed_mps": round(sum(row["mean_speed_mps"] for row in rows) / max(len(rows), 1), 4) if rows else 0.0,
        "mean_throughput_proxy": round(sum(row["throughput_proxy"] for row in rows) / max(len(rows), 1), 4) if rows else 0.0,
        "use_prediction": int(bool(use_prediction)),
        "scenario_run_id": str(scenario_run_id or ""),
        "prediction_available_share": round(
            sum(row.get("prediction_available", 0) for row in rows) / max(len(rows), 1),
            4,
        ) if rows else 0.0,
        "prediction_fallback_share": round(
            sum(row.get("prediction_fallback_used", 0) for row in rows) / max(len(rows), 1),
            4,
        ) if rows else 0.0,
        "prediction_ready_steps": len(ready_rows),
        "prediction_ready_mean_queue": round(
            sum(row["queue_sum"] for row in ready_rows) / max(len(ready_rows), 1),
            4,
        ) if ready_rows else 0.0,
        "prediction_ready_mean_reward": round(
            sum(row["reward"] for row in ready_rows) / max(len(ready_rows), 1),
            6,
        ) if ready_rows else 0.0,
        "prediction_ready_mean_speed_kmh": round(
            sum(row["mean_speed_mps"] for row in ready_rows) / max(len(ready_rows), 1) * 3.6,
            4,
        ) if ready_rows else 0.0,
        "pre_event_mean_queue": round(sum(_float(row.get("queue_sum")) for row in pre_event_rows) / max(len(pre_event_rows), 1), 4)
        if pre_event_rows else 0.0,
        "post_event_mean_queue": round(sum(_float(row.get("queue_sum")) for row in post_event_rows) / max(len(post_event_rows), 1), 4)
        if post_event_rows else 0.0,
        "post_event_first_300s_mean_queue": round(
            sum(_float(row.get("queue_sum")) for row in post_event_300_rows) / max(len(post_event_300_rows), 1),
            4,
        ) if post_event_300_rows else 0.0,
        "first_switch_after_event_s": first_switch_after_event_s,
        "reward_mode": str(env.reward_mode),
        "use_prediction_reward": int(bool(env.use_prediction_reward)),
        "out": str(destination),
    }
    return summary


class LoadedDqnPolicy:
    name = "dqn"

    def __init__(self, model_path: str | Path):
        try:
            from stable_baselines3 import DQN
        except ImportError as exc:
            raise RuntimeError("stable-baselines3 is required to evaluate DQN policies.") from exc
        path = Path(model_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"DQN model not found: {path}")
        self.model = DQN.load(str(path))

    def act(self, observation: Any, info: dict[str, Any]) -> int:
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)


def _load_policy(policy_name: str, model_path: str | Path | None):
    normalized = (policy_name or "").strip().lower()
    if normalized == "dqn":
        if not model_path:
            raise ValueError("--model-path is required when --policy dqn")
        return LoadedDqnPolicy(model_path)
    return make_policy(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a single-intersection signal control policy.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rl_signal_config.json"))
    parser.add_argument("--policy", choices=["webster", "max_pressure", "dqn"], default="webster")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--sim-end", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-prediction", default="false")
    parser.add_argument("--use-prediction-reward", default="")
    parser.add_argument("--reward-mode", default="")
    parser.add_argument("--scenario-run-id", default="")
    args = parser.parse_args()
    summary = evaluate_policy(
        args.config,
        args.policy,
        args.sim_end,
        args.out,
        args.seed,
        args.model_path,
        _parse_bool(args.use_prediction),
        (
            _parse_bool(args.use_prediction_reward)
            if str(args.use_prediction_reward).strip() != ""
            else None
        ),
        (args.reward_mode or "").strip() or None,
        (args.scenario_run_id or "").strip() or None,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
