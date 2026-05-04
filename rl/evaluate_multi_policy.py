from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .env import PROJECT_ROOT
from .multi_dqn_shared import load_model_artifact, masked_q_values
from .multi_env import MultiSignalControlEnv


def evaluate_multi_policy(
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
    reference_csv: str | Path | None = None,
) -> dict[str, Any]:
    env = MultiSignalControlEnv(
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
            if policy.name == "webster":
                observation, reward, done, info = env.advance_without_control()
            else:
                actions = policy.act(observation, env.last_info)
                observation, reward, done, info = env.step(actions)
            total_reward += reward
            switch_count += int(info.get("switch_count", 0))
            per_tls_queue = {
                tls_id: round(
                    sum(float(item.get("queue_sum", 0.0)) for item in tls_info.get("phase_stats", [])),
                    4,
                )
                for tls_id, tls_info in info.get("per_tls", {}).items()
            }
            rows.append(
                {
                    "policy": policy.name,
                    "sim_time_s": round(float(info.get("sim_time_s", 0.0)), 3),
                    "reward": round(float(reward), 6),
                    "queue_sum": round(float(info.get("cluster_queue_sum", 0.0)), 4),
                    "vehicle_count": int(info.get("vehicle_count", 0)),
                    "mean_speed_mps": round(float(info.get("mean_speed_mps", 0.0)), 4),
                    "switch_count": int(info.get("switch_count", 0)),
                    "prediction_ready": int(bool(info.get("prediction_ready"))),
                    "prediction_snapshots": int(info.get("prediction_snapshots", 0)),
                    "scenario_run_id": str(info.get("scenario_run_id", "")),
                    "scenario_id": str(info.get("scenario_id", "")),
                    "event_type": str(info.get("event_type", "")),
                    "signal_variant": str(info.get("signal_variant", "")),
                    "reward_mode": str(info.get("reward_mode", "")),
                    "action_signature": json.dumps(info.get("action_by_tls", {}), ensure_ascii=False, sort_keys=True),
                    "per_tls_queue": json.dumps(per_tls_queue, ensure_ascii=False, sort_keys=True),
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

    event_start_s = _scenario_event_start(rows)
    pre_event_rows = [row for row in rows if _float(row.get("sim_time_s")) < event_start_s] if event_start_s > 0 else rows
    post_event_rows = [row for row in rows if _float(row.get("sim_time_s")) >= event_start_s] if event_start_s > 0 else rows
    post_event_300_rows = [
        row for row in rows
        if event_start_s <= _float(row.get("sim_time_s")) < (event_start_s + 300.0)
    ] if event_start_s > 0 else rows
    first_switch_after_event_s = ""
    if event_start_s > 0:
        for row in rows:
            if _float(row.get("sim_time_s")) >= event_start_s and int(_float(row.get("switch_count"))) > 0:
                first_switch_after_event_s = str(round(_float(row.get("sim_time_s")), 3))
                break

    per_tls_mean_queue = _mean_per_tls_queue(rows)
    summary = {
        "policy": policy.name,
        "steps": len(rows),
        "total_reward": round(float(total_reward), 6),
        "mean_reward": round(float(total_reward / max(len(rows), 1)), 6),
        "switch_count": int(switch_count),
        "mean_queue": round(sum(_float(row.get("queue_sum")) for row in rows) / max(len(rows), 1), 4) if rows else 0.0,
        "max_queue": round(max((_float(row.get("queue_sum")) for row in rows), default=0.0), 4),
        "mean_speed_kmh": round(
            sum(_float(row.get("mean_speed_mps")) for row in rows) / max(len(rows), 1) * 3.6,
            4,
        ) if rows else 0.0,
        "prediction_ready_share": round(
            sum(int(_float(row.get("prediction_ready"))) for row in rows) / max(len(rows), 1),
            4,
        ) if rows else 0.0,
        "pre_event_mean_queue": round(sum(_float(row.get("queue_sum")) for row in pre_event_rows) / max(len(pre_event_rows), 1), 4)
        if pre_event_rows else 0.0,
        "post_event_mean_queue": round(sum(_float(row.get("queue_sum")) for row in post_event_rows) / max(len(post_event_rows), 1), 4)
        if post_event_rows else 0.0,
        "post_event_first_300s_mean_queue": round(
            sum(_float(row.get("queue_sum")) for row in post_event_300_rows) / max(len(post_event_300_rows), 1),
            4,
        ) if post_event_300_rows else 0.0,
        "first_switch_after_event_s": first_switch_after_event_s,
        "per_tls_mean_queue": per_tls_mean_queue,
        "reward_mode": str(rows[0].get("reward_mode", "")) if rows else "",
        "out": str(destination),
    }
    if reference_csv:
        summary["matches_max_pressure_stepwise"] = int(_matches_reference(rows, reference_csv))
    return summary


class WebsterMultiPolicy:
    name = "webster"

    def act(self, observation: Any, info: dict[str, Any]) -> dict[str, int]:
        return {tls_id: 0 for tls_id in info.get("tls_ids", [])}


class MaxPressureMultiPolicy:
    name = "max_pressure"

    def act(self, observation: Any, info: dict[str, Any]) -> dict[str, int]:
        actions = {}
        per_tls = info.get("per_tls", {})
        for tls_id in info.get("tls_ids", []):
            tls_info = per_tls.get(tls_id, {})
            phases = list(tls_info.get("legal_green_phases", []))
            stats = list(tls_info.get("phase_stats", []))
            if not phases or not stats:
                actions[tls_id] = 0
                continue
            pressure_by_phase = {
                int(item.get("phase_id")): float(item.get("queue_sum", 0.0)) + float(item.get("arrival_flow_sum", 0.0))
                for item in stats
            }
            best_phase = max(phases, key=lambda phase: pressure_by_phase.get(int(phase), 0.0))
            try:
                actions[tls_id] = phases.index(best_phase) + 1
            except ValueError:
                actions[tls_id] = 0
        return actions


class LoadedMultiDqnPolicy:
    name = "dqn_multi"

    def __init__(self, model_path: str | Path):
        self.model, payload = load_model_artifact(model_path, device="cpu")
        self.model.eval()
        self.cluster_tls_ids = list(payload.get("cluster_tls_ids", []))

    def act(self, observation: Any, info: dict[str, Any]) -> dict[str, int]:
        obs_tensor = torch.as_tensor(np.asarray(observation), dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(obs_tensor)
        action_masks = info.get("action_masks", {})
        actions = {}
        for index, tls_id in enumerate(info.get("tls_ids", self.cluster_tls_ids)):
            mask = torch.as_tensor(np.asarray(action_masks.get(tls_id, []), dtype=np.float32)).unsqueeze(0)
            masked = masked_q_values(q_values[index:index + 1], mask).squeeze(0)
            actions[tls_id] = int(torch.argmax(masked).item())
        return actions


def _load_policy(policy_name: str, model_path: str | Path | None):
    normalized = (policy_name or "").strip().lower()
    if normalized == "webster":
        return WebsterMultiPolicy()
    if normalized == "max_pressure":
        return MaxPressureMultiPolicy()
    if normalized == "dqn":
        if not model_path:
            raise ValueError("--model-path is required when --policy dqn")
        return LoadedMultiDqnPolicy(model_path)
    raise ValueError(f"unknown multi policy: {policy_name}")


def _scenario_event_start(rows: list[dict[str, Any]]) -> float:
    for row in rows:
        event_type = str(row.get("event_type", "")).strip()
        if event_type:
            return 1200.0
    return 0.0


def _mean_per_tls_queue(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        payload = json.loads(str(row.get("per_tls_queue", "{}")))
        for tls_id, value in payload.items():
            totals[str(tls_id)] = totals.get(str(tls_id), 0.0) + float(value)
            counts[str(tls_id)] = counts.get(str(tls_id), 0) + 1
    return {
        tls_id: round(total / max(counts.get(tls_id, 1), 1), 4)
        for tls_id, total in totals.items()
    }


def _matches_reference(rows: list[dict[str, Any]], reference_csv: str | Path) -> bool:
    source = Path(reference_csv)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    if not source.exists():
        return False
    with source.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        reference_rows = list(reader)
    if len(reference_rows) != len(rows):
        return False
    for row, ref in zip(rows, reference_rows):
        if str(row.get("action_signature", "")) != str(ref.get("action_signature", "")):
            return False
    return True


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a multi-intersection control policy.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rl_multi_signal_config_v1.json"))
    parser.add_argument("--policy", choices=["webster", "max_pressure", "dqn"], default="webster")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--sim-end", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-prediction", default="false")
    parser.add_argument("--use-prediction-reward", default="")
    parser.add_argument("--reward-mode", default="")
    parser.add_argument("--scenario-run-id", default="")
    parser.add_argument("--reference-csv", default=None)
    args = parser.parse_args()
    summary = evaluate_multi_policy(
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
        args.reference_csv,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
