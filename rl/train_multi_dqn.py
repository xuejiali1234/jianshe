from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .env import PROJECT_ROOT
from .multi_dqn_shared import (
    SharedMultiTLSQNetwork,
    masked_q_values,
    save_model_artifact,
    select_actions,
)
from .multi_env import MultiSignalControlEnv


@dataclass
class ReplayTransition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: float
    action_mask: np.ndarray
    next_action_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.items: deque[ReplayTransition] = deque(maxlen=self.capacity)

    def add(self, transition: ReplayTransition) -> None:
        self.items.append(transition)

    def sample(self, batch_size: int, rng: random.Random) -> list[ReplayTransition]:
        indices = rng.sample(range(len(self.items)), int(batch_size))
        return [self.items[index] for index in indices]

    def __len__(self) -> int:
        return len(self.items)


def train_multi_dqn(
    config_path: str | Path,
    timesteps: int,
    seed: int,
    use_prediction: bool,
    use_prediction_reward: bool | None,
    reward_mode: str | None,
    sim_end: int | None,
    out_dir: str | Path,
    smoke_test: bool,
    device: str,
    report_dir: str | Path | None = None,
    run_name: str | None = None,
    checkpoint_every: int = 0,
    scenario_run_id: str | None = None,
) -> dict[str, Any]:
    config_file = _project_path(config_path)
    raw_config = json.loads(config_file.read_text(encoding="utf-8"))
    train_config = dict(raw_config.get("train", {}))
    output_dir = _project_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_root = _project_path(report_dir) if report_dir else PROJECT_ROOT / "reports" / "rl_signal_control"
    report_root.mkdir(parents=True, exist_ok=True)
    suffix = _safe_name(run_name or ("pred_v1" if use_prediction else "no_pred"))
    log_path = report_root / f"dqn_training_log_{suffix}.csv"
    summary_path = output_dir / f"dqn_training_summary_{suffix}.json"
    checkpoint_dir = output_dir / "checkpoints" / suffix
    if int(checkpoint_every) > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = MultiSignalControlEnv(
        config_file,
        use_prediction_features=use_prediction,
        use_prediction_reward=use_prediction_reward,
        reward_mode=reward_mode,
        scenario_run_id=scenario_run_id,
    )
    if sim_end is not None:
        env.episode_s = int(sim_end)

    rng = random.Random(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    hidden_sizes = tuple(int(item) for item in train_config.get("hidden_sizes", [128, 128]))
    observation_size = int(env.local_observation_size)
    action_count = int(env.max_action_count)
    tls_ids = list(env.cluster_tls_ids)

    model = SharedMultiTLSQNetwork(observation_size, action_count, hidden_sizes).to(torch_device)
    target_model = SharedMultiTLSQNetwork(observation_size, action_count, hidden_sizes).to(torch_device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config.get("learning_rate", 5e-4)))

    gamma = float(train_config.get("gamma", 0.99))
    batch_size = int(train_config.get("batch_size", 64))
    buffer_size = int(train_config.get("buffer_size", 50000))
    learning_starts = int(train_config.get("learning_starts", 500))
    train_freq = int(train_config.get("train_freq", 4))
    target_update_interval = int(train_config.get("target_update_interval", 500))
    epsilon_start = float(train_config.get("epsilon_start", 1.0))
    epsilon_end = float(train_config.get("epsilon_end", 0.05))
    epsilon_decay_steps = int(train_config.get("epsilon_decay_steps", 10000))
    grad_clip = float(train_config.get("grad_clip", 1.0))
    if smoke_test:
        batch_size = min(batch_size, 32)
        buffer_size = min(buffer_size, 2000)
        learning_starts = min(learning_starts, 32)

    replay = ReplayBuffer(buffer_size)
    observation = env.reset(seed=seed)
    log_rows = []
    losses: list[float] = []

    for step in range(1, int(timesteps) + 1):
        epsilon = _epsilon(step, epsilon_start, epsilon_end, epsilon_decay_steps)
        action_masks = dict(env.last_info.get("action_masks", {}))
        actions = select_actions(model, observation, action_masks, tls_ids, epsilon, torch_device, rng)
        next_observation, reward_mean, done, info = env.step(actions)
        next_action_masks = dict(info.get("action_masks", {}))

        for index, tls_id in enumerate(tls_ids):
            replay.add(
                ReplayTransition(
                    obs=np.asarray(observation[index], dtype=np.float32),
                    action=int(actions[tls_id]),
                    reward=float(info.get("per_tls_reward", {}).get(tls_id, reward_mean)),
                    next_obs=np.asarray(next_observation[index], dtype=np.float32),
                    done=1.0 if done else 0.0,
                    action_mask=np.asarray(action_masks.get(tls_id, [1] * action_count), dtype=np.float32),
                    next_action_mask=np.asarray(next_action_masks.get(tls_id, [1] * action_count), dtype=np.float32),
                )
            )

        loss_value = None
        if len(replay) >= max(batch_size, learning_starts) and step % max(train_freq, 1) == 0:
            batch = replay.sample(batch_size, rng)
            loss_value = _optimize_batch(
                model,
                target_model,
                optimizer,
                batch,
                gamma,
                torch_device,
                grad_clip,
            )
            losses.append(float(loss_value))
        if step % max(target_update_interval, 1) == 0:
            target_model.load_state_dict(model.state_dict())

        log_rows.append(
            {
                "timestep": int(step),
                "reward": round(float(reward_mean), 6),
                "queue_sum": round(float(info.get("cluster_queue_sum", 0.0)), 4),
                "vehicle_count": int(info.get("vehicle_count", 0)),
                "mean_speed_mps": round(float(info.get("mean_speed_mps", 0.0)), 4),
                "switch_count": int(info.get("switch_count", 0)),
                "prediction_ready": int(bool(info.get("prediction_ready"))),
                "prediction_snapshots": int(info.get("prediction_snapshots", 0)),
                "prediction_latency_ms": round(float(info.get("prediction_latency_ms", 0.0)), 3),
                "scenario_run_id": str(info.get("scenario_run_id", "")),
                "scenario_id": str(info.get("scenario_id", "")),
                "event_type": str(info.get("event_type", "")),
                "signal_variant": str(info.get("signal_variant", "")),
                "reward_mode": str(info.get("reward_mode", "")),
                "mean_coordination_penalty": round(float(info.get("mean_coordination_penalty", 0.0)), 6),
                "epsilon": round(float(epsilon), 6),
                "loss": round(float(loss_value), 6) if loss_value is not None else "",
            }
        )

        observation = next_observation
        if done:
            observation = env.reset(seed=seed + step)

        if int(checkpoint_every) > 0 and step % int(checkpoint_every) == 0:
            checkpoint_path = checkpoint_dir / f"dqn_multi_tls_{suffix}_{step}_steps.pt"
            save_model_artifact(
                checkpoint_path,
                model,
                observation_size,
                action_count,
                tls_ids,
                hidden_sizes,
                {
                    "timestep": int(step),
                    "config": str(config_file),
                    "run_name": suffix,
                    "use_prediction_features": bool(use_prediction),
                    "use_prediction_reward": bool(
                        raw_config.get("use_prediction_reward", False)
                        if use_prediction_reward is None
                        else use_prediction_reward
                    ),
                    "reward_mode": str(reward_mode or raw_config.get("reward_mode", "current_pressure_v1")),
                },
            )

    env.close()

    model_path = output_dir / f"dqn_multi_tls_{suffix}.pt"
    save_model_artifact(
        model_path,
        model,
        observation_size,
        action_count,
        tls_ids,
        hidden_sizes,
        {
            "timesteps": int(timesteps),
            "config": str(config_file),
            "run_name": suffix,
            "use_prediction_features": bool(use_prediction),
            "use_prediction_reward": bool(
                raw_config.get("use_prediction_reward", False)
                if use_prediction_reward is None
                else use_prediction_reward
            ),
            "reward_mode": str(reward_mode or raw_config.get("reward_mode", "current_pressure_v1")),
            "scenario_run_id": str(scenario_run_id or ""),
            "mean_training_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        },
    )
    _write_training_log(log_path, log_rows)
    summary = {
        "status": "ok",
        "algorithm": "shared_multi_tls_dqn",
        "cluster_tls_ids": tls_ids,
        "use_prediction": bool(use_prediction),
        "use_prediction_reward": bool(
            raw_config.get("use_prediction_reward", False)
            if use_prediction_reward is None
            else use_prediction_reward
        ),
        "reward_mode": str(reward_mode or raw_config.get("reward_mode", "current_pressure_v1")),
        "timesteps": int(timesteps),
        "seed": int(seed),
        "sim_end": sim_end,
        "scenario_run_id": str(scenario_run_id or ""),
        "model_path": str(model_path),
        "training_log": str(log_path),
        "checkpoint_dir": str(checkpoint_dir) if int(checkpoint_every) > 0 else "",
        "observation_size": int(observation_size),
        "action_count": int(action_count),
        "mean_training_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        "note": "Smoke-test run" if smoke_test else "Training run",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _optimize_batch(
    model: SharedMultiTLSQNetwork,
    target_model: SharedMultiTLSQNetwork,
    optimizer: torch.optim.Optimizer,
    batch: list[ReplayTransition],
    gamma: float,
    device: torch.device,
    grad_clip: float,
) -> float:
    obs = torch.as_tensor(np.stack([item.obs for item in batch]), dtype=torch.float32, device=device)
    actions = torch.as_tensor([item.action for item in batch], dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(np.stack([item.next_obs for item in batch]), dtype=torch.float32, device=device)
    dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=device)
    action_mask = torch.as_tensor(np.stack([item.action_mask for item in batch]), dtype=torch.float32, device=device)
    next_action_mask = torch.as_tensor(np.stack([item.next_action_mask for item in batch]), dtype=torch.float32, device=device)

    q_values = masked_q_values(model(obs), action_mask)
    current_q = q_values.gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q_values = masked_q_values(target_model(next_obs), next_action_mask)
        next_q = torch.max(next_q_values, dim=1).values
        target = rewards + (1.0 - dones) * gamma * next_q

    loss = F.smooth_l1_loss(current_q, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
    optimizer.step()
    return float(loss.item())


def _write_training_log(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else ["timestep"])
        writer.writeheader()
        writer.writerows(rows)


def _epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return float(end)
    ratio = min(max(int(step), 0) / float(decay_steps), 1.0)
    return float(start + (end - start) * ratio)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _safe_name(value: str) -> str:
    cleaned = []
    for char in str(value).strip():
        cleaned.append(char if char.isalnum() or char in {"_", "-"} else "_")
    result = "".join(cleaned).strip("_")
    return result or "run"


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a shared multi-TLS DQN policy.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rl_multi_signal_config_v1.json"))
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-prediction", default="false")
    parser.add_argument("--use-prediction-reward", default="")
    parser.add_argument("--reward-mode", default="")
    parser.add_argument("--sim-end", type=int, default=None)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "models" / "artifacts_rl_multi"))
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "reports" / "rl_signal_control_multi"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--scenario-run-id", default="")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    summary = train_multi_dqn(
        args.config,
        args.timesteps,
        args.seed,
        _parse_bool(args.use_prediction),
        (
            _parse_bool(args.use_prediction_reward)
            if str(args.use_prediction_reward).strip() != ""
            else None
        ),
        (args.reward_mode or "").strip() or None,
        args.sim_end,
        args.out_dir,
        args.smoke_test,
        args.device,
        args.report_dir,
        args.run_name,
        args.checkpoint_every,
        (args.scenario_run_id or "").strip() or None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
