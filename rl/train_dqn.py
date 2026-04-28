from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .env import PROJECT_ROOT, SumoSignalGymEnv


class TrainingLogCallback:
    def __init__(self, log_path: Path):
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError as exc:
            raise RuntimeError("stable-baselines3 is required to train DQN policies.") from exc

        class _Callback(BaseCallback):
            def __init__(self, destination: Path):
                super().__init__()
                self.destination = destination
                self.fp = None
                self.writer = None

            def _on_training_start(self) -> None:
                self.destination.parent.mkdir(parents=True, exist_ok=True)
                self.fp = self.destination.open("w", newline="", encoding="utf-8")
                self.writer = csv.DictWriter(
                    self.fp,
                    fieldnames=[
                        "timestep",
                        "reward",
                        "queue_sum",
                        "arrival_flow_sum",
                        "vehicle_count",
                        "mean_speed_mps",
                        "switch_applied",
                        "current_phase",
                        "transition_fallback",
                        "prediction_available",
                        "prediction_snapshots",
                        "prediction_fallback_used",
                        "prediction_latency_ms",
                    ],
                )
                self.writer.writeheader()

            def _on_step(self) -> bool:
                infos = self.locals.get("infos") or [{}]
                rewards = self.locals.get("rewards") or [0.0]
                info = infos[0] if infos else {}
                queue_sum = sum(float(item.get("queue_sum", 0.0)) for item in info.get("phase_stats", []))
                arrival_sum = sum(float(item.get("arrival_flow_sum", 0.0)) for item in info.get("phase_stats", []))
                if self.writer:
                    self.writer.writerow(
                        {
                            "timestep": int(self.num_timesteps),
                            "reward": round(float(rewards[0]), 6),
                            "queue_sum": round(queue_sum, 4),
                            "arrival_flow_sum": round(arrival_sum, 4),
                            "vehicle_count": int(info.get("vehicle_count", 0)),
                            "mean_speed_mps": round(float(info.get("mean_speed_mps", 0.0)), 4),
                            "switch_applied": int(bool(info.get("switch_applied"))),
                            "current_phase": int(info.get("current_phase", -1)),
                            "transition_fallback": int(bool(info.get("transition_fallback"))),
                            "prediction_available": int(bool(info.get("prediction_available"))),
                            "prediction_snapshots": int(info.get("prediction_snapshots", 0)),
                            "prediction_fallback_used": int(bool(info.get("prediction_fallback_used"))),
                            "prediction_latency_ms": round(float(info.get("prediction_latency_ms", 0.0)), 3),
                        }
                    )
                return True

            def _on_training_end(self) -> None:
                if self.fp:
                    self.fp.close()

        self.callback = _Callback(log_path)


def train_dqn(
    config_path: str | Path,
    timesteps: int,
    seed: int,
    use_prediction: bool,
    sim_end: int | None,
    out_dir: str | Path,
    smoke_test: bool,
    device: str,
    report_dir: str | Path | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required. Install it in traffic_pred with: pip install stable-baselines3 gymnasium"
        ) from exc

    config_file = _project_path(config_path)
    raw_config = json.loads(config_file.read_text(encoding="utf-8"))
    train_config = dict(raw_config.get("train", {}))
    sb3_config = dict(train_config.get("sb3", {}))
    suffix = _safe_name(run_name) if run_name else ("pred_v2" if use_prediction else "no_pred")
    output_dir = _project_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_root = _project_path(report_dir) if report_dir else PROJECT_ROOT / "reports" / "rl_signal_control"
    report_root.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        SumoSignalGymEnv(config_file, sim_end=sim_end, use_prediction_features=use_prediction)
    )
    params = _dqn_params(train_config, sb3_config, timesteps, seed, smoke_test)
    model_path = output_dir / f"dqn_signal_single_tls_{suffix}.zip"
    log_path = report_root / f"dqn_training_log_{suffix}.csv"
    summary_path = output_dir / f"dqn_training_summary_{suffix}.json"
    callback = TrainingLogCallback(log_path).callback

    try:
        model = DQN(
            policy=str(sb3_config.get("policy", "MlpPolicy")),
            env=env,
            seed=seed,
            verbose=1 if smoke_test else 0,
            device=device,
            **params,
        )
        model.learn(total_timesteps=int(timesteps), callback=callback, progress_bar=False)
        model.save(str(model_path))
    finally:
        env.close()

    summary = {
        "status": "ok",
        "algorithm": "DQN",
        "use_prediction": bool(use_prediction),
        "run_name": suffix,
        "timesteps": int(timesteps),
        "seed": int(seed),
        "sim_end": sim_end,
        "model_path": str(model_path),
        "training_log": str(log_path),
        "target_tls_id": raw_config.get("target_tls_id"),
        "device": device,
        "sb3_params": params,
        "note": "Smoke-test run" if smoke_test else "Training run",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _dqn_params(
    train_config: dict[str, Any],
    sb3_config: dict[str, Any],
    timesteps: int,
    seed: int,
    smoke_test: bool,
) -> dict[str, Any]:
    learning_starts = int(sb3_config.get("learning_starts", train_config.get("learning_starts", 1000)))
    buffer_size = int(sb3_config.get("buffer_size", train_config.get("buffer_size", 50000)))
    batch_size = int(sb3_config.get("batch_size", train_config.get("batch_size", 64)))
    if smoke_test:
        learning_starts = min(learning_starts, 10)
        buffer_size = min(buffer_size, 1000)
        batch_size = min(batch_size, 32)
    learning_starts = min(learning_starts, max(1, int(timesteps) // 4))
    return {
        "learning_rate": float(sb3_config.get("learning_rate", train_config.get("learning_rate", 0.0005))),
        "gamma": float(sb3_config.get("gamma", train_config.get("gamma", 0.99))),
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "batch_size": batch_size,
        "target_update_interval": int(sb3_config.get("target_update_interval", 500)),
        "exploration_fraction": float(sb3_config.get("exploration_fraction", 0.3)),
        "exploration_initial_eps": float(sb3_config.get("exploration_initial_eps", train_config.get("epsilon_start", 1.0))),
        "exploration_final_eps": float(sb3_config.get("exploration_final_eps", train_config.get("epsilon_end", 0.05))),
        "train_freq": int(sb3_config.get("train_freq", 4)),
    }


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
    parser = argparse.ArgumentParser(description="Train a Stable-Baselines3 DQN signal-control policy.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rl_signal_config.json"))
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-prediction", default="false")
    parser.add_argument("--sim-end", type=int, default=None)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "models" / "artifacts_rl"))
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "reports" / "rl_signal_control"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    summary = train_dqn(
        args.config,
        args.timesteps,
        args.seed,
        _parse_bool(args.use_prediction),
        args.sim_end,
        args.out_dir,
        args.smoke_test,
        args.device,
        args.report_dir,
        args.run_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
