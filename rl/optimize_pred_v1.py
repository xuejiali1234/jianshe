from __future__ import annotations

import argparse
import json

from .env import PROJECT_ROOT
from .optimize_pred_v2 import optimize_pred_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DQN-pred-v1 optimization sweep.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rl_signal_config.json"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--sim-end", type=int, default=1800)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "rl_signal_control" / "full_v3_pred_control_v1"),
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "models" / "artifacts_rl" / "full_v3_pred_control_v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    summary = optimize_pred_v2(
        args.config,
        args.rounds,
        args.timesteps,
        args.sim_end,
        args.device,
        args.report_dir,
        args.artifact_dir,
        args.seed,
        "pred_v1",
        args.smoke_test,
        args.checkpoint_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
