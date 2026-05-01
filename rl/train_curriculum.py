from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .env import PROJECT_ROOT
from .train_dqn import train_dqn


def run_curriculum(
    config_path: str | Path,
    device: str,
    smoke_test: bool = False,
    stage1_timesteps: int | None = None,
    stage2_timesteps: int | None = None,
    seed: int | None = None,
    sim_end: int | None = None,
    out_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg_path = _project_path(config_path)
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    stage1_cfg = str(raw["stage1_config"])
    stage2_cfg = str(raw["stage2_config"])
    stage1_steps = int(stage1_timesteps if stage1_timesteps is not None else raw.get("stage1_timesteps", 3000))
    stage2_steps = int(stage2_timesteps if stage2_timesteps is not None else raw.get("stage2_timesteps", 7000))
    if smoke_test:
        stage1_steps = min(stage1_steps, 400)
        stage2_steps = min(stage2_steps, 600)

    training_seed = int(seed if seed is not None else raw.get("seed", 42))
    episode_end = int(sim_end if sim_end is not None else raw.get("sim_end", 1800))
    use_prediction = bool(raw.get("use_prediction", True))
    use_prediction_reward = bool(raw.get("use_prediction_reward", True))
    reward_mode = str(raw.get("reward_mode", "anticipatory_delta_pressure_v2"))
    output_dir = _project_path(out_dir or raw.get("out_dir", "models/artifacts_rl/anticipatory_v3_curriculum"))
    reports_dir = _project_path(report_dir or raw.get("report_dir", "reports/rl_signal_control/anticipatory_v3_curriculum"))
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    stage1_run_name = str(raw.get("stage1_run_name", "curriculum_stage1"))
    stage2_run_name = str(raw.get("stage2_run_name", "curriculum_final"))
    checkpoint_every = int(raw.get("checkpoint_every", 1000))

    stage1 = train_dqn(
        stage1_cfg,
        stage1_steps,
        training_seed,
        use_prediction,
        use_prediction_reward,
        reward_mode,
        episode_end,
        output_dir,
        smoke_test,
        device,
        reports_dir,
        stage1_run_name,
        None,
        checkpoint_every,
        None,
    )

    stage2 = train_dqn(
        stage2_cfg,
        stage2_steps,
        training_seed,
        use_prediction,
        use_prediction_reward,
        reward_mode,
        episode_end,
        output_dir,
        smoke_test,
        device,
        reports_dir,
        stage2_run_name,
        stage1["model_path"],
        checkpoint_every,
        None,
    )

    summary = {
        "status": "ok",
        "mode": "curriculum",
        "smoke_test": bool(smoke_test),
        "device": device,
        "stage1": stage1,
        "stage2": stage2,
    }
    summary_path = output_dir / f"{stage2_run_name}_curriculum_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-stage RL curriculum training: default warmup then mixed fine-tune.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rl_curriculum_anticipatory_v3.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--stage1-timesteps", type=int, default=None)
    parser.add_argument("--stage2-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sim-end", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    result = run_curriculum(
        args.config,
        args.device,
        smoke_test=args.smoke_test,
        stage1_timesteps=args.stage1_timesteps,
        stage2_timesteps=args.stage2_timesteps,
        seed=args.seed,
        sim_end=args.sim_end,
        out_dir=args.out_dir,
        report_dir=args.report_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
