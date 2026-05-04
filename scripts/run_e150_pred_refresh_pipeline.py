from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.env import PROJECT_ROOT
from rl.evaluate_multi_policy import evaluate_multi_policy
from rl.evaluate_policy import evaluate_policy
from rl.train_dqn import train_dqn
from rl.train_multi_dqn import train_multi_dqn


SCENARIOS: list[tuple[str, str | None]] = [
    ("default", None),
    ("incident", "S5_incident_mainline_edge_closure_scale_1p30_seed_11"),
    ("control_short_cycle", "S3_control_short_cycle_scale_1p30_seed_11"),
]


@dataclass
class SinglePipelineSpec:
    config_path: Path
    out_dir: Path
    report_dir: Path
    run_name: str
    timesteps: int
    sim_end: int
    checkpoint_every: int
    use_prediction: bool
    use_prediction_reward: bool
    reward_mode: str
    seed: int
    device: str


@dataclass
class MultiPipelineSpec:
    config_path: Path
    out_dir: Path
    report_dir: Path
    run_name: str
    timesteps: int
    sim_end: int
    checkpoint_every: int
    use_prediction: bool
    use_prediction_reward: bool
    reward_mode: str
    seed: int
    device: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the e150 RL pred refresh pipeline: train, evaluate, and scan checkpoints."
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--single-timesteps", type=int, default=10000)
    parser.add_argument("--multi-timesteps", type=int, default=5000)
    parser.add_argument("--sim-end", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single-checkpoint-every", type=int, default=1000)
    parser.add_argument("--multi-checkpoint-every", type=int, default=1000)
    parser.add_argument("--single-run-name", default="tls12254692358_pred_v1_e150")
    parser.add_argument("--multi-run-name", default="west_multi_pred_v1_e150")
    parser.add_argument(
        "--output-tag",
        default="e150_refresh",
        help="Used in generated config and summary file names.",
    )
    args = parser.parse_args()

    generated_dir = PROJECT_ROOT / "configs" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    prediction_config = _ensure_prediction_config_e150(
        generated_dir / f"prediction_config_full_v3_plus_lane_incident_v1_selected_{args.output_tag}.json"
    )
    single_config = _clone_rl_config_with_prediction(
        PROJECT_ROOT / "configs" / "rl_signal_config_12254692358_v1.json",
        generated_dir / f"rl_signal_config_12254692358_v1_{args.output_tag}.json",
        prediction_config,
    )
    multi_config = _clone_rl_config_with_prediction(
        PROJECT_ROOT / "configs" / "rl_multi_signal_config_west_v1.json",
        generated_dir / f"rl_multi_signal_config_west_v1_{args.output_tag}.json",
        prediction_config,
    )

    single_spec = SinglePipelineSpec(
        config_path=single_config,
        out_dir=PROJECT_ROOT / "models" / "artifacts_rl" / f"single_12254692358_{args.output_tag}",
        report_dir=PROJECT_ROOT / "reports" / "rl_signal_control" / f"single_12254692358_{args.output_tag}",
        run_name=str(args.single_run_name),
        timesteps=int(args.single_timesteps),
        sim_end=int(args.sim_end),
        checkpoint_every=int(args.single_checkpoint_every),
        use_prediction=True,
        use_prediction_reward=True,
        reward_mode="anticipatory_delta_pressure_v2",
        seed=int(args.seed),
        device=str(args.device),
    )
    multi_spec = MultiPipelineSpec(
        config_path=multi_config,
        out_dir=PROJECT_ROOT / "models" / "artifacts_rl_multi" / f"west_v1_{args.output_tag}",
        report_dir=PROJECT_ROOT / "reports" / "rl_signal_control_multi" / f"west_v1_{args.output_tag}",
        run_name=str(args.multi_run_name),
        timesteps=int(args.multi_timesteps),
        sim_end=int(args.sim_end),
        checkpoint_every=int(args.multi_checkpoint_every),
        use_prediction=True,
        use_prediction_reward=True,
        reward_mode="anticipatory_delta_pressure_v2",
        seed=int(args.seed),
        device=str(args.device),
    )

    pipeline_dir = PROJECT_ROOT / "reports" / "rl_automation" / args.output_tag
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    _log(f"pipeline output tag: {args.output_tag}")
    _log(f"generated prediction config: {_relpath(prediction_config)}")
    _log(f"generated single rl config: {_relpath(single_config)}")
    _log(f"generated multi rl config: {_relpath(multi_config)}")

    single_train_summary = run_single_training_and_scan(single_spec, pipeline_dir)
    multi_train_summary = run_multi_training_and_scan(multi_spec, pipeline_dir)

    combined_summary = {
        "status": "ok",
        "prediction_config": _relpath(prediction_config),
        "single": single_train_summary,
        "multi": multi_train_summary,
    }
    summary_path = pipeline_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(combined_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(combined_summary, ensure_ascii=False, indent=2))


def run_single_training_and_scan(spec: SinglePipelineSpec, pipeline_dir: Path) -> dict[str, Any]:
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    spec.report_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[single] train start: run_name={spec.run_name}, timesteps={spec.timesteps}")

    train_summary = train_dqn(
        config_path=spec.config_path,
        timesteps=spec.timesteps,
        seed=spec.seed,
        use_prediction=spec.use_prediction,
        use_prediction_reward=spec.use_prediction_reward,
        reward_mode=spec.reward_mode,
        sim_end=spec.sim_end,
        out_dir=spec.out_dir,
        smoke_test=False,
        device=spec.device,
        report_dir=spec.report_dir,
        run_name=spec.run_name,
        resume_from=None,
        checkpoint_every=spec.checkpoint_every,
        scenario_run_id=None,
    )

    final_model_path = spec.out_dir / f"dqn_signal_single_tls_{spec.run_name}.zip"
    final_eval_dir = spec.report_dir / "final_eval"
    final_eval_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[single] final model saved: {final_model_path}")
    _log("[single] final evaluation start")
    final_eval = evaluate_single_model(
        config_path=spec.config_path,
        model_path=final_model_path,
        out_dir=final_eval_dir,
        use_prediction=spec.use_prediction,
        use_prediction_reward=spec.use_prediction_reward,
        reward_mode=spec.reward_mode,
        sim_end=spec.sim_end,
        seed=spec.seed,
    )

    checkpoint_dir = spec.out_dir / "checkpoints" / spec.run_name
    scan_dir = spec.report_dir / "checkpoint_scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[single] checkpoint scan start: {checkpoint_dir}")
    scan_summary = scan_single_checkpoints(
        config_path=spec.config_path,
        checkpoint_dir=checkpoint_dir,
        out_dir=scan_dir,
        use_prediction=spec.use_prediction,
        use_prediction_reward=spec.use_prediction_reward,
        reward_mode=spec.reward_mode,
        sim_end=spec.sim_end,
        seed=spec.seed,
    )

    result = {
        "train_summary": train_summary,
        "final_model_path": str(final_model_path),
        "final_eval": final_eval,
        "checkpoint_scan": scan_summary,
    }
    (pipeline_dir / "single_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_multi_training_and_scan(spec: MultiPipelineSpec, pipeline_dir: Path) -> dict[str, Any]:
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    spec.report_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[multi] train start: run_name={spec.run_name}, timesteps={spec.timesteps}")

    train_summary = train_multi_dqn(
        config_path=spec.config_path,
        timesteps=spec.timesteps,
        seed=spec.seed,
        use_prediction=spec.use_prediction,
        use_prediction_reward=spec.use_prediction_reward,
        reward_mode=spec.reward_mode,
        sim_end=spec.sim_end,
        out_dir=spec.out_dir,
        smoke_test=False,
        device=spec.device,
        report_dir=spec.report_dir,
        run_name=spec.run_name,
        checkpoint_every=spec.checkpoint_every,
        scenario_run_id=None,
    )

    final_model_path = spec.out_dir / f"dqn_multi_tls_{spec.run_name}.pt"
    final_eval_dir = spec.report_dir / "final_eval"
    final_eval_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[multi] final model saved: {final_model_path}")
    _log("[multi] final evaluation start")
    final_eval = evaluate_multi_model(
        config_path=spec.config_path,
        model_path=final_model_path,
        out_dir=final_eval_dir,
        use_prediction=spec.use_prediction,
        use_prediction_reward=spec.use_prediction_reward,
        reward_mode=spec.reward_mode,
        sim_end=spec.sim_end,
        seed=spec.seed,
    )

    checkpoint_dir = spec.out_dir / "checkpoints" / spec.run_name
    scan_dir = spec.report_dir / "checkpoint_scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[multi] checkpoint scan start: {checkpoint_dir}")
    scan_summary = scan_multi_checkpoints(
        config_path=spec.config_path,
        checkpoint_dir=checkpoint_dir,
        out_dir=scan_dir,
        use_prediction=spec.use_prediction,
        use_prediction_reward=spec.use_prediction_reward,
        reward_mode=spec.reward_mode,
        sim_end=spec.sim_end,
        seed=spec.seed,
    )

    result = {
        "train_summary": train_summary,
        "final_model_path": str(final_model_path),
        "final_eval": final_eval,
        "checkpoint_scan": scan_summary,
    }
    (pipeline_dir / "multi_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def evaluate_single_model(
    config_path: Path,
    model_path: Path,
    out_dir: Path,
    use_prediction: bool,
    use_prediction_reward: bool,
    reward_mode: str,
    sim_end: int,
    seed: int,
) -> dict[str, Any]:
    results = {}
    for scenario_label, scenario_run_id in SCENARIOS:
        out_csv = out_dir / f"{scenario_label}_eval.csv"
        results[scenario_label] = evaluate_policy(
            config_path=config_path,
            policy_name="dqn",
            sim_end=sim_end,
            out_path=out_csv,
            seed=seed,
            model_path=model_path,
            use_prediction=use_prediction,
            use_prediction_reward=use_prediction_reward,
            reward_mode=reward_mode,
            scenario_run_id=scenario_run_id,
        )
    return results


def evaluate_multi_model(
    config_path: Path,
    model_path: Path,
    out_dir: Path,
    use_prediction: bool,
    use_prediction_reward: bool,
    reward_mode: str,
    sim_end: int,
    seed: int,
) -> dict[str, Any]:
    results = {}
    for scenario_label, scenario_run_id in SCENARIOS:
        out_csv = out_dir / f"{scenario_label}_eval.csv"
        results[scenario_label] = evaluate_multi_policy(
            config_path=config_path,
            policy_name="dqn",
            sim_end=sim_end,
            out_path=out_csv,
            seed=seed,
            model_path=model_path,
            use_prediction=use_prediction,
            use_prediction_reward=use_prediction_reward,
            reward_mode=reward_mode,
            scenario_run_id=scenario_run_id,
        )
    return results


def scan_single_checkpoints(
    config_path: Path,
    checkpoint_dir: Path,
    out_dir: Path,
    use_prediction: bool,
    use_prediction_reward: bool,
    reward_mode: str,
    sim_end: int,
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    checkpoint_paths = sorted(checkpoint_dir.glob("*.zip"), key=_checkpoint_step_from_name)
    _log(f"[single] checkpoint count: {len(checkpoint_paths)}")
    for checkpoint_path in checkpoint_paths:
        step = _checkpoint_step_from_name(checkpoint_path)
        _log(f"[single] evaluate checkpoint {step}")
        scenario_results = evaluate_single_model(
            config_path=config_path,
            model_path=checkpoint_path,
            out_dir=out_dir / f"checkpoint_{step}",
            use_prediction=use_prediction,
            use_prediction_reward=use_prediction_reward,
            reward_mode=reward_mode,
            sim_end=sim_end,
            seed=seed,
        )
        summary_row = {
            "checkpoint_step": step,
            "model_path": str(checkpoint_path),
        }
        queue_sum = 0.0
        for scenario_label, _scenario_run_id in SCENARIOS:
            metrics = scenario_results[scenario_label]
            summary_row[f"{scenario_label}_mean_reward"] = metrics.get("mean_reward", 0.0)
            summary_row[f"{scenario_label}_mean_queue"] = metrics.get("mean_queue", 0.0)
            summary_row[f"{scenario_label}_mean_speed_mps"] = metrics.get("mean_speed_mps", 0.0)
            summary_row[f"{scenario_label}_switch_count"] = metrics.get("switch_count", 0)
            summary_row[f"{scenario_label}_post_event_first_300s_mean_queue"] = metrics.get(
                "post_event_first_300s_mean_queue", 0.0
            )
            summary_row[f"{scenario_label}_first_switch_after_event_s"] = metrics.get(
                "first_switch_after_event_s", ""
            )
            queue_sum += float(metrics.get("mean_queue", 0.0))
        summary_row["queue_sum_3scenarios"] = round(queue_sum, 4)
        rows.append(summary_row)

    csv_path = out_dir / "checkpoint_summary.csv"
    _write_csv(csv_path, rows)
    best = min(rows, key=lambda row: float(row["queue_sum_3scenarios"])) if rows else {}
    summary = {
        "checkpoint_count": len(rows),
        "summary_csv": str(csv_path),
        "best_checkpoint_step": best.get("checkpoint_step"),
        "best_model_path": best.get("model_path"),
        "best_queue_sum_3scenarios": best.get("queue_sum_3scenarios"),
    }
    (out_dir / "checkpoint_summary.json").write_text(
        json.dumps({"rows": rows, "best": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def scan_multi_checkpoints(
    config_path: Path,
    checkpoint_dir: Path,
    out_dir: Path,
    use_prediction: bool,
    use_prediction_reward: bool,
    reward_mode: str,
    sim_end: int,
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    checkpoint_paths = sorted(checkpoint_dir.glob("*.pt"), key=_checkpoint_step_from_name)
    _log(f"[multi] checkpoint count: {len(checkpoint_paths)}")
    for checkpoint_path in checkpoint_paths:
        step = _checkpoint_step_from_name(checkpoint_path)
        _log(f"[multi] evaluate checkpoint {step}")
        scenario_results = evaluate_multi_model(
            config_path=config_path,
            model_path=checkpoint_path,
            out_dir=out_dir / f"checkpoint_{step}",
            use_prediction=use_prediction,
            use_prediction_reward=use_prediction_reward,
            reward_mode=reward_mode,
            sim_end=sim_end,
            seed=seed,
        )
        summary_row = {
            "checkpoint_step": step,
            "model_path": str(checkpoint_path),
        }
        queue_sum = 0.0
        for scenario_label, _scenario_run_id in SCENARIOS:
            metrics = scenario_results[scenario_label]
            summary_row[f"{scenario_label}_mean_reward"] = metrics.get("mean_reward", 0.0)
            summary_row[f"{scenario_label}_mean_queue"] = metrics.get("mean_queue", 0.0)
            summary_row[f"{scenario_label}_max_queue"] = metrics.get("max_queue", 0.0)
            summary_row[f"{scenario_label}_mean_speed_kmh"] = metrics.get("mean_speed_kmh", 0.0)
            summary_row[f"{scenario_label}_switch_count"] = metrics.get("switch_count", 0)
            summary_row[f"{scenario_label}_post_event_first_300s_mean_queue"] = metrics.get(
                "post_event_first_300s_mean_queue", 0.0
            )
            summary_row[f"{scenario_label}_first_switch_after_event_s"] = metrics.get(
                "first_switch_after_event_s", ""
            )
            queue_sum += float(metrics.get("mean_queue", 0.0))
        summary_row["queue_sum_3scenarios"] = round(queue_sum, 4)
        rows.append(summary_row)

    csv_path = out_dir / "checkpoint_summary.csv"
    _write_csv(csv_path, rows)
    best = min(rows, key=lambda row: float(row["queue_sum_3scenarios"])) if rows else {}
    summary = {
        "checkpoint_count": len(rows),
        "summary_csv": str(csv_path),
        "best_checkpoint_step": best.get("checkpoint_step"),
        "best_model_path": best.get("model_path"),
        "best_queue_sum_3scenarios": best.get("queue_sum_3scenarios"),
    }
    (out_dir / "checkpoint_summary.json").write_text(
        json.dumps({"rows": rows, "best": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _ensure_prediction_config_e150(destination: Path) -> Path:
    source = PROJECT_ROOT / "configs" / "prediction_config_full_v3_plus_lane_incident_v1_selected.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["artifact_dir"] = "models/artifacts_full_v3_plus_lane_incident_v1_selected_e150"
    payload["metrics_file"] = "reports/full_v3_plus_lane_incident_v1_selected_e150/metrics.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def _clone_rl_config_with_prediction(source: Path, destination: Path, prediction_config: Path) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["prediction_config"] = _relpath(prediction_config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_step_from_name(path: Path) -> int:
    stem = path.stem
    for part in reversed(stem.split("_")):
        if part.isdigit():
            return int(part)
    digits = "".join(char for char in stem if char.isdigit())
    return int(digits) if digits else 0


def _relpath(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()
