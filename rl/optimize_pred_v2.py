from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .env import PROJECT_ROOT
from .evaluate_policy import evaluate_policy
from .train_dqn import train_dqn


@dataclass
class RoundSpec:
    name: str
    use_prediction: bool
    reward_updates: dict[str, float] = field(default_factory=dict)
    sb3_updates: dict[str, float | int | str] = field(default_factory=dict)
    source: str = "manual"


def optimize_pred_v2(
    config_path: str | Path,
    rounds: int,
    timesteps: int,
    sim_end: int,
    device: str,
    report_dir: str | Path,
    artifact_dir: str | Path,
    seed: int,
    prediction_label: str = "pred_v1",
    smoke_test: bool = False,
    checkpoint_every: int = 0,
) -> dict[str, Any]:
    config_file = _project_path(config_path)
    report_root = _project_path(report_dir)
    artifact_root = _project_path(artifact_dir)
    report_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    base_config = json.loads(config_file.read_text(encoding="utf-8"))
    label = _safe_name(prediction_label or "pred_v1")
    specs = _round_specs(max(1, min(int(rounds), 5)), label)
    completed: list[dict[str, Any]] = []
    pred_results: list[tuple[RoundSpec, dict[str, Any]]] = []

    for index, spec in enumerate(specs, start=1):
        if spec.name == "no_pred_matched_best":
            if not pred_results:
                continue
            best_spec, best_metrics = min(
                pred_results,
                key=lambda item: (
                    float(item[1].get("mean_queue", 1e9)),
                    -float(item[1].get("mean_reward", -1e9)),
                ),
            )
            spec = RoundSpec(
                name=spec.name,
                use_prediction=False,
                reward_updates=dict(best_spec.reward_updates),
                sb3_updates=dict(best_spec.sb3_updates),
                source=f"matched_{best_spec.name}",
            )

        print(f"[{index}/{len(specs)}] {spec.name} use_prediction={spec.use_prediction}")
        round_config = _build_round_config(base_config, spec)
        round_config_path = report_root / "configs" / f"{spec.name}.json"
        round_config_path.parent.mkdir(parents=True, exist_ok=True)
        round_config_path.write_text(
            json.dumps(round_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        train_summary = train_dqn(
            config_path=round_config_path,
            timesteps=timesteps,
            seed=seed,
            use_prediction=spec.use_prediction,
            use_prediction_reward=None,
            sim_end=sim_end,
            out_dir=artifact_root,
            smoke_test=smoke_test,
            device=device,
            report_dir=report_root,
            run_name=spec.name,
            resume_from=None,
            checkpoint_every=checkpoint_every,
        )
        eval_path = report_root / f"{spec.name}_eval.csv"
        eval_summary = evaluate_policy(
            config_path=round_config_path,
            policy_name="dqn",
            sim_end=sim_end,
            out_path=eval_path,
            seed=seed,
            model_path=train_summary["model_path"],
            use_prediction=spec.use_prediction,
            use_prediction_reward=None,
        )
        eval_metrics = _metrics_from_eval_csv(eval_path)
        row = {
            "round": spec.name,
            "source": spec.source,
            "use_prediction": int(spec.use_prediction),
            "timesteps": int(timesteps),
            "sim_end": int(sim_end),
            "seed": int(seed),
            "device": device,
            "config_path": str(round_config_path),
            "model_path": train_summary["model_path"],
            "training_log": train_summary["training_log"],
            "eval_path": str(eval_path),
            "reward_updates": json.dumps(spec.reward_updates, ensure_ascii=False, sort_keys=True),
            "sb3_updates": json.dumps(spec.sb3_updates, ensure_ascii=False, sort_keys=True),
            **eval_summary,
            **eval_metrics,
        }
        completed.append(row)
        if spec.use_prediction:
            pred_results.append((spec, row))
        _write_summary(report_root / "sweep_summary.csv", completed)
        _write_summary_markdown(report_root / "sweep_summary.md", completed)

    best_pred = _best_row([row for row in completed if int(row.get("use_prediction", 0)) == 1])
    best_overall = _best_row(completed)
    summary = {
        "status": "ok",
        "rounds_requested": int(rounds),
        "rounds_completed": len(completed),
        "timesteps": int(timesteps),
        "sim_end": int(sim_end),
        "device": device,
        "prediction_label": label,
        "checkpoint_every": int(checkpoint_every),
        "report_dir": str(report_root),
        "artifact_dir": str(artifact_root),
        "best_pred_round": best_pred.get("round", "") if best_pred else "",
        "best_overall_round": best_overall.get("round", "") if best_overall else "",
        "summary_csv": str(report_root / "sweep_summary.csv"),
        "summary_md": str(report_root / "sweep_summary.md"),
        "note": "Smoke-test sweep" if smoke_test else "Full DQN prediction-control sweep",
    }
    (report_root / "sweep_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _round_specs(rounds: int, prediction_label: str = "pred_v1") -> list[RoundSpec]:
    label = _safe_name(prediction_label or "pred_v1")
    base = [
        RoundSpec(name=f"{label}_long_baseline", use_prediction=True),
        RoundSpec(
            name=f"{label}_low_switch_penalty",
            use_prediction=True,
            reward_updates={"switch": 0.01},
        ),
        RoundSpec(
            name=f"{label}_pressure_boost",
            use_prediction=True,
            reward_updates={"queue": 1.2, "pressure": 0.8, "switch": 0.02},
        ),
        RoundSpec(
            name=f"{label}_explore_more",
            use_prediction=True,
            reward_updates={"switch": 0.02},
            sb3_updates={"exploration_fraction": 0.55, "exploration_final_eps": 0.08},
        ),
        RoundSpec(name="no_pred_matched_best", use_prediction=False, source="matched_best_pred"),
    ]
    return base[:rounds]


def _build_round_config(base_config: dict[str, Any], spec: RoundSpec) -> dict[str, Any]:
    payload = copy.deepcopy(base_config)
    reward_weights = dict(payload.get("reward_weights", {}))
    reward_weights.update(spec.reward_updates)
    payload["reward_weights"] = reward_weights

    train_config = dict(payload.get("train", {}))
    sb3_config = dict(train_config.get("sb3", {}))
    sb3_config.update(spec.sb3_updates)
    train_config["sb3"] = sb3_config
    payload["train"] = train_config
    payload["use_prediction_features"] = bool(spec.use_prediction)
    payload["optimization_round"] = spec.name
    return payload


def _metrics_from_eval_csv(path: Path) -> dict[str, Any]:
    rows = _read_csv(path)
    if not rows:
        return {
            "eval_steps": 0,
            "eval_mean_queue": 0.0,
            "eval_max_queue": 0.0,
            "eval_mean_reward": 0.0,
            "eval_mean_speed_kmh": 0.0,
            "eval_switch_count": 0,
            "eval_prediction_available_share": 0.0,
            "eval_prediction_fallback_share": 0.0,
        }
    queue = [_float(row.get("queue_sum")) for row in rows]
    reward = [_float(row.get("reward")) for row in rows]
    speed = [_float(row.get("mean_speed_mps")) for row in rows]
    switch_count = sum(int(_float(row.get("switch_applied"))) for row in rows)
    pred_available = [_float(row.get("prediction_available")) for row in rows]
    pred_fallback = [_float(row.get("prediction_fallback_used")) for row in rows]
    latency = [_float(row.get("prediction_latency_ms")) for row in rows]
    return {
        "eval_steps": len(rows),
        "eval_mean_queue": round(_mean(queue), 4),
        "eval_max_queue": round(max(queue), 4),
        "eval_mean_reward": round(_mean(reward), 6),
        "eval_mean_speed_kmh": round(_mean(speed) * 3.6, 4),
        "eval_switch_count": switch_count,
        "eval_prediction_available_share": round(_mean(pred_available), 4),
        "eval_prediction_fallback_share": round(_mean(pred_fallback), 4),
        "eval_prediction_latency_ms": round(_mean(latency), 4),
    }


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            float(row.get("eval_mean_queue", row.get("mean_queue", 1e9))),
            -float(row.get("eval_mean_reward", row.get("mean_reward", -1e9))),
        ),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_summary_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    headers = [
        "round",
        "source",
        "use_prediction",
        "timesteps",
        "eval_mean_queue",
        "eval_max_queue",
        "eval_mean_reward",
        "eval_mean_speed_kmh",
        "eval_switch_count",
        "eval_prediction_available_share",
        "eval_prediction_fallback_share",
    ]
    lines = [
        "# DQN Prediction-Enhanced Sweep Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _safe_name(value: str) -> str:
    cleaned = []
    for char in str(value).strip():
        cleaned.append(char if char.isalnum() or char in {"_", "-"} else "_")
    result = "".join(cleaned).strip("_")
    return result or "pred_v1"


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DQN prediction-enhanced optimization sweep.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rl_signal_config.json"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--sim-end", type=int, default=1800)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "reports" / "rl_signal_control" / "pred_v1_sweep"))
    parser.add_argument("--artifact-dir", default=str(PROJECT_ROOT / "models" / "artifacts_rl" / "pred_v1_sweep"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prediction-label", default="pred_v1")
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
        args.prediction_label,
        args.smoke_test,
        args.checkpoint_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
