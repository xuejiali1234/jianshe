from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .env import PROJECT_ROOT


POLICY_SOURCES = [
    ("Webster", "formal", "webster_1800_eval.csv"),
    ("MaxPressure", "formal", "max_pressure_1800_eval.csv"),
    ("DQN-no-pred", "formal", "dqn_no_pred_eval.csv"),
    ("DQN-pred-v1", "formal", "dqn_pred_v1_eval.csv"),
    ("DQN-pred-v1-smoke", "smoke", "dqn_pred_v1_smoke_eval.csv"),
    ("DQN-pred-v2", "formal", "dqn_pred_v2_eval.csv"),
    ("DQN-pred-v2-smoke", "smoke", "dqn_pred_v2_smoke_eval.csv"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RL signal-control evaluation and training logs.")
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "reports" / "rl_signal_control"))
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()
    report_dir = _project_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = _build_policy_summary(report_dir)
    _write_csv(report_dir / "policy_comparison_1800.csv", summary_rows)
    _write_markdown(
        report_dir / "policy_comparison_1800.md",
        summary_rows,
        title="RL Signal Control Policy Comparison",
        note="Formal rows use the same 1800s evaluation naming convention. Smoke rows are interface checks only.",
    )

    has_prediction_result = any(
        row.get("policy") in {"DQN-pred-v1", "DQN-pred-v1-smoke", "DQN-pred-v2", "DQN-pred-v2-smoke"}
        or row.get("prediction_available_share", "0.00") not in {"", "0.00"}
        for row in summary_rows
    )
    prediction_rows = list(summary_rows) if has_prediction_result else []
    _write_csv(report_dir / "policy_comparison_with_prediction.csv", prediction_rows)
    _write_markdown(
        report_dir / "policy_comparison_with_prediction.md",
        prediction_rows,
        title="RL Prediction-Enhanced Policy Comparison",
        note="DQN-pred-v1 is the current stable prediction-enhanced policy target; DQN-pred-v2 rows are legacy exploratory results.",
    )

    figure_paths = [
        _plot_training_curves(
            report_dir / "dqn_training_log_no_pred.csv",
            report_dir,
            args.window,
            "no_pred",
            "DQN-no-pred Training Curves",
        ),
        _plot_training_curves(
            report_dir / "dqn_training_log_pred_v1.csv",
            report_dir,
            args.window,
            "pred_v1",
            "DQN-pred-v1 Training Curves",
        ),
        _plot_training_curves(
            report_dir / "dqn_training_log_pred_v2.csv",
            report_dir,
            args.window,
            "pred_v2_legacy",
            "DQN-pred-v2 Legacy Training Curves",
        ),
    ]

    print(f"policy_csv={report_dir / 'policy_comparison_1800.csv'}")
    print(f"policy_md={report_dir / 'policy_comparison_1800.md'}")
    print(f"prediction_csv={report_dir / 'policy_comparison_with_prediction.csv'}")
    print(f"prediction_md={report_dir / 'policy_comparison_with_prediction.md'}")
    for figure_path in figure_paths:
        if figure_path:
            print(f"training_figure={figure_path}")


def _build_policy_summary(report_dir: Path) -> list[dict[str, str]]:
    rows = []
    baseline_queue = None
    baseline_reward = None
    for name, run_type, filename in POLICY_SOURCES:
        path = report_dir / filename
        records = _read_csv(path)
        if not records:
            continue
        queue_values = [_float(row.get("queue_sum")) for row in records]
        reward_values = [_float(row.get("reward")) for row in records]
        speed_values = [_float(row.get("mean_speed_mps")) for row in records]
        switch_count = sum(int(_float(row.get("switch_applied"))) for row in records)
        fallback_count = sum(int(_float(row.get("transition_fallback"))) for row in records)
        program_mismatch_count = sum(int(_float(row.get("transition_program_mismatch"))) for row in records)
        prediction_available = [_float(row.get("prediction_available")) for row in records]
        prediction_fallback = [_float(row.get("prediction_fallback_used")) for row in records]
        prediction_latency = [_float(row.get("prediction_latency_ms")) for row in records]
        prediction_snapshots = [_float(row.get("prediction_snapshots")) for row in records]
        prediction_ready = [
            1.0
            if int(_float(row.get("prediction_ready"))) == 1
            or (
                int(_float(row.get("prediction_available"))) == 1
                and int(_float(row.get("prediction_fallback_used"))) == 0
            )
            else 0.0
            for row in records
        ]
        ready_records = [
            row for row in records
            if int(_float(row.get("prediction_ready"))) == 1
            or (
                int(_float(row.get("prediction_available"))) == 1
                and int(_float(row.get("prediction_fallback_used"))) == 0
            )
        ]

        row = {
            "policy": name,
            "run_type": run_type,
            "source_file": filename,
            "steps": str(len(records)),
            "sim_start_s": records[0].get("sim_time_s", ""),
            "sim_end_s": records[-1].get("sim_time_s", ""),
            "mean_reward": _fmt(_mean(reward_values), 4),
            "mean_queue_veh": _fmt(_mean(queue_values)),
            "max_queue_veh": _fmt(max(queue_values) if queue_values else 0.0),
            "mean_speed_mps": _fmt(_mean(speed_values)),
            "mean_speed_kmh": _fmt(_mean(speed_values) * 3.6),
            "switch_count": str(switch_count),
            "transition_fallback_count": str(fallback_count),
            "transition_program_mismatch_count": str(program_mismatch_count),
            "prediction_available_share": _fmt(_mean(prediction_available)),
            "prediction_ready_share": _fmt(_mean(prediction_ready)),
            "prediction_fallback_share": _fmt(_mean(prediction_fallback)),
            "max_prediction_snapshots": str(int(max(prediction_snapshots) if prediction_snapshots else 0)),
            "mean_prediction_latency_ms": _fmt(_mean(prediction_latency), 3),
            "prediction_ready_steps": str(len(ready_records)),
            "prediction_ready_mean_queue_veh": _fmt(
                _mean([_float(item.get("queue_sum")) for item in ready_records])
            ),
            "prediction_ready_mean_reward": _fmt(
                _mean([_float(item.get("reward")) for item in ready_records]),
                4,
            ),
            "prediction_ready_mean_speed_kmh": _fmt(
                _mean([_float(item.get("mean_speed_mps")) for item in ready_records]) * 3.6
            ),
        }
        if name == "Webster":
            baseline_queue = float(row["mean_queue_veh"])
            baseline_reward = float(row["mean_reward"])
            row["queue_reduction_vs_webster_pct"] = "0.00"
            row["reward_delta_vs_webster"] = "0.0000"
        elif baseline_queue is not None and baseline_reward is not None:
            reduction = (baseline_queue - float(row["mean_queue_veh"])) / max(baseline_queue, 1e-9) * 100.0
            row["queue_reduction_vs_webster_pct"] = _fmt(reduction)
            row["reward_delta_vs_webster"] = _fmt(float(row["mean_reward"]) - baseline_reward, 4)
        else:
            row["queue_reduction_vs_webster_pct"] = ""
            row["reward_delta_vs_webster"] = ""
        rows.append(row)
    return rows


def _plot_training_curves(
    log_path: Path,
    report_dir: Path,
    window: int,
    suffix: str,
    title: str,
) -> Path | None:
    records = _read_csv(log_path)
    if not records:
        return None

    timesteps = [int(_float(row.get("timestep"))) for row in records]
    series = [
        ("Reward", [_float(row.get("reward")) for row in records], "#00a6fb"),
        ("Queue (veh)", [_float(row.get("queue_sum")) for row in records], "#f77f00"),
        ("Speed (km/h)", [_float(row.get("mean_speed_mps")) * 3.6 for row in records], "#2a9d8f"),
        ("Switch rate", _rolling_mean([_float(row.get("switch_applied")) for row in records], window), "#d62828"),
        (
            "Transition fallback rate",
            _rolling_mean([_float(row.get("transition_fallback")) for row in records], window),
            "#6d597a",
        ),
    ]
    if any("prediction_available" in row for row in records):
        series.append(
            (
                "Prediction available",
                _rolling_mean([_float(row.get("prediction_available")) for row in records], window),
                "#7cb518",
            )
        )
    if any("prediction_fallback_used" in row for row in records):
        series.append(
            (
                "Prediction fallback",
                _rolling_mean([_float(row.get("prediction_fallback_used")) for row in records], window),
                "#8e44ad",
            )
        )

    fig, axes = plt.subplots(len(series), 1, figsize=(12, max(8, 2.2 * len(series))), sharex=True)
    if len(series) == 1:
        axes = [axes]
    for ax, (axis_title, values, color) in zip(axes, series):
        ax.plot(timesteps, _rolling_mean(values, window), color=color, linewidth=1.8)
        ax.set_ylabel(axis_title)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Training timestep")
    fig.suptitle(f"{title} (rolling window={window})", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figure_path = report_dir / f"dqn_training_curves_{suffix}.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    result = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        result.append(running_sum / min(index + 1, window))
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, str]], title: str, note: str) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = [
        f"# {title}",
        "",
        note,
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


def _fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
