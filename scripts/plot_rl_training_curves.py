from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RL training curves from a training log CSV.")
    parser.add_argument("--log", required=True, help="Path to the training log CSV.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument("--window", type=int, default=100, help="Rolling mean window.")
    parser.add_argument("--title", default="", help="Optional figure title.")
    parser.add_argument(
        "--metric",
        choices=["reward", "queue_sum", "mean_speed_mps", "switch_applied", "switch_count", "loss"],
        default="",
        help="If set, plot only one standardized metric panel.",
    )
    parser.add_argument("--legend-loc", default="upper right", help="Legend location for single-metric plots.")
    parser.add_argument("--font-scale", type=float, default=1.0, help="Font scale for single-metric plots.")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)
    records = _read_csv(log_path)
    if not records:
        raise SystemExit(f"No records found in {log_path}")

    timesteps = [int(_float(row.get("timestep"))) for row in records]
    series = [
        ("Reward", [_float(row.get("reward")) for row in records], "#00a6fb"),
        ("Queue (veh)", [_float(row.get("queue_sum")) for row in records], "#f77f00"),
        ("Speed (km/h)", [_float(row.get("mean_speed_mps")) * 3.6 for row in records], "#2a9d8f"),
    ]

    if any("switch_applied" in row for row in records):
        series.append(
            ("Switch rate", _rolling_mean([_float(row.get("switch_applied")) for row in records], args.window), "#d62828")
        )
    elif any("switch_count" in row for row in records):
        series.append(
            ("Switch count / step", _rolling_mean([_float(row.get("switch_count")) for row in records], args.window), "#d62828")
        )

    if any("transition_fallback" in row for row in records):
        series.append(
            (
                "Transition fallback rate",
                _rolling_mean([_float(row.get("transition_fallback")) for row in records], args.window),
                "#6d597a",
            )
        )
    if any("prediction_available" in row for row in records):
        series.append(
            (
                "Prediction available",
                _rolling_mean([_float(row.get("prediction_available")) for row in records], args.window),
                "#7cb518",
            )
        )
    if any("prediction_ready" in row for row in records):
        series.append(
            (
                "Prediction ready",
                _rolling_mean([_float(row.get("prediction_ready")) for row in records], args.window),
                "#1982c4",
            )
        )
    if any("prediction_fallback_used" in row for row in records):
        series.append(
            (
                "Prediction fallback",
                _rolling_mean([_float(row.get("prediction_fallback_used")) for row in records], args.window),
                "#8e44ad",
            )
        )

    loss_values = [_float_or_none(row.get("loss")) for row in records]
    if any(value is not None for value in loss_values):
        dense_loss = _forward_fill(loss_values)
        series.append(("Loss", _rolling_mean(dense_loss, args.window), "#264653"))

    if args.metric:
        _plot_single_metric(
            records=records,
            timesteps=timesteps,
            metric=args.metric,
            window=args.window,
            out_path=out_path,
            legend_loc=args.legend_loc,
            font_scale=float(args.font_scale),
        )
    else:
        fig, axes = plt.subplots(len(series), 1, figsize=(12, max(8, 2.2 * len(series))), sharex=True)
        if len(series) == 1:
            axes = [axes]

        for ax, (label, values, color) in zip(axes, series):
            ax.plot(timesteps, values, color=color, linewidth=1.8)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)

        axes[-1].set_xlabel("Training timestep")
        if args.title.strip():
            fig.suptitle(args.title.strip(), fontsize=15)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
        else:
            fig.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
    print(str(out_path))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    result: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        result.append(running_sum / min(index + 1, window))
    return result


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_or_none(value: object) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _forward_fill(values: Iterable[float | None]) -> list[float]:
    result: list[float] = []
    last = 0.0
    for value in values:
        if value is not None:
            last = float(value)
        result.append(last)
    return result


def _plot_single_metric(
    records: list[dict[str, str]],
    timesteps: list[int],
    metric: str,
    window: int,
    out_path: Path,
    legend_loc: str,
    font_scale: float,
) -> None:
    metric_map = {
        "reward": ("Reward", [_float(row.get("reward")) for row in records], "#1f77b4"),
        "queue_sum": ("Queue length (veh)", [_float(row.get("queue_sum")) for row in records], "#ff7f0e"),
        "mean_speed_mps": ("Speed (km/h)", [_float(row.get("mean_speed_mps")) * 3.6 for row in records], "#2a9d8f"),
        "switch_applied": ("Switch rate", [_float(row.get("switch_applied")) for row in records], "#d62828"),
        "switch_count": ("Switch count / step", [_float(row.get("switch_count")) for row in records], "#d62828"),
        "loss": ("Loss", _forward_fill([_float_or_none(row.get("loss")) for row in records]), "#264653"),
    }
    ylabel, values, color = metric_map[metric]
    smooth = _rolling_mean(values, window)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.1,
            "font.size": 18 * font_scale,
            "axes.labelsize": 20 * font_scale,
            "xtick.labelsize": 18 * font_scale,
            "ytick.labelsize": 18 * font_scale,
            "legend.fontsize": 16 * font_scale,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.plot(timesteps, values, color=color, linewidth=1.0, alpha=0.18, label="raw")
    ax.plot(timesteps, smooth, color=color, linewidth=2.6, label=f"rolling mean ({window})")

    ax.set_xlabel("Training timestep")
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc, frameon=False)
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.1)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=0.9)
    ax.grid(False)
    x_min = float(timesteps[0])
    x_max = float(timesteps[-1])
    y_min = float(min(min(values), min(smooth)))
    y_max = float(max(max(values), max(smooth)))
    if y_max <= y_min:
        y_max = y_min + 1.0

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    x_ticks = np.linspace(x_min, x_max, 6)
    x_tick_labels = [str(int(round(item))) for item in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)

    y_ticks = np.linspace(y_min, y_max, 6)
    ax.set_yticks(y_ticks)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
