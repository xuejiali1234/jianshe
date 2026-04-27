from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "batch_movement_aggregates.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "movement_speed_diagnostics.json"
DEFAULT_BOTTLENECKS = PROJECT_ROOT / "reports" / "movement_bottlenecks.csv"
DEFAULT_APPROACHES = PROJECT_ROOT / "reports" / "movement_approach_bottlenecks.csv"


def diagnose_movement_data(
    csv_path: str | Path = DEFAULT_CSV,
    manifest_path: str | Path | None = None,
    summary_path: str | Path = DEFAULT_SUMMARY,
    bottleneck_path: str | Path = DEFAULT_BOTTLENECKS,
    approach_path: str | Path = DEFAULT_APPROACHES,
    top_n: int = 30,
) -> dict[str, Any]:
    source = Path(csv_path)
    if not source.exists():
        raise FileNotFoundError(f"movement CSV not found: {source}")

    df = pd.read_csv(source, low_memory=False)
    required = {
        "scenario_id",
        "tls_id",
        "incoming_edge",
        "movement_id",
        "turn_type",
        "arrival_flow",
        "discharge_flow",
        "speed_kmh",
        "occupancy",
        "queue_veh",
        "signal_state",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"movement CSV missing required columns: {', '.join(missing)}")

    df["is_green"] = df["signal_state"].fillna("").astype(str).str.contains("G|g", regex=True)
    active = df[df["speed_kmh"] > 0]
    occupied = df[df["occupancy"] > 0]
    speed_summary = {
        "all_rows_mean_speed": _mean(df["speed_kmh"]),
        "active_mean_speed": _mean(active["speed_kmh"]),
        "occupied_mean_speed": _mean(occupied["speed_kmh"]),
        "arrival_weighted_speed": _weighted_mean(df, "speed_kmh", "arrival_flow"),
        "discharge_weighted_speed": _weighted_mean(df, "speed_kmh", "discharge_flow"),
        "occupancy_weighted_speed": _weighted_mean(df, "speed_kmh", "occupancy"),
        "active_row_share": round(float(len(active) / max(len(df), 1)), 4),
    }
    scenario_speed = (
        active.groupby("scenario_id")["speed_kmh"].mean().round(4).to_dict()
        if not active.empty
        else {}
    )
    target_columns = [column for column in ["arrival_flow", "mean_speed_mps", "speed_kmh", "queue_veh"] if column in df.columns]
    target_summary = {
        column: {
            "mean": _mean(df[column]),
            "p50": _quantile(df[column], 0.5),
            "p90": _quantile(df[column], 0.9),
            "max": round(float(df[column].max()), 4) if len(df[column]) else 0.0,
            "nonzero_share": round(float((df[column] != 0).mean()), 4) if len(df[column]) else 0.0,
        }
        for column in target_columns
    }
    scenario_counts = (
        df.groupby("scenario_id")["run_id"].nunique().astype(int).to_dict()
        if "run_id" in df.columns
        else df["scenario_id"].value_counts().astype(int).to_dict()
    )
    event_counts = (
        df.groupby("event_type")["run_id"].nunique().astype(int).to_dict()
        if "event_type" in df.columns and "run_id" in df.columns
        else {}
    )
    signal_variant_counts = (
        df.groupby("signal_variant")["run_id"].nunique().astype(int).to_dict()
        if "signal_variant" in df.columns and "run_id" in df.columns
        else {}
    )
    turn_type_summary = (
        df.groupby("turn_type", dropna=False)
        .agg(
            rows=("movement_id", "size"),
            movement_count=("movement_id", "nunique"),
            arrival_flow_mean=("arrival_flow", "mean"),
            arrival_flow_sum=("arrival_flow", "sum"),
            queue_veh_mean=("queue_veh", "mean"),
            queue_veh_max=("queue_veh", "max"),
            speed_kmh_mean=("speed_kmh", "mean"),
        )
        .round(4)
        .reset_index()
        .to_dict(orient="records")
    )
    tls_sample_summary = (
        df.groupby("tls_id", dropna=False)
        .agg(
            rows=("movement_id", "size"),
            movement_count=("movement_id", "nunique"),
            arrival_flow_sum=("arrival_flow", "sum"),
            queue_veh_mean=("queue_veh", "mean"),
            queue_veh_max=("queue_veh", "max"),
        )
        .round(4)
        .reset_index()
        .to_dict(orient="records")
    )
    manifest_summary: dict[str, Any] = {}
    if manifest_path:
        manifest_source = Path(manifest_path)
        if manifest_source.exists():
            manifest = pd.read_csv(manifest_source, low_memory=False).fillna("")
            manifest_summary = {
                "manifest": str(manifest_source),
                "run_count": int(len(manifest)),
                "status_counts": manifest["status"].value_counts().astype(int).to_dict()
                if "status" in manifest.columns
                else {},
                "scenario_counts": manifest["scenario_id"].value_counts().astype(int).to_dict()
                if "scenario_id" in manifest.columns
                else {},
                "event_type_counts": manifest["event_type"].value_counts().astype(int).to_dict()
                if "event_type" in manifest.columns
                else {},
                "signal_variant_counts": manifest["signal_variant"].value_counts().astype(int).to_dict()
                if "signal_variant" in manifest.columns
                else {},
            }

    movement_stats = (
        df.groupby(["tls_id", "incoming_edge", "movement_id", "turn_type"], dropna=False)
        .agg(
            speed_kmh=("speed_kmh", "mean"),
            queue_mean=("queue_veh", "mean"),
            queue_max=("queue_veh", "max"),
            arrival_flow=("arrival_flow", "sum"),
            discharge_flow=("discharge_flow", "sum"),
            occupancy=("occupancy", "mean"),
            green_share=("is_green", "mean"),
        )
        .reset_index()
    )
    movement_stats["residual"] = movement_stats["arrival_flow"] - movement_stats["discharge_flow"]
    movement_stats["discharge_ratio"] = movement_stats["discharge_flow"] / (
        movement_stats["arrival_flow"] + 1e-9
    )
    movement_stats["bottleneck_score"] = (
        movement_stats["queue_mean"] * 1.8
        + movement_stats["queue_max"] * 0.15
        + movement_stats["arrival_flow"].clip(lower=0) / 10000.0
        + movement_stats["residual"].clip(lower=0) / 1000.0
        + (1.0 - movement_stats["green_share"]).clip(lower=0) * 4.0
    )
    movement_top = movement_stats.sort_values(
        ["bottleneck_score", "queue_mean", "arrival_flow"],
        ascending=False,
    ).head(top_n)

    approach_stats = (
        df.groupby(["tls_id", "incoming_edge"], dropna=False)
        .agg(
            speed_kmh=("speed_kmh", "mean"),
            queue_mean=("queue_veh", "mean"),
            queue_max=("queue_veh", "max"),
            arrival_flow=("arrival_flow", "sum"),
            discharge_flow=("discharge_flow", "sum"),
            occupancy=("occupancy", "mean"),
            green_share=("is_green", "mean"),
        )
        .reset_index()
    )
    approach_stats["residual"] = approach_stats["arrival_flow"] - approach_stats["discharge_flow"]
    approach_stats["discharge_ratio"] = approach_stats["discharge_flow"] / (
        approach_stats["arrival_flow"] + 1e-9
    )
    approach_stats["bottleneck_score"] = (
        approach_stats["queue_mean"] * 1.8
        + approach_stats["queue_max"] * 0.15
        + approach_stats["arrival_flow"].clip(lower=0) / 12000.0
        + approach_stats["residual"].clip(lower=0) / 1200.0
        + (1.0 - approach_stats["green_share"]).clip(lower=0) * 3.0
    )
    approach_top = approach_stats.sort_values(
        ["bottleneck_score", "queue_mean", "arrival_flow"],
        ascending=False,
    ).head(top_n)

    summary = {
        "status": "ok",
        "csv": str(source),
        "rows": int(len(df)),
        "run_count": int(df["run_id"].nunique()) if "run_id" in df.columns else None,
        "movement_count": int(df["movement_id"].nunique()),
        "step_count": int(df["step"].nunique()) if "step" in df.columns else None,
        "scenario_counts": scenario_counts,
        "event_counts": event_counts,
        "signal_variant_counts": signal_variant_counts,
        "target_summary": target_summary,
        "turn_type_summary": turn_type_summary,
        "tls_sample_summary": tls_sample_summary,
        "manifest_summary": manifest_summary,
        "speed_summary": speed_summary,
        "active_speed_by_scenario": scenario_speed,
        "top_bottleneck_movements": movement_top.to_dict(orient="records"),
        "top_bottleneck_approaches": approach_top.to_dict(orient="records"),
    }

    summary_out = Path(summary_path)
    bottleneck_out = Path(bottleneck_path)
    approach_out = Path(approach_path)
    for path in [summary_out, bottleneck_out, approach_out]:
        path.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    movement_top.to_csv(bottleneck_out, index=False, encoding="utf-8")
    approach_top.to_csv(approach_out, index=False, encoding="utf-8")
    return summary


def _mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return round(float(series.mean()), 4)


def _quantile(series: pd.Series, q: float) -> float:
    if series.empty:
        return 0.0
    return round(float(series.quantile(q)), 4)


def _weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    weight_sum = float(df[weight_col].sum())
    if weight_sum <= 1e-9:
        return 0.0
    return round(float((df[value_col] * df[weight_col]).sum() / weight_sum), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose movement-level speed and bottlenecks.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--out", default=None, help="Alias for --summary.")
    parser.add_argument("--bottlenecks", default=str(DEFAULT_BOTTLENECKS))
    parser.add_argument("--approaches", default=str(DEFAULT_APPROACHES))
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()
    summary_path = args.out or args.summary
    summary = diagnose_movement_data(
        args.csv,
        args.manifest,
        summary_path,
        args.bottlenecks,
        args.approaches,
        args.top_n,
    )
    print(json.dumps(summary["speed_summary"], ensure_ascii=False, indent=2))
    print(f"movement bottlenecks: {args.bottlenecks}")
    print(f"approach bottlenecks: {args.approaches}")


if __name__ == "__main__":
    main()
