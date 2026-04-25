from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PredictionConfig


TARGET_TO_COLUMN = {
    "flow": "flow",
    "speed": "speed_mps",
    "queue": "queue",
}
INCIDENT_COLUMN = "incident_flag"


@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_feature_names: list[str]
    edge_ids: list[str]
    targets: list[str]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    sample_run_ids: np.ndarray
    sample_scenario_ids: np.ndarray
    sample_target_steps: np.ndarray
    sample_target_timestamps: np.ndarray
    sample_incident_flags: np.ndarray

    def save(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "sliding_windows.npz",
            X=self.X,
            y=self.y,
            x_mean=self.x_mean,
            x_std=self.x_std,
            y_mean=self.y_mean,
            y_std=self.y_std,
            train_idx=self.train_idx,
            val_idx=self.val_idx,
            test_idx=self.test_idx,
            sample_run_ids=self.sample_run_ids,
            sample_scenario_ids=self.sample_scenario_ids,
            sample_target_steps=self.sample_target_steps,
            sample_target_timestamps=self.sample_target_timestamps,
            sample_incident_flags=self.sample_incident_flags,
        )
        metadata = {
            "feature_names": self.feature_names,
            "target_feature_names": self.target_feature_names,
            "edge_ids": self.edge_ids,
            "targets": self.targets,
            "shape": {"X": list(self.X.shape), "y": list(self.y.shape)},
            "splits": {
                "train": self.train_idx.tolist(),
                "val": self.val_idx.tolist(),
                "test": self.test_idx.tolist(),
            },
        }
        (out / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, input_dir: str | Path) -> "DatasetBundle":
        src = Path(input_dir)
        arrays = np.load(src / "sliding_windows.npz", allow_pickle=True)
        metadata = json.loads((src / "metadata.json").read_text(encoding="utf-8"))
        return cls(
            X=arrays["X"],
            y=arrays["y"],
            feature_names=list(metadata["feature_names"]),
            target_feature_names=list(metadata["target_feature_names"]),
            edge_ids=list(metadata["edge_ids"]),
            targets=list(metadata["targets"]),
            x_mean=arrays["x_mean"],
            x_std=arrays["x_std"],
            y_mean=arrays["y_mean"],
            y_std=arrays["y_std"],
            train_idx=arrays["train_idx"],
            val_idx=arrays["val_idx"],
            test_idx=arrays["test_idx"],
            sample_run_ids=arrays["sample_run_ids"],
            sample_scenario_ids=arrays["sample_scenario_ids"],
            sample_target_steps=arrays["sample_target_steps"],
            sample_target_timestamps=arrays["sample_target_timestamps"],
            sample_incident_flags=arrays["sample_incident_flags"],
        )

    def metadata_for_artifact(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "target_feature_names": self.target_feature_names,
            "edge_ids": self.edge_ids,
            "targets": self.targets,
            "x_mean": self.x_mean.tolist(),
            "x_std": self.x_std.tolist(),
            "y_mean": self.y_mean.tolist(),
            "y_std": self.y_std.tolist(),
            "history_steps": int(self.X.shape[1]),
            "horizon_steps": int(self.y.shape[1]),
            "input_size": int(self.X.shape[2]),
            "output_size": int(self.y.shape[2]),
        }


def build_dataset_from_csv(
    csv_path: str | Path,
    config: PredictionConfig,
) -> DatasetBundle:
    df = pd.read_csv(csv_path)
    required = {
        "run_id",
        "scenario_id",
        "timestamp",
        "step",
        "edge_id",
        "flow",
        "speed_mps",
        "queue",
        "incident_flag",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

    edge_ids = list(config.observed_edges)
    targets = list(config.targets)
    feature_names = [
        f"{edge_id}__{feature}"
        for edge_id in edge_ids
        for feature in [*targets, INCIDENT_COLUMN]
    ] + ["tod_sin", "tod_cos"]
    target_feature_names = [f"{edge_id}__{target}" for edge_id in edge_ids for target in targets]

    L = config.history_steps
    H = config.horizon_steps

    X_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    sample_run_ids: list[str] = []
    sample_scenario_ids: list[str] = []
    sample_target_steps: list[int] = []
    sample_target_timestamps: list[str] = []
    sample_incident_flags: list[bool] = []
    run_snapshot_counts: dict[str, int] = {}
    run_window_counts: dict[str, int] = {}

    for run_id, run_df in df.groupby("run_id", sort=True):
        input_snapshots: list[np.ndarray] = []
        target_snapshots: list[np.ndarray] = []
        snapshot_steps: list[int] = []
        snapshot_timestamps: list[str] = []
        snapshot_incident_flags: list[bool] = []
        snapshot_scenario_id = ""

        for _, group in run_df.groupby(["step", "timestamp"], sort=True):
            by_edge = {row.edge_id: row for row in group.itertuples(index=False)}
            if any(edge_id not in by_edge for edge_id in edge_ids):
                continue

            timestamp = str(group.iloc[0]["timestamp"])
            tod_sin, tod_cos = encode_time_of_day(timestamp)
            snapshot_scenario_id = str(group.iloc[0]["scenario_id"])
            input_values: list[float] = []
            output_values: list[float] = []
            incident_any = False

            for edge_id in edge_ids:
                row = by_edge[edge_id]
                for target in targets:
                    value = float(getattr(row, TARGET_TO_COLUMN[target]))
                    input_values.append(value)
                    output_values.append(value)
                incident_value = float(getattr(row, INCIDENT_COLUMN, 0.0) or 0.0)
                input_values.append(incident_value)
                incident_any = incident_any or incident_value > 0.0

            input_values.extend([tod_sin, tod_cos])
            input_snapshots.append(np.asarray(input_values, dtype=np.float32))
            target_snapshots.append(np.asarray(output_values, dtype=np.float32))
            snapshot_steps.append(int(group.iloc[0]["step"]))
            snapshot_timestamps.append(timestamp)
            snapshot_incident_flags.append(incident_any)

        run_snapshot_counts[str(run_id)] = len(input_snapshots)
        if len(input_snapshots) < L + H:
            run_window_counts[str(run_id)] = 0
            continue

        X_series = np.stack(input_snapshots, axis=0)
        y_series = np.stack(target_snapshots, axis=0)
        window_count = 0
        for start in range(0, X_series.shape[0] - L - H + 1):
            target_start = start + L
            X_rows.append(X_series[start:target_start])
            y_rows.append(y_series[target_start : target_start + H])
            sample_run_ids.append(str(run_id))
            sample_scenario_ids.append(snapshot_scenario_id)
            sample_target_steps.append(int(snapshot_steps[target_start]))
            sample_target_timestamps.append(str(snapshot_timestamps[target_start]))
            sample_incident_flags.append(any(snapshot_incident_flags[target_start : target_start + H]))
            window_count += 1
        run_window_counts[str(run_id)] = window_count

    if not X_rows:
        details = ", ".join(
            f"{run_id}:snapshots={run_snapshot_counts[run_id]}"
            for run_id in sorted(run_snapshot_counts)
        )
        raise ValueError(
            f"No valid sliding-window samples found for L={L}, H={H}. Per-run snapshots: {details}"
        )

    X_arr = np.stack(X_rows).astype(np.float32)
    y_arr = np.stack(y_rows).astype(np.float32)
    sample_run_ids_arr = np.asarray(sample_run_ids)
    sample_scenario_ids_arr = np.asarray(sample_scenario_ids)
    sample_target_steps_arr = np.asarray(sample_target_steps, dtype=np.int64)
    sample_target_timestamps_arr = np.asarray(sample_target_timestamps)
    sample_incident_flags_arr = np.asarray(sample_incident_flags, dtype=bool)

    train_idx, val_idx, test_idx = split_indices_by_run(
        sample_run_ids_arr,
        sample_incident_flags_arr,
    )

    x_train_flat = X_arr[train_idx].reshape(-1, X_arr.shape[-1])
    y_train_flat = y_arr[train_idx].reshape(-1, y_arr.shape[-1])
    x_mean = x_train_flat.mean(axis=0).astype(np.float32)
    x_std = x_train_flat.std(axis=0).astype(np.float32)
    y_mean = y_train_flat.mean(axis=0).astype(np.float32)
    y_std = y_train_flat.std(axis=0).astype(np.float32)
    x_std[x_std < 1e-6] = 1.0
    y_std[y_std < 1e-6] = 1.0

    return DatasetBundle(
        X=X_arr,
        y=y_arr,
        feature_names=feature_names,
        target_feature_names=target_feature_names,
        edge_ids=edge_ids,
        targets=targets,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        sample_run_ids=sample_run_ids_arr,
        sample_scenario_ids=sample_scenario_ids_arr,
        sample_target_steps=sample_target_steps_arr,
        sample_target_timestamps=sample_target_timestamps_arr,
        sample_incident_flags=sample_incident_flags_arr,
    )


def split_indices_by_run(
    sample_run_ids: np.ndarray,
    sample_incident_flags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered_run_ids = list(dict.fromkeys(sample_run_ids.tolist()))
    if len(ordered_run_ids) < 3:
        raise ValueError(f"Need at least 3 run_ids for train/val/test split; got {len(ordered_run_ids)}")

    run_has_incident = {
        run_id: bool(np.any(sample_incident_flags[sample_run_ids == run_id]))
        for run_id in ordered_run_ids
    }
    incident_runs = [run_id for run_id in ordered_run_ids if run_has_incident[run_id]]
    regular_runs = [run_id for run_id in ordered_run_ids if not run_has_incident[run_id]]

    if incident_runs and regular_runs:
        train_regular, val_regular, test_regular = time_ordered_split(regular_runs)
        train_incident, val_incident, test_incident = time_ordered_split(incident_runs)
        train_runs = [*train_regular, *train_incident]
        val_runs = [*val_regular, *val_incident]
        test_runs = [*test_regular, *test_incident]
    else:
        train_runs, val_runs, test_runs = time_ordered_split(ordered_run_ids)

    train_set = set(train_runs)
    val_set = set(val_runs)
    test_set = set(test_runs)

    train_idx = np.asarray(
        [idx for idx, run_id in enumerate(sample_run_ids.tolist()) if run_id in train_set],
        dtype=np.int64,
    )
    val_idx = np.asarray(
        [idx for idx, run_id in enumerate(sample_run_ids.tolist()) if run_id in val_set],
        dtype=np.int64,
    )
    test_idx = np.asarray(
        [idx for idx, run_id in enumerate(sample_run_ids.tolist()) if run_id in test_set],
        dtype=np.int64,
    )
    return train_idx, val_idx, test_idx


def time_ordered_split(items: list[str]) -> tuple[list[str], list[str], list[str]]:
    n_items = len(items)
    if n_items == 14:
        train_end = 10
        val_end = 12
    else:
        train_end = max(1, int(n_items * 0.7))
        val_end = max(train_end + 1, int(n_items * 0.85))
        if n_items - val_end < 1:
            val_end = n_items - 1
    return items[:train_end], items[train_end:val_end], items[val_end:]


def encode_time_of_day(timestamp: str) -> tuple[float, float]:
    dt = datetime.fromisoformat(timestamp)
    seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
    angle = (2.0 * math.pi * seconds) / 86400.0
    return float(math.sin(angle)), float(math.cos(angle))


def scale_X(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def scale_y(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((y - mean) / std).astype(np.float32)


def inverse_scale_y(y_scaled: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (y_scaled * std + mean).astype(np.float32)


def window_to_matrix(
    window: list[Any],
    edge_ids: list[str],
    targets: list[str],
    config: PredictionConfig,
) -> np.ndarray:
    rows: list[list[float]] = []
    inferred_start = datetime.fromisoformat(config.simulation_start_iso)
    for step_index, raw_step in enumerate(window):
        step = _as_dict(raw_step)
        by_edge = {
            _as_dict(node).get("edge_id"): _as_dict(node)
            for node in step.get("nodes", [])
        }
        values: list[float] = []
        for edge_id in edge_ids:
            node = by_edge.get(edge_id, {})
            for target in targets:
                value = node.get(target)
                if target == "speed" and value is None:
                    value = node.get("speed_mps")
                values.append(float(value or 0.0))
            values.append(float(node.get("incident_flag") or 0.0))

        timestamp = step.get("timestamp")
        if timestamp:
            tod_sin, tod_cos = encode_time_of_day(str(timestamp))
        else:
            inferred_timestamp = inferred_start + timedelta(
                seconds=step_index * config.sample_interval_s
            )
            tod_sin, tod_cos = encode_time_of_day(inferred_timestamp.isoformat())
        values.extend([tod_sin, tod_cos])
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def matrix_to_prediction_payload(
    y: np.ndarray,
    edge_ids: list[str],
    targets: list[str],
    model_name: str,
    horizon: int | None = None,
) -> dict[str, Any]:
    if y.ndim != 2:
        raise ValueError(f"Expected y with shape [H, O], got {y.shape}")
    horizon_steps = horizon or y.shape[0]
    if horizon_steps <= y.shape[0]:
        y_out = y[:horizon_steps]
    else:
        pad = np.repeat(y[-1:, :], horizon_steps - y.shape[0], axis=0)
        y_out = np.concatenate([y, pad], axis=0)

    nodes = []
    width = len(targets)
    for edge_index, edge_id in enumerate(edge_ids):
        node_payload: dict[str, Any] = {"edge_id": edge_id}
        offset = edge_index * width
        for target_index, target in enumerate(targets):
            values = y_out[:, offset + target_index]
            node_payload[f"pred_{target}"] = [float(v) for v in values]
        nodes.append(node_payload)

    return {
        "model": model_name,
        "horizon": list(range(1, horizon_steps + 1)),
        "nodes": nodes,
    }


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)
