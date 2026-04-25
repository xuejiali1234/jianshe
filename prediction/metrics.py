from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum(np.abs(y_true)))
    wape = float(np.sum(np.abs(err)) / denom) if denom > 1e-8 else 0.0
    result = {"mae": mae, "rmse": rmse, "wape": wape}
    for horizon in (5, 10, 15):
        idx = min(horizon, y_true.shape[1])
        window_err = y_pred[:, :idx] - y_true[:, :idx]
        window_abs = np.abs(window_err)
        window_denom = float(np.sum(np.abs(y_true[:, :idx])))
        result[f"mae_h{horizon}"] = float(np.mean(window_abs))
        result[f"rmse_h{horizon}"] = float(np.sqrt(np.mean(window_err**2)))
        result[f"wape_h{horizon}"] = (
            float(np.sum(window_abs) / window_denom) if window_denom > 1e-8 else 0.0
        )
    return result


def append_metrics_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "split",
        "n_samples",
        "run_count",
        "subset",
        "mae",
        "rmse",
        "wape",
        "mae_h5",
        "rmse_h5",
        "wape_h5",
        "mae_h10",
        "rmse_h10",
        "wape_h10",
        "mae_h15",
        "rmse_h15",
        "wape_h15",
        "note",
    ]
    with out.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
