from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import load_prediction_config
from .dataset import (
    DatasetBundle,
    build_dataset_from_csv,
    inverse_scale_y,
    scale_X,
    scale_y,
)
from .metrics import append_metrics_csv, regression_metrics
from .schemas import PredictRequest
from .service import PredictionService
from .torch_models import LSTMForecaster, TransformerForecaster


NOTE = "P4 movement-level control-feature closed-loop pipeline"
ARTIFACT_FILES = [
    "xgboost_model.joblib",
    "lstm_model.pt",
    "transformer_v1_model.pt",
    "best_model.pt",
    "best_model.joblib",
    "model_registry.json",
]
REPORT_FILES = [
    "metrics.csv",
    "control_feature_ablation.csv",
    "smoke_test_summary.json",
    "p4_training_summary.json",
]


def train_all(
    config_path: str | Path = "configs/prediction_config.json",
    csv_path: str | Path = "data/raw/batch_movement_aggregates.csv",
    dataset_dir: str | Path = "data/datasets/p4_movement_control",
    artifact_dir: str | Path = "models/artifacts",
    report_dir: str | Path = "reports",
) -> dict[str, Any]:
    config = load_prediction_config(config_path)
    dataset = build_dataset_from_csv(csv_path, config, include_control_features=True)
    ablation_dataset = build_dataset_from_csv(csv_path, config, include_control_features=False)

    artifact_root = Path(artifact_dir)
    report_root = Path(report_dir)
    archive_dir = archive_existing_outputs(dataset_dir, artifact_root, report_root)
    if archive_dir:
        print(f"archived previous training outputs to {archive_dir}")

    dataset.save(dataset_dir)
    ablation_dataset_dir = Path(dataset_dir).with_name(f"{Path(dataset_dir).name}_no_control")
    ablation_dataset.save(ablation_dataset_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    ablation_artifact_root = artifact_root / "control_feature_ablation_off"
    ablation_artifact_root.mkdir(parents=True, exist_ok=True)
    (report_root / "figures").mkdir(parents=True, exist_ok=True)

    metrics_rows, predictions = train_model_suite(dataset, config.device, artifact_root)
    ablation_rows, ablation_predictions = train_model_suite(
        ablation_dataset,
        config.device,
        ablation_artifact_root,
    )

    append_metrics_csv(report_root / "metrics.csv", metrics_rows)
    write_control_ablation_csv(
        report_root / "control_feature_ablation.csv",
        [
            annotate_feature_set(row, "with_control") for row in metrics_rows
        ]
        + [
            annotate_feature_set(row, "without_control") for row in ablation_rows
        ],
    )
    best = select_best(metrics_rows)
    if best:
        write_best_artifact(best["model"], artifact_root)

    plot_prediction_comparison(
        dataset.y[dataset.test_idx],
        predictions,
        dataset,
        report_root / "figures" / "prediction_comparison_overall.png",
    )
    plot_incident_vs_normal(
        dataset,
        predictions,
        best["model"] if best else next(iter(predictions), "ha_baseline"),
        report_root / "figures" / "normal_vs_incident_prediction.png",
    )
    plot_subset_metric_bars(
        metrics_rows,
        report_root / "figures" / "incident_subset_metrics.png",
    )
    plot_control_feature_ablation(
        report_root / "control_feature_ablation.csv",
        report_root / "figures" / "control_feature_ablation.png",
    )

    summary = {
        "status": "ok",
        "note": NOTE,
        "archive_dir": str(archive_dir) if archive_dir else "",
        "dataset": {
            "X_shape": list(dataset.X.shape),
            "y_shape": list(dataset.y.shape),
            "n_train": int(len(dataset.train_idx)),
            "n_val": int(len(dataset.val_idx)),
            "n_test": int(len(dataset.test_idx)),
            "input_features": int(dataset.X.shape[-1]),
            "output_features": int(dataset.y.shape[-1]),
        },
        "ablation_dataset": {
            "X_shape": list(ablation_dataset.X.shape),
            "y_shape": list(ablation_dataset.y.shape),
            "input_features": int(ablation_dataset.X.shape[-1]),
            "output_features": int(ablation_dataset.y.shape[-1]),
        },
        "base_demand_factor": config.base_demand_factor,
        "control_features_enabled": True,
        "best_model": best["model"] if best else "ha_baseline",
        "metrics_path": str(report_root / "metrics.csv"),
        "control_ablation_path": str(report_root / "control_feature_ablation.csv"),
    }
    (report_root / "p4_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def train_model_suite(
    dataset: DatasetBundle,
    device: str,
    artifact_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}

    test_idx = dataset.test_idx
    y_true_test = dataset.y[test_idx]
    subset_masks = build_subset_masks(dataset, test_idx)

    ha_pred = predict_ha(dataset, test_idx)
    predictions["ha_baseline"] = ha_pred
    metrics_rows.extend(rows_for_subsets("ha_baseline", dataset, test_idx, y_true_test, ha_pred, subset_masks))

    try:
        xgb_pred = train_xgboost(dataset, artifact_root)
        predictions["xgboost"] = xgb_pred
        metrics_rows.extend(rows_for_subsets("xgboost", dataset, test_idx, y_true_test, xgb_pred, subset_masks))
    except Exception as exc:
        metrics_rows.append(error_row("xgboost", str(exc)))

    try:
        lstm_pred = train_torch_model("lstm", dataset, device, artifact_root)
        predictions["lstm"] = lstm_pred
        metrics_rows.extend(rows_for_subsets("lstm", dataset, test_idx, y_true_test, lstm_pred, subset_masks))
    except Exception as exc:
        metrics_rows.append(error_row("lstm", str(exc)))

    try:
        transformer_pred = train_torch_model(
            "transformer_v1",
            dataset,
            device,
            artifact_root,
        )
        predictions["transformer_v1"] = transformer_pred
        metrics_rows.extend(
            rows_for_subsets("transformer_v1", dataset, test_idx, y_true_test, transformer_pred, subset_masks)
        )
    except Exception as exc:
        metrics_rows.append(error_row("transformer_v1", str(exc)))

    return metrics_rows, predictions


def annotate_feature_set(row: dict[str, Any], feature_set: str) -> dict[str, Any]:
    annotated = dict(row)
    annotated["feature_set"] = feature_set
    return annotated


def archive_existing_outputs(
    dataset_dir: str | Path,
    artifact_root: Path,
    report_root: Path,
) -> Path | None:
    existing_paths: list[tuple[Path, Path]] = []
    dataset_path = Path(dataset_dir)
    if dataset_path.exists():
        existing_paths.append((dataset_path, Path("datasets") / dataset_path.name))

    for filename in ARTIFACT_FILES:
        path = artifact_root / filename
        if path.exists():
            existing_paths.append((path, Path("artifacts") / filename))

    for filename in REPORT_FILES:
        path = report_root / filename
        if path.exists():
            existing_paths.append((path, Path("reports") / filename))
    figures_dir = report_root / "figures"
    if figures_dir.exists():
        existing_paths.append((figures_dir, Path("reports") / "figures"))

    if not existing_paths:
        return None

    archive_dir = Path("data/archive") / f"training_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for source, relative_target in existing_paths:
        target = archive_dir / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    return archive_dir


def build_subset_masks(dataset: DatasetBundle, indices: np.ndarray) -> dict[str, np.ndarray]:
    incident_flags = dataset.sample_incident_flags[indices].astype(bool)
    signal_variants = dataset.sample_signal_variants[indices].astype(str)
    return {
        "overall": np.ones(len(indices), dtype=bool),
        "non_incident": ~incident_flags,
        "incident": incident_flags,
        "control_perturbation": signal_variants != "webster_base",
    }


def predict_ha(dataset: DatasetBundle, indices: np.ndarray) -> np.ndarray:
    X = dataset.X[indices]
    mean_history = []
    input_indices = target_input_indices(dataset)
    for sample in X:
        edge_values = []
        for edge_id in dataset.edge_ids:
            for target in dataset.targets:
                edge_values.append(float(sample[:, input_indices[(edge_id, target)]].mean()))
        mean_history.append(edge_values)
    repeated = np.asarray(mean_history, dtype=np.float32)[:, None, :]
    return np.repeat(repeated, dataset.y.shape[1], axis=1).astype(np.float32)


def target_input_indices(dataset: DatasetBundle) -> dict[tuple[str, str], int]:
    return {
        (edge_id, target): dataset.feature_names.index(f"{edge_id}__{target}")
        for edge_id in dataset.edge_ids
        for target in dataset.targets
    }


def train_xgboost(dataset: DatasetBundle, artifact_root: Path) -> np.ndarray:
    from sklearn.decomposition import PCA
    from xgboost import XGBRegressor

    X_train, y_train = flatten_scaled(dataset, dataset.train_idx)
    X_test = scale_X(dataset.X[dataset.test_idx], dataset.x_mean, dataset.x_std).reshape(
        len(dataset.test_idx),
        -1,
    )
    target_reducer = None
    y_fit = y_train
    max_native_outputs = int(os.environ.get("TRAFFIC_XGB_NATIVE_OUTPUT_LIMIT", "2000"))
    if y_train.shape[1] > max_native_outputs:
        n_components = min(
            int(os.environ.get("TRAFFIC_XGB_PCA_COMPONENTS", "64")),
            max(1, y_train.shape[0] - 1),
            y_train.shape[1],
        )
        target_reducer = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        y_fit = target_reducer.fit_transform(y_train)

    model = XGBRegressor(
        n_estimators=int(os.environ.get("TRAFFIC_XGB_ESTIMATORS", "8")),
        max_depth=int(os.environ.get("TRAFFIC_XGB_MAX_DEPTH", "2")),
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        multi_strategy="multi_output_tree",
    )
    model.fit(X_train, y_fit)
    pred_scaled_flat = np.asarray(model.predict(X_test), dtype=np.float32)
    if target_reducer is not None:
        pred_scaled_flat = target_reducer.inverse_transform(pred_scaled_flat).astype(np.float32)
    pred_scaled = pred_scaled_flat.reshape(
        len(dataset.test_idx),
        dataset.y.shape[1],
        dataset.y.shape[2],
    )
    pred = inverse_scale_y(pred_scaled, dataset.y_mean, dataset.y_std)
    artifact = {
        "kind": "xgboost",
        "model_name": "xgboost",
        "xgboost_multi_strategy": "multi_output_tree",
        "target_reducer": target_reducer,
        "target_reducer_kind": "pca" if target_reducer is not None else "",
        "model": model,
        **dataset.metadata_for_artifact(),
    }
    joblib.dump(artifact, artifact_root / "xgboost_model.joblib")
    return pred


def train_torch_model(
    kind: str,
    dataset: DatasetBundle,
    device: str,
    artifact_root: Path,
) -> np.ndarray:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    X_train = scale_X(dataset.X[dataset.train_idx], dataset.x_mean, dataset.x_std)
    y_train = scale_y(dataset.y[dataset.train_idx], dataset.y_mean, dataset.y_std)
    X_val = scale_X(dataset.X[dataset.val_idx], dataset.x_mean, dataset.x_std)
    y_val = scale_y(dataset.y[dataset.val_idx], dataset.y_mean, dataset.y_std)
    X_test = scale_X(dataset.X[dataset.test_idx], dataset.x_mean, dataset.x_std)

    input_size = dataset.X.shape[-1]
    output_size = dataset.y.shape[-1]
    horizon_steps = dataset.y.shape[1]
    if kind == "lstm":
        model_config = {
            "input_size": input_size,
            "output_size": output_size,
            "horizon_steps": horizon_steps,
            "hidden_size": 64,
            "num_layers": 1,
            "dropout": 0.1,
        }
        model = LSTMForecaster(**model_config)
    elif kind == "transformer_v1":
        model_config = {
            "input_size": input_size,
            "output_size": output_size,
            "horizon_steps": horizon_steps,
            "d_model": 128,
            "n_heads": 4,
            "num_layers": 3,
            "dropout": 0.1,
        }
        model = TransformerForecaster(**model_config)
    else:
        raise ValueError(f"Unknown torch model kind: {kind}")

    model.to(device)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=min(64, len(train_ds)), shuffle=True)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_val = math.inf
    patience = int(os.environ.get("TRAFFIC_TRAIN_PATIENCE", "4"))
    stale_epochs = 0
    max_epochs = int(os.environ.get("TRAFFIC_TRAIN_EPOCHS", "12"))

    for _ in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(X_val).to(device))
            val_loss = float(criterion(val_pred, torch.from_numpy(y_val).to(device)).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred_scaled = model(torch.from_numpy(X_test).to(device)).detach().cpu().numpy()
    pred = inverse_scale_y(test_pred_scaled, dataset.y_mean, dataset.y_std)

    artifact = {
        "kind": kind,
        "model_name": kind,
        "model_config": model_config,
        "state_dict": model.state_dict(),
        **dataset.metadata_for_artifact(),
    }
    torch.save(artifact, artifact_root / f"{kind}_model.pt")
    return pred


def flatten_scaled(dataset: DatasetBundle, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = scale_X(dataset.X[indices], dataset.x_mean, dataset.x_std)
    y = scale_y(dataset.y[indices], dataset.y_mean, dataset.y_std)
    return X.reshape(len(indices), -1), y.reshape(len(indices), -1)


def rows_for_subsets(
    model: str,
    dataset: DatasetBundle,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subset_masks: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_run_ids = dataset.sample_run_ids[indices]
    for subset_name, mask in subset_masks.items():
        if not np.any(mask):
            continue
        metrics = regression_metrics(y_true[mask], y_pred[mask])
        metrics.update(
            {
                "model": model,
                "split": "test",
                "subset": subset_name,
                "n_samples": int(np.sum(mask)),
                "run_count": int(len(set(sample_run_ids[mask].tolist()))),
                "note": NOTE,
            }
        )
        rows.append(metrics)
    return rows


def error_row(model: str, message: str) -> dict[str, Any]:
    return {
        "model": model,
        "split": "test",
        "subset": "overall",
        "n_samples": 0,
        "run_count": 0,
        "note": f"failed: {message}",
    }


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        row
        for row in rows
        if row.get("subset") == "overall" and isinstance(row.get("mae"), float)
    ]
    if not valid:
        return None
    return min(valid, key=lambda row: row["mae"])


def write_best_artifact(model_name: str, artifact_root: Path) -> None:
    source_by_model = {
        "xgboost": artifact_root / "xgboost_model.joblib",
        "lstm": artifact_root / "lstm_model.pt",
        "transformer_v1": artifact_root / "transformer_v1_model.pt",
    }
    source = source_by_model.get(model_name)
    active_artifact = source.name if source and source.exists() else ""
    active_alias = ""
    if source and source.exists():
        dest_name = "best_model.joblib" if source.suffix == ".joblib" else "best_model.pt"
        dest_path = artifact_root / dest_name
        try:
            if dest_path.exists() and dest_path.resolve() != source.resolve():
                dest_path.unlink()
            if dest_path.resolve() != source.resolve():
                shutil.copyfile(source, dest_path)
            active_alias = dest_name
        except PermissionError as exc:
            print(
                f"Could not refresh compatibility alias {dest_path.name}; "
                f"using registry pointer to {source.name}: {exc}"
            )

    registry = {
        "active_model": model_name,
        "active_artifact": active_artifact,
        "active_alias": active_alias,
        "note": NOTE,
    }
    (artifact_root / "model_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def plot_prediction_comparison(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    dataset: DatasetBundle,
    output_path: Path,
) -> None:
    if y_true.size == 0 or not predictions:
        return
    feature_idx = 0
    plt.figure(figsize=(9, 5))
    plt.plot(y_true[0, :, feature_idx], label="true", linewidth=2)
    for name, pred in predictions.items():
        plt.plot(pred[0, :, feature_idx], label=name, linestyle="--")
    plt.title(f"Forecast Comparison: {dataset.target_feature_names[feature_idx]}")
    plt.xlabel("horizon step")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_incident_vs_normal(
    dataset: DatasetBundle,
    predictions: dict[str, np.ndarray],
    model_name: str,
    output_path: Path,
) -> None:
    if not predictions or model_name not in predictions:
        return
    pred = predictions[model_name]
    incident_mask = dataset.sample_incident_flags[dataset.test_idx].astype(bool)
    non_incident_mask = ~incident_mask
    if not np.any(incident_mask) or not np.any(non_incident_mask):
        return

    incident_pos = int(np.flatnonzero(incident_mask)[0])
    normal_pos = int(np.flatnonzero(non_incident_mask)[0])
    feature_offset = 1  # first edge speed_mps

    plt.figure(figsize=(9, 5))
    plt.plot(dataset.y[dataset.test_idx][normal_pos, :, feature_offset] * 3.6, label="normal true", linewidth=2)
    plt.plot(pred[normal_pos, :, feature_offset] * 3.6, label="normal pred", linestyle="--", linewidth=2)
    plt.plot(dataset.y[dataset.test_idx][incident_pos, :, feature_offset] * 3.6, label="incident true", linewidth=2)
    plt.plot(pred[incident_pos, :, feature_offset] * 3.6, label="incident pred", linestyle="--", linewidth=2)
    plt.title(f"Normal vs Incident Speed Forecast ({model_name})")
    plt.xlabel("horizon step")
    plt.ylabel("km/h")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_subset_metric_bars(rows: list[dict[str, Any]], output_path: Path) -> None:
    relevant = [
        row for row in rows if row.get("subset") in {"overall", "incident"} and isinstance(row.get("mae"), float)
    ]
    if not relevant:
        return
    models = sorted({row["model"] for row in relevant})
    overall = {
        row["model"]: row["mae"]
        for row in relevant
        if row.get("subset") == "overall"
    }
    incident = {
        row["model"]: row["mae"]
        for row in relevant
        if row.get("subset") == "incident"
    }
    x = np.arange(len(models))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, [overall.get(model, 0.0) for model in models], width=width, label="overall")
    plt.bar(x + width / 2, [incident.get(model, 0.0) for model in models], width=width, label="incident")
    plt.xticks(x, models)
    plt.ylabel("MAE")
    plt.title("Overall vs Incident MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_control_ablation_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature_set",
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
        import csv

        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_control_feature_ablation(csv_path: str | Path, output_path: Path) -> None:
    import csv

    path = Path(csv_path)
    if not path.exists():
        return

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if row.get("subset") in {"overall", "control_perturbation"} and row.get("mae"):
                try:
                    row["mae_value"] = float(row["mae"])
                except ValueError:
                    continue
                rows.append(row)
    if not rows:
        return

    labels = []
    values = []
    colors = []
    for row in rows:
        labels.append(f"{row['model']}\n{row['feature_set']}\n{row['subset']}")
        values.append(row["mae_value"])
        colors.append("#27a2ff" if row["feature_set"] == "with_control" else "#a0a7b5")

    plt.figure(figsize=(max(10, len(labels) * 0.55), 5))
    plt.bar(np.arange(len(values)), values, color=colors)
    plt.xticks(np.arange(len(values)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("MAE")
    plt.title("Control Feature Ablation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_smoke_test(
    config_path: str | Path = "configs/prediction_config.json",
    dataset_dir: str | Path = "data/datasets/p4_movement_control",
) -> dict[str, Any]:
    service = PredictionService(load_prediction_config(config_path))
    dataset = DatasetBundle.load(dataset_dir)
    idx = int(dataset.test_idx[0])
    window = []
    input_indices = target_input_indices(dataset)
    for step in dataset.X[idx]:
        items = []
        for edge_index, edge_id in enumerate(dataset.edge_ids):
            node = {"movement_id": edge_id} if dataset.observation_level == "movement" else {"edge_id": edge_id}
            if dataset.entity_metadata and edge_id in dataset.entity_metadata:
                node.update(dataset.entity_metadata[edge_id])
            for target in dataset.targets:
                node[target] = float(step[input_indices[(edge_id, target)]])
            incident_feature = f"{edge_id}__incident_flag"
            if incident_feature in dataset.feature_names:
                node["incident_flag"] = float(step[dataset.feature_names.index(incident_feature)])
            for control_feature in ("phase_id", "phase_elapsed_s", "green_remaining_s"):
                feature_name = f"{edge_id}__{control_feature}"
                if feature_name in dataset.feature_names:
                    node[control_feature] = float(step[dataset.feature_names.index(feature_name)])
            items.append(node)
        if dataset.observation_level == "movement":
            window.append({"timestamp": None, "movements": items})
        else:
            window.append({"timestamp": None, "nodes": items})
    payload = service.predict_request(PredictRequest(window=window, horizon=15))
    expected_key = "movements" if dataset.observation_level == "movement" else "nodes"
    if payload["horizon"][-1] != 15 or len(payload.get(expected_key, [])) != len(dataset.edge_ids):
        raise RuntimeError("Smoke API prediction returned an unexpected shape")
    return {
        "status": "ok",
        "model": payload["model"],
        "observation_level": dataset.observation_level,
        expected_key: len(payload.get(expected_key, [])),
        "legacy_nodes": len(payload.get("nodes", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train low-demand incident-aware traffic prediction models.")
    parser.add_argument("--config", default="configs/prediction_config.json")
    parser.add_argument("--csv", default="data/raw/batch_movement_aggregates.csv")
    parser.add_argument("--dataset-dir", default="data/datasets/p4_movement_control")
    parser.add_argument("--artifact-dir", default="models/artifacts")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    summary = train_all(
        args.config,
        args.csv,
        args.dataset_dir,
        args.artifact_dir,
        args.report_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.smoke_test:
        print(json.dumps(run_smoke_test(args.config, args.dataset_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
