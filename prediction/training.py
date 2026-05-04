from __future__ import annotations

import argparse
import csv
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
from .graph_utils import load_movement_adjacency, load_movement_attention_bias
from .metrics import append_metrics_csv, regression_metrics
from .schemas import PredictRequest
from .service import PredictionService
from .torch_models import LSTMForecaster, SpatioTemporalTransformerForecaster, TransformerForecaster


NOTE = "P4 movement-level control-feature closed-loop pipeline"
V2_VARIANTS = {
    "phase_embed_only": {
        "phase_embedding": True,
        "graph_bias": False,
        "queue_weight": False,
        "control_feature_scheme": "phase_embed_graph_v1",
        "graph_bias_scale": 0.0,
    },
    "graph_bias_only": {
        "phase_embedding": False,
        "graph_bias": True,
        "queue_weight": False,
        "control_feature_scheme": "phase_state_v1",
        "graph_bias_scale": 0.15,
    },
    "queue_weight_only": {
        "phase_embedding": False,
        "graph_bias": False,
        "queue_weight": True,
        "control_feature_scheme": "phase_state_v1",
        "graph_bias_scale": 0.0,
    },
}
ARTIFACT_FILES = [
    "xgboost_model.joblib",
    "lstm_model.pt",
    "transformer_v1_model.pt",
    "transformer_v2_model.pt",
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
    train_ablation: bool = True,
    control_feature_scheme: str | None = None,
) -> dict[str, Any]:
    config = load_prediction_config(config_path)
    if control_feature_scheme:
        config.control_feature_scheme = control_feature_scheme
    dataset = build_dataset_from_csv(csv_path, config, include_control_features=True)
    ablation_dataset = (
        build_dataset_from_csv(csv_path, config, include_control_features=False)
        if train_ablation
        else None
    )

    artifact_root = Path(artifact_dir)
    report_root = Path(report_dir)
    archive_dir = archive_existing_outputs(dataset_dir, artifact_root, report_root)
    if archive_dir:
        print(f"archived previous training outputs to {archive_dir}")

    dataset.save(dataset_dir)
    ablation_dataset_dir = Path(dataset_dir).with_name(f"{Path(dataset_dir).name}_no_control")
    if ablation_dataset is not None:
        ablation_dataset.save(ablation_dataset_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    ablation_artifact_root = artifact_root / "control_feature_ablation_off"
    if ablation_dataset is not None:
        ablation_artifact_root.mkdir(parents=True, exist_ok=True)
    (report_root / "figures").mkdir(parents=True, exist_ok=True)

    metrics_rows, predictions = train_model_suite(
        dataset,
        config.device,
        artifact_root,
        report_root / "figures",
    )
    ablation_rows: list[dict[str, Any]] = []
    if ablation_dataset is not None:
        ablation_rows, _ = train_model_suite(
            ablation_dataset,
            config.device,
            ablation_artifact_root,
            report_root / "figures" / "no_control",
        )

    append_metrics_csv(report_root / "metrics.csv", metrics_rows)
    if ablation_dataset is not None:
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
    plot_continuous_high_flow_series(
        dataset,
        predictions,
        report_root / "figures" / "prediction_comparison_continuous_high_flow.png",
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
    if ablation_dataset is not None:
        plot_control_feature_ablation(
            report_root / "control_feature_ablation.csv",
            report_root / "figures" / "control_feature_ablation.png",
        )
    write_feature_diagnostics_csv(
        dataset,
        predictions,
        report_root / "phase_state_target_turn_diagnostics.csv",
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
        "ablation_dataset": (
            {
                "X_shape": list(ablation_dataset.X.shape),
                "y_shape": list(ablation_dataset.y.shape),
                "input_features": int(ablation_dataset.X.shape[-1]),
                "output_features": int(ablation_dataset.y.shape[-1]),
            }
            if ablation_dataset is not None
            else {}
        ),
        "base_demand_factor": config.base_demand_factor,
        "control_features_enabled": True,
        "control_feature_scheme": dataset.control_feature_scheme,
        "best_model": best["model"] if best else "ha_baseline",
        "metrics_path": str(report_root / "metrics.csv"),
        "control_ablation_path": str(report_root / "control_feature_ablation.csv")
        if ablation_dataset is not None
        else "",
        "diagnostics_path": str(report_root / "phase_state_target_turn_diagnostics.csv"),
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
    loss_history_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    loss_histories: dict[str, list[dict[str, Any]]] = {}

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
        lstm_pred, lstm_history = train_torch_model(
            "lstm",
            dataset,
            device,
            artifact_root,
            loss_history_dir=loss_history_dir,
        )
        predictions["lstm"] = lstm_pred
        loss_histories["lstm"] = lstm_history
        metrics_rows.extend(rows_for_subsets("lstm", dataset, test_idx, y_true_test, lstm_pred, subset_masks))
    except Exception as exc:
        metrics_rows.append(error_row("lstm", str(exc)))

    try:
        transformer_pred, transformer_history = train_torch_model(
            "transformer_v1",
            dataset,
            device,
            artifact_root,
            loss_history_dir=loss_history_dir,
        )
        predictions["transformer_v1"] = transformer_pred
        loss_histories["transformer_v1"] = transformer_history
        metrics_rows.extend(
            rows_for_subsets("transformer_v1", dataset, test_idx, y_true_test, transformer_pred, subset_masks)
        )
    except Exception as exc:
        metrics_rows.append(error_row("transformer_v1", str(exc)))

    try:
        transformer_v2_pred, transformer_v2_history = train_torch_model(
            "transformer_v2",
            dataset,
            device,
            artifact_root,
            loss_history_dir=loss_history_dir,
        )
        predictions["transformer_v2"] = transformer_v2_pred
        loss_histories["transformer_v2"] = transformer_v2_history
        metrics_rows.extend(
            rows_for_subsets("transformer_v2", dataset, test_idx, y_true_test, transformer_v2_pred, subset_masks)
        )
    except Exception as exc:
        metrics_rows.append(error_row("transformer_v2", str(exc)))

    if loss_history_dir is not None and loss_histories:
        plot_torch_training_loss_curves(
            loss_histories,
            loss_history_dir / "torch_training_loss_curves.png",
        )

    return metrics_rows, predictions


def train_v2_ablation_suite(
    config_path: str | Path,
    csv_path: str | Path,
    dataset_root: str | Path,
    artifact_root: str | Path,
    report_root: str | Path,
) -> dict[str, Any]:
    root_dataset = Path(dataset_root)
    root_artifact = Path(artifact_root)
    root_report = Path(report_root)
    root_dataset.mkdir(parents=True, exist_ok=True)
    root_artifact.mkdir(parents=True, exist_ok=True)
    root_report.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    predictions_by_variant: dict[str, np.ndarray] = {}
    datasets_by_variant: dict[str, DatasetBundle] = {}
    for variant_name in ["phase_embed_only", "graph_bias_only", "queue_weight_only"]:
        variant = V2_VARIANTS[variant_name]
        config = load_prediction_config(config_path)
        config.control_feature_scheme = str(variant["control_feature_scheme"])
        dataset = build_dataset_from_csv(csv_path, config, include_control_features=True)
        dataset_dir = root_dataset / variant_name
        artifact_dir = root_artifact / variant_name
        dataset.save(dataset_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        pred, _history = train_torch_model(
            "transformer_v2",
            dataset,
            config.device,
            artifact_dir,
            v2_variant_name=variant_name,
            v2_variant=variant,
            loss_history_dir=root_report / "figures" / variant_name,
        )
        predictions_by_variant[variant_name] = pred
        datasets_by_variant[variant_name] = dataset
        rows = rows_for_subsets(
            f"transformer_v2_{variant_name}",
            dataset,
            dataset.test_idx,
            dataset.y[dataset.test_idx],
            pred,
            build_subset_masks(dataset, dataset.test_idx),
        )
        summary_rows.extend([{**row, "variant": variant_name} for row in rows])

    write_v2_ablation_summary(root_report / "v2_ablation_summary.csv", summary_rows)
    best_variant = select_best_v2_variant(summary_rows)
    stable_rows: list[dict[str, Any]] = []
    if best_variant:
        stable_variant = dict(V2_VARIANTS[best_variant])
        config = load_prediction_config(config_path)
        config.control_feature_scheme = str(stable_variant["control_feature_scheme"])
        dataset = build_dataset_from_csv(csv_path, config, include_control_features=True)
        dataset.save(root_dataset / "v2_stable_selected")
        artifact_dir = root_artifact / "v2_stable_selected"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        pred, _history = train_torch_model(
            "transformer_v2",
            dataset,
            config.device,
            artifact_dir,
            v2_variant_name="v2_stable_selected",
            v2_variant=stable_variant,
            loss_history_dir=root_report / "figures" / "v2_stable_selected",
        )
        stable_rows = rows_for_subsets(
            "transformer_v2_v2_stable_selected",
            dataset,
            dataset.test_idx,
            dataset.y[dataset.test_idx],
            pred,
            build_subset_masks(dataset, dataset.test_idx),
        )
        write_v2_ablation_summary(
            root_report / "v2_stable_selected_metrics.csv",
            [{**row, "variant": "v2_stable_selected"} for row in stable_rows],
        )

    summary = {
        "status": "ok",
        "variants": list(V2_VARIANTS.keys()),
        "best_variant": best_variant or "",
        "summary_path": str(root_report / "v2_ablation_summary.csv"),
        "stable_metrics_path": str(root_report / "v2_stable_selected_metrics.csv") if stable_rows else "",
    }
    (root_report / "v2_ablation_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


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
    v2_variant_name: str | None = None,
    v2_variant: dict[str, Any] | None = None,
    loss_history_dir: Path | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
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
    graph_path: Path | None = None
    optimizer_config = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": int(os.environ.get("TRAFFIC_TRAIN_EPOCHS", "12")),
        "patience": int(os.environ.get("TRAFFIC_TRAIN_PATIENCE", "4")),
        "grad_clip": None,
    }
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
    elif kind == "transformer_v2":
        v2_variant = v2_variant or default_v2_variant(dataset.control_feature_scheme)
        time_feature_size, entity_input_size = infer_v2_feature_layout(dataset)
        graph_path = resolve_movement_graph_path()
        phase_id_feature_index = feature_index_in_entity(dataset, "phase_id_embed") if v2_variant.get("phase_embedding") else None
        signal_state_feature_index = feature_index_in_entity(dataset, "signal_state_embed") if v2_variant.get("phase_embedding") else None
        max_phase_id = max_embedding_value(dataset, phase_id_feature_index)
        max_signal_state_id = max_embedding_value(dataset, signal_state_feature_index)
        graph_adjacency = None
        graph_attention_bias = None
        if graph_path is not None:
            if v2_variant.get("graph_bias"):
                graph_attention_bias = load_movement_attention_bias(
                    graph_path,
                    dataset.edge_ids,
                    scale=float(v2_variant.get("graph_bias_scale", 0.15)),
                )
            else:
                graph_adjacency = load_movement_adjacency(graph_path, dataset.edge_ids)
        model_config = {
            "input_size": input_size,
            "output_size": output_size,
            "horizon_steps": horizon_steps,
            "num_entities": len(dataset.edge_ids),
            "entity_input_size": entity_input_size,
            "target_size": len(dataset.targets),
            "time_feature_size": time_feature_size,
            "d_model": 128,
            "n_heads": 4,
            "temporal_layers": 2,
            "spatial_layers": 2 if dataset.control_feature_scheme == "phase_embed_graph_v1" else 1,
            "dropout": 0.05,
            "graph_adjacency": graph_adjacency,
            "graph_attention_bias": graph_attention_bias,
            "phase_id_feature_index": phase_id_feature_index,
            "max_phase_id": max_phase_id,
            "signal_state_feature_index": signal_state_feature_index,
            "max_signal_state_id": max_signal_state_id,
        }
        model = SpatioTemporalTransformerForecaster(**model_config)
        optimizer_config = {
            "lr": float(os.environ.get("TRAFFIC_TRAIN_LR_V2", "3e-4")),
            "weight_decay": float(os.environ.get("TRAFFIC_TRAIN_WD_V2", "5e-5")),
            "max_epochs": int(os.environ.get("TRAFFIC_TRAIN_EPOCHS_V2", "40")),
            "patience": int(os.environ.get("TRAFFIC_TRAIN_PATIENCE_V2", "8")),
            "grad_clip": float(os.environ.get("TRAFFIC_TRAIN_GRAD_CLIP_V2", "1.0")),
        }
    else:
        raise ValueError(f"Unknown torch model kind: {kind}")

    model.to(device)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=min(64, len(train_ds)), shuffle=True)
    batches_per_epoch = max(len(train_loader), 1)
    criterion = nn.HuberLoss()
    target_weights = (
        build_output_target_weights(dataset, device)
        if kind == "transformer_v2" and v2_variant and v2_variant.get("queue_weight")
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"],
    )
    best_state = None
    best_val = math.inf
    patience = optimizer_config["patience"]
    stale_epochs = 0
    max_epochs = optimizer_config["max_epochs"]
    loss_history: list[dict[str, Any]] = []
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = weighted_huber_loss(pred, yb, target_weights) if target_weights is not None else criterion(pred, yb)
            loss.backward()
            if optimizer_config["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(optimizer_config["grad_clip"]),
                )
            optimizer.step()
            batch_size = int(xb.shape[0])
            train_loss_sum += float(loss.item()) * batch_size
            train_sample_count += batch_size

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(X_val).to(device))
            y_val_tensor = torch.from_numpy(y_val).to(device)
            val_loss_tensor = (
                weighted_huber_loss(val_pred, y_val_tensor, target_weights)
                if target_weights is not None
                else criterion(val_pred, y_val_tensor)
            )
            val_loss = float(val_loss_tensor.item())
        train_loss = train_loss_sum / max(train_sample_count, 1)
        is_best = val_loss < best_val
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        loss_history.append(
            {
                "epoch": int(epoch),
                "update_step": int(epoch * batches_per_epoch),
                "train_loss": round(float(train_loss), 8),
                "val_loss": round(float(val_loss), 8),
                "best_val_loss": round(float(best_val), 8),
                "is_best": bool(is_best),
            }
        )
        if not is_best:
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred_scaled = model(torch.from_numpy(X_test).to(device)).detach().cpu().numpy()
    pred = inverse_scale_y(test_pred_scaled, dataset.y_mean, dataset.y_std)

    artifact_model_config = sanitize_torch_model_config(
        kind,
        model_config,
        graph_path=graph_path if kind == "transformer_v2" else None,
        control_feature_scheme=dataset.control_feature_scheme,
        v2_variant_name=v2_variant_name,
        v2_variant=v2_variant,
    )

    artifact = {
        "kind": kind,
        "model_name": kind,
        "model_config": artifact_model_config,
        "state_dict": model.state_dict(),
        "loss_history": loss_history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "epochs_trained": int(len(loss_history)),
        **dataset.metadata_for_artifact(),
    }
    torch.save(artifact, artifact_root / f"{kind}_model.pt")
    if loss_history_dir is not None:
        write_loss_history_csv(
            loss_history,
            loss_history_dir / f"{kind}_loss_history.csv",
            kind,
        )
        plot_single_training_loss_curve(
            loss_history,
            loss_history_dir / f"{kind}_training_loss.png",
            kind,
        )
    return pred, loss_history


def infer_v2_feature_layout(dataset: DatasetBundle) -> tuple[int, int]:
    time_feature_names = ["tod_sin", "tod_cos"]
    time_feature_size = 2 if dataset.feature_names[-2:] == time_feature_names else 0
    if time_feature_size != 2:
        raise ValueError(
            "transformer_v2 requires feature_names to end with ['tod_sin', 'tod_cos']"
        )

    num_entities = len(dataset.edge_ids)
    if num_entities <= 0:
        raise ValueError("transformer_v2 requires at least one entity")

    entity_body_size = len(dataset.feature_names) - time_feature_size
    if entity_body_size <= 0 or entity_body_size % num_entities != 0:
        raise ValueError(
            f"Bad V2 feature layout: feature_names={len(dataset.feature_names)}, "
            f"entities={num_entities}, time_feature_size={time_feature_size}"
        )

    entity_input_size = entity_body_size // num_entities
    expected_total = entity_input_size * num_entities + time_feature_size
    if expected_total != len(dataset.feature_names):
        raise ValueError(
            f"Bad V2 feature count: expected_total={expected_total}, "
            f"feature_names={len(dataset.feature_names)}"
        )

    for entity_id in dataset.edge_ids[:3]:
        prefix = f"{entity_id}__"
        prefix_count = sum(1 for feature_name in dataset.feature_names if feature_name.startswith(prefix))
        if prefix_count != entity_input_size:
            raise ValueError(
                f"Entity {entity_id} has {prefix_count} features, expected {entity_input_size}"
            )

    expected_features = list(dataset.per_movement_input_features or [])
    if expected_features:
        first_entity = dataset.edge_ids[0]
        expected_feature_names = [f"{first_entity}__{feature}" for feature in expected_features]
        actual_feature_names = [
            feature_name
            for feature_name in dataset.feature_names
            if feature_name.startswith(f"{first_entity}__")
        ]
        if actual_feature_names != expected_feature_names:
            raise ValueError(
                "transformer_v2 feature order mismatch for first entity: "
                f"expected {expected_feature_names}, got {actual_feature_names}"
            )

    if output_size := dataset.y.shape[-1]:
        expected_output_size = num_entities * len(dataset.targets)
        if output_size != expected_output_size:
            raise ValueError(
                f"Bad V2 output size: expected {expected_output_size}, got {output_size}"
            )

    return time_feature_size, entity_input_size


def feature_index_in_entity(dataset: DatasetBundle, feature: str) -> int | None:
    features = list(dataset.per_movement_input_features or [])
    if feature not in features:
        return None
    return features.index(feature)


def max_embedding_value(dataset: DatasetBundle, feature_index: int | None) -> int:
    if feature_index is None:
        return 0
    entity_feature_size = len(dataset.per_movement_input_features or [])
    if entity_feature_size <= 0:
        return 0
    columns = [
        entity_index * entity_feature_size + feature_index
        for entity_index in range(len(dataset.edge_ids))
    ]
    values = dataset.X[:, :, columns]
    valid = values[values >= 0]
    if valid.size == 0:
        return 0
    return max(0, int(np.nanmax(valid)))


def build_output_target_weights(dataset: DatasetBundle, device: str):
    import torch

    weights = []
    for _entity_id in dataset.edge_ids:
        for target in dataset.targets:
            weights.append(1.2 if target == "queue_veh" else 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device).view(1, 1, -1)


def weighted_huber_loss(pred, target, target_weights):
    import torch.nn.functional as F

    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * target_weights).mean()


def default_v2_variant(control_feature_scheme: str) -> dict[str, Any]:
    if control_feature_scheme == "phase_embed_graph_v1":
        return {
            "phase_embedding": True,
            "graph_bias": True,
            "queue_weight": True,
            "control_feature_scheme": "phase_embed_graph_v1",
            "graph_bias_scale": 1.0,
        }
    return {
        "phase_embedding": False,
        "graph_bias": False,
        "queue_weight": False,
        "control_feature_scheme": control_feature_scheme,
        "graph_bias_scale": 0.0,
    }


def resolve_movement_graph_path() -> Path | None:
    candidate = Path(load_prediction_config("configs/prediction_config.json").movement_graph_file)
    return candidate if candidate.exists() else None


def sanitize_torch_model_config(
    kind: str,
    model_config: dict[str, Any],
    graph_path: Path | None = None,
    control_feature_scheme: str = "phase_state_v1",
    v2_variant_name: str | None = None,
    v2_variant: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sanitized = dict(model_config)
    sanitized.pop("graph_adjacency", None)
    sanitized.pop("graph_attention_bias", None)
    if kind == "transformer_v2":
        sanitized["graph_enabled"] = graph_path is not None and bool(model_config.get("graph_adjacency") is not None)
        sanitized["graph_bias_enabled"] = graph_path is not None and bool(model_config.get("graph_attention_bias") is not None)
        sanitized["graph_bias_scale"] = float((v2_variant or {}).get("graph_bias_scale", 0.0))
        sanitized["graph_path"] = str(graph_path) if graph_path is not None else ""
        sanitized["control_feature_scheme"] = control_feature_scheme
        sanitized["v2_experiment_variant"] = v2_variant_name or ""
        sanitized["phase_embedding_enabled"] = bool((v2_variant or {}).get("phase_embedding", False))
        sanitized["queue_weight_enabled"] = bool((v2_variant or {}).get("queue_weight", False))
    return sanitized


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
        "transformer_v2": artifact_root / "transformer_v2_model.pt",
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


def write_loss_history_csv(
    history: list[dict[str, Any]],
    output_path: Path,
    model_name: str,
) -> None:
    if not history:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "epoch", "update_step", "train_loss", "val_loss", "best_val_loss", "is_best"]
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({"model": model_name, **row})


def plot_single_training_loss_curve(
    history: list[dict[str, Any]],
    output_path: Path,
    model_name: str,
) -> None:
    if not history:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_values = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    first_epoch = x_values[0]
    last_epoch = x_values[-1]
    y_min = min(min(train_loss), min(val_loss))
    y_max = max(max(train_loss), max(val_loss))

    # Keep axis ends aligned with visible tick ends and avoid auto padding.
    x_ticks = np.array([first_epoch, 30, 60, 90, 120, last_epoch], dtype=float)
    y_tick_count = 5
    y_start = math.floor(y_min * 1000.0) / 1000.0
    y_end = math.ceil(y_max * 1000.0) / 1000.0
    y_ticks = np.linspace(y_start, y_end, y_tick_count)

    with plt.rc_context(
        {
            "font.family": "Times New Roman",
            "font.size": 22,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 22,
            "axes.unicode_minus": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
        ax.set_facecolor("white")
        ax.plot(x_values, train_loss, label="train loss", linewidth=2.0, color="#1f77b4")
        ax.plot(x_values, val_loss, label="val loss", linewidth=2.0, color="#ff7f0e")

        ax.set_xlabel("epoch", fontname="Times New Roman")
        ax.set_ylabel("Huber loss", fontname="Times New Roman")

        ax.set_xlim(float(x_ticks[0]), float(x_ticks[-1]))
        ax.set_ylim(float(y_ticks[0]), float(y_ticks[-1]))
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.margins(x=0.0, y=0.0)

        ax.tick_params(direction="in", length=4.0, width=0.8, colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)

        ax.grid(False)
        ax.legend(loc="upper right", frameon=False, borderaxespad=0.6)

        fig.savefig(output_path, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)


def plot_torch_training_loss_curves(
    histories: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    available = {name: rows for name, rows in histories.items() if rows}
    if not available:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(available), 1, figsize=(9, max(4, 3.0 * len(available))), sharex=False)
    if len(available) == 1:
        axes = [axes]
    for ax, (model_name, history) in zip(axes, available.items()):
        x_values = [int(row["epoch"]) for row in history]
        train_loss = [float(row["train_loss"]) for row in history]
        val_loss = [float(row["val_loss"]) for row in history]
        best_points = [int(row["epoch"]) for row in history if row.get("is_best")]
        best_point = best_points[-1] if best_points else x_values[-1]
        x_label = "epoch"
        ax.plot(x_values, train_loss, label="train", linewidth=1.8)
        ax.plot(x_values, val_loss, label="val", linewidth=1.8)
        ax.axvline(best_point, color="#d62728", linestyle="--", linewidth=1.0)
        ax.set_title(model_name)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Huber loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle("Torch Model Training Loss Curves", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_prediction_comparison(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    dataset: DatasetBundle,
    output_path: Path,
) -> None:
    if y_true.size == 0 or not predictions:
        return

    target_size = len(dataset.targets)
    try:
        arrival_offset = dataset.targets.index("arrival_flow")
    except ValueError:
        arrival_offset = 0

    incoming_to_cols: dict[str, list[int]] = {}
    incoming_to_tls: dict[str, str] = {}
    for entity_index, entity_id in enumerate(dataset.edge_ids):
        meta = (dataset.entity_metadata or {}).get(entity_id, {})
        incoming_edge = str(meta.get("incoming_edge") or entity_id)
        incoming_to_cols.setdefault(incoming_edge, []).append(entity_index * target_size + arrival_offset)
        incoming_to_tls.setdefault(incoming_edge, str(meta.get("tls_id") or "unknown"))

    if not incoming_to_cols:
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
        return

    best_sample_pos = 0
    best_incoming_edge = next(iter(incoming_to_cols))
    best_score = -1.0
    for sample_pos in range(y_true.shape[0]):
        for incoming_edge, cols in incoming_to_cols.items():
            score = float(y_true[sample_pos, :, cols].sum())
            if score > best_score:
                best_score = score
                best_sample_pos = sample_pos
                best_incoming_edge = incoming_edge

    selected_cols = incoming_to_cols[best_incoming_edge]
    tls_id = incoming_to_tls.get(best_incoming_edge, "unknown")
    movement_count = len(selected_cols)
    y_true_series = y_true[best_sample_pos, :, selected_cols].sum(axis=1)

    plt.figure(figsize=(9, 5))
    plt.plot(y_true_series, label="true", linewidth=2)
    for name, pred in predictions.items():
        pred_series = pred[best_sample_pos, :, selected_cols].sum(axis=1)
        plt.plot(pred_series, label=name, linestyle="--")
    plt.title(
        "Forecast Comparison: "
        f"TLS {tls_id} / incoming {best_incoming_edge} / arrival_flow "
        f"(sum of {movement_count} movements)"
    )
    plt.xlabel("horizon step")
    plt.ylabel("arrival flow")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_continuous_high_flow_series(
    dataset: DatasetBundle,
    predictions: dict[str, np.ndarray],
    output_path: Path,
    target_name: str = "arrival_flow",
    horizon_offset: int = 0,
) -> None:
    if not predictions or len(dataset.test_idx) == 0:
        return

    target_size = len(dataset.targets)
    try:
        target_offset = dataset.targets.index(target_name)
    except ValueError:
        target_offset = 0

    incoming_to_cols: dict[str, list[int]] = {}
    incoming_to_tls: dict[str, str] = {}
    for entity_index, entity_id in enumerate(dataset.edge_ids):
        meta = (dataset.entity_metadata or {}).get(entity_id, {})
        incoming_edge = str(meta.get("incoming_edge") or entity_id)
        incoming_to_cols.setdefault(incoming_edge, []).append(entity_index * target_size + target_offset)
        incoming_to_tls.setdefault(incoming_edge, str(meta.get("tls_id") or "unknown"))

    y_true = dataset.y[dataset.test_idx]
    run_ids = dataset.sample_run_ids[dataset.test_idx]
    timestamps = dataset.sample_target_timestamps[dataset.test_idx]
    steps = dataset.sample_target_steps[dataset.test_idx]

    best_group: tuple[str, str] | None = None
    best_score = -1.0
    for run_id in dict.fromkeys(run_ids.tolist()):
        run_mask = run_ids == run_id
        if not np.any(run_mask):
            continue
        for incoming_edge, cols in incoming_to_cols.items():
            score = float(y_true[run_mask, horizon_offset, :][:, cols].sum())
            if score > best_score:
                best_score = score
                best_group = (str(run_id), incoming_edge)

    if best_group is None:
        return

    selected_run_id, selected_incoming_edge = best_group
    selected_cols = incoming_to_cols[selected_incoming_edge]
    selected_tls = incoming_to_tls.get(selected_incoming_edge, "unknown")
    group_mask = run_ids == selected_run_id
    group_positions = np.flatnonzero(group_mask)
    if group_positions.size == 0:
        return
    group_positions = group_positions[np.argsort(steps[group_positions])]

    true_series = y_true[group_positions, horizon_offset, :][:, selected_cols].sum(axis=1)
    series_by_model = {
        name: pred[group_positions, horizon_offset, :][:, selected_cols].sum(axis=1)
        for name, pred in predictions.items()
    }

    interval_seconds = 60.0
    if len(group_positions) >= 2:
        try:
            ts0 = datetime.fromisoformat(str(timestamps[group_positions[0]]))
            ts1 = datetime.fromisoformat(str(timestamps[group_positions[1]]))
            inferred_interval = abs((ts1 - ts0).total_seconds())
            if inferred_interval > 0:
                interval_seconds = inferred_interval
        except ValueError:
            interval_seconds = 60.0

    y_label = f"{target_name} per 60s"
    if target_name == "arrival_flow":
        flow_scale = 3600.0 / interval_seconds
        true_series = true_series * flow_scale
        series_by_model = {name: series * flow_scale for name, series in series_by_model.items()}
        y_label = "Traffic flow q (veh/h)"

    labels = []
    for pos in group_positions:
        raw_ts = str(timestamps[pos])
        try:
            labels.append(datetime.fromisoformat(raw_ts).strftime("%H:%M"))
        except ValueError:
            labels.append(raw_ts)

    x = np.arange(len(group_positions))
    all_series = [true_series, *series_by_model.values()]
    y_min = min(float(np.min(series)) for series in all_series)
    y_max = max(float(np.max(series)) for series in all_series)
    y_start = math.floor(y_min)
    y_end = math.ceil(y_max)
    if y_end <= y_start:
        y_end = y_start + 1
    y_ticks = np.linspace(y_start, y_end, 5)

    tick_count = min(6, len(x))
    tick_idx = np.linspace(0, len(x) - 1, tick_count, dtype=int)
    tick_idx = np.unique(tick_idx)

    color_map = {
        "true": "#1f77b4",
        "ha_baseline": "#4c78a8",
        "xgboost": "#f58518",
        "lstm": "#54a24b",
        "transformer_v1": "#e45756",
        "transformer_v2": "#b279a2",
    }

    with plt.rc_context(
        {
            "font.family": "Times New Roman",
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.unicode_minus": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
        ax.set_facecolor("white")
        ax.plot(x, true_series, label="true", linewidth=2.4, color=color_map["true"])
        for name, series in series_by_model.items():
            ax.plot(
                x,
                series,
                label=name,
                linestyle="--",
                linewidth=2.0,
                color=color_map.get(name),
            )

        ax.set_xlabel("time", fontname="Times New Roman")
        ax.set_ylabel(y_label, fontname="Times New Roman")
        ax.set_xlim(float(tick_idx[0]), float(tick_idx[-1]))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([labels[idx] for idx in tick_idx], rotation=0)
        ax.set_ylim(float(y_ticks[0]), float(y_ticks[-1]))
        ax.set_yticks(y_ticks)
        ax.margins(x=0.0, y=0.0)
        ax.tick_params(direction="in", length=4.0, width=0.8, colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)
        ax.grid(False)
        ax.legend(loc="upper right", frameon=False, borderaxespad=0.5)
        fig.savefig(output_path, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)


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
    preferred_order = ["ha_baseline", "xgboost", "lstm", "transformer_v1", "transformer_v2"]
    present_models = {str(row["model"]) for row in relevant}
    ordered_models = [model for model in preferred_order if model in present_models]
    remaining_models = sorted(present_models - set(ordered_models))
    models = ordered_models + remaining_models
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
    overall_values = [overall.get(model, 0.0) for model in models]
    incident_values = [incident.get(model, 0.0) for model in models]
    all_values = [*overall_values, *incident_values]
    y_max = max(all_values) if all_values else 1.0
    y_top = math.ceil(y_max * 10.0) / 10.0
    if y_top <= 0:
        y_top = 1.0
    y_ticks = np.linspace(0.0, y_top, 5)
    display_labels = {
        "ha_baseline": "HA",
        "lstm": "LSTM",
        "xgboost": "XGBoost",
        "transformer_v1": "Transformer V1",
        "transformer_v2": "Transformer V2",
    }

    with plt.rc_context(
        {
            "font.family": "Times New Roman",
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.unicode_minus": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
        ax.set_facecolor("white")
        ax.bar(
            x - width / 2,
            overall_values,
            width=width,
            label="overall",
            color="#4c78a8",
        )
        ax.bar(
            x + width / 2,
            incident_values,
            width=width,
            label="incident",
            color="#f58518",
        )
        ax.set_ylabel("MAE", fontname="Times New Roman")
        # Category axis: leave balanced visual padding on both sides.
        ax.set_xlim(float(x[0] - 0.6), float(x[-1] + 0.6))
        ax.set_xticks(x)
        ax.set_xticklabels([display_labels.get(model, model) for model in models], rotation=0)
        ax.set_ylim(0.0, float(y_ticks[-1]))
        ax.set_yticks(y_ticks)
        ax.margins(x=0.0, y=0.0)
        ax.tick_params(direction="in", length=4.0, width=0.8, colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)
        ax.grid(False)
        ax.legend(loc="upper right", frameon=False, borderaxespad=0.5)
        fig.savefig(output_path, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)


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


def write_v2_ablation_summary(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "model",
        "split",
        "subset",
        "n_samples",
        "run_count",
        "mae",
        "rmse",
        "wape",
        "mae_h5",
        "mae_h10",
        "mae_h15",
        "note",
    ]
    with out.open("w", newline="", encoding="utf-8") as fp:
        import csv

        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def select_best_v2_variant(rows: list[dict[str, Any]]) -> str | None:
    valid = [
        row
        for row in rows
        if row.get("subset") == "overall" and isinstance(row.get("mae"), float)
    ]
    if not valid:
        return None
    best = min(valid, key=lambda row: float(row["mae"]))
    return str(best.get("variant") or "")


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


def write_feature_diagnostics_csv(
    dataset: DatasetBundle,
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    if not predictions:
        return

    import csv

    y_true = dataset.y[dataset.test_idx]
    rows: list[dict[str, Any]] = []
    target_size = len(dataset.targets)
    turn_type_to_indices: dict[str, list[int]] = {}
    for entity_index, entity_id in enumerate(dataset.edge_ids):
        turn_type = str((dataset.entity_metadata or {}).get(entity_id, {}).get("turn_type", "unknown") or "unknown")
        turn_type_to_indices.setdefault(turn_type, []).append(entity_index)

    for model_name, y_pred in predictions.items():
        for target_index, target_name in enumerate(dataset.targets):
            cols = np.arange(target_index, y_true.shape[2], target_size)
            target_metrics = regression_metrics(y_true[:, :, cols], y_pred[:, :, cols])
            rows.append(
                {
                    "group_type": "target",
                    "group_name": target_name,
                    "model": model_name,
                    "mae": target_metrics.get("mae"),
                    "rmse": target_metrics.get("rmse"),
                    "wape": target_metrics.get("wape"),
                    "n_entities": int(len(cols)),
                }
            )

        for turn_type, entity_indices in sorted(turn_type_to_indices.items()):
            if not entity_indices:
                continue
            cols: list[int] = []
            for entity_index in entity_indices:
                base = entity_index * target_size
                cols.extend(range(base, base + target_size))
            turn_metrics = regression_metrics(y_true[:, :, cols], y_pred[:, :, cols])
            rows.append(
                {
                    "group_type": "turn_type",
                    "group_name": turn_type,
                    "model": model_name,
                    "mae": turn_metrics.get("mae"),
                    "rmse": turn_metrics.get("rmse"),
                    "wape": turn_metrics.get("wape"),
                    "n_entities": int(len(entity_indices)),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["group_type", "group_name", "model", "mae", "rmse", "wape", "n_entities"],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_smoke_test(
    config_path: str | Path = "configs/prediction_config.json",
    dataset_dir: str | Path = "data/datasets/p4_movement_control",
    artifact_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    csv_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    metrics_path = Path(report_dir) / "metrics.csv" if report_dir else None
    service = PredictionService(
        load_prediction_config(config_path),
        artifact_dir=artifact_dir,
        metrics_path=metrics_path,
        batch_csv_path=csv_path,
        manifest_path=manifest_path,
    )
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
    parser.add_argument("--scenario-manifest", default=None)
    parser.add_argument(
        "--control-feature-scheme",
        default=None,
        choices=["phase_state_v1", "phase_embed_graph_v1"],
        help="Override the prediction config control feature scheme for this training run.",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="After successful training, point prediction_config.json to this CSV/artifact/report set.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the no-control-feature ablation run for faster V2 validation.",
    )
    parser.add_argument(
        "--v2-ablation",
        action="store_true",
        help="Run V2-only component ablations instead of the full model suite.",
    )
    args = parser.parse_args()

    if args.v2_ablation:
        summary = train_v2_ablation_suite(
            args.config,
            args.csv,
            args.dataset_dir,
            args.artifact_dir,
            args.report_dir,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    summary = train_all(
        args.config,
        args.csv,
        args.dataset_dir,
        args.artifact_dir,
        args.report_dir,
        train_ablation=not args.skip_ablation,
        control_feature_scheme=args.control_feature_scheme,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.update_config:
        update_prediction_config_paths(
            args.config,
            args.csv,
            args.artifact_dir,
            args.report_dir,
            args.scenario_manifest,
            args.control_feature_scheme,
        )
    if args.smoke_test:
        print(
            json.dumps(
                run_smoke_test(
                    args.config,
                    args.dataset_dir,
                    args.artifact_dir,
                    args.report_dir,
                    args.csv,
                    args.scenario_manifest,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )


def update_prediction_config_paths(
    config_path: str | Path,
    csv_path: str | Path,
    artifact_dir: str | Path,
    report_dir: str | Path,
    manifest_path: str | Path | None = None,
    control_feature_scheme: str | None = None,
) -> None:
    path = Path(config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["batch_csv_file"] = _as_posix(csv_path)
    payload["artifact_dir"] = _as_posix(artifact_dir)
    payload["metrics_file"] = _as_posix(Path(report_dir) / "metrics.csv")
    if manifest_path:
        payload["scenario_manifest_file"] = _as_posix(manifest_path)
    if control_feature_scheme:
        payload["control_feature_scheme"] = control_feature_scheme
    payload["active_model_from_registry"] = True
    payload["preferred_model"] = "transformer_v2"
    payload["fallback_model"] = "ha_baseline"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_posix(path_value: str | Path) -> str:
    return Path(path_value).as_posix()


if __name__ == "__main__":
    main()
