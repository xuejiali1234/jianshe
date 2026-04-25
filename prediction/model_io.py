from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .baselines import HistoricalAveragePredictor
from .config import PredictionConfig
from .dataset import (
    inverse_scale_y,
    matrix_to_prediction_payload,
    scale_X,
    window_to_matrix,
)
from .torch_models import build_torch_model


ARTIFACT_FILES = {
    "xgboost": "xgboost_model.joblib",
    "lstm": "lstm_model.pt",
    "transformer_v1": "transformer_v1_model.pt",
}


class ArtifactPredictor:
    def __init__(
        self,
        model_name: str,
        kind: str,
        artifact: dict[str, Any],
        config: PredictionConfig,
        artifact_path: Path,
    ):
        self.model_name = model_name
        self.kind = kind
        self.artifact = artifact
        self.config = config
        self.artifact_path = artifact_path
        self.edge_ids = artifact["edge_ids"]
        self.targets = artifact["targets"]
        self.feature_names = list(artifact.get("feature_names", []))
        self.target_feature_names = list(artifact.get("target_feature_names", []))
        self.x_mean = np.asarray(artifact.get("x_mean", artifact.get("mean")), dtype=np.float32)
        self.x_std = np.asarray(artifact.get("x_std", artifact.get("std")), dtype=np.float32)
        self.y_mean = np.asarray(artifact.get("y_mean", artifact.get("mean")), dtype=np.float32)
        self.y_std = np.asarray(artifact.get("y_std", artifact.get("std")), dtype=np.float32)
        self.history_steps = int(artifact["history_steps"])
        self.horizon_steps = int(artifact["horizon_steps"])
        self._torch_model = None

    @classmethod
    def load_best(
        cls,
        config: PredictionConfig,
        artifact_dir: str | Path,
    ) -> "ArtifactPredictor | None":
        root = Path(artifact_dir)
        manifest_path = root / "model_registry.json"
        candidates: list[Path] = []
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            active = manifest.get("active_artifact")
            if active:
                candidates.append(root / active)
            active_model = manifest.get("active_model")
            if active_model in ARTIFACT_FILES:
                candidates.append(root / ARTIFACT_FILES[active_model])
        candidates.extend([root / "best_model.pt", root / "best_model.joblib"])
        for filename in ARTIFACT_FILES.values():
            candidates.append(root / filename)

        for path in candidates:
            if not path.exists():
                continue
            try:
                return cls._load_from_path(config, path)
            except Exception as exc:
                print(f"Failed to load trained predictor {path}: {exc}")
        return None

    @classmethod
    def load_named(
        cls,
        config: PredictionConfig,
        artifact_dir: str | Path,
        model_name: str,
    ) -> "ArtifactPredictor | None":
        filename = ARTIFACT_FILES.get(model_name)
        if not filename:
            return None
        path = Path(artifact_dir) / filename
        if not path.exists():
            return None
        return cls._load_from_path(config, path)

    @classmethod
    def _load_from_path(
        cls,
        config: PredictionConfig,
        path: Path,
    ) -> "ArtifactPredictor":
        if path.suffix == ".joblib":
            artifact = joblib.load(path)
            return cls(
                artifact.get("model_name", "xgboost"),
                artifact["kind"],
                artifact,
                config,
                path,
            )
        if path.suffix == ".pt":
            import torch

            artifact = torch.load(
                path,
                map_location=config.device,
                weights_only=False,
            )
            return cls(
                artifact.get("model_name", artifact["kind"]),
                artifact["kind"],
                artifact,
                config,
                path,
            )
        raise ValueError(f"Unsupported artifact suffix: {path.suffix}")

    def predict(self, window: list[Any], horizon: int | None = None) -> dict[str, Any]:
        horizon_steps = horizon or self.config.horizon_steps
        if len(window) < self.history_steps:
            raise ValueError(
                f"Trained predictor requires {self.history_steps} history steps; "
                f"got {len(window)}"
            )
        matrix = window_to_matrix(
            window[-self.history_steps :],
            self.edge_ids,
            self.targets,
            self.config,
        )
        X = scale_X(matrix[None, :, :], self.x_mean, self.x_std)

        if self.kind == "xgboost":
            pred_scaled = self.artifact["model"].predict(X.reshape(1, -1))
            pred_scaled = pred_scaled.reshape(1, self.horizon_steps, -1)
        elif self.kind in {"lstm", "transformer_v1"}:
            import torch

            model = self._get_torch_model()
            with torch.no_grad():
                tensor = torch.from_numpy(X).to(self.config.device)
                pred_scaled = model(tensor).detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported artifact kind: {self.kind}")

        pred = inverse_scale_y(pred_scaled, self.y_mean, self.y_std)[0]
        return matrix_to_prediction_payload(
            pred,
            self.edge_ids,
            self.targets,
            self.model_name,
            horizon_steps,
        )

    def _get_torch_model(self):
        if self._torch_model is None:
            model = build_torch_model(self.kind, self.artifact["model_config"])
            model.load_state_dict(self.artifact["state_dict"])
            model.to(self.config.device)
            model.eval()
            self._torch_model = model
        return self._torch_model


def load_active_or_fallback(
    config: PredictionConfig,
    artifact_dir: str | Path,
) -> tuple[Any, str]:
    trained = ArtifactPredictor.load_best(config, artifact_dir)
    if trained is not None:
        return trained, trained.model_name
    fallback = HistoricalAveragePredictor(config)
    return fallback, fallback.model_name


def discover_available_models(artifact_dir: str | Path) -> list[str]:
    root = Path(artifact_dir)
    models = []
    for model_name, filename in ARTIFACT_FILES.items():
        if (root / filename).exists():
            models.append(model_name)
    return models
