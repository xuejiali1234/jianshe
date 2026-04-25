from __future__ import annotations

import csv
from collections import deque
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .baselines import HistoricalAveragePredictor
from .config import PredictionConfig
from .model_io import ArtifactPredictor, discover_available_models
from .schemas import PredictRequest, ScenarioCompareRequest


class PredictionService:
    def __init__(
        self,
        config: PredictionConfig,
        artifact_dir: str | Path = "models/artifacts",
        metrics_path: str | Path = "reports/metrics.csv",
        batch_csv_path: str | Path = "data/raw/batch_edge_aggregates.csv",
        manifest_path: str | Path = "data/raw/scenarios/manifest.csv",
    ):
        self.config = config
        self.artifact_dir = Path(artifact_dir)
        self.metrics_path = Path(metrics_path)
        self.batch_csv_path = Path(batch_csv_path)
        self.manifest_path = Path(manifest_path)
        self.fallback_predictor = HistoricalAveragePredictor(config)
        self.available_models = list(
            dict.fromkeys([self.fallback_predictor.model_name, *discover_available_models(self.artifact_dir)])
        )
        self.model_metrics = self._load_model_metrics()
        self._predictor_cache: dict[str, ArtifactPredictor] = {}
        self._manifest_cache: pd.DataFrame | None = None
        self._manifest_cache_mtime: float | None = None
        self._batch_cache: pd.DataFrame | None = None
        self._batch_cache_mtime: float | None = None
        self.trained_predictor = ArtifactPredictor.load_best(config, self.artifact_dir)
        if self.trained_predictor is not None:
            self._predictor_cache[self.trained_predictor.model_name] = self.trained_predictor
        self.predictor = self.trained_predictor or self.fallback_predictor
        self.active_model = self.predictor.model_name
        self.history = deque(maxlen=config.history_steps)
        self.latest_observation: dict[str, Any] | None = None
        self.latest_prediction = self._attach_prediction_meta(
            self.fallback_predictor.predict([], config.horizon_steps),
            0,
        )
        if self.active_model not in self.available_models:
            self.available_models.append(self.active_model)
        registry_active_model = self._read_registry_active_model()
        if (
            registry_active_model
            and registry_active_model in self.available_models
            and registry_active_model != self.active_model
        ):
            self.switch_model(registry_active_model)

    def config_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "config": self.config.public_dict(),
            "active_model": self.active_model,
            "available_models": self.available_models,
            "model_metrics": self.model_metrics,
            "metrics_split": "test",
            "trained_artifact": str(self.trained_predictor.artifact_path)
            if self.trained_predictor
            else None,
            "scenario_compare_available": self.batch_csv_path.exists() and self.manifest_path.exists(),
        }

    def latest_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "observation": self.latest_observation,
            "prediction": self.latest_prediction,
            "history_size": len(self.history),
        }

    def update_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.latest_observation = observation
        self.history.append(observation)
        self.latest_prediction = self._attach_prediction_meta(
            self._predict_with_fallback(list(self.history), self.config.horizon_steps),
            len(self.history),
        )
        return self.latest_prediction

    def predict_request(self, request: PredictRequest) -> dict[str, Any]:
        horizon = request.horizon or self.config.horizon_steps
        return self._attach_prediction_meta(
            self._predict_with_fallback(request.window, horizon),
            len(request.window),
        )

    def predict_request_with_model(
        self,
        request: PredictRequest,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        predictor = self._resolve_predictor(model_name or self.active_model)
        horizon = request.horizon or self.config.horizon_steps
        payload = self._predict_with_specific_predictor(predictor, request.window, horizon)
        return self._attach_prediction_meta(payload, len(request.window), model_name or self.active_model)

    def switch_model(self, model_name: str) -> dict[str, Any]:
        model_name = (model_name or "").strip()
        predictor = self._resolve_predictor(model_name)
        self.trained_predictor = predictor if isinstance(predictor, ArtifactPredictor) else None
        self.predictor = predictor
        self.active_model = predictor.model_name
        self._write_registry()
        self.latest_prediction = self._attach_prediction_meta(
            self._predict_with_specific_predictor(self.predictor, list(self.history), self.config.horizon_steps),
            len(self.history),
        )
        payload = self.config_payload()
        payload["prediction"] = self.latest_prediction
        return payload

    def scenario_runs_payload(self) -> dict[str, Any]:
        manifest = self._load_manifest()
        if manifest is None or manifest.empty:
            return {
                "status": "ok",
                "baseline_runs": [],
                "incident_runs": [],
            }

        manifest = manifest.fillna("")
        baseline_df = manifest[manifest["incident_type"].astype(str).str.strip() == ""].copy()
        incident_df = manifest[manifest["incident_type"].astype(str).str.strip() != ""].copy()

        baseline_runs = [
            {
                "run_id": row.run_id,
                "scenario_id": row.scenario_id,
                "seed": int(row.seed),
                "demand_scale": float(row.demand_scale),
                "base_demand_factor": float(row.base_demand_factor or self.config.base_demand_factor),
            }
            for row in baseline_df.itertuples(index=False)
        ]

        incident_runs = []
        for row in incident_df.itertuples(index=False):
            recommended_baseline = baseline_df[
                (baseline_df["seed"].astype(str) == str(row.seed))
                & (baseline_df["demand_scale"].astype(float) == float(row.demand_scale))
            ]
            recommended_baseline_run_id = (
                str(recommended_baseline.iloc[0]["run_id"]) if not recommended_baseline.empty else ""
            )
            incident_runs.append(
                {
                    "run_id": row.run_id,
                    "scenario_id": row.scenario_id,
                    "seed": int(row.seed),
                    "demand_scale": float(row.demand_scale),
                    "base_demand_factor": float(row.base_demand_factor or self.config.base_demand_factor),
                    "incident_type": row.incident_type,
                    "incident_start_s": int(row.incident_start_s),
                    "incident_end_s": int(row.incident_end_s),
                    "affected_edges": [
                        edge_id for edge_id in str(row.affected_edges).split("|") if edge_id
                    ],
                    "recommended_baseline_run_id": recommended_baseline_run_id,
                }
            )

        return {
            "status": "ok",
            "baseline_runs": baseline_runs,
            "incident_runs": incident_runs,
        }

    def scenario_compare_payload(self, request: ScenarioCompareRequest) -> dict[str, Any]:
        manifest = self._load_manifest()
        batch_df = self._load_batch_csv()
        if manifest is None or batch_df is None or manifest.empty or batch_df.empty:
            raise ValueError("Scenario compare requires batch CSV and manifest outputs")

        incident_row = manifest.loc[manifest["run_id"] == request.incident_run_id]
        if incident_row.empty:
            raise ValueError(f"Unknown incident run_id: {request.incident_run_id}")
        incident_meta = incident_row.iloc[0]
        anchor_step = int(incident_meta.get("incident_start_s") or 0)
        affected_edges = [
            edge_id for edge_id in str(incident_meta.get("affected_edges") or "").split("|") if edge_id
        ]
        horizon = request.horizon or self.config.horizon_steps

        baseline_window, _baseline_future = self._build_window_for_run(
            batch_df,
            request.baseline_run_id,
            anchor_step,
            horizon,
        )
        incident_window, _incident_future = self._build_window_for_run(
            batch_df,
            request.incident_run_id,
            anchor_step,
            horizon,
        )
        model_name = request.model_name or self.active_model
        baseline_pred = self.predict_request_with_model(
            PredictRequest(window=baseline_window, horizon=horizon),
            model_name,
        )
        incident_pred = self.predict_request_with_model(
            PredictRequest(window=incident_window, horizon=horizon),
            model_name,
        )
        delta = self._build_delta_payload(
            baseline_pred,
            incident_pred,
            request.edge_id,
        )
        return {
            "status": "ok",
            "edge_id": request.edge_id,
            "model_name": model_name,
            "baseline_run_id": request.baseline_run_id,
            "incident_run_id": request.incident_run_id,
            "affected_edges": affected_edges,
            "incident_type": str(incident_meta.get("incident_type") or ""),
            "anchor_step": anchor_step,
            "baseline_pred": baseline_pred,
            "incident_pred": incident_pred,
            "delta": delta,
        }

    def _build_delta_payload(
        self,
        baseline_pred: dict[str, Any],
        incident_pred: dict[str, Any],
        edge_id: str,
    ) -> dict[str, Any]:
        baseline_node = next(
            (node for node in baseline_pred["nodes"] if node["edge_id"] == edge_id),
            baseline_pred["nodes"][0],
        )
        incident_node = next(
            (node for node in incident_pred["nodes"] if node["edge_id"] == edge_id),
            incident_pred["nodes"][0],
        )
        return {
            "edge_id": edge_id,
            "delta_flow": [
                float(incident - baseline)
                for baseline, incident in zip(
                    baseline_node.get("pred_flow", []),
                    incident_node.get("pred_flow", []),
                )
            ],
            "delta_speed": [
                float(incident - baseline)
                for baseline, incident in zip(
                    baseline_node.get("pred_speed", []),
                    incident_node.get("pred_speed", []),
                )
            ],
            "delta_queue": [
                float(incident - baseline)
                for baseline, incident in zip(
                    baseline_node.get("pred_queue", []),
                    incident_node.get("pred_queue", []),
                )
            ],
        }

    def _build_window_for_run(
        self,
        batch_df: pd.DataFrame,
        run_id: str,
        forecast_start_step: int,
        horizon: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        run_df = batch_df.loc[batch_df["run_id"] == run_id].copy()
        if run_df.empty:
            raise ValueError(f"Unknown run_id in batch CSV: {run_id}")

        snapshots: list[dict[str, Any]] = []
        for (step, timestamp), group in run_df.groupby(["step", "timestamp"], sort=True):
            by_edge = {row.edge_id: row for row in group.itertuples(index=False)}
            if any(edge_id not in by_edge for edge_id in self.config.observed_edges):
                continue
            nodes = []
            for edge_id in self.config.observed_edges:
                row = by_edge[edge_id]
                nodes.append(
                    {
                        "edge_id": edge_id,
                        "flow": float(row.flow),
                        "speed": float(row.speed_mps),
                        "speed_mps": float(row.speed_mps),
                        "queue": float(row.queue),
                        "incident_flag": float(getattr(row, "incident_flag", 0.0) or 0.0),
                    }
                )
            snapshots.append(
                {
                    "step": int(step),
                    "timestamp": str(timestamp),
                    "nodes": nodes,
                }
            )

        if len(snapshots) < self.config.history_steps + horizon:
            raise ValueError(
                f"Run {run_id} has only {len(snapshots)} snapshots; need at least "
                f"{self.config.history_steps + horizon}"
            )

        target_idx = next(
            (idx for idx, snapshot in enumerate(snapshots) if snapshot["step"] >= forecast_start_step),
            len(snapshots) - horizon,
        )
        target_idx = max(target_idx, self.config.history_steps)
        target_idx = min(target_idx, len(snapshots) - horizon)
        window = snapshots[target_idx - self.config.history_steps : target_idx]
        future = snapshots[target_idx : target_idx + horizon]
        return window, future

    def _predict_with_fallback(self, window: list[Any], horizon: int) -> dict[str, Any]:
        return self._predict_with_specific_predictor(self.predictor, window, horizon)

    def _predict_with_specific_predictor(
        self,
        predictor: HistoricalAveragePredictor | ArtifactPredictor,
        window: list[Any],
        horizon: int,
    ) -> dict[str, Any]:
        if isinstance(predictor, ArtifactPredictor):
            if len(window) >= self.config.history_steps:
                try:
                    return predictor.predict(window, horizon)
                except Exception as exc:
                    print(f"Trained predictor failed; falling back to HA baseline: {exc}")
            return self.fallback_predictor.predict(window, horizon)
        return predictor.predict(window, horizon)

    def _resolve_predictor(self, model_name: str) -> HistoricalAveragePredictor | ArtifactPredictor:
        model_name = (model_name or "").strip()
        if model_name not in self.available_models:
            raise ValueError(f"Unknown prediction model: {model_name}")
        if model_name == self.fallback_predictor.model_name:
            return self.fallback_predictor
        predictor = self._predictor_cache.get(model_name)
        if predictor is None:
            predictor = ArtifactPredictor.load_named(self.config, self.artifact_dir, model_name)
            if predictor is None:
                raise ValueError(f"Prediction artifact for model '{model_name}' is not available")
            self._predictor_cache[model_name] = predictor
        return predictor

    def _attach_prediction_meta(
        self,
        payload: dict[str, Any],
        history_size: int,
        active_model_name: str | None = None,
    ) -> dict[str, Any]:
        enriched = dict(payload)
        enriched["history_size"] = int(history_size)
        enriched["history_required"] = int(self.config.history_steps)
        enriched["active_model"] = active_model_name or self.active_model
        return enriched

    def _write_registry(self) -> None:
        active_artifact = self.trained_predictor.artifact_path.name if self.trained_predictor else ""
        registry = {
            "active_model": self.active_model,
            "active_artifact": active_artifact,
            "active_alias": "",
        }
        note = "active model selected from dashboard"
        registry_path = self.artifact_dir / "model_registry.json"
        if registry_path.exists():
            try:
                existing = json.loads(registry_path.read_text(encoding="utf-8"))
                if existing.get("note"):
                    note = existing["note"]
            except Exception:
                pass
        registry["note"] = note
        registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _read_registry_active_model(self) -> str | None:
        registry_path = self.artifact_dir / "model_registry.json"
        if not registry_path.exists():
            return None
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        active_model = registry.get("active_model")
        return str(active_model) if active_model else None

    def _load_model_metrics(self) -> dict[str, dict[str, float | int | str]]:
        if not self.metrics_path.exists():
            return {}

        metrics: dict[str, dict[str, float | int | str]] = {}
        try:
            with self.metrics_path.open("r", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    model_name = (row.get("model") or "").strip()
                    split = (row.get("split") or "").strip()
                    subset = (row.get("subset") or "overall").strip()
                    if not model_name or split != "test" or subset not in {"", "overall"}:
                        continue
                    metrics[model_name] = {
                        "split": split,
                        "subset": subset or "overall",
                        "n_samples": self._safe_int(row.get("n_samples")),
                        "run_count": self._safe_int(row.get("run_count")),
                        "mae": self._safe_float(row.get("mae")),
                        "rmse": self._safe_float(row.get("rmse")),
                        "wape": self._safe_float(row.get("wape")),
                    }
        except Exception as exc:
            print(f"Failed to load model metrics from {self.metrics_path}: {exc}")
            return {}
        return metrics

    def _load_manifest(self) -> pd.DataFrame | None:
        if not self.manifest_path.exists():
            return None
        mtime = self.manifest_path.stat().st_mtime
        if self._manifest_cache is not None and self._manifest_cache_mtime == mtime:
            return self._manifest_cache
        df = pd.read_csv(self.manifest_path)
        for column, default in {
            "base_demand_factor": self.config.base_demand_factor,
            "incident_type": "",
            "incident_start_s": 0,
            "incident_end_s": 0,
            "affected_edges": "",
        }.items():
            if column not in df.columns:
                df[column] = default
        self._manifest_cache = df
        self._manifest_cache_mtime = mtime
        return df

    def _load_batch_csv(self) -> pd.DataFrame | None:
        if not self.batch_csv_path.exists():
            return None
        mtime = self.batch_csv_path.stat().st_mtime
        if self._batch_cache is not None and self._batch_cache_mtime == mtime:
            return self._batch_cache
        df = pd.read_csv(self.batch_csv_path)
        if "incident_flag" not in df.columns:
            df["incident_flag"] = 0
        self._batch_cache = df
        self._batch_cache_mtime = mtime
        return df

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
