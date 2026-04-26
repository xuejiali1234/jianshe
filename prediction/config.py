import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_OBSERVED_EDGES = [
    "472652453#3",
    "472652453#2",
    "-472652453#4",
    "-472652453#4.38",
    "472652453#1",
    "472652453#4",
    "1324604419#0",
    "1324604419#1",
]

DEFAULT_PER_EDGE_INPUT_FEATURES = [
    "flow",
    "speed",
    "queue",
    "incident_flag",
    "phase_id",
    "phase_elapsed_s",
    "green_remaining_s",
]

DEFAULT_PER_MOVEMENT_INPUT_FEATURES = [
    "arrival_flow",
    "discharge_flow",
    "mean_speed",
    "occupancy",
    "queue_veh",
    "queue_meter",
    "incident_flag",
    "phase_id",
    "phase_elapsed_s",
    "green_remaining_s",
]


def detect_torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class PredictionConfig:
    observed_edges: list[str] = field(default_factory=lambda: DEFAULT_OBSERVED_EDGES.copy())
    observation_level: str = "edge"
    movement_config_file: str = "configs/movement_config.json"
    batch_csv_file: str = "data/raw/batch_edge_aggregates.csv"
    scenario_manifest_file: str = "data/raw/scenarios/manifest.csv"
    artifact_dir: str = "models/artifacts"
    metrics_file: str = "reports/metrics.csv"
    sample_interval_s: int = 60
    history_steps: int = 12
    horizon_steps: int = 15
    targets: list[str] = field(default_factory=lambda: ["flow", "speed", "queue"])
    model: str = "ha_baseline"
    device: str = "auto"
    simulation_start_iso: str = "2026-04-17T08:00:00"
    base_demand_factor: float = 0.25
    sumo_net_file: str = "data/processed/czq_tls_webster.net.xml"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "PredictionConfig":
        allowed = set(cls.__dataclass_fields__.keys())
        values = {key: value for key, value in raw.items() if key in allowed}
        config = cls(**values)
        if config.device == "auto":
            config.device = detect_torch_device()
        return config

    def public_dict(self) -> dict[str, Any]:
        per_entity_features = (
            DEFAULT_PER_MOVEMENT_INPUT_FEATURES
            if self.observation_level == "movement"
            else DEFAULT_PER_EDGE_INPUT_FEATURES
        )
        input_feature_count = len(self.observed_edges) * len(per_entity_features) + 2
        return {
            "observed_edges": self.observed_edges,
            "observation_level": self.observation_level,
            "movement_config_file": self.movement_config_file,
            "batch_csv_file": self.batch_csv_file,
            "scenario_manifest_file": self.scenario_manifest_file,
            "artifact_dir": self.artifact_dir,
            "metrics_file": self.metrics_file,
            "sample_interval_s": self.sample_interval_s,
            "history_steps": self.history_steps,
            "horizon_steps": self.horizon_steps,
            "targets": self.targets,
            "model": self.model,
            "device": self.device,
            "simulation_start_iso": self.simulation_start_iso,
            "base_demand_factor": self.base_demand_factor,
            "sumo_net_file": self.sumo_net_file,
            "demand_regime": f"baseline_x{self.base_demand_factor:.2f}",
            "control_features_enabled": True,
            "per_edge_input_features": DEFAULT_PER_EDGE_INPUT_FEATURES,
            "per_movement_input_features": DEFAULT_PER_MOVEMENT_INPUT_FEATURES,
            "per_entity_input_features": per_entity_features,
            "global_input_features": ["tod_sin", "tod_cos"],
            "input_feature_count": input_feature_count,
        }


def load_prediction_config(path: str | Path) -> PredictionConfig:
    config_path = Path(path)
    if not config_path.exists():
        return PredictionConfig.from_mapping({})

    with config_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    return PredictionConfig.from_mapping(raw)
