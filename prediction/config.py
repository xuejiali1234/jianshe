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


def detect_torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class PredictionConfig:
    observed_edges: list[str] = field(default_factory=lambda: DEFAULT_OBSERVED_EDGES.copy())
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
        return {
            "observed_edges": self.observed_edges,
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
        }


def load_prediction_config(path: str | Path) -> PredictionConfig:
    config_path = Path(path)
    if not config_path.exists():
        return PredictionConfig.from_mapping({})

    with config_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    return PredictionConfig.from_mapping(raw)
