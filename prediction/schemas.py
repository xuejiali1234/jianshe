from typing import Optional

from pydantic import BaseModel, Field


class PredictionNodeInput(BaseModel):
    edge_id: str
    flow: Optional[float] = None
    speed: Optional[float] = None
    speed_mps: Optional[float] = None
    queue: Optional[float] = None
    incident_flag: Optional[float] = None


class PredictionWindowStep(BaseModel):
    timestamp: Optional[str] = None
    nodes: list[PredictionNodeInput] = Field(default_factory=list)


class PredictRequest(BaseModel):
    window: list[PredictionWindowStep] = Field(default_factory=list)
    horizon: Optional[int] = None


class PredictionModelSwitchRequest(BaseModel):
    model_name: str


class ScenarioCompareRequest(BaseModel):
    baseline_run_id: str
    incident_run_id: str
    edge_id: str
    model_name: Optional[str] = None
    horizon: Optional[int] = None
