from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RealTrafficRecord(BaseModel):
    movement_id: Optional[str] = None
    detector_id: Optional[str] = None

    tls_id: Optional[str] = None
    incoming_edge: Optional[str] = None
    outgoing_edge: Optional[str] = None
    turn_type: Optional[str] = None

    arrival_flow: float = 0.0
    discharge_flow: float = 0.0
    mean_speed_mps: Optional[float] = None
    speed_kmh: Optional[float] = None
    queue_veh: float = 0.0
    queue_meter: float = 0.0
    occupancy: float = 0.0

    incident_flag: int = 0
    phase_id: int = -1
    phase_elapsed_s: float = 0.0
    green_remaining_s: float = 0.0
    signal_state: str = ""


class RealTrafficSnapshotRequest(BaseModel):
    source: str = "real_api"
    timestamp: Optional[str] = None
    step: Optional[int] = None
    records: list[RealTrafficRecord] = Field(default_factory=list)
