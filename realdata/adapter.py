from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from prediction.movement_collector import MOVEMENT_CSV_FIELDS, movement_rows_to_legacy_nodes
from sim.movement_tools import load_movement_config

from .schemas import RealTrafficRecord, RealTrafficSnapshotRequest


DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "use_for_prediction": False,
    "sample_interval_s": 60,
    "mode": "api",
    "write_csv": True,
    "csv_path": "data/real/realtime_realdata_snapshots.csv",
    "movement_config_file": "configs/movement_config.json",
    "detector_map_file": "configs/real_detector_map.example.json",
    "fill_missing_movements": True,
    "missing_fill_policy": "last_or_zero",
}


class RealDataAdapter:
    def __init__(self, project_root: Path, config_path: Path):
        self.project_root = Path(project_root)
        self.config_path = Path(config_path)
        self.config = {**DEFAULT_CONFIG, **self._load_json(self.config_path)}

        self.movement_config_path = self._resolve_path(
            self.config.get("movement_config_file", DEFAULT_CONFIG["movement_config_file"])
        )
        self.detector_map_path = self._resolve_path(
            self.config.get("detector_map_file", DEFAULT_CONFIG["detector_map_file"])
        )
        self.csv_path = self._resolve_path(self.config.get("csv_path", DEFAULT_CONFIG["csv_path"]))

        movement_payload = load_movement_config(self.movement_config_path)
        self.movements = list(movement_payload.get("movements", []))
        self.movement_by_id = {
            str(movement.get("movement_id", "")): movement
            for movement in self.movements
            if movement.get("movement_id")
        }
        self.detector_map = self._load_json(self.detector_map_path)
        self.last_rows_by_movement_id: dict[str, dict[str, Any]] = {}
        self.last_ignored_records = 0

    def config_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "enabled": bool(self.config.get("enabled", True)),
            "use_for_prediction": bool(self.config.get("use_for_prediction", False)),
            "sample_interval_s": int(self.config.get("sample_interval_s", 60) or 60),
            "mode": str(self.config.get("mode", "api")),
            "write_csv": bool(self.config.get("write_csv", True)),
            "csv_path": str(self.csv_path),
            "movement_config_file": str(self.movement_config_path),
            "detector_map_file": str(self.detector_map_path),
            "movement_count": len(self.movements),
            "detector_count": len(self.detector_map),
            "fill_missing_movements": bool(self.config.get("fill_missing_movements", True)),
            "missing_fill_policy": str(self.config.get("missing_fill_policy", "last_or_zero")),
        }

    def build_snapshot(self, req: RealTrafficSnapshotRequest) -> dict[str, Any]:
        timestamp = req.timestamp or datetime.now().isoformat(timespec="seconds")
        step = int(req.step if req.step is not None else datetime.now().timestamp())

        rows_by_id: dict[str, dict[str, Any]] = {}
        ignored_records = 0
        for record in req.records:
            row = self._record_to_movement_row(record)
            if row is None:
                ignored_records += 1
                continue
            rows_by_id[row["movement_id"]] = row

        if self.config.get("fill_missing_movements", True):
            for movement in self.movements:
                movement_id = str(movement.get("movement_id", ""))
                if movement_id and movement_id not in rows_by_id:
                    rows_by_id[movement_id] = self._missing_row(movement)

        movement_rows = [
            rows_by_id[str(movement.get("movement_id", ""))]
            for movement in self.movements
            if str(movement.get("movement_id", "")) in rows_by_id
        ]
        extra_rows = [
            row
            for movement_id, row in rows_by_id.items()
            if movement_id not in self.movement_by_id
        ]
        movement_rows.extend(extra_rows)

        snapshot = {
            "run_id": "real_api",
            "scenario_id": "real_online",
            "seed": "",
            "demand_scale": "",
            "base_demand_factor": "",
            "signal_variant": "real_signal",
            "event_type": "",
            "event_policy": "",
            "speed_factor": "",
            "incident_type": "",
            "incident_start_s": "",
            "incident_end_s": "",
            "affected_edges": [],
            "timestamp": timestamp,
            "step": step,
            "source": req.source,
            "records_received": len(req.records),
            "records_ignored": ignored_records,
            "movements": movement_rows,
            "nodes": movement_rows_to_legacy_nodes(movement_rows),
        }

        self.last_ignored_records = ignored_records
        self.last_rows_by_movement_id = {
            row["movement_id"]: row
            for row in movement_rows
        }

        if self.config.get("write_csv", True):
            self._append_csv(snapshot)

        return snapshot

    def _record_to_movement_row(self, record: RealTrafficRecord) -> dict[str, Any] | None:
        movement_id = (record.movement_id or "").strip()
        mapped = self._mapped_detector(record.detector_id)
        if not movement_id and mapped:
            movement_id = str(mapped.get("movement_id", "") or "").strip()
        if not movement_id:
            return None

        movement_meta = dict(self.movement_by_id.get(movement_id, {}))
        if mapped:
            mapped_meta = {key: value for key, value in mapped.items() if key != "movement_id"}
            movement_meta = {**movement_meta, **mapped_meta}
        if movement_id not in self.movement_by_id and not movement_meta:
            return None

        mean_speed_mps = record.mean_speed_mps
        if mean_speed_mps is None and record.speed_kmh is not None:
            mean_speed_mps = float(record.speed_kmh) / 3.6
        if mean_speed_mps is None:
            mean_speed_mps = 0.0
        speed_kmh = float(record.speed_kmh) if record.speed_kmh is not None else float(mean_speed_mps) * 3.6

        return {
            "movement_id": movement_id,
            "tls_id": record.tls_id or movement_meta.get("tls_id", ""),
            "incoming_edge": record.incoming_edge or movement_meta.get("incoming_edge", ""),
            "outgoing_edge": record.outgoing_edge or movement_meta.get("outgoing_edge", ""),
            "turn_type": record.turn_type or movement_meta.get("turn_type", ""),
            "lane_ids": list(movement_meta.get("lane_ids", [])),
            "arrival_flow": float(record.arrival_flow),
            "discharge_flow": float(record.discharge_flow),
            "mean_speed": float(mean_speed_mps),
            "mean_speed_mps": float(mean_speed_mps),
            "speed_kmh": speed_kmh,
            "occupancy": float(record.occupancy),
            "queue_veh": float(record.queue_veh),
            "queue_meter": float(record.queue_meter),
            "incident_flag": int(record.incident_flag),
            "phase_id": int(record.phase_id),
            "phase_elapsed_s": float(record.phase_elapsed_s),
            "green_remaining_s": float(record.green_remaining_s),
            "signal_state": str(record.signal_state or ""),
            "zone_quality": "real",
        }

    def _missing_row(self, movement: dict[str, Any]) -> dict[str, Any]:
        movement_id = str(movement.get("movement_id", ""))
        if self.config.get("missing_fill_policy") == "last_or_zero":
            last_row = self.last_rows_by_movement_id.get(movement_id)
            if last_row is not None:
                row = dict(last_row)
                row["zone_quality"] = "real_last_filled"
                return row

        return {
            "movement_id": movement_id,
            "tls_id": movement.get("tls_id", ""),
            "incoming_edge": movement.get("incoming_edge", ""),
            "outgoing_edge": movement.get("outgoing_edge", ""),
            "turn_type": movement.get("turn_type", ""),
            "lane_ids": list(movement.get("lane_ids", [])),
            "arrival_flow": 0.0,
            "discharge_flow": 0.0,
            "mean_speed": 0.0,
            "mean_speed_mps": 0.0,
            "speed_kmh": 0.0,
            "occupancy": 0.0,
            "queue_veh": 0.0,
            "queue_meter": 0.0,
            "incident_flag": 0,
            "phase_id": -1,
            "phase_elapsed_s": 0.0,
            "green_remaining_s": 0.0,
            "signal_state": "",
            "zone_quality": "real_missing_filled",
        }

    def _append_csv(self, snapshot: dict[str, Any]) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self._csv_header_matches()
        with self.csv_path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=MOVEMENT_CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            for movement in snapshot["movements"]:
                writer.writerow(
                    {
                        "run_id": snapshot["run_id"],
                        "scenario_id": snapshot.get("scenario_id", ""),
                        "seed": snapshot.get("seed", ""),
                        "demand_scale": snapshot.get("demand_scale", ""),
                        "base_demand_factor": snapshot.get("base_demand_factor", ""),
                        "signal_variant": snapshot.get("signal_variant", "real_signal"),
                        "event_type": snapshot.get("event_type", ""),
                        "event_policy": snapshot.get("event_policy", ""),
                        "speed_factor": snapshot.get("speed_factor", ""),
                        "incident_type": snapshot.get("incident_type", ""),
                        "incident_start_s": snapshot.get("incident_start_s", ""),
                        "incident_end_s": snapshot.get("incident_end_s", ""),
                        "affected_edges": "|".join(snapshot.get("affected_edges", [])),
                        "timestamp": snapshot["timestamp"],
                        "step": snapshot["step"],
                        "movement_id": movement["movement_id"],
                        "tls_id": movement["tls_id"],
                        "incoming_edge": movement["incoming_edge"],
                        "outgoing_edge": movement["outgoing_edge"],
                        "turn_type": movement["turn_type"],
                        "lane_ids": "|".join(movement.get("lane_ids", [])),
                        "arrival_flow": movement["arrival_flow"],
                        "discharge_flow": movement["discharge_flow"],
                        "mean_speed_mps": round(float(movement["mean_speed_mps"]), 6),
                        "speed_kmh": round(float(movement["speed_kmh"]), 6),
                        "occupancy": round(float(movement["occupancy"]), 6),
                        "queue_veh": movement["queue_veh"],
                        "queue_meter": round(float(movement["queue_meter"]), 6),
                        "incident_flag": movement["incident_flag"],
                        "phase_id": movement["phase_id"],
                        "phase_elapsed_s": round(float(movement["phase_elapsed_s"]), 6),
                        "green_remaining_s": round(float(movement["green_remaining_s"]), 6),
                        "signal_state": movement.get("signal_state", ""),
                        "zone_quality": movement.get("zone_quality", ""),
                    }
                )

    def _csv_header_matches(self) -> bool:
        if not self.csv_path.exists():
            return False
        with self.csv_path.open("r", encoding="utf-8", errors="ignore") as fp:
            first_line = fp.readline().strip()
        return [field.strip() for field in first_line.split(",")] == MOVEMENT_CSV_FIELDS

    def _mapped_detector(self, detector_id: str | None) -> dict[str, Any]:
        if not detector_id:
            return {}
        mapped = self.detector_map.get(detector_id)
        return dict(mapped) if isinstance(mapped, dict) else {}

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
