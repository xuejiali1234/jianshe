from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

FULL_V3_CSV = RAW_DIR / "batch_movement_aggregates_full_v3.csv"
FULL_V3_MANIFEST = RAW_DIR / "scenarios_full_v3" / "manifest.csv"
LANE_INCIDENT_CSV = RAW_DIR / "batch_movement_aggregates_lane_incident_v1.csv"
LANE_INCIDENT_MANIFEST = RAW_DIR / "scenarios_lane_incident_v1" / "manifest.csv"

SELECTED_INCIDENT_TYPES = [
    "north_j9_e13",
    "north_j9_minus_e13",
    "west_minus_e21_32",
]

SELECTED_CSV = RAW_DIR / "batch_movement_aggregates_lane_incident_v1_selected.csv"
SELECTED_MANIFEST = RAW_DIR / "scenarios_lane_incident_v1_selected" / "manifest.csv"
MERGED_CSV = RAW_DIR / "batch_movement_aggregates_full_v3_plus_lane_incident_v1_selected.csv"
MERGED_MANIFEST = RAW_DIR / "scenarios_full_v3_plus_lane_incident_v1_selected" / "manifest.csv"
SUMMARY_JSON = RAW_DIR / "scenarios_full_v3_plus_lane_incident_v1_selected" / "merge_summary.json"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _align_manifest_columns(full_manifest: pd.DataFrame, selected_manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_columns: list[str] = []
    for column in list(full_manifest.columns) + list(selected_manifest.columns):
        if column not in all_columns:
            all_columns.append(column)
    for frame in (full_manifest, selected_manifest):
        for column in all_columns:
            if column not in frame.columns:
                frame[column] = "" if column != "lane_closure_count" else 0
    return full_manifest[all_columns], selected_manifest[all_columns]


def main() -> None:
    full_csv = _read_csv(FULL_V3_CSV)
    full_manifest = _read_csv(FULL_V3_MANIFEST)
    lane_csv = _read_csv(LANE_INCIDENT_CSV)
    lane_manifest = _read_csv(LANE_INCIDENT_MANIFEST)

    selected_lane_csv = lane_csv[lane_csv["incident_type"].isin(SELECTED_INCIDENT_TYPES)].copy()
    selected_lane_manifest = lane_manifest[lane_manifest["incident_type"].isin(SELECTED_INCIDENT_TYPES)].copy()

    full_manifest_aligned, selected_manifest_aligned = _align_manifest_columns(
        full_manifest.copy(),
        selected_lane_manifest.copy(),
    )

    merged_csv = pd.concat([full_csv, selected_lane_csv], ignore_index=True)
    merged_manifest = pd.concat([full_manifest_aligned, selected_manifest_aligned], ignore_index=True)

    SELECTED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MERGED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    selected_lane_csv.to_csv(SELECTED_CSV, index=False, encoding="utf-8")
    selected_manifest_aligned.to_csv(SELECTED_MANIFEST, index=False, encoding="utf-8")
    merged_csv.to_csv(MERGED_CSV, index=False, encoding="utf-8")
    merged_manifest.to_csv(MERGED_MANIFEST, index=False, encoding="utf-8")

    summary = {
        "selected_incident_types": SELECTED_INCIDENT_TYPES,
        "selected_incremental_runs": int(selected_lane_manifest["run_id"].nunique()),
        "selected_incremental_rows": int(len(selected_lane_csv)),
        "merged_runs": int(merged_manifest["run_id"].nunique()),
        "merged_rows": int(len(merged_csv)),
        "selected_csv": str(SELECTED_CSV.relative_to(PROJECT_ROOT)),
        "selected_manifest": str(SELECTED_MANIFEST.relative_to(PROJECT_ROOT)),
        "merged_csv": str(MERGED_CSV.relative_to(PROJECT_ROOT)),
        "merged_manifest": str(MERGED_MANIFEST.relative_to(PROJECT_ROOT)),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
