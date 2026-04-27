from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prediction import load_prediction_config
from sim import resolve_runtime_net_file
from sim.movement_tools import build_movement_config, load_movement_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "movement_quality_report.csv"


def inspect_movement_catalog(
    config_path: str | Path,
    movement_config_path: str | Path | None = None,
    report_path: str | Path = DEFAULT_REPORT,
    rebuild: bool = False,
) -> dict[str, Any]:
    config = load_prediction_config(config_path)
    movement_path = Path(movement_config_path or PROJECT_ROOT / config.movement_config_file)
    if not movement_path.is_absolute():
        movement_path = PROJECT_ROOT / movement_path

    if rebuild or not movement_path.exists():
        net_file = resolve_runtime_net_file(PROJECT_ROOT, config.sumo_net_file)
        build_movement_config(
            net_file,
            config.observed_edges,
            movement_path,
            PROJECT_ROOT / "data" / "processed" / "movement_map.csv",
        )

    payload = load_movement_config(movement_path)
    movements = list(payload.get("movements", []))
    by_tls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for movement in movements:
        by_tls[str(movement.get("tls_id", ""))].append(movement)
        by_incoming[str(movement.get("incoming_edge", ""))].append(movement)

    report = Path(report_path)
    if not report.is_absolute():
        report = PROJECT_ROOT / report
    report.parent.mkdir(parents=True, exist_ok=True)
    write_quality_report(report, movements)

    missing_edges = [
        edge_id for edge_id in config.observed_edges if edge_id not in by_incoming
    ]
    short_edges = sorted(
        {
            str(movement.get("incoming_edge", ""))
            for movement in movements
            if str(movement.get("zone_quality", "")) != "ok"
        }
    )
    movement_ids = [str(movement.get("movement_id", "")) for movement in movements]
    duplicate_ids = sorted(
        movement_id
        for movement_id, count in Counter(movement_ids).items()
        if movement_id and count > 1
    )
    phase_missing = [
        str(movement.get("movement_id", ""))
        for movement in movements
        if _safe_int(movement.get("phase_id"), -1) < 0
        or not movement.get("green_phase_ids")
    ]
    short_zone_movements = [
        str(movement.get("movement_id", ""))
        for movement in movements
        if _safe_float(movement.get("zone_length_m"), 0.0) < 80.0
    ]
    zone_quality_summary = dict(
        Counter(str(movement.get("zone_quality", "unknown") or "unknown") for movement in movements)
    )
    tls_summary = [
        {
            "tls_id": tls_id,
            "movement_count": len(items),
            "incoming_edges": len({item.get("incoming_edge") for item in items}),
            "turn_counts": turn_counts(items),
        }
        for tls_id, items in sorted(by_tls.items())
    ]
    return {
        "status": "ok",
        "movement_count": len(movements),
        "turn_type_counts": payload.get("turn_type_counts", turn_counts(movements)),
        "tls_count": len(by_tls),
        "tls_summary": tls_summary,
        "duplicate_movement_ids": duplicate_ids,
        "duplicate_movement_id_count": len(duplicate_ids),
        "phase_missing_count": len(phase_missing),
        "phase_missing_movements": phase_missing[:50],
        "short_zone_count": len(short_zone_movements),
        "short_zone_movements": short_zone_movements[:50],
        "short_upstream_count": len(short_edges),
        "zone_quality_summary": zone_quality_summary,
        "observed_edges_without_movements": missing_edges,
        "short_upstream_edges": short_edges,
        "quality_report": str(report),
    }


def write_quality_report(report: Path, movements: list[dict[str, Any]]) -> None:
    fieldnames = [
        "tls_id",
        "incoming_edge",
        "movement_id",
        "turn_type",
        "outgoing_edge",
        "link_indexes",
        "green_phase_ids",
        "lane_ids",
        "incoming_length_m",
        "zone_length_m",
        "zone_lane_m",
        "zone_quality",
        "upstream_edges",
    ]
    with report.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for movement in movements:
            writer.writerow(
                {
                    "tls_id": movement.get("tls_id", ""),
                    "incoming_edge": movement.get("incoming_edge", ""),
                    "movement_id": movement.get("movement_id", ""),
                    "turn_type": movement.get("turn_type", ""),
                    "outgoing_edge": movement.get("outgoing_edge", ""),
                    "link_indexes": "|".join(str(v) for v in movement.get("link_indexes", [])),
                    "green_phase_ids": "|".join(str(v) for v in movement.get("green_phase_ids", [])),
                    "lane_ids": "|".join(str(v) for v in movement.get("lane_ids", [])),
                    "incoming_length_m": movement.get("incoming_length_m", ""),
                    "zone_length_m": movement.get("zone_length_m", ""),
                    "zone_lane_m": movement.get("zone_lane_m", ""),
                    "zone_quality": movement.get("zone_quality", ""),
                    "upstream_edges": "|".join(str(v) for v in movement.get("upstream_edges", [])),
                }
            )


def turn_counts(movements: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for movement in movements:
        turn_type = str(movement.get("turn_type", ""))
        counts[turn_type] = counts.get(turn_type, 0) + 1
    return counts


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def print_human_summary(summary: dict[str, Any]) -> None:
    print(f"movement_count={summary['movement_count']}")
    print(f"tls_count={summary['tls_count']}")
    print(f"turn_type_counts={summary['turn_type_counts']}")
    print(f"quality_report={summary['quality_report']}")
    if summary["observed_edges_without_movements"]:
        print("WARNING observed edges without movements:")
        for edge_id in summary["observed_edges_without_movements"]:
            print(f"  - {edge_id}")
    if summary["short_upstream_edges"]:
        print("WARNING short upstream zones:")
        for edge_id in summary["short_upstream_edges"]:
            print(f"  - {edge_id}")
    print("\nTLS summary:")
    for item in summary["tls_summary"]:
        print(
            f"  {item['tls_id']}: movements={item['movement_count']} "
            f"incoming_edges={item['incoming_edges']} turns={item['turn_counts']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect movement catalog by traffic light.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "prediction_config.json"))
    parser.add_argument("--movement-config", default=None)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--out", default=None, help="Write machine-readable JSON summary to this path.")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    args = parser.parse_args()

    summary = inspect_movement_catalog(
        args.config,
        args.movement_config,
        args.report,
        args.rebuild,
    )
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_human_summary(summary)


if __name__ == "__main__":
    main()
