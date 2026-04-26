from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.signal_timing import build_webster_signal_net, discover_intersection_junction_ids


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "czq.net.xml"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "czq_tls_webster.net.xml"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "czq_tls_webster_summary.json"
DEFAULT_CONTROL_CONFIG = PROJECT_ROOT / "configs" / "signal_control_config.json"
DEFAULT_MOVEMENT_CSV = PROJECT_ROOT / "data" / "raw" / "batch_movement_aggregates.csv"
DEFAULT_MOVEMENT_CONFIG = PROJECT_ROOT / "configs" / "movement_config.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a signalized SUMO network with Webster-style fixed-time plans.",
    )
    parser.add_argument("--source-net", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-net", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--control-config", type=Path, default=DEFAULT_CONTROL_CONFIG)
    parser.add_argument("--movement-csv", type=Path, default=None)
    parser.add_argument("--movement-config", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["config", "auto-all-intersections"],
        default=None,
        help="Override TLS selection mode. config uses signal_control_config.json.",
    )
    args = parser.parse_args()

    control_config = {}
    if args.control_config.exists():
        control_config = json.loads(args.control_config.read_text(encoding="utf-8"))

    mode = args.mode or control_config.get("tls_selection_mode", "manual")
    disabled_tls_ids = control_config.get("disabled_tls_junction_ids") or []
    manual_tls_ids = control_config.get("manual_tls_junction_ids")
    adjustment_enabled = bool(control_config.get("movement_signal_adjustment_enabled", True))
    movement_csv = args.movement_csv or Path(control_config.get("movement_signal_adjustment_csv", DEFAULT_MOVEMENT_CSV))
    movement_config = args.movement_config or Path(
        control_config.get("movement_signal_adjustment_config", DEFAULT_MOVEMENT_CONFIG)
    )
    if mode == "auto_all_intersections" or mode == "auto-all-intersections":
        min_incoming = int(control_config.get("auto_tls_min_incoming_edges", 2))
        min_outgoing = int(control_config.get("auto_tls_min_outgoing_edges", 2))
        manual_tls_ids = discover_intersection_junction_ids(
            args.source_net,
            min_incoming_edges=min_incoming,
            min_outgoing_edges=min_outgoing,
            excluded_junction_ids=disabled_tls_ids,
        )
        print(
            f"Auto TLS candidates: {len(manual_tls_ids)} "
            f"(incoming>={min_incoming}, outgoing>={min_outgoing})"
        )

    output_net, summaries = build_webster_signal_net(
        args.source_net,
        args.output_net,
        manual_tls_ids=manual_tls_ids,
        disabled_tls_ids=disabled_tls_ids,
        summary_path=args.summary,
        movement_csv=movement_csv if adjustment_enabled and movement_csv.exists() else None,
        movement_config_file=movement_config if adjustment_enabled and movement_config.exists() else None,
    )
    print(f"Built signalized net: {output_net}")
    print(f"TLS count: {len(summaries)}")
    for summary in summaries:
        print(
            f"{summary.tls_id}: cycle={summary.cycle_time}s, "
            f"greens={summary.green_phase_durations}, approaches={summary.approach_count}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
