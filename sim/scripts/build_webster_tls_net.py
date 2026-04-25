from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.signal_timing import build_webster_signal_net


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "czq.net.xml"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "czq_tls_webster.net.xml"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "czq_tls_webster_summary.json"
DEFAULT_CONTROL_CONFIG = PROJECT_ROOT / "configs" / "signal_control_config.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a signalized SUMO network with Webster-style fixed-time plans.",
    )
    parser.add_argument("--source-net", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-net", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--control-config", type=Path, default=DEFAULT_CONTROL_CONFIG)
    args = parser.parse_args()

    control_config = {}
    if args.control_config.exists():
        control_config = json.loads(args.control_config.read_text(encoding="utf-8"))

    output_net, summaries = build_webster_signal_net(
        args.source_net,
        args.output_net,
        manual_tls_ids=control_config.get("manual_tls_junction_ids"),
        disabled_tls_ids=control_config.get("disabled_tls_junction_ids"),
        summary_path=args.summary,
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
