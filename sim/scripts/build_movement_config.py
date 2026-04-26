from __future__ import annotations

import argparse
import json
from pathlib import Path

from prediction import load_prediction_config
from sim import resolve_runtime_net_file
from sim.movement_tools import build_movement_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build signalized movement dictionary from SUMO net.xml.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "prediction_config.json"))
    parser.add_argument("--output-json", default=str(PROJECT_ROOT / "configs" / "movement_config.json"))
    parser.add_argument("--output-csv", default=str(PROJECT_ROOT / "data" / "processed" / "movement_map.csv"))
    args = parser.parse_args()

    config = load_prediction_config(args.config)
    net_file = resolve_runtime_net_file(PROJECT_ROOT, config.sumo_net_file)
    payload = build_movement_config(
        net_file,
        config.observed_edges,
        args.output_json,
        args.output_csv,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "movement_count": payload["movement_count"],
                "turn_type_counts": payload["turn_type_counts"],
                "observed_edges_without_movements": payload["observed_edges_without_movements"],
                "short_upstream_edges": payload["short_upstream_edges"],
                "json": args.output_json,
                "csv": args.output_csv,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
