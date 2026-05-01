from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from sim.movement_tools import load_movement_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOVEMENT_CONFIG = PROJECT_ROOT / "configs" / "movement_config.json"
DEFAULT_NET_FILE = PROJECT_ROOT / "data" / "processed" / "czq_tls_webster.net.xml"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "tls_coordination_graph_v1.json"
DEFAULT_TLS_IDS = ["12257758211", "1704063278", "1702536336"]


def build_tls_coordination_graph(
    movement_config_path: str | Path = DEFAULT_MOVEMENT_CONFIG,
    net_file: str | Path = DEFAULT_NET_FILE,
    out_path: str | Path = DEFAULT_OUT,
    tls_ids: list[str] | None = None,
) -> dict[str, Any]:
    movement_source = Path(movement_config_path)
    if not movement_source.is_absolute():
        movement_source = PROJECT_ROOT / movement_source
    net_source = Path(net_file)
    if not net_source.is_absolute():
        net_source = PROJECT_ROOT / net_source
    payload = load_movement_config(movement_source)
    cluster_tls_ids = [str(tls_id) for tls_id in (tls_ids or DEFAULT_TLS_IDS)]
    movements = [
        movement
        for movement in payload.get("movements", [])
        if str(movement.get("tls_id", "")) in cluster_tls_ids
    ]
    tls_meta: dict[str, dict[str, Any]] = {
        tls_id: {
            "incoming_edges": sorted(
                {str(movement.get("incoming_edge", "")) for movement in movements if str(movement.get("tls_id", "")) == tls_id}
            ),
            "outgoing_edges": sorted(
                {str(movement.get("outgoing_edge", "")) for movement in movements if str(movement.get("tls_id", "")) == tls_id}
            ),
            "legal_green_phases": [],
        }
        for tls_id in cluster_tls_ids
    }

    phase_map = _green_phase_ids_from_net(net_source, cluster_tls_ids)
    for tls_id in cluster_tls_ids:
        tls_meta[tls_id]["legal_green_phases"] = sorted(phase_map.get(tls_id, []))

    by_tls = {
        tls_id: [
            movement
            for movement in movements
            if str(movement.get("tls_id", "")) == tls_id
        ]
        for tls_id in cluster_tls_ids
    }
    directed_edges = []
    same_corridor = []
    neighbors = {tls_id: {"upstream": [], "downstream": []} for tls_id in cluster_tls_ids}

    for from_tls in cluster_tls_ids:
        from_outgoing = set(tls_meta[from_tls]["outgoing_edges"])
        for to_tls in cluster_tls_ids:
            if from_tls == to_tls:
                continue
            to_incoming = set(tls_meta[to_tls]["incoming_edges"])
            shared = sorted(from_outgoing & to_incoming)
            if not shared:
                continue
            relation = {
                "from_tls_id": from_tls,
                "to_tls_id": to_tls,
                "relation": "upstream_downstream",
                "via_edges": shared,
            }
            directed_edges.append(relation)
            neighbors[from_tls]["downstream"].append(to_tls)
            neighbors[to_tls]["upstream"].append(from_tls)
            for edge_id in shared:
                same_corridor.append(
                    {
                        "tls_a": from_tls,
                        "tls_b": to_tls,
                        "shared_edge": edge_id,
                        "relation": "same_corridor",
                    }
                )

    for tls_id in cluster_tls_ids:
        neighbors[tls_id]["upstream"] = sorted(set(neighbors[tls_id]["upstream"]))
        neighbors[tls_id]["downstream"] = sorted(set(neighbors[tls_id]["downstream"]))

    graph = {
        "status": "ok",
        "version": 1,
        "movement_config": str(movement_source),
        "net_file": str(net_source),
        "tls_ids": cluster_tls_ids,
        "directed_edges": directed_edges,
        "shared_corridor_edges": same_corridor,
        "neighbors": neighbors,
        "tls_meta": tls_meta,
    }

    destination = Path(out_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return graph


def _green_phase_ids_from_net(net_file: Path, tls_ids: list[str]) -> dict[str, set[int]]:
    result = {str(tls_id): set() for tls_id in tls_ids}
    root = ET.parse(net_file).getroot()
    for tls_id in tls_ids:
        logic = root.find(f".//tlLogic[@id='{tls_id}']")
        if logic is None:
            continue
        for index, phase in enumerate(logic.findall("phase")):
            state = str(phase.attrib.get("state", ""))
            duration = float(phase.attrib.get("duration", 0.0))
            if duration >= 5.0 and ("g" in state or "G" in state) and "y" not in state:
                result[tls_id].add(index)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed multi-TLS coordination graph for RL V1.")
    parser.add_argument("--movement-config", default=str(DEFAULT_MOVEMENT_CONFIG))
    parser.add_argument("--net-file", default=str(DEFAULT_NET_FILE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--tls-ids", nargs="*", default=DEFAULT_TLS_IDS)
    args = parser.parse_args()
    graph = build_tls_coordination_graph(
        movement_config_path=args.movement_config,
        net_file=args.net_file,
        out_path=args.out,
        tls_ids=list(args.tls_ids),
    )
    print(
        json.dumps(
            {
                "status": graph["status"],
                "tls_ids": graph["tls_ids"],
                "directed_edge_count": len(graph["directed_edges"]),
                "out": args.out,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
