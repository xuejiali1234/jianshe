from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sim.movement_tools import load_movement_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOVEMENT_CONFIG = PROJECT_ROOT / "configs" / "movement_config.json"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "movement_graph.json"
RELATION_TYPES = [
    "same_tls",
    "same_incoming_edge",
    "same_phase",
    "upstream_downstream",
    "conflict_same_tls_different_phase",
]


def build_movement_graph(
    movement_config_path: str | Path = DEFAULT_MOVEMENT_CONFIG,
    out_path: str | Path = DEFAULT_OUT,
) -> dict[str, Any]:
    source = Path(movement_config_path)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    payload = load_movement_config(source)
    movements = list(payload.get("movements", []))
    movement_ids = [str(movement.get("movement_id", "")) for movement in movements if movement.get("movement_id")]

    by_tls = _group_by(movements, "tls_id")
    by_incoming = _group_by(movements, "incoming_edge")
    by_phase: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_incoming_edge: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for movement in movements:
        incoming = str(movement.get("incoming_edge", ""))
        if incoming:
            by_incoming_edge[incoming].append(movement)
        for phase_id in _green_phase_ids(movement):
            by_phase[(str(movement.get("tls_id", "")), phase_id)].append(movement)

    relations: dict[str, set[tuple[str, str]]] = {name: set() for name in RELATION_TYPES}
    _add_group_relation(relations["same_tls"], by_tls)
    _add_group_relation(relations["same_incoming_edge"], by_incoming)
    _add_group_relation(relations["same_phase"], by_phase)
    _add_upstream_downstream(relations["upstream_downstream"], movements, by_incoming_edge)
    _add_conflict_relation(relations["conflict_same_tls_different_phase"], by_tls)

    adjacency = {
        movement_id: {relation: [] for relation in RELATION_TYPES}
        for movement_id in movement_ids
    }
    for relation, edges in relations.items():
        for src, dst in edges:
            if src == dst:
                continue
            adjacency.setdefault(src, {name: [] for name in RELATION_TYPES})[relation].append(dst)

    for relation_map in adjacency.values():
        for relation in RELATION_TYPES:
            relation_map[relation] = sorted(set(relation_map.get(relation, [])))

    graph = {
        "status": "ok",
        "movement_config": str(source),
        "movement_count": len(movement_ids),
        "relation_types": RELATION_TYPES,
        "relation_counts": {
            relation: len(edges)
            for relation, edges in relations.items()
        },
        "adjacency": adjacency,
    }

    destination = Path(out_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return graph


def _group_by(movements: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for movement in movements:
        value = str(movement.get(key, ""))
        if value:
            groups[value].append(movement)
    return groups


def _movement_id(movement: dict[str, Any]) -> str:
    return str(movement.get("movement_id", ""))


def _add_group_relation(
    relation: set[tuple[str, str]],
    groups: dict[Any, list[dict[str, Any]]],
) -> None:
    for items in groups.values():
        ids = [_movement_id(item) for item in items if _movement_id(item)]
        for src in ids:
            for dst in ids:
                if src != dst:
                    relation.add((src, dst))


def _add_upstream_downstream(
    relation: set[tuple[str, str]],
    movements: list[dict[str, Any]],
    by_incoming_edge: dict[str, list[dict[str, Any]]],
) -> None:
    for movement in movements:
        src = _movement_id(movement)
        outgoing = str(movement.get("outgoing_edge", ""))
        if not src or not outgoing:
            continue
        for downstream in by_incoming_edge.get(outgoing, []):
            dst = _movement_id(downstream)
            if dst and dst != src:
                relation.add((src, dst))


def _add_conflict_relation(
    relation: set[tuple[str, str]],
    by_tls: dict[str, list[dict[str, Any]]],
) -> None:
    for movements in by_tls.values():
        for src_movement in movements:
            src = _movement_id(src_movement)
            src_phases = set(_green_phase_ids(src_movement))
            if not src or not src_phases:
                continue
            for dst_movement in movements:
                dst = _movement_id(dst_movement)
                if not dst or dst == src:
                    continue
                dst_phases = set(_green_phase_ids(dst_movement))
                if dst_phases and src_phases.isdisjoint(dst_phases):
                    relation.add((src, dst))


def _green_phase_ids(movement: dict[str, Any]) -> list[int]:
    values = movement.get("green_phase_ids", [])
    if not values and movement.get("phase_id") is not None:
        values = [movement.get("phase_id")]
    result = []
    for value in values:
        try:
            phase_id = int(value)
        except (TypeError, ValueError):
            continue
        if phase_id >= 0:
            result.append(phase_id)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build movement graph relations for RL and future ST models.")
    parser.add_argument("--movement-config", default=str(DEFAULT_MOVEMENT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    graph = build_movement_graph(args.movement_config, args.out)
    print(
        json.dumps(
            {
                "status": graph["status"],
                "movement_count": graph["movement_count"],
                "relation_counts": graph["relation_counts"],
                "out": args.out,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
