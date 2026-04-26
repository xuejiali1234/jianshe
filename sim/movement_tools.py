from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any


TURN_TYPES = {"l", "s", "r"}
DEFAULT_ARRIVAL_START_M = 50.0
DEFAULT_DETECTION_UPSTREAM_M = 150.0
MIN_USABLE_ZONE_M = 80.0


def safe_movement_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_")
    return token or "empty"


def make_movement_id(tls_id: str, incoming_edge: str, turn_type: str, outgoing_edge: str) -> str:
    return "__".join(
        [
            safe_movement_token(tls_id),
            safe_movement_token(incoming_edge),
            safe_movement_token(turn_type),
            safe_movement_token(outgoing_edge),
        ]
    )


def lane_to_edge_id(lane_id: str) -> str:
    if "_" not in lane_id:
        return lane_id
    edge_id, lane_index = lane_id.rsplit("_", 1)
    return edge_id if lane_index.isdigit() else lane_id


def load_movement_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_movement_rows(path: str | Path) -> list[dict[str, Any]]:
    data = load_movement_config(path)
    return list(data.get("movements", []))


def build_movement_config(
    net_file: str | Path,
    observed_edges: list[str],
    output_json: str | Path = "configs/movement_config.json",
    output_csv: str | Path = "data/processed/movement_map.csv",
    arrival_start_m: float = DEFAULT_ARRIVAL_START_M,
    detection_upstream_m: float = DEFAULT_DETECTION_UPSTREAM_M,
) -> dict[str, Any]:
    """Build a movement dictionary from controlled SUMO connections.

    The dictionary keeps only signal-controlled left/through/right movements from
    the configured incoming edges. SUMO U-turn connections (dir=t) are excluded.
    """

    from sim import configure_sumo_python_path

    configure_sumo_python_path()
    import sumolib

    net_path = Path(net_file)
    if not net_path.exists():
        raise FileNotFoundError(f"SUMO net file not found: {net_path}")

    tree = ET.parse(net_path)
    root = tree.getroot()
    observed_set = set(observed_edges)
    tl_green_phases = _collect_green_phases(root)
    grouped: "OrderedDict[tuple[str, str, str, str], dict[str, Any]]" = OrderedDict()

    for conn in root.findall(".//connection"):
        tls_id = conn.attrib.get("tl", "").strip()
        incoming_edge = conn.attrib.get("from", "").strip()
        outgoing_edge = conn.attrib.get("to", "").strip()
        turn_type = conn.attrib.get("dir", "").strip()
        if not tls_id or incoming_edge not in observed_set or turn_type not in TURN_TYPES:
            continue

        key = (tls_id, incoming_edge, turn_type, outgoing_edge)
        item = grouped.setdefault(
            key,
            {
                "tls_id": tls_id,
                "incoming_edge": incoming_edge,
                "outgoing_edge": outgoing_edge,
                "turn_type": turn_type,
                "lane_ids": set(),
                "link_indexes": set(),
            },
        )
        from_lane = conn.attrib.get("fromLane", "").strip()
        if from_lane:
            item["lane_ids"].add(f"{incoming_edge}_{from_lane}")
        link_index = conn.attrib.get("linkIndex", "").strip()
        if link_index != "":
            try:
                item["link_indexes"].add(int(link_index))
            except ValueError:
                pass

    net = sumolib.net.readNet(str(net_path))
    movements: list[dict[str, Any]] = []
    warning_edges: list[str] = []
    movement_id_counts: dict[str, int] = {}

    for data in grouped.values():
        incoming_edge = data["incoming_edge"]
        tls_id = data["tls_id"]
        link_indexes = sorted(data["link_indexes"])
        green_phase_ids = _green_phases_for_links(tl_green_phases.get(tls_id, []), link_indexes)
        zone = _build_zone_metadata(net, incoming_edge, detection_upstream_m)
        zone_quality = "ok" if zone["max_upstream_m"] >= MIN_USABLE_ZONE_M else "short_upstream"
        if zone_quality != "ok":
            warning_edges.append(incoming_edge)

        movement_id = make_movement_id(
            tls_id,
            incoming_edge,
            data["turn_type"],
            data["outgoing_edge"],
        )
        count = movement_id_counts.get(movement_id, 0)
        movement_id_counts[movement_id] = count + 1
        if count:
            movement_id = f"{movement_id}_{count + 1}"

        movements.append(
            {
                "movement_id": movement_id,
                "tls_id": tls_id,
                "incoming_edge": incoming_edge,
                "outgoing_edge": data["outgoing_edge"],
                "turn_type": data["turn_type"],
                "link_index": link_indexes[0] if link_indexes else -1,
                "link_indexes": link_indexes,
                "lane_ids": sorted(data["lane_ids"]),
                "phase_id": green_phase_ids[0] if green_phase_ids else -1,
                "green_phase_ids": green_phase_ids,
                "incoming_length_m": round(zone["incoming_length_m"], 3),
                "zone_length_m": round(zone["max_upstream_m"], 3),
                "arrival_start_m": float(arrival_start_m),
                "arrival_end_m": float(detection_upstream_m),
                "queue_start_m": 0.0,
                "queue_end_m": float(detection_upstream_m),
                "zone_quality": zone_quality,
                "zone_edges": zone["zone_edges"],
                "upstream_edges": zone["zone_edges"][1:],
                "zone_edge_offsets": zone["zone_edge_offsets"],
                "zone_edge_lengths": zone["zone_edge_lengths"],
                "zone_lane_m": round(zone["zone_lane_m"], 3),
            }
        )

    observed_without_movements = sorted(
        edge_id
        for edge_id in observed_edges
        if not any(m["incoming_edge"] == edge_id for m in movements)
    )
    turn_type_counts: dict[str, int] = {}
    for movement in movements:
        turn_type_counts[movement["turn_type"]] = turn_type_counts.get(movement["turn_type"], 0) + 1

    payload = {
        "version": 1,
        "observation_level": "movement",
        "source_net_file": str(net_path),
        "arrival_start_m": float(arrival_start_m),
        "detection_upstream_m": float(detection_upstream_m),
        "turn_types": sorted(TURN_TYPES),
        "movement_count": len(movements),
        "turn_type_counts": turn_type_counts,
        "observed_edges": list(observed_edges),
        "observed_edges_without_movements": observed_without_movements,
        "short_upstream_edges": sorted(set(warning_edges)),
        "movements": movements,
    }
    write_movement_outputs(payload, output_json, output_csv)
    return payload


def write_movement_outputs(
    payload: dict[str, Any],
    output_json: str | Path,
    output_csv: str | Path,
) -> None:
    json_path = Path(output_json)
    csv_path = Path(output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "movement_id",
        "tls_id",
        "incoming_edge",
        "outgoing_edge",
        "turn_type",
        "link_index",
        "link_indexes",
        "lane_ids",
        "phase_id",
        "green_phase_ids",
        "incoming_length_m",
        "zone_length_m",
        "zone_lane_m",
        "zone_quality",
        "zone_edges",
        "upstream_edges",
        "zone_edge_offsets",
        "arrival_start_m",
        "arrival_end_m",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for movement in payload.get("movements", []):
            writer.writerow(
                {
                    key: "|".join(str(v) for v in movement.get(key, []))
                    if key in {"link_indexes", "lane_ids", "green_phase_ids"}
                    else json.dumps(movement.get(key, {}), ensure_ascii=False)
                    if key == "zone_edge_offsets"
                    else "|".join(str(v) for v in movement.get(key, []))
                    if key in {"zone_edges", "upstream_edges"}
                    else movement.get(key, "")
                    for key in fieldnames
                }
            )


def _collect_green_phases(root: ET.Element) -> dict[str, list[tuple[int, str]]]:
    result: dict[str, list[tuple[int, str]]] = {}
    for tl_logic in root.findall(".//tlLogic"):
        tls_id = tl_logic.attrib.get("id", "")
        phases: list[tuple[int, str]] = []
        for phase_idx, phase in enumerate(tl_logic.findall("phase")):
            phases.append((phase_idx, phase.attrib.get("state", "")))
        result[tls_id] = phases
    return result


def _green_phases_for_links(
    phases: list[tuple[int, str]],
    link_indexes: list[int],
) -> list[int]:
    green_phase_ids: list[int] = []
    for phase_idx, state in phases:
        for link_index in link_indexes:
            if 0 <= link_index < len(state) and state[link_index] in {"G", "g"}:
                green_phase_ids.append(phase_idx)
                break
    return green_phase_ids


def _build_zone_metadata(net: Any, incoming_edge_id: str, max_upstream_m: float) -> dict[str, Any]:
    try:
        incoming_edge = net.getEdge(incoming_edge_id)
    except Exception:
        return {
            "incoming_length_m": 0.0,
            "max_upstream_m": 0.0,
            "zone_edges": [],
            "zone_edge_offsets": {},
            "zone_edge_lengths": {},
            "zone_lane_m": 1.0,
        }

    queue = deque([(incoming_edge, 0.0)])
    visited_offsets: dict[str, float] = {}
    edge_lengths: dict[str, float] = {}
    zone_edges: list[str] = []
    max_upstream = 0.0
    zone_lane_m = 0.0

    while queue:
        edge, offset = queue.popleft()
        edge_id = edge.getID()
        if edge_id in visited_offsets and visited_offsets[edge_id] <= offset:
            continue
        visited_offsets[edge_id] = offset
        zone_edges.append(edge_id)
        edge_length = float(edge.getLength())
        edge_lengths[edge_id] = edge_length
        lane_count = max(1, len(edge.getLanes()))
        included_length = max(0.0, min(edge_length, max_upstream_m - offset))
        zone_lane_m += included_length * lane_count
        max_upstream = max(max_upstream, min(max_upstream_m, offset + edge_length))

        if offset + edge_length >= max_upstream_m:
            continue
        try:
            incoming_map = edge.getIncoming()
        except Exception:
            incoming_map = {}
        for pred_edge in incoming_map.keys():
            queue.append((pred_edge, offset + edge_length))

    return {
        "incoming_length_m": float(incoming_edge.getLength()),
        "max_upstream_m": float(max_upstream),
        "zone_edges": zone_edges,
        "zone_edge_offsets": {edge_id: float(offset) for edge_id, offset in visited_offsets.items()},
        "zone_edge_lengths": {edge_id: float(length) for edge_id, length in edge_lengths.items()},
        "zone_lane_m": max(1.0, float(zone_lane_m)),
    }
