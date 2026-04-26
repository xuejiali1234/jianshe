from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

from sim.route_tools import sanitize_route_file
from sim.validation import configure_sumo_python_path

configure_sumo_python_path()

import sumolib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NET = PROJECT_ROOT / "czq.net.xml"
DEFAULT_SOURCE_ROUTE = PROJECT_ROOT / "czq_demand.rou.xml"
DEFAULT_OUTPUT_ROUTE = PROJECT_ROOT / "czq_demand.rou.xml"
DEFAULT_ARCHIVE_DIR = PROJECT_ROOT / "data" / "archive" / "route_backups"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "route_spatial_config.json"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "spatial_route_summary.json"
DEFAULT_HIGH_INPUT_EDGES = [
    "-E15",
    "-E17",
    "E40",
    "-E39",
    "E31",
    "-E33",
]
DEFAULT_LOW_INPUT_EDGES = [
    "-E4",
    "143009830#0.1574",
    "-143009874.1041",
    "E23.226",
    "-E27",
    "-472652453#5.1331",
    "E29",
]
DEFAULT_RELIEF_DESTINATION_EDGES = [
    "-E34.60",
    "-E35.87",
    "-472652453#1",
    "158074689#3",
    "-158074689#3",
]
DEFAULT_DISCOURAGED_EDGES = {
    "E3": 2.4,
    "E5": 2.6,
    "E6": 2.8,
    "-E23.55": 3.0,
    "-472652453#1": 1.35,
    "472652453#1": 2.1,
    "472652453#2": 2.15,
    "472652453#3": 2.15,
    "158176575#3.57": 2.1,
    "158176575#3": 1.9,
    "158074689#1": 2.3,
    "158074689#3": 1.75,
    "-158074689#3": 1.75,
    "143009830#0.1693": 1.8,
}
DEFAULT_ENCOURAGED_EDGES = {
    "-E34.60": 0.78,
    "-E35.87": 0.74,
    "-472652453#1": 0.92,
    "158074689#3": 0.9,
    "-158074689#3": 0.9,
}


@dataclass(frozen=True)
class EdgeInfo:
    edge_id: str
    lane_count: int
    x: float
    y: float
    perimeter_distance: float
    is_perimeter: bool


def _edge_center(edge: object) -> tuple[float, float]:
    shape = edge.getShape()
    return (
        sum(point[0] for point in shape) / len(shape),
        sum(point[1] for point in shape) / len(shape),
    )


def _build_edge_info(
    net: object,
    perimeter_margin: float,
    min_perimeter_lanes: int,
) -> dict[str, EdgeInfo]:
    regular_edges = [
        edge
        for edge in net.getEdges()
        if edge.getFunction() != "internal" and edge.getShape()
    ]
    centers = [(edge, *_edge_center(edge)) for edge in regular_edges]
    xmin = min(center[1] for center in centers)
    xmax = max(center[1] for center in centers)
    ymin = min(center[2] for center in centers)
    ymax = max(center[2] for center in centers)
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)

    edge_info: dict[str, EdgeInfo] = {}
    for edge, x, y in centers:
        perimeter_distance = min(
            (x - xmin) / width,
            (xmax - x) / width,
            (y - ymin) / height,
            (ymax - y) / height,
        )
        is_perimeter = (
            perimeter_distance <= perimeter_margin
            and edge.getLaneNumber() >= min_perimeter_lanes
        )
        edge_info[edge.getID()] = EdgeInfo(
            edge_id=edge.getID(),
            lane_count=edge.getLaneNumber(),
            x=x,
            y=y,
            perimeter_distance=perimeter_distance,
            is_perimeter=is_perimeter,
        )

    return edge_info


def _path_ids(path: list[object] | tuple[object, ...] | None) -> list[str]:
    if not path:
        return []
    return [edge.getID() for edge in path]


def _path_distance(path_ids: list[str], edge_info: dict[str, EdgeInfo]) -> float:
    if not path_ids:
        return 1.0
    missing_penalty = 0.5
    return sum(edge_info.get(edge_id).perimeter_distance if edge_id in edge_info else missing_penalty for edge_id in path_ids) / len(path_ids)


def _internal_share(path_ids: list[str], edge_info: dict[str, EdgeInfo]) -> float:
    if not path_ids:
        return 1.0
    internal_count = sum(
        1
        for edge_id in path_ids
        if not edge_info.get(edge_id, EdgeInfo(edge_id, 1, 0.0, 0.0, 1.0, False)).is_perimeter
    )
    return internal_count / len(path_ids)


def _valid_shortest_path(net: object, from_edge_id: str, to_edge_id: str) -> list[str]:
    try:
        path, _cost = net.getShortestPath(net.getEdge(from_edge_id), net.getEdge(to_edge_id))
    except Exception:
        return []
    return _path_ids(path)


def _path_score(
    path_ids: list[str],
    edge_info: dict[str, EdgeInfo],
    discouraged_edges: dict[str, float] | None = None,
    encouraged_edges: dict[str, float] | None = None,
) -> float:
    if not path_ids:
        return 1e9
    discouraged_edges = discouraged_edges or {}
    encouraged_edges = encouraged_edges or {}
    score = float(len(path_ids))
    score += 5.0 * _internal_share(path_ids, edge_info)
    score += 1.5 * _path_distance(path_ids, edge_info)
    for edge_id in path_ids:
        score *= discouraged_edges.get(edge_id, 1.0)
        score *= encouraged_edges.get(edge_id, 1.0)
    return max(score, 1e-6)


def _weighted_choice(routes: list[list[str]], scores: list[float], rng: random.Random) -> list[str]:
    weights = [1.0 / max(score, 1e-6) for score in scores]
    total = sum(weights)
    threshold = rng.random() * total
    current = 0.0
    for route, weight in zip(routes, weights):
        current += weight
        if current >= threshold:
            return route
    return routes[-1]


def _sample_routes(
    net: object,
    edge_info: dict[str, EdgeInfo],
    origins: list[str],
    destinations: list[str],
    count: int,
    max_internal_share: float,
    min_length: int,
    rng: random.Random,
    require_unique: bool = False,
    discouraged_edges: dict[str, float] | None = None,
    encouraged_edges: dict[str, float] | None = None,
) -> list[list[str]]:
    routes: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = max(2000, count * 180)
    candidate_routes: list[list[str]] = []
    candidate_scores: list[float] = []

    for _ in range(attempts):
        if len(routes) >= count:
            break
        start = rng.choice(origins)
        end = rng.choice(destinations)
        if start == end:
            continue
        path_ids = _valid_shortest_path(net, start, end)
        if len(path_ids) < min_length:
            continue
        route_key = tuple(path_ids)
        if require_unique and route_key in seen:
            continue
        if _internal_share(path_ids, edge_info) > max_internal_share:
            continue
        candidate_routes.append(path_ids)
        candidate_scores.append(_path_score(path_ids, edge_info, discouraged_edges, encouraged_edges))
        seen.add(route_key)
        if len(candidate_routes) >= max(12, count * 4):
            routes.append(_weighted_choice(candidate_routes, candidate_scores, rng))
            candidate_routes.clear()
            candidate_scores.clear()

    while len(routes) < count and candidate_routes:
        routes.append(_weighted_choice(candidate_routes, candidate_scores, rng))

    if len(routes) < count:
        raise RuntimeError(
            f"Only built {len(routes)}/{count} routes. "
            "Try relaxing perimeter_margin or max_internal_share."
        )
    return routes


def _filter_existing_edges(net: object, edge_ids: list[str], label: str) -> list[str]:
    valid: list[str] = []
    missing: list[str] = []
    unusable: list[str] = []
    for edge_id in edge_ids:
        try:
            edge = net.getEdge(edge_id)
            if edge.getFunction() != "internal" and edge.getOutgoing():
                valid.append(edge_id)
            else:
                unusable.append(edge_id)
        except Exception:
            missing.append(edge_id)
    if missing:
        print(f"warning: ignored missing {label} edges: {missing}")
    if unusable:
        print(f"warning: ignored unusable {label} origin edges: {unusable}")
    if not valid:
        raise ValueError(f"No valid {label} edges remain after filtering.")
    return valid


def _filter_existing_route_edges(net: object, edge_ids: list[str], label: str) -> list[str]:
    valid: list[str] = []
    missing: list[str] = []
    for edge_id in edge_ids:
        try:
            edge = net.getEdge(edge_id)
            if edge.getFunction() != "internal":
                valid.append(edge_id)
        except Exception:
            missing.append(edge_id)
    if missing:
        print(f"warning: ignored missing {label} edges: {missing}")
    return valid


def _filter_existing_weight_map(net: object, weights: dict[str, float], label: str) -> dict[str, float]:
    valid: dict[str, float] = {}
    missing: list[str] = []
    for edge_id, weight in weights.items():
        try:
            edge = net.getEdge(edge_id)
            if edge.getFunction() != "internal":
                valid[edge_id] = float(weight)
        except Exception:
            missing.append(edge_id)
    if missing:
        print(f"warning: ignored missing {label} edges: {missing}")
    return valid


def _allocate_integer_flow(total: int, count: int, rng: random.Random) -> list[int]:
    weights = [rng.uniform(0.65, 1.35) for _ in range(count)]
    raw = [total * weight / sum(weights) for weight in weights]
    values = [max(1, int(value)) for value in raw]
    diff = total - sum(values)
    order = sorted(range(count), key=lambda idx: raw[idx] - int(raw[idx]), reverse=True)

    while diff > 0:
        for idx in order:
            if diff <= 0:
                break
            values[idx] += 1
            diff -= 1
    while diff < 0:
        for idx in sorted(range(count), key=lambda item: values[item], reverse=True):
            if diff >= 0:
                break
            if values[idx] > 1:
                values[idx] -= 1
                diff += 1

    return values


def _split_integer(total: int, parts: int) -> list[int]:
    if parts <= 0:
        return []
    base = total // parts
    values = [base for _ in range(parts)]
    for index in range(total - base * parts):
        values[index] += 1
    return values


def _sample_routes_by_origin(
    net: object,
    edge_info: dict[str, EdgeInfo],
    origins: list[str],
    destinations: list[str],
    route_count: int,
    total_vehs_per_hour: int,
    max_internal_share: float,
    min_length: int,
    rng: random.Random,
    discouraged_edges: dict[str, float] | None = None,
    encouraged_edges: dict[str, float] | None = None,
) -> tuple[list[list[str]], list[int], dict[str, int]]:
    origin_route_counts = _split_integer(route_count, len(origins))
    origin_totals = _split_integer(total_vehs_per_hour, len(origins))
    routes: list[list[str]] = []
    flows: list[int] = []
    origin_flow_totals: dict[str, int] = {}

    for origin, count, total in zip(origins, origin_route_counts, origin_totals):
        if count <= 0 or total <= 0:
            continue
        origin_routes = []
        relax_options = [
            max_internal_share,
            0.55,
            0.82,
            1.0,
        ]
        for relaxed_share in relax_options:
            try:
                origin_routes = _sample_routes(
                    net,
                    edge_info,
                    origins=[origin],
                    destinations=destinations,
                    count=count,
                    max_internal_share=max(relaxed_share, max_internal_share),
                    min_length=min_length,
                    rng=rng,
                    discouraged_edges=discouraged_edges,
                    encouraged_edges=encouraged_edges,
                )
                break
            except RuntimeError:
                continue
        if not origin_routes:
            raise RuntimeError(f"Could not build routes for origin {origin}.")
        origin_flows = _allocate_integer_flow(total, count, rng)
        routes.extend(origin_routes)
        flows.extend(origin_flows)
        origin_flow_totals[origin] = sum(origin_flows)

    return routes, flows, origin_flow_totals


def _read_base_total(route_path: Path) -> int:
    root = ET.fromstring(sanitize_route_file(route_path))
    flows = root.findall("flow")
    return int(round(sum(float(flow.get("vehsPerHour", "0")) for flow in flows)))


def _write_route_file(
    output_path: Path,
    route_groups: list[tuple[str, list[list[str]], list[int]]],
) -> None:
    root = ET.Element("routes")
    ET.SubElement(root, "vType", {"id": "passenger_car", "vClass": "passenger"})

    for group_name, routes, flows in route_groups:
        for route_index, (path_ids, vehs_per_hour) in enumerate(zip(routes, flows)):
            route_id = f"r_{group_name}_{route_index:03d}"
            ET.SubElement(root, "route", {"id": route_id, "edges": " ".join(path_ids)})
            ET.SubElement(
                root,
                "flow",
                {
                    "id": f"f_{group_name}_{route_index:03d}",
                    "type": "passenger_car",
                    "route": route_id,
                    "begin": "0",
                    "end": "3600",
                    "vehsPerHour": str(vehs_per_hour),
                    "departSpeed": "max",
                    "departLane": "best",
                },
            )

    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def rebalance_routes(
    net_path: Path,
    source_route: Path,
    output_route: Path,
    archive_dir: Path,
    config_path: Path,
    summary_path: Path,
    perimeter_share: float,
    perimeter_margin: float,
    min_perimeter_lanes: int,
    perimeter_route_count: int,
    internal_route_count: int,
    high_input_edges: list[str],
    low_input_edges: list[str],
    high_input_share: float,
    relief_destination_edges: list[str],
    relief_destination_weight: int,
    discouraged_edges: dict[str, float],
    encouraged_edges: dict[str, float],
    seed: int,
    overwrite: bool,
) -> dict[str, object]:
    rng = random.Random(seed)
    net = sumolib.net.readNet(str(net_path))
    edge_info = _build_edge_info(net, perimeter_margin, min_perimeter_lanes)
    perimeter_edges = sorted(edge_id for edge_id, info in edge_info.items() if info.is_perimeter)
    internal_edges = sorted(edge_id for edge_id, info in edge_info.items() if not info.is_perimeter)
    high_origins = _filter_existing_edges(net, high_input_edges, "high-input")
    low_origins = _filter_existing_edges(net, low_input_edges, "low-input")
    relief_destinations = _filter_existing_route_edges(net, relief_destination_edges, "relief destination")
    discouraged_edge_weights = _filter_existing_weight_map(net, discouraged_edges, "discouraged")
    encouraged_edge_weights = _filter_existing_weight_map(net, encouraged_edges, "encouraged")
    weighted_perimeter_destinations = list(perimeter_edges)
    if relief_destinations and relief_destination_weight > 0:
        weighted_perimeter_destinations.extend(relief_destinations * int(relief_destination_weight))

    high_input_share = max(0.0, min(1.0, high_input_share))
    high_perimeter_count = max(1, int(round(perimeter_route_count * high_input_share)))
    low_perimeter_count = max(1, perimeter_route_count - high_perimeter_count)
    high_internal_count = max(1, int(round(internal_route_count * high_input_share)))
    low_internal_count = max(1, internal_route_count - high_internal_count)

    base_total = _read_base_total(source_route)
    perimeter_total = int(round(base_total * perimeter_share))
    internal_total = base_total - perimeter_total
    high_perimeter_total = int(round(perimeter_total * high_input_share))
    low_perimeter_total = perimeter_total - high_perimeter_total
    high_internal_total = int(round(internal_total * high_input_share))
    low_internal_total = internal_total - high_internal_total

    high_perimeter_routes, high_perimeter_flows, high_perimeter_origin_totals = _sample_routes_by_origin(
        net,
        edge_info,
        origins=high_origins,
        destinations=weighted_perimeter_destinations,
        route_count=high_perimeter_count,
        total_vehs_per_hour=high_perimeter_total,
        max_internal_share=0.28,
        min_length=4,
        rng=rng,
        discouraged_edges=discouraged_edge_weights,
        encouraged_edges=encouraged_edge_weights,
    )
    low_perimeter_routes, low_perimeter_flows, low_perimeter_origin_totals = _sample_routes_by_origin(
        net,
        edge_info,
        origins=low_origins,
        destinations=weighted_perimeter_destinations,
        route_count=low_perimeter_count,
        total_vehs_per_hour=low_perimeter_total,
        max_internal_share=0.28,
        min_length=4,
        rng=rng,
        discouraged_edges=discouraged_edge_weights,
        encouraged_edges=encouraged_edge_weights,
    )
    high_internal_routes, high_internal_flows, high_internal_origin_totals = _sample_routes_by_origin(
        net,
        edge_info,
        origins=high_origins,
        destinations=internal_edges,
        route_count=high_internal_count,
        total_vehs_per_hour=high_internal_total,
        max_internal_share=0.82,
        min_length=3,
        rng=rng,
        discouraged_edges=discouraged_edge_weights,
        encouraged_edges=encouraged_edge_weights,
    )
    low_internal_routes, low_internal_flows, low_internal_origin_totals = _sample_routes_by_origin(
        net,
        edge_info,
        origins=low_origins,
        destinations=internal_edges,
        route_count=low_internal_count,
        total_vehs_per_hour=low_internal_total,
        max_internal_share=0.82,
        min_length=3,
        rng=rng,
        discouraged_edges=discouraged_edge_weights,
        encouraged_edges=encouraged_edge_weights,
    )

    if output_route.exists() and overwrite:
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = archive_dir / f"{output_route.stem}_before_spatial_{timestamp}{output_route.suffix}"
        shutil.copy2(output_route, backup_path)
    elif output_route.exists() and not overwrite:
        raise FileExistsError(f"{output_route} exists. Pass --overwrite to replace it.")

    route_groups = [
        ("perimeter_high", high_perimeter_routes, high_perimeter_flows),
        ("perimeter_low", low_perimeter_routes, low_perimeter_flows),
        ("internal_high", high_internal_routes, high_internal_flows),
        ("internal_low", low_internal_routes, low_internal_flows),
    ]
    _write_route_file(output_route, route_groups)
    perimeter_routes = high_perimeter_routes + low_perimeter_routes
    internal_routes = high_internal_routes + low_internal_routes
    perimeter_flows = high_perimeter_flows + low_perimeter_flows
    internal_flows = high_internal_flows + low_internal_flows
    high_total = sum(high_perimeter_flows) + sum(high_internal_flows)
    low_total = sum(low_perimeter_flows) + sum(low_internal_flows)
    origin_input_totals: dict[str, int] = {}
    for totals in [
        high_perimeter_origin_totals,
        low_perimeter_origin_totals,
        high_internal_origin_totals,
        low_internal_origin_totals,
    ]:
        for origin, value in totals.items():
            origin_input_totals[origin] = origin_input_totals.get(origin, 0) + value

    summary = {
        "status": "ok",
        "net": str(net_path),
        "source_route": str(source_route),
        "output_route": str(output_route),
        "seed": seed,
        "perimeter_share": perimeter_share,
        "internal_share": round(1.0 - perimeter_share, 4),
        "perimeter_margin": perimeter_margin,
        "min_perimeter_lanes": min_perimeter_lanes,
        "perimeter_edge_count": len(perimeter_edges),
        "internal_edge_count": len(internal_edges),
        "perimeter_route_count": perimeter_route_count,
        "internal_route_count": internal_route_count,
        "high_input_edges": high_origins,
        "low_input_edges": low_origins,
        "high_input_share": high_input_share,
        "low_input_share": round(1.0 - high_input_share, 4),
        "relief_destination_edges": relief_destinations,
        "relief_destination_weight": relief_destination_weight,
        "discouraged_edges": discouraged_edge_weights,
        "encouraged_edges": encouraged_edge_weights,
        "base_total_vehs_per_hour": base_total,
        "perimeter_total_vehs_per_hour": sum(perimeter_flows),
        "internal_total_vehs_per_hour": sum(internal_flows),
        "high_input_total_vehs_per_hour": high_total,
        "low_input_total_vehs_per_hour": low_total,
        "origin_input_totals_vehs_per_hour": dict(sorted(origin_input_totals.items())),
        "runtime_total_at_base_demand_0p25": round(base_total * 0.25),
        "runtime_high_input_at_base_demand_0p25": round(high_total * 0.25),
        "runtime_low_input_at_base_demand_0p25": round(low_total * 0.25),
        "perimeter_route_mean_internal_share": round(
            sum(_internal_share(route, edge_info) for route in perimeter_routes) / len(perimeter_routes),
            4,
        ),
        "internal_route_mean_internal_share": round(
            sum(_internal_share(route, edge_info) for route in internal_routes) / len(internal_routes),
            4,
        ),
        "perimeter_route_mean_distance": round(
            sum(_path_distance(route, edge_info) for route in perimeter_routes) / len(perimeter_routes),
            4,
        ),
        "internal_route_mean_distance": round(
            sum(_path_distance(route, edge_info) for route in internal_routes) / len(internal_routes),
            4,
        ),
        "perimeter_edges_sample": perimeter_edges[:40],
        "internal_edges_sample": internal_edges[:40],
    }

    config_payload = {
        "description": "Spatial route rebalance: 70% vehicles on perimeter-oriented routes, 30% to internal destinations.",
        "perimeter_share": perimeter_share,
        "perimeter_margin": perimeter_margin,
        "min_perimeter_lanes": min_perimeter_lanes,
        "perimeter_route_count": perimeter_route_count,
        "internal_route_count": internal_route_count,
        "high_input_share": high_input_share,
        "low_input_share": round(1.0 - high_input_share, 4),
        "high_input_edges": high_origins,
        "low_input_edges": low_origins,
        "relief_destination_edges": relief_destinations,
        "relief_destination_weight": relief_destination_weight,
        "discouraged_edges": discouraged_edge_weights,
        "encouraged_edges": encouraged_edge_weights,
        "seed": seed,
    }
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebalance SUMO demand routes by spatial perimeter/internal split.")
    parser.add_argument("--net", type=Path, default=DEFAULT_NET)
    parser.add_argument("--source-route", type=Path, default=DEFAULT_SOURCE_ROUTE)
    parser.add_argument("--output-route", type=Path, default=DEFAULT_OUTPUT_ROUTE)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--perimeter-share", type=float, default=0.70)
    parser.add_argument("--perimeter-margin", type=float, default=0.10)
    parser.add_argument("--min-perimeter-lanes", type=int, default=2)
    parser.add_argument("--perimeter-route-count", type=int, default=84)
    parser.add_argument("--internal-route-count", type=int, default=36)
    parser.add_argument("--high-input-share", type=float, default=0.62)
    parser.add_argument("--high-input-edges", nargs="*", default=DEFAULT_HIGH_INPUT_EDGES)
    parser.add_argument("--low-input-edges", nargs="*", default=DEFAULT_LOW_INPUT_EDGES)
    parser.add_argument("--relief-destination-edges", nargs="*", default=DEFAULT_RELIEF_DESTINATION_EDGES)
    parser.add_argument("--relief-destination-weight", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = rebalance_routes(
        net_path=args.net,
        source_route=args.source_route,
        output_route=args.output_route,
        archive_dir=args.archive_dir,
        config_path=args.config,
        summary_path=args.summary,
        perimeter_share=args.perimeter_share,
        perimeter_margin=args.perimeter_margin,
        min_perimeter_lanes=args.min_perimeter_lanes,
        perimeter_route_count=args.perimeter_route_count,
        internal_route_count=args.internal_route_count,
        high_input_edges=args.high_input_edges,
        low_input_edges=args.low_input_edges,
        high_input_share=args.high_input_share,
        relief_destination_edges=args.relief_destination_edges,
        relief_destination_weight=args.relief_destination_weight,
        discouraged_edges=DEFAULT_DISCOURAGED_EDGES,
        encouraged_edges=DEFAULT_ENCOURAGED_EDGES,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
