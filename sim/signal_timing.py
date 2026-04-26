from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from .validation import configure_sumo_python_path

configure_sumo_python_path()

import sumolib


@dataclass
class WebsterTlsSummary:
    tls_id: str
    approach_count: int
    incoming_edges: list[str]
    total_incoming_lanes: int
    proxy_y: float
    loss_time: int
    cycle_time: int
    green_phase_indices: list[int]
    green_phase_weights: list[float]
    green_phase_durations: list[int]
    base_green_phase_durations: list[int] = field(default_factory=list)
    phase_adjustments: list[dict[str, object]] = field(default_factory=list)


def discover_intersection_junction_ids(
    source_net: str | Path,
    min_incoming_edges: int = 2,
    min_outgoing_edges: int = 2,
    excluded_junction_ids: list[str] | None = None,
) -> list[str]:
    """Return broad signal candidates for manual screening.

    The intentionally broad default includes all non-internal junctions with at
    least two incoming and two outgoing regular edges. This may include some
    road split/merge nodes, which is acceptable for the current workflow because
    the user wants to add first and manually delete later.
    """
    excluded = set(excluded_junction_ids or [])
    net = sumolib.net.readNet(str(source_net))
    candidates: list[str] = []

    for node in net.getNodes():
        node_id = node.getID()
        if node_id in excluded:
            continue
        if node.getType() == "internal":
            continue
        incoming_edges = [
            edge
            for edge in node.getIncoming()
            if edge.getFunction() != "internal"
        ]
        outgoing_edges = [
            edge
            for edge in node.getOutgoing()
            if edge.getFunction() != "internal"
        ]
        if (
            len(incoming_edges) >= min_incoming_edges
            and len(outgoing_edges) >= min_outgoing_edges
        ):
            candidates.append(node_id)

    return sorted(candidates)


def _phase_is_green(state: str) -> bool:
    return any(char == "G" for char in state)


def _target_proxy_y(approach_count: int, total_incoming_lanes: int) -> float:
    base = 0.58 + 0.05 * max(0, approach_count - 2) + 0.008 * max(0, total_incoming_lanes - 6)
    return max(0.66, min(0.84, base))


def _cycle_bounds(approach_count: int) -> tuple[int, int]:
    if approach_count >= 4:
        return 70, 120
    if approach_count == 3:
        return 60, 100
    return 50, 90


def _phase_weight(
    phase_state: str,
    link_to_lane: dict[int, object],
) -> float:
    lane_weights: dict[str, float] = {}
    for link_index, signal in enumerate(phase_state):
        if signal not in "Gg":
            continue
        lane = link_to_lane.get(link_index)
        if lane is None:
            continue
        lane_id = lane.getID()
        current = lane_weights.get(lane_id, 0.0)
        lane_weights[lane_id] = max(current, 1.0 if signal == "G" else 0.7)
    return float(sum(lane_weights.values()))


def _allocate_greens(
    weights: list[float],
    effective_green: int,
) -> list[int]:
    if not weights:
        return []

    total_weight = sum(weights) or float(len(weights))
    max_weight = max(weights) if weights else 1.0
    min_greens = [12 if weight >= 0.45 * max_weight else 5 for weight in weights]
    remaining = max(effective_green - sum(min_greens), 0)

    raw = [
        min_green + remaining * (weight / total_weight)
        for min_green, weight in zip(min_greens, weights)
    ]
    rounded = [int(value) for value in raw]
    deficit = effective_green - sum(rounded)
    order = sorted(
        range(len(raw)),
        key=lambda index: raw[index] - rounded[index],
        reverse=True,
    )
    while deficit > 0 and order:
        for index in order:
            if deficit <= 0:
                break
            rounded[index] += 1
            deficit -= 1

    while deficit < 0:
        candidates = [
            idx
            for idx in sorted(range(len(rounded)), key=lambda item: rounded[item], reverse=True)
            if rounded[idx] > min_greens[idx]
        ]
        if not candidates:
            break
        for index in candidates:
            if deficit >= 0:
                break
            if rounded[index] > min_greens[index]:
                rounded[index] -= 1
                deficit += 1

    return rounded


def _cap_adjusted_greens(
    base_durations: list[int],
    adjusted_durations: list[int],
    effective_green: int,
    min_green: int = 8,
    max_increase_ratio: float = 0.35,
) -> list[int]:
    if not adjusted_durations:
        return []

    caps = [
        max(min_green, int(round(base * (1.0 + max_increase_ratio))))
        for base in base_durations
    ]
    values = [
        min(max(int(value), min_green), cap)
        for value, cap in zip(adjusted_durations, caps)
    ]

    while sum(values) > effective_green:
        candidates = sorted(
            [
                index
                for index, value in enumerate(values)
                if value > min_green
            ],
            key=lambda index: values[index],
            reverse=True,
        )
        if not candidates:
            break
        for index in candidates:
            if sum(values) <= effective_green:
                break
            values[index] -= 1

    while sum(values) < effective_green:
        candidates = [
            index
            for index, value in sorted(
                enumerate(values),
                key=lambda item: caps[item[0]] - item[1],
                reverse=True,
            )
            if values[index] < caps[index]
        ]
        if not candidates:
            values[values.index(max(values))] += effective_green - sum(values)
            break
        for index in candidates:
            if sum(values) >= effective_green:
                break
            values[index] += 1

    return values


BOTTLENECK_MOVEMENT_TARGETS = {
    ("12257758211", "143009830#0.1693", "s"),
    ("12254671324", "E5", "l"),
    ("12254692358", "158074689#1", "l"),
    ("J55", "-E38", "r"),
    ("12254671324", "E3", "r"),
}


def load_phase_weight_adjustments(
    movement_csv: str | Path | None,
    movement_config_file: str | Path | None,
    max_boost: float = 0.35,
) -> dict[str, dict[int, dict[str, object]]]:
    if not movement_csv or not movement_config_file:
        return {}

    csv_path = Path(movement_csv)
    config_path = Path(movement_config_file)
    if not csv_path.exists() or not config_path.exists():
        return {}

    import pandas as pd

    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty or "movement_id" not in df.columns:
        return {}

    movement_payload = json.loads(config_path.read_text(encoding="utf-8"))
    movement_by_id = {
        str(movement["movement_id"]): movement
        for movement in movement_payload.get("movements", [])
    }
    df["is_green"] = df["signal_state"].fillna("").astype(str).str.contains("G|g", regex=True)
    grouped = (
        df.groupby(["movement_id", "tls_id", "incoming_edge", "turn_type"], dropna=False)
        .agg(
            arrival_flow=("arrival_flow", "sum"),
            discharge_flow=("discharge_flow", "sum"),
            queue_mean=("queue_veh", "mean"),
            queue_max=("queue_veh", "max"),
            green_share=("is_green", "mean"),
            speed_kmh=("speed_kmh", "mean"),
        )
        .reset_index()
    )
    grouped["residual"] = grouped["arrival_flow"] - grouped["discharge_flow"]

    adjustments: dict[str, dict[int, dict[str, object]]] = {}
    for row in grouped.itertuples(index=False):
        tls_id = str(row.tls_id)
        incoming_edge = str(row.incoming_edge)
        turn_type = str(row.turn_type)
        priority = (tls_id, incoming_edge, turn_type) in BOTTLENECK_MOVEMENT_TARGETS
        automatic = (
            float(row.arrival_flow) >= 1000.0
            and float(row.queue_mean) >= 4.0
            and float(row.green_share) <= 0.25
        )
        if not priority and not automatic:
            continue

        movement = movement_by_id.get(str(row.movement_id), {})
        phase_ids = [
            int(phase_id)
            for phase_id in movement.get("green_phase_ids", [])
            if _valid_phase_id(phase_id)
        ]
        if not phase_ids and _valid_phase_id(movement.get("phase_id")):
            phase_ids = [int(movement["phase_id"])]
        if not phase_ids:
            continue

        queue_component = min(0.14, max(0.0, float(row.queue_mean)) / 100.0)
        residual_component = min(0.08, max(0.0, float(row.residual)) / 25000.0)
        green_component = min(0.10, max(0.0, 0.25 - float(row.green_share)) * 0.5)
        priority_component = 0.14 if priority else 0.04
        boost = min(max_boost, priority_component + queue_component + residual_component + green_component)

        for phase_id in phase_ids:
            phase_adjustments = adjustments.setdefault(tls_id, {})
            item = phase_adjustments.setdefault(
                phase_id,
                {"boost": 0.0, "reasons": []},
            )
            item["boost"] = min(max_boost, float(item["boost"]) + boost)
            item["reasons"].append(
                {
                    "movement_id": str(row.movement_id),
                    "incoming_edge": incoming_edge,
                    "turn_type": turn_type,
                    "arrival_flow": round(float(row.arrival_flow), 3),
                    "queue_mean": round(float(row.queue_mean), 3),
                    "queue_max": round(float(row.queue_max), 3),
                    "green_share": round(float(row.green_share), 4),
                    "residual": round(float(row.residual), 3),
                    "priority": priority,
                    "boost": round(boost, 4),
                }
            )
    return adjustments


def _valid_phase_id(value: object) -> bool:
    try:
        return int(value) >= 0
    except (TypeError, ValueError):
        return False


def _build_tls_summaries(
    net_path: Path,
    phase_weight_adjustments: dict[str, dict[int, dict[str, object]]] | None = None,
) -> list[WebsterTlsSummary]:
    net = sumolib.net.readNet(str(net_path), withPrograms=True)
    summaries: list[WebsterTlsSummary] = []
    phase_weight_adjustments = phase_weight_adjustments or {}

    for tls in net.getTrafficLights():
        programs = list(tls.getPrograms().values())
        if not programs:
            continue

        phases = programs[0].getPhases()
        connections = tls.getConnections()
        link_to_lane = {connection[2]: connection[0] for connection in connections}

        incoming_edges: dict[str, object] = {}
        for connection in connections:
            lane = connection[0]
            incoming_edges[lane.getEdge().getID()] = lane.getEdge()

        green_phase_indices = [
            index
            for index, phase in enumerate(phases)
            if _phase_is_green(phase.state)
        ]
        if not green_phase_indices:
            continue

        green_phase_weights = [
            _phase_weight(phases[index].state, link_to_lane)
            for index in green_phase_indices
        ]
        if not any(green_phase_weights):
            green_phase_weights = [1.0 for _ in green_phase_indices]

        loss_time = int(
            round(
                sum(
                    phase.duration
                    for index, phase in enumerate(phases)
                    if index not in green_phase_indices
                )
            )
        )
        approach_count = len(incoming_edges)
        total_incoming_lanes = sum(edge.getLaneNumber() for edge in incoming_edges.values())
        proxy_y = _target_proxy_y(approach_count, total_incoming_lanes)
        cycle_time = int(round((1.5 * loss_time + 5.0) / max(1e-3, (1.0 - proxy_y))))
        lower_bound, upper_bound = _cycle_bounds(approach_count)
        cycle_time = max(lower_bound, min(upper_bound, cycle_time))
        effective_green = max(cycle_time - loss_time, len(green_phase_indices) * 5)
        base_green_phase_durations = _allocate_greens(green_phase_weights, effective_green)
        tls_adjustments = phase_weight_adjustments.get(tls.getID(), {})
        adjusted_weights: list[float] = []
        phase_adjustments: list[dict[str, object]] = []
        for phase_index, weight in zip(green_phase_indices, green_phase_weights):
            adjustment = tls_adjustments.get(phase_index, {})
            boost = float(adjustment.get("boost", 0.0)) if adjustment else 0.0
            adjusted_weights.append(weight * (1.0 + boost))
            if boost > 0:
                phase_adjustments.append(
                    {
                        "phase_index": phase_index,
                        "boost": round(boost, 4),
                        "reasons": adjustment.get("reasons", []),
                    }
                )

        green_phase_durations = _allocate_greens(adjusted_weights, effective_green)
        if phase_adjustments:
            green_phase_durations = _cap_adjusted_greens(
                base_green_phase_durations,
                green_phase_durations,
                effective_green,
            )
        cycle_time = loss_time + sum(green_phase_durations)

        summaries.append(
            WebsterTlsSummary(
                tls_id=tls.getID(),
                approach_count=approach_count,
                incoming_edges=sorted(incoming_edges.keys()),
                total_incoming_lanes=total_incoming_lanes,
                proxy_y=round(proxy_y, 4),
                loss_time=loss_time,
                cycle_time=cycle_time,
                green_phase_indices=green_phase_indices,
                green_phase_weights=[round(value, 3) for value in adjusted_weights],
                green_phase_durations=green_phase_durations,
                base_green_phase_durations=base_green_phase_durations,
                phase_adjustments=phase_adjustments,
            )
        )

    return summaries


def build_webster_signal_net(
    source_net: str | Path,
    output_net: str | Path,
    manual_tls_ids: list[str] | None = None,
    disabled_tls_ids: list[str] | None = None,
    summary_path: str | Path | None = None,
    movement_csv: str | Path | None = None,
    movement_config_file: str | Path | None = None,
) -> tuple[Path, list[WebsterTlsSummary]]:
    source_net_path = Path(source_net)
    output_net_path = Path(output_net)
    output_net_path.parent.mkdir(parents=True, exist_ok=True)

    temp_guess_path = output_net_path.with_name(output_net_path.stem + "_guess.net.xml")
    command = [
        "netconvert",
        "--sumo-net-file",
        str(source_net_path),
        "--tls.default-type",
        "static",
        "--tls.rebuild",
        "--tls.layout",
        "opposites",
        "--tls.yellow.time",
        "3",
        "--tls.allred.time",
        "1",
        "-o",
        str(temp_guess_path),
    ]
    if manual_tls_ids:
        command.extend(["--tls.set", ",".join(manual_tls_ids)])
    else:
        command.append("--tls.guess")
    if disabled_tls_ids:
        command.extend(["--tls.unset", ",".join(disabled_tls_ids)])
    subprocess.run(command, check=True, capture_output=True, text=True)

    phase_adjustments = load_phase_weight_adjustments(movement_csv, movement_config_file)
    summaries = _build_tls_summaries(temp_guess_path, phase_adjustments)
    tree = ET.parse(temp_guess_path)
    root = tree.getroot()

    summary_by_id = {summary.tls_id: summary for summary in summaries}
    for tl_logic in root.findall("tlLogic"):
        summary = summary_by_id.get(tl_logic.attrib.get("id", ""))
        if summary is None:
            continue
        phases = tl_logic.findall("phase")
        for phase_index, duration in zip(summary.green_phase_indices, summary.green_phase_durations):
            phases[phase_index].set("duration", str(int(duration)))

    tree.write(output_net_path, encoding="utf-8", xml_declaration=True)

    if summary_path:
        summary_output = Path(summary_path)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "tls_id": summary.tls_id,
                "approach_count": summary.approach_count,
                "incoming_edges": summary.incoming_edges,
                "total_incoming_lanes": summary.total_incoming_lanes,
                "proxy_y": summary.proxy_y,
                "loss_time": summary.loss_time,
                "cycle_time": summary.cycle_time,
                "green_phase_indices": summary.green_phase_indices,
                "green_phase_weights": summary.green_phase_weights,
                "green_phase_durations": summary.green_phase_durations,
                "base_green_phase_durations": summary.base_green_phase_durations,
                "phase_adjustments": summary.phase_adjustments,
            }
            for summary in summaries
        ]
        summary_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    try:
        temp_guess_path.unlink()
    except (FileNotFoundError, PermissionError):
        pass

    return output_net_path, summaries
