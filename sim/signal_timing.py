from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
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


def _build_tls_summaries(net_path: Path) -> list[WebsterTlsSummary]:
    net = sumolib.net.readNet(str(net_path), withPrograms=True)
    summaries: list[WebsterTlsSummary] = []

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
        green_phase_durations = _allocate_greens(green_phase_weights, effective_green)
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
                green_phase_weights=[round(value, 3) for value in green_phase_weights],
                green_phase_durations=green_phase_durations,
            )
        )

    return summaries


def build_webster_signal_net(
    source_net: str | Path,
    output_net: str | Path,
    manual_tls_ids: list[str] | None = None,
    disabled_tls_ids: list[str] | None = None,
    summary_path: str | Path | None = None,
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

    summaries = _build_tls_summaries(temp_guess_path)
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
