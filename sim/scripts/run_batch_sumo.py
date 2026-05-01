from __future__ import annotations

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from prediction import EdgeRealtimeCollector, MovementRealtimeCollector, load_prediction_config
from sim import configure_sumo_python_path, resolve_runtime_net_file, write_scaled_route_file
from sim.movement_tools import build_movement_config

configure_sumo_python_path()

import sumolib
import traci


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_ROUTE = PROJECT_ROOT / "czq_demand.rou.xml"
SCENARIO_DIR = PROJECT_ROOT / "data" / "raw" / "scenarios"
ROUTE_DIR = SCENARIO_DIR / "routes"
NET_DIR = SCENARIO_DIR / "nets"
MANIFEST_PATH = SCENARIO_DIR / "manifest.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "batch_movement_aggregates.csv"
LEGACY_EDGE_OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "batch_edge_aggregates.csv"
ARCHIVE_ROOT = PROJECT_ROOT / "data" / "archive"

EVENT_WINDOW = (1200, 2100)
VSL_SPEED_FACTOR = 0.25
VSL_TEMPLATES = {
    "mainline_drop": ("472652453#2", "472652453#3"),
    "downstream_drop": ("1324604419#0", "1324604419#1"),
}
INCIDENT_CLOSURE_TEMPLATES = {
    "mainline_edge_closure": ("472652453#2", "472652453#3"),
    "downstream_edge_closure": ("1324604419#0", "1324604419#1"),
}
LANE_INCIDENT_TEMPLATES = {
    "north_j9_e13": {
        "affected_edges": ("E13",),
        "affected_lanes": ("E13_1",),
    },
    "north_j9_minus_e13": {
        "affected_edges": ("-E13",),
        "affected_lanes": ("-E13_1",),
    },
    "west_minus_e21_32": {
        "affected_edges": ("-E21.32",),
        "affected_lanes": ("-E21.32_1",),
    },
    "west_e26_73": {
        "affected_edges": ("E26.73",),
        "affected_lanes": ("E26.73_1",),
    },
}
EVENT_VCLASS_BLOCKLIST = ("passenger",)
SIGNAL_VARIANT_GREEN_SCALES = {
    "webster_base": 1.0,
    "short_cycle": 0.80,
    "long_cycle": 1.25,
}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    demand_scale: float
    seed: int
    incident_type: str = ""
    incident_start_s: int = 0
    incident_end_s: int = 0
    affected_edges: tuple[str, ...] = field(default_factory=tuple)
    affected_lanes: tuple[str, ...] = field(default_factory=tuple)
    signal_variant: str = "webster_base"
    event_type: str = ""
    event_policy: str = ""
    speed_factor: float = 1.0

    @property
    def run_id(self) -> str:
        scale_tag = f"{self.demand_scale:.2f}".replace(".", "p")
        if self.scenario_id == "S3_control":
            return f"S3_control_{self.signal_variant}_scale_{scale_tag}_seed_{self.seed}"
        if self.scenario_id == "S4_vsl":
            return f"S4_vsl_{self.incident_type}_scale_{scale_tag}_seed_{self.seed}"
        if self.scenario_id == "S5_incident":
            return f"S5_incident_{self.incident_type}_scale_{scale_tag}_seed_{self.seed}"
        if self.scenario_id == "S5_lane_incident":
            return f"S5_lane_incident_{self.incident_type}_scale_{scale_tag}_seed_{self.seed}"
        return f"{self.scenario_id}_scale_{scale_tag}_seed_{self.seed}"

    @property
    def is_event(self) -> bool:
        return bool(self.event_type and self.affected_edges)


def default_scenarios(preset: str = "full") -> list[Scenario]:
    groups = [
        ("S1_normal", [0.60, 0.80, 1.00]),
        ("S2_peak", [1.10, 1.20, 1.30]),
        ("S3_congested", [1.40, 1.50]),
    ]
    seeds = [11, 22, 33]
    regular = [
        Scenario(scenario_id, scale, seed)
        for scenario_id, scales in groups
        for scale in scales
        for seed in seeds
    ]
    event_scales = [1.10, 1.30, 1.50]
    control = [
        Scenario(
            scenario_id="S3_control",
            demand_scale=scale,
            seed=seed,
            signal_variant=signal_variant,
        )
        for signal_variant in ("short_cycle", "long_cycle")
        for scale in (1.10, 1.30)
        for seed in seeds
    ]
    vsl = [
        Scenario(
            scenario_id="S4_vsl",
            demand_scale=scale,
            seed=seed,
            incident_type=incident_type,
            incident_start_s=EVENT_WINDOW[0],
            incident_end_s=EVENT_WINDOW[1],
            affected_edges=tuple(affected_edges),
            event_type="vsl_speed_drop",
            event_policy="variable_speed_limit",
            speed_factor=VSL_SPEED_FACTOR,
        )
        for incident_type, affected_edges in VSL_TEMPLATES.items()
        for scale in event_scales
        for seed in seeds
    ]
    closures = [
        Scenario(
            scenario_id="S5_incident",
            demand_scale=scale,
            seed=seed,
            incident_type=incident_type,
            incident_start_s=EVENT_WINDOW[0],
            incident_end_s=EVENT_WINDOW[1],
            affected_edges=tuple(affected_edges),
            event_type="incident_closure",
            event_policy="edge_passenger_closure",
            speed_factor=0.0,
        )
        for incident_type, affected_edges in INCIDENT_CLOSURE_TEMPLATES.items()
        for scale in event_scales
        for seed in seeds
    ]
    lane_incidents = [
        Scenario(
            scenario_id="S5_lane_incident",
            demand_scale=scale,
            seed=seed,
            incident_type=incident_type,
            incident_start_s=EVENT_WINDOW[0],
            incident_end_s=EVENT_WINDOW[1],
            affected_edges=tuple(template["affected_edges"]),
            affected_lanes=tuple(template["affected_lanes"]),
            event_type="incident_lane_closure",
            event_policy="single_lane_passenger_closure",
            speed_factor=0.0,
        )
        for incident_type, template in LANE_INCIDENT_TEMPLATES.items()
        for scale in (1.30, 1.50)
        for seed in seeds
    ]
    scenarios = [*regular, *vsl, *closures, *control]
    if preset == "full":
        return scenarios
    if preset == "fast_v2":
        return [
            scenario
            for scenario in scenarios
            if scenario.scenario_id != "S4_vsl"
            or abs(scenario.demand_scale - 1.30) < 1e-6
        ]
    if preset == "lane_incident_v1":
        return lane_incidents
    raise ValueError(f"Unknown scenario preset: {preset}")


def output_csv_for_collector(collector_mode: str, output_csv: Path | None = None) -> Path:
    if output_csv is not None:
        return output_csv
    return OUTPUT_CSV if collector_mode == "movement" else LEGACY_EDGE_OUTPUT_CSV


def paths_for_scenario_dir(scenario_dir: Path | None = None) -> tuple[Path, Path, Path]:
    scenario_dir = scenario_dir or SCENARIO_DIR
    return scenario_dir / "manifest.csv", scenario_dir / "routes", scenario_dir / "nets"


def archive_existing_outputs(
    collector_mode: str,
    output_csv: Path | None = None,
    manifest_path: Path | None = None,
    route_dir: Path | None = None,
    net_dir: Path | None = None,
) -> Path | None:
    manifest_path = manifest_path or MANIFEST_PATH
    route_dir = route_dir or ROUTE_DIR
    net_dir = net_dir or NET_DIR
    existing = [
        path
        for path in [output_csv_for_collector(collector_mode, output_csv), manifest_path, route_dir, net_dir]
        if path.exists()
    ]
    if not existing:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = ARCHIVE_ROOT / f"raw_outputs_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for path in existing:
        target = archive_dir / path.name
        if path.is_dir():
            shutil.move(str(path), str(target))
        else:
            shutil.move(str(path), str(target))
    return archive_dir


def write_manifest(rows: list[dict], manifest_path: Path | None = None) -> None:
    manifest_path = manifest_path or MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "run_id",
                "scenario_id",
                "seed",
                "demand_scale",
                "base_demand_factor",
                "signal_variant",
                "event_type",
                "event_policy",
                "speed_factor",
                "incident_type",
                "incident_start_s",
                "incident_end_s",
                "affected_edges",
                "affected_lanes",
                "lane_closure_count",
                "route_file",
                "net_file",
                "status",
                "snapshots",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def prepare_signal_variant_net(source_net: Path, signal_variant: str, net_dir: Path | None = None) -> Path:
    signal_variant = signal_variant or "webster_base"
    if signal_variant == "webster_base":
        return source_net
    if signal_variant not in SIGNAL_VARIANT_GREEN_SCALES:
        raise ValueError(f"Unknown signal_variant: {signal_variant}")

    net_dir = net_dir or NET_DIR
    net_dir.mkdir(parents=True, exist_ok=True)
    output_path = net_dir / f"{source_net.stem}_{signal_variant}.net.xml"
    if output_path.exists():
        return output_path

    green_scale = SIGNAL_VARIANT_GREEN_SCALES[signal_variant]
    tree = ET.parse(source_net)
    root = tree.getroot()
    for phase in root.findall(".//tlLogic/phase"):
        state = phase.attrib.get("state", "")
        if not any(char in "Gg" for char in state):
            continue
        try:
            duration = float(phase.attrib.get("duration", "0"))
        except ValueError:
            continue
        phase.set("duration", str(max(5, int(round(duration * green_scale)))))
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def collect_lane_disallowed_baseline(
    scenario: Scenario,
    net_obj: sumolib.net.Net,
) -> dict[str, tuple[str, ...]]:
    if scenario.event_type not in {"incident_closure", "incident_lane_closure"}:
        return {}
    lane_disallowed: dict[str, tuple[str, ...]] = {}
    lane_ids: list[str] = []
    if scenario.event_type == "incident_lane_closure":
        lane_ids.extend(scenario.affected_lanes)
    else:
        for edge_id in scenario.affected_edges:
            try:
                edge = net_obj.getEdge(edge_id)
            except Exception:
                continue
            lane_ids.extend(lane.getID() for lane in edge.getLanes())
    for lane_id in lane_ids:
        if lane_id in lane_disallowed:
            continue
        try:
            net_obj.getLane(lane_id)
        except Exception:
            continue
        try:
            lane_disallowed[lane_id] = tuple(traci.lane.getDisallowed(lane_id))
        except Exception:
            lane_disallowed[lane_id] = tuple()
    return lane_disallowed


def apply_event_controls(
    scenario: Scenario,
    sim_time_s: float,
    baseline_speeds: dict[str, float],
    baseline_lane_disallowed: dict[str, tuple[str, ...]],
    event_applied: bool,
) -> tuple[bool, set[str]]:
    if not scenario.is_event:
        return False, set()

    event_active = scenario.incident_start_s <= sim_time_s <= scenario.incident_end_s
    affected = set(scenario.affected_edges)

    if event_active and not event_applied:
        if scenario.event_type == "vsl_speed_drop":
            speed_factor = scenario.speed_factor if scenario.speed_factor > 0 else VSL_SPEED_FACTOR
            for edge_id, base_speed in baseline_speeds.items():
                traci.edge.setMaxSpeed(edge_id, max(0.1, base_speed * speed_factor))
        elif scenario.event_type in {"incident_closure", "incident_lane_closure"}:
            for lane_id, disallowed in baseline_lane_disallowed.items():
                merged = sorted(set(disallowed) | set(EVENT_VCLASS_BLOCKLIST))
                traci.lane.setDisallowed(lane_id, merged)
        event_applied = True
    elif not event_active and event_applied:
        restore_event_controls(scenario, baseline_speeds, baseline_lane_disallowed)
        event_applied = False

    return event_applied, affected if event_active else set()


def restore_event_controls(
    scenario: Scenario,
    baseline_speeds: dict[str, float],
    baseline_lane_disallowed: dict[str, tuple[str, ...]],
) -> None:
    if scenario.event_type == "vsl_speed_drop":
        for edge_id, base_speed in baseline_speeds.items():
            traci.edge.setMaxSpeed(edge_id, base_speed)
    elif scenario.event_type in {"incident_closure", "incident_lane_closure"}:
        for lane_id, disallowed in baseline_lane_disallowed.items():
            traci.lane.setDisallowed(lane_id, list(disallowed))


def run_one_scenario(
    scenario: Scenario,
    config_path: Path,
    sim_end: int,
    base_demand_factor: float,
    net_file: Path,
    collector_mode: str,
    movement_config_path: Path | None = None,
    output_csv: Path | None = None,
    route_dir: Path | None = None,
    net_dir: Path | None = None,
) -> dict:
    config = load_prediction_config(config_path)
    route_dir = route_dir or ROUTE_DIR
    route_dir.mkdir(parents=True, exist_ok=True)
    route_path = route_dir / f"{scenario.run_id}.rou.xml"
    scenario_net_file = prepare_signal_variant_net(net_file, scenario.signal_variant, net_dir)
    effective_scale = base_demand_factor * scenario.demand_scale
    write_scaled_route_file(BASE_ROUTE, route_path, effective_scale)

    collector_kwargs = {
        "run_id": scenario.run_id,
        "scenario_id": scenario.scenario_id,
        "seed": scenario.seed,
        "demand_scale": scenario.demand_scale,
        "base_demand_factor": base_demand_factor,
        "signal_variant": scenario.signal_variant,
        "event_type": scenario.event_type,
        "event_policy": scenario.event_policy,
        "speed_factor": scenario.speed_factor,
        "incident_type": scenario.incident_type,
        "incident_start_s": scenario.incident_start_s or "",
        "incident_end_s": scenario.incident_end_s or "",
        "affected_edges": list(scenario.affected_edges),
    }
    if collector_mode == "movement":
        collector = MovementRealtimeCollector(
            config,
            output_csv_for_collector(collector_mode, output_csv),
            **collector_kwargs,
            project_root=PROJECT_ROOT,
            net_file=scenario_net_file,
            movement_config_path=movement_config_path,
        )
    elif collector_mode == "edge":
        collector = EdgeRealtimeCollector(
            config,
            output_csv_for_collector(collector_mode, output_csv),
            **collector_kwargs,
        )
    else:
        raise ValueError(f"Unknown collector mode: {collector_mode}")
    sumo_binary = sumolib.checkBinary("sumo")
    cmd = [
        sumo_binary,
        "-n",
        str(scenario_net_file),
        "-r",
        str(route_path),
        "--begin",
        "0",
        "--end",
        str(sim_end),
        "--seed",
        str(scenario.seed),
        "--no-warnings",
        "--ignore-route-errors",
        "--time-to-teleport",
        "15",
        "--ignore-junction-blocker",
        "5",
        "--time-to-impatience",
        "10",
        "--default.action-step-length",
        "0.5",
        "--device.rerouting.probability",
        "0.8",
        "--device.rerouting.period",
        "30",
        "--device.rerouting.adaptation-interval",
        "10",
    ]

    net_obj = sumolib.net.readNet(str(scenario_net_file))
    baseline_speeds = {
        edge_id: float(net_obj.getEdge(edge_id).getSpeed())
        for edge_id in scenario.affected_edges
    }

    snapshots = 0
    event_applied = False
    try:
        traci.start(cmd)
        baseline_lane_disallowed = collect_lane_disallowed_baseline(scenario, net_obj)
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0 and step < sim_end:
            traci.simulationStep()
            sim_time_s = float(traci.simulation.getTime())
            event_applied, event_edges = apply_event_controls(
                scenario,
                sim_time_s,
                baseline_speeds,
                baseline_lane_disallowed,
                event_applied,
            )
            snapshot = collector.record_step(
                traci,
                step,
                sim_time_s,
                incident_edges=event_edges,
            )
            if snapshot:
                snapshots += 1
            step += 1

        if event_applied:
            restore_event_controls(scenario, baseline_speeds, baseline_lane_disallowed)
        traci.close()
        return {
            "run_id": scenario.run_id,
            "scenario_id": scenario.scenario_id,
            "seed": scenario.seed,
            "demand_scale": scenario.demand_scale,
            "base_demand_factor": base_demand_factor,
            "signal_variant": scenario.signal_variant,
            "event_type": scenario.event_type,
            "event_policy": scenario.event_policy,
            "speed_factor": scenario.speed_factor,
            "incident_type": scenario.incident_type,
            "incident_start_s": scenario.incident_start_s or "",
            "incident_end_s": scenario.incident_end_s or "",
            "affected_edges": "|".join(scenario.affected_edges),
            "affected_lanes": "|".join(scenario.affected_lanes),
            "lane_closure_count": len(scenario.affected_lanes),
            "route_file": str(route_path),
            "net_file": str(scenario_net_file),
            "status": "ok",
            "snapshots": snapshots,
            "message": "",
        }
    except Exception as exc:
        try:
            if event_applied:
                restore_event_controls(scenario, baseline_speeds, locals().get("baseline_lane_disallowed", {}))
            traci.close()
        except Exception:
            pass
        return {
            "run_id": scenario.run_id,
            "scenario_id": scenario.scenario_id,
            "seed": scenario.seed,
            "demand_scale": scenario.demand_scale,
            "base_demand_factor": base_demand_factor,
            "signal_variant": scenario.signal_variant,
            "event_type": scenario.event_type,
            "event_policy": scenario.event_policy,
            "speed_factor": scenario.speed_factor,
            "incident_type": scenario.incident_type,
            "incident_start_s": scenario.incident_start_s or "",
            "incident_end_s": scenario.incident_end_s or "",
            "affected_edges": "|".join(scenario.affected_edges),
            "affected_lanes": "|".join(scenario.affected_lanes),
            "lane_closure_count": len(scenario.affected_lanes),
            "route_file": str(route_path),
            "net_file": str(scenario_net_file),
            "status": "error",
            "snapshots": snapshots,
            "message": str(exc),
        }


def run_batch(
    config_path: Path,
    sim_end: int,
    limit: int | None,
    overwrite: bool,
    collector_mode: str = "movement",
    movement_config_path: Path | None = None,
    output_csv: Path | None = None,
    scenario_dir: Path | None = None,
    scenario_filter: str = "",
    scenario_preset: str = "full",
) -> list[dict]:
    config = load_prediction_config(config_path)
    runtime_net_file = resolve_runtime_net_file(PROJECT_ROOT, config.sumo_net_file)
    collector_mode = collector_mode.lower().strip()
    if collector_mode not in {"movement", "edge"}:
        raise ValueError("--collector must be either 'movement' or 'edge'")
    movement_config_path = movement_config_path or PROJECT_ROOT / getattr(
        config,
        "movement_config_file",
        "configs/movement_config.json",
    )
    if not movement_config_path.is_absolute():
        movement_config_path = PROJECT_ROOT / movement_config_path
    if collector_mode == "movement":
        build_movement_config(
            runtime_net_file,
            config.observed_edges,
            movement_config_path,
            PROJECT_ROOT / "data" / "processed" / "movement_map.csv",
        )

    selected_output_csv = output_csv_for_collector(collector_mode, output_csv)
    manifest_path, route_dir, net_dir = paths_for_scenario_dir(scenario_dir)
    if not overwrite and any(path.exists() for path in [selected_output_csv, manifest_path, route_dir]):
        raise RuntimeError(
            "Existing batch outputs detected. Re-run with --overwrite to archive the current raw outputs "
            "before generating the low-demand event dataset."
        )
    archive_dir = (
        archive_existing_outputs(collector_mode, output_csv, manifest_path, route_dir, net_dir)
        if overwrite
        else None
    )
    if archive_dir:
        print(f"archived previous raw outputs to {archive_dir}")

    scenarios = default_scenarios(scenario_preset)
    if scenario_filter:
        needle = scenario_filter.lower()
        scenarios = [
            scenario for scenario in scenarios
            if needle in scenario.run_id.lower()
            or needle in scenario.scenario_id.lower()
            or needle in scenario.event_type.lower()
            or needle in scenario.incident_type.lower()
        ]
    if limit is not None:
        scenarios = scenarios[:limit]

    manifest_rows = []
    total = len(scenarios)
    for index, scenario in enumerate(scenarios, start=1):
        print(f"[{index}/{total}] {scenario.run_id}")
        row = run_one_scenario(
            scenario,
            config_path,
            sim_end,
            config.base_demand_factor,
            runtime_net_file,
            collector_mode,
            movement_config_path,
            selected_output_csv,
            route_dir,
            net_dir,
        )
        print(f"  status={row['status']} snapshots={row['snapshots']} {row['message']}")
        manifest_rows.append(row)
        write_manifest(manifest_rows, manifest_path)
    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run demand-scale, VSL, closure, and control SUMO batch simulations.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "prediction_config.json"))
    parser.add_argument("--sim-end", type=int, default=3600)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N scenarios for testing.")
    parser.add_argument("--overwrite", action="store_true", help="Archive existing raw outputs before running.")
    parser.add_argument("--collector", choices=["movement", "edge"], default="movement")
    parser.add_argument("--movement-config", default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--scenario-dir", type=Path, default=None)
    parser.add_argument(
        "--scenario-preset",
        choices=["full", "fast_v2", "lane_incident_v1"],
        default="full",
        help="Use full 72-run suite, fast_v2 reduced suite, or lane_incident_v1 24-run incremental lane-closure suite.",
    )
    parser.add_argument(
        "--scenario-filter",
        default="",
        help="Run only scenarios whose run_id/scenario_id/event_type/incident_type contains this text.",
    )
    args = parser.parse_args()

    rows = run_batch(
        Path(args.config),
        args.sim_end,
        args.limit,
        args.overwrite,
        args.collector,
        Path(args.movement_config) if args.movement_config else None,
        args.output_csv,
        args.scenario_dir,
        args.scenario_filter,
        args.scenario_preset,
    )
    ok_count = sum(1 for row in rows if row["status"] == "ok")
    print(f"finished {ok_count}/{len(rows)} scenarios")
    print(f"collector: {args.collector}")
    print(f"scenario preset: {args.scenario_preset}")
    print(f"csv: {output_csv_for_collector(args.collector, args.output_csv)}")
    print(f"manifest: {paths_for_scenario_dir(args.scenario_dir)[0]}")


if __name__ == "__main__":
    main()
