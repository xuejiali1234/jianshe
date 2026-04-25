from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from prediction import EdgeRealtimeCollector, load_prediction_config
from sim import configure_sumo_python_path, resolve_runtime_net_file, write_scaled_route_file

configure_sumo_python_path()

import sumolib
import traci


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_ROUTE = PROJECT_ROOT / "czq_demand.rou.xml"
SCENARIO_DIR = PROJECT_ROOT / "data" / "raw" / "scenarios"
ROUTE_DIR = SCENARIO_DIR / "routes"
MANIFEST_PATH = SCENARIO_DIR / "manifest.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "batch_edge_aggregates.csv"
ARCHIVE_ROOT = PROJECT_ROOT / "data" / "archive"

INCIDENT_WINDOW = (1200, 2100)
INCIDENT_SPEED_FACTOR = 0.25
INCIDENT_TEMPLATES = {
    "mainline_drop": ("472652453#2", "472652453#3"),
    "downstream_drop": ("1324604419#0", "1324604419#1"),
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

    @property
    def run_id(self) -> str:
        scale_tag = f"{self.demand_scale:.2f}".replace(".", "p")
        if self.incident_type:
            return f"S4_incident_{self.incident_type}_scale_{scale_tag}_seed_{self.seed}"
        return f"{self.scenario_id}_scale_{scale_tag}_seed_{self.seed}"

    @property
    def is_incident(self) -> bool:
        return bool(self.incident_type and self.affected_edges)


def default_scenarios() -> list[Scenario]:
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
    incident_scales = [1.10, 1.30, 1.50]
    incident = [
        Scenario(
            scenario_id="S4_incident",
            demand_scale=scale,
            seed=seed,
            incident_type=incident_type,
            incident_start_s=INCIDENT_WINDOW[0],
            incident_end_s=INCIDENT_WINDOW[1],
            affected_edges=tuple(affected_edges),
        )
        for incident_type, affected_edges in INCIDENT_TEMPLATES.items()
        for scale in incident_scales
        for seed in seeds
    ]
    return [*regular, *incident]


def archive_existing_outputs() -> Path | None:
    existing = [path for path in [OUTPUT_CSV, MANIFEST_PATH, ROUTE_DIR] if path.exists()]
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


def write_manifest(rows: list[dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "run_id",
                "scenario_id",
                "seed",
                "demand_scale",
                "base_demand_factor",
                "incident_type",
                "incident_start_s",
                "incident_end_s",
                "affected_edges",
                "route_file",
                "status",
                "snapshots",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def apply_incident_controls(
    scenario: Scenario,
    sim_time_s: float,
    baseline_speeds: dict[str, float],
    incident_applied: bool,
) -> tuple[bool, set[str]]:
    if not scenario.is_incident:
        return False, set()

    incident_active = scenario.incident_start_s <= sim_time_s <= scenario.incident_end_s
    affected = set(scenario.affected_edges)

    if incident_active and not incident_applied:
        for edge_id, base_speed in baseline_speeds.items():
            traci.edge.setMaxSpeed(edge_id, max(0.1, base_speed * INCIDENT_SPEED_FACTOR))
        incident_applied = True
    elif not incident_active and incident_applied:
        for edge_id, base_speed in baseline_speeds.items():
            traci.edge.setMaxSpeed(edge_id, base_speed)
        incident_applied = False

    return incident_applied, affected if incident_active else set()


def run_one_scenario(
    scenario: Scenario,
    config_path: Path,
    sim_end: int,
    base_demand_factor: float,
    net_file: Path,
) -> dict:
    config = load_prediction_config(config_path)
    route_path = ROUTE_DIR / f"{scenario.run_id}.rou.xml"
    effective_scale = base_demand_factor * scenario.demand_scale
    write_scaled_route_file(BASE_ROUTE, route_path, effective_scale)

    collector = EdgeRealtimeCollector(
        config,
        OUTPUT_CSV,
        run_id=scenario.run_id,
        scenario_id=scenario.scenario_id,
        seed=scenario.seed,
        demand_scale=scenario.demand_scale,
        base_demand_factor=base_demand_factor,
        incident_type=scenario.incident_type,
        incident_start_s=scenario.incident_start_s or "",
        incident_end_s=scenario.incident_end_s or "",
        affected_edges=list(scenario.affected_edges),
    )
    sumo_binary = sumolib.checkBinary("sumo")
    cmd = [
        sumo_binary,
        "-n",
        str(net_file),
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

    net_obj = sumolib.net.readNet(str(net_file))
    baseline_speeds = {
        edge_id: float(net_obj.getEdge(edge_id).getSpeed())
        for edge_id in scenario.affected_edges
    }

    snapshots = 0
    incident_applied = False
    try:
        traci.start(cmd)
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0 and step < sim_end:
            traci.simulationStep()
            sim_time_s = float(traci.simulation.getTime())
            incident_applied, incident_edges = apply_incident_controls(
                scenario,
                sim_time_s,
                baseline_speeds,
                incident_applied,
            )
            snapshot = collector.record_step(
                traci,
                step,
                sim_time_s,
                incident_edges=incident_edges,
            )
            if snapshot:
                snapshots += 1
            step += 1

        if incident_applied:
            for edge_id, base_speed in baseline_speeds.items():
                traci.edge.setMaxSpeed(edge_id, base_speed)
        traci.close()
        return {
            "run_id": scenario.run_id,
            "scenario_id": scenario.scenario_id,
            "seed": scenario.seed,
            "demand_scale": scenario.demand_scale,
            "base_demand_factor": base_demand_factor,
            "incident_type": scenario.incident_type,
            "incident_start_s": scenario.incident_start_s or "",
            "incident_end_s": scenario.incident_end_s or "",
            "affected_edges": "|".join(scenario.affected_edges),
            "route_file": str(route_path),
            "status": "ok",
            "snapshots": snapshots,
            "message": "",
        }
    except Exception as exc:
        try:
            if incident_applied:
                for edge_id, base_speed in baseline_speeds.items():
                    traci.edge.setMaxSpeed(edge_id, base_speed)
            traci.close()
        except Exception:
            pass
        return {
            "run_id": scenario.run_id,
            "scenario_id": scenario.scenario_id,
            "seed": scenario.seed,
            "demand_scale": scenario.demand_scale,
            "base_demand_factor": base_demand_factor,
            "incident_type": scenario.incident_type,
            "incident_start_s": scenario.incident_start_s or "",
            "incident_end_s": scenario.incident_end_s or "",
            "affected_edges": "|".join(scenario.affected_edges),
            "route_file": str(route_path),
            "status": "error",
            "snapshots": snapshots,
            "message": str(exc),
        }


def run_batch(
    config_path: Path,
    sim_end: int,
    limit: int | None,
    overwrite: bool,
) -> list[dict]:
    config = load_prediction_config(config_path)
    runtime_net_file = resolve_runtime_net_file(PROJECT_ROOT, config.sumo_net_file)
    if not overwrite and any(path.exists() for path in [OUTPUT_CSV, MANIFEST_PATH, ROUTE_DIR]):
        raise RuntimeError(
            "Existing batch outputs detected. Re-run with --overwrite to archive the current raw outputs "
            "before generating the low-demand incident dataset."
        )
    archive_dir = archive_existing_outputs() if overwrite else None
    if archive_dir:
        print(f"archived previous raw outputs to {archive_dir}")

    scenarios = default_scenarios()
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
        )
        print(f"  status={row['status']} snapshots={row['snapshots']} {row['message']}")
        manifest_rows.append(row)
        write_manifest(manifest_rows)
    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run demand-scale and incident SUMO batch simulations.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "prediction_config.json"))
    parser.add_argument("--sim-end", type=int, default=3600)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N scenarios for testing.")
    parser.add_argument("--overwrite", action="store_true", help="Archive existing raw outputs before running.")
    args = parser.parse_args()

    rows = run_batch(Path(args.config), args.sim_end, args.limit, args.overwrite)
    ok_count = sum(1 for row in rows if row["status"] == "ok")
    print(f"finished {ok_count}/{len(rows)} scenarios")
    print(f"csv: {OUTPUT_CSV}")
    print(f"manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
