from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any


def configure_sumo_python_path() -> None:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        sumo_exe = shutil.which("sumo")
        if sumo_exe:
            sumo_home = str(Path(sumo_exe).resolve().parent.parent)
            os.environ["SUMO_HOME"] = sumo_home

    if sumo_home:
        tools_dir = Path(sumo_home) / "tools"
        if tools_dir.exists():
            tools_path = str(tools_dir)
            if tools_path not in sys.path:
                sys.path.append(tools_path)


def validate_prediction_runtime(
    project_root: Path,
    net_file: Path,
    net_obj: Any,
    observed_edges: list[str],
    sumo_binary: str,
) -> None:
    missing_files = [
        name
        for name in [
            "czq_demand.rou.xml",
            "intersection.sumocfg",
            "vtypes.add.xml",
        ]
        if not (project_root / name).exists()
    ]
    if missing_files:
        raise RuntimeError(f"Missing SUMO project files: {', '.join(missing_files)}")

    if not net_file.exists():
        raise RuntimeError(f"SUMO net file was not found: {net_file}")

    resolved_sumo = Path(sumo_binary) if sumo_binary else None
    if not sumo_binary or (not resolved_sumo.exists() and not shutil.which(sumo_binary)):
        raise RuntimeError("SUMO binary was not found. Check PATH or SUMO_HOME.")

    existing_edges = {edge.getID() for edge in net_obj.getEdges()}
    missing_edges = [edge_id for edge_id in observed_edges if edge_id not in existing_edges]
    if missing_edges:
        raise RuntimeError(
            f"Configured prediction edges are missing from {net_file.name}: "
            + ", ".join(missing_edges)
        )
