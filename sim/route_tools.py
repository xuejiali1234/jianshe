from __future__ import annotations

import re
from pathlib import Path


VEH_PER_HOUR_PATTERN = re.compile(r'vehsPerHour="([0-9.]+)"')


def sanitize_route_text(text: str) -> str:
    lines = text.splitlines()
    closing_indices = [
        index for index, line in enumerate(lines) if line.strip() == "</routes>"
    ]
    if len(closing_indices) <= 1:
        return text if text.endswith("\n") else text + "\n"
    first_closing = closing_indices[0]
    return "\n".join(lines[: first_closing + 1]) + "\n"


def sanitize_route_file(source: str | Path) -> str:
    source_path = Path(source)
    return sanitize_route_text(source_path.read_text(encoding="utf-8-sig"))


def scale_route_demands_text(text: str, scale_factor: float) -> str:
    factor = max(float(scale_factor), 0.0)

    def replace(match: re.Match[str]) -> str:
        original = float(match.group(1))
        scaled = max(1, int(round(original * factor)))
        return f'vehsPerHour="{scaled}"'

    return VEH_PER_HOUR_PATTERN.sub(replace, text)


def write_scaled_route_file(
    source: str | Path,
    destination: str | Path,
    scale_factor: float = 1.0,
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    text = sanitize_route_file(source)
    text = scale_route_demands_text(text, scale_factor)
    destination_path.write_text(text, encoding="utf-8")
    return destination_path


def prepare_runtime_route_file(
    source: str | Path,
    output_dir: str | Path,
    scale_factor: float = 1.0,
    output_name: str = "czq_demand_runtime.rou.xml",
) -> Path:
    return write_scaled_route_file(
        source,
        Path(output_dir) / output_name,
        scale_factor=scale_factor,
    )


def summarize_route_demand(
    source: str | Path,
    scale_factor: float = 1.0,
) -> dict[str, float]:
    text = sanitize_route_file(source)
    values = [float(value) for value in VEH_PER_HOUR_PATTERN.findall(text)]
    base_total = float(sum(values))
    scaled_total = float(sum(max(1, int(round(value * scale_factor))) for value in values))
    return {
        "flow_count": float(len(values)),
        "base_total_vehs_per_hour": base_total,
        "scaled_total_vehs_per_hour": scaled_total,
    }
