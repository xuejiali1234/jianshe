from __future__ import annotations

from pathlib import Path


DEFAULT_BASE_NET = Path("czq.net.xml")
DEFAULT_SIGNALIZED_NET = Path("data/processed/czq_tls_webster.net.xml")


def resolve_runtime_net_file(
    project_root: Path,
    preferred_path: str | Path | None = None,
) -> Path:
    candidates: list[Path] = []
    if preferred_path:
        preferred = Path(preferred_path)
        candidates.append(preferred if preferred.is_absolute() else project_root / preferred)
    candidates.append(project_root / DEFAULT_SIGNALIZED_NET)
    candidates.append(project_root / DEFAULT_BASE_NET)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return project_root / DEFAULT_BASE_NET
