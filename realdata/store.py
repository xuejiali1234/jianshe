from __future__ import annotations

from collections import deque
from typing import Any


class RealDataStore:
    def __init__(self, maxlen: int = 120):
        self.latest_snapshot: dict[str, Any] | None = None
        self.history = deque(maxlen=maxlen)

    def update(self, snapshot: dict[str, Any]) -> None:
        self.latest_snapshot = snapshot
        self.history.append(snapshot)

    def latest_payload(self) -> dict[str, Any]:
        if self.latest_snapshot is None:
            return {
                "status": "empty",
                "message": "No real traffic data received yet.",
                "snapshot": None,
                "history_size": 0,
            }
        return {
            "status": "ok",
            "snapshot": self.latest_snapshot,
            "history_size": len(self.history),
        }
