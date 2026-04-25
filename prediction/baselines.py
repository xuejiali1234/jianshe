from __future__ import annotations

from statistics import mean
from typing import Any

from .config import PredictionConfig


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


def _node_value(node: dict[str, Any], target: str) -> float:
    if target == "speed":
        value = node.get("speed")
        if value is None:
            value = node.get("speed_mps")
    else:
        value = node.get(target)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class HistoricalAveragePredictor:
    """MVP baseline: HA when enough history exists, otherwise last value."""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.model_name = "ha_baseline"

    def predict(self, window: list[Any], horizon: int | None = None) -> dict[str, Any]:
        horizon_steps = horizon or self.config.horizon_steps
        ordered_edges = list(self.config.observed_edges)
        history_by_edge = {
            edge_id: {target: [] for target in self.config.targets}
            for edge_id in ordered_edges
        }

        for raw_step in window:
            step = _as_dict(raw_step)
            for raw_node in step.get("nodes", []):
                node = _as_dict(raw_node)
                edge_id = node.get("edge_id")
                if not edge_id:
                    continue
                if edge_id not in history_by_edge:
                    ordered_edges.append(edge_id)
                    history_by_edge[edge_id] = {
                        target: [] for target in self.config.targets
                    }
                for target in self.config.targets:
                    history_by_edge[edge_id][target].append(_node_value(node, target))

        nodes = []
        for edge_id in ordered_edges:
            edge_payload = {"edge_id": edge_id}
            for target in self.config.targets:
                values = history_by_edge[edge_id][target]
                if not values:
                    value = 0.0
                elif len(values) >= self.config.history_steps:
                    value = float(mean(values[-self.config.history_steps :]))
                else:
                    value = float(values[-1])
                edge_payload[f"pred_{target}"] = [value] * horizon_steps
            nodes.append(edge_payload)

        return {
            "model": self.model_name,
            "horizon": list(range(1, horizon_steps + 1)),
            "nodes": nodes,
        }
