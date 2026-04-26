from __future__ import annotations

from statistics import mean
from pathlib import Path
from typing import Any

from .config import PredictionConfig
from sim.movement_tools import load_movement_config


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


def _node_value(node: dict[str, Any], target: str) -> float:
    if target == "speed":
        value = node.get("speed")
        if value is None:
            value = node.get("speed_mps")
    elif target == "mean_speed":
        value = node.get("mean_speed")
        if value is None:
            value = node.get("mean_speed_mps", node.get("speed_mps"))
    elif target == "queue":
        value = node.get("queue")
        if value is None:
            value = node.get("queue_veh")
    elif target == "flow":
        value = node.get("flow")
        if value is None:
            value = node.get("arrival_flow")
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
        observation_level = getattr(self.config, "observation_level", "edge")
        ordered_entities, entity_metadata = self._ordered_entities(observation_level)
        history_by_entity = {
            entity_id: {target: [] for target in self.config.targets}
            for entity_id in ordered_entities
        }

        for raw_step in window:
            step = _as_dict(raw_step)
            raw_items = step.get("movements", []) if observation_level == "movement" else step.get("nodes", [])
            for raw_item in raw_items:
                item = _as_dict(raw_item)
                entity_id = item.get("movement_id") if observation_level == "movement" else item.get("edge_id")
                if not entity_id:
                    continue
                if entity_id not in history_by_entity:
                    ordered_entities.append(entity_id)
                    history_by_entity[entity_id] = {
                        target: [] for target in self.config.targets
                    }
                for target in self.config.targets:
                    history_by_entity[entity_id][target].append(_node_value(item, target))

        entity_payloads = []
        for entity_id in ordered_entities:
            entity_payload: dict[str, Any] = (
                {"movement_id": entity_id, **entity_metadata.get(entity_id, {})}
                if observation_level == "movement"
                else {"edge_id": entity_id}
            )
            for target in self.config.targets:
                values = history_by_entity[entity_id][target]
                if not values:
                    value = 0.0
                elif len(values) >= self.config.history_steps:
                    value = float(mean(values[-self.config.history_steps :]))
                else:
                    value = float(values[-1])
                entity_payload[f"pred_{target}"] = [value] * horizon_steps
            entity_payloads.append(entity_payload)

        payload = {
            "model": self.model_name,
            "horizon": list(range(1, horizon_steps + 1)),
        }
        if observation_level == "movement":
            payload["movements"] = entity_payloads
            payload["nodes"] = self._legacy_nodes_from_movement_predictions(entity_payloads, horizon_steps)
            payload["observation_level"] = "movement"
        else:
            payload["nodes"] = entity_payloads
        return payload

    def _ordered_entities(self, observation_level: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
        if observation_level != "movement":
            return list(self.config.observed_edges), {}

        movement_path = Path(getattr(self.config, "movement_config_file", "configs/movement_config.json"))
        if not movement_path.exists():
            return [], {}
        try:
            payload = load_movement_config(movement_path)
        except Exception:
            return [], {}
        metadata = {
            str(movement["movement_id"]): {
                "incoming_edge": movement.get("incoming_edge", ""),
                "outgoing_edge": movement.get("outgoing_edge", ""),
                "turn_type": movement.get("turn_type", ""),
                "tls_id": movement.get("tls_id", ""),
            }
            for movement in payload.get("movements", [])
        }
        return list(metadata.keys()), metadata

    def _legacy_nodes_from_movement_predictions(
        self,
        movements: list[dict[str, Any]],
        horizon_steps: int,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for movement in movements:
            edge_id = movement.get("incoming_edge") or movement.get("movement_id")
            item = grouped.setdefault(
                edge_id,
                {
                    "edge_id": edge_id,
                    "pred_flow": [0.0] * horizon_steps,
                    "_speed_values": [],
                    "pred_queue": [0.0] * horizon_steps,
                },
            )
            flow_values = movement.get("pred_arrival_flow") or movement.get("pred_flow") or []
            speed_values = movement.get("pred_mean_speed") or movement.get("pred_speed") or []
            queue_values = movement.get("pred_queue_veh") or movement.get("pred_queue") or []
            for idx, value in enumerate(flow_values[:horizon_steps]):
                item["pred_flow"][idx] += float(value)
            if speed_values:
                item["_speed_values"].append([float(value) for value in speed_values[:horizon_steps]])
            for idx, value in enumerate(queue_values[:horizon_steps]):
                item["pred_queue"][idx] += float(value)

        nodes = []
        for item in grouped.values():
            if item["_speed_values"]:
                item["pred_speed"] = [
                    float(mean(values))
                    for values in zip(*item["_speed_values"])
                ]
            else:
                item["pred_speed"] = [0.0] * horizon_steps
            item.pop("_speed_values", None)
            nodes.append(item)
        return nodes
