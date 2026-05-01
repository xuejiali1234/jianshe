from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


RELATION_WEIGHTS: dict[str, float] = {
    "same_incoming_edge": 1.0,
    "upstream_downstream": 0.8,
    "same_phase": 0.5,
    "same_tls": 0.25,
    "conflict_same_tls_different_phase": 0.0,
}

ATTENTION_BIAS_WEIGHTS: dict[str, float] = {
    "same_incoming_edge": 1.0,
    "upstream_downstream": 0.8,
    "same_phase": 0.5,
    "same_tls": 0.25,
    "conflict_same_tls_different_phase": 0.0,
}


def load_movement_adjacency(
    graph_path: str | Path,
    entity_ids: list[str],
    include_self: bool = True,
) -> torch.Tensor:
    path = Path(graph_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    adjacency_payload = payload.get("adjacency", {})

    missing = [entity_id for entity_id in entity_ids if entity_id not in adjacency_payload]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"movement_graph is missing {len(missing)} entities from dataset ordering: {preview}"
        )

    index_by_entity = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}
    size = len(entity_ids)
    adjacency = np.zeros((size, size), dtype=np.float32)

    if include_self:
        np.fill_diagonal(adjacency, 1.0)

    for src_entity, relations in adjacency_payload.items():
        src_idx = index_by_entity.get(src_entity)
        if src_idx is None:
            continue
        for relation_type, dst_entities in relations.items():
            weight = RELATION_WEIGHTS.get(relation_type, 0.0)
            if weight <= 0.0:
                continue
            for dst_entity in dst_entities:
                dst_idx = index_by_entity.get(dst_entity)
                if dst_idx is None:
                    continue
                adjacency[src_idx, dst_idx] = max(adjacency[src_idx, dst_idx], weight)

    row_sums = adjacency.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-6] = 1.0
    adjacency = adjacency / row_sums
    return torch.tensor(adjacency, dtype=torch.float32)


def load_movement_attention_bias(
    graph_path: str | Path,
    entity_ids: list[str],
    include_self: bool = True,
    scale: float = 1.0,
) -> torch.Tensor:
    path = Path(graph_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    adjacency_payload = payload.get("adjacency", {})

    missing = [entity_id for entity_id in entity_ids if entity_id not in adjacency_payload]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"movement_graph is missing {len(missing)} entities from dataset ordering: {preview}"
        )

    index_by_entity = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}
    size = len(entity_ids)
    bias = np.zeros((size, size), dtype=np.float32)
    if include_self:
        np.fill_diagonal(bias, 0.5)

    for src_entity, relations in adjacency_payload.items():
        src_idx = index_by_entity.get(src_entity)
        if src_idx is None:
            continue
        for relation_type, dst_entities in relations.items():
            weight = ATTENTION_BIAS_WEIGHTS.get(relation_type, 0.0)
            if weight <= 0.0:
                continue
            for dst_entity in dst_entities:
                dst_idx = index_by_entity.get(dst_entity)
                if dst_idx is None:
                    continue
                bias[src_idx, dst_idx] = max(bias[src_idx, dst_idx], weight)

    return torch.tensor(bias * float(scale), dtype=torch.float32)
