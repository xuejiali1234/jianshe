from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn


MASK_FILL_VALUE = -1e9


class SharedMultiTLSQNetwork(nn.Module):
    def __init__(self, observation_size: int, action_count: int, hidden_sizes: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = int(observation_size)
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_features, int(hidden)))
            layers.append(nn.ReLU())
            in_features = int(hidden)
        layers.append(nn.Linear(in_features, int(action_count)))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class MultiTLSModelArtifact:
    observation_size: int
    action_count: int
    cluster_tls_ids: list[str]
    hidden_sizes: tuple[int, ...]
    model_state_dict: dict[str, Any]
    metadata: dict[str, Any]


def masked_q_values(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    mask = action_mask.to(dtype=torch.bool)
    fill = torch.full_like(q_values, MASK_FILL_VALUE)
    return torch.where(mask, q_values, fill)


def select_actions(
    model: SharedMultiTLSQNetwork,
    observation_matrix: np.ndarray,
    action_masks: dict[str, list[int]],
    tls_ids: list[str],
    epsilon: float,
    device: torch.device,
    rng: random.Random,
) -> dict[str, int]:
    obs_tensor = torch.as_tensor(observation_matrix, dtype=torch.float32, device=device)
    with torch.no_grad():
        q_values = model(obs_tensor).detach().cpu().numpy()
    actions: dict[str, int] = {}
    for index, tls_id in enumerate(tls_ids):
        mask = np.asarray(action_masks.get(tls_id, []), dtype=np.int32)
        valid_actions = [action for action, enabled in enumerate(mask.tolist()) if enabled]
        if not valid_actions:
            actions[tls_id] = 0
            continue
        if rng.random() < float(epsilon):
            actions[tls_id] = int(rng.choice(valid_actions))
            continue
        masked = q_values[index].copy()
        masked[mask <= 0] = MASK_FILL_VALUE
        actions[tls_id] = int(np.argmax(masked))
    return actions


def save_model_artifact(
    path: str | Path,
    model: SharedMultiTLSQNetwork,
    observation_size: int,
    action_count: int,
    cluster_tls_ids: list[str],
    hidden_sizes: tuple[int, ...],
    metadata: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "observation_size": int(observation_size),
        "action_count": int(action_count),
        "cluster_tls_ids": list(cluster_tls_ids),
        "hidden_sizes": list(hidden_sizes),
        "model_state_dict": model.state_dict(),
        "metadata": dict(metadata),
    }
    torch.save(payload, destination)


def load_model_artifact(path: str | Path, device: str | torch.device = "cpu") -> tuple[SharedMultiTLSQNetwork, dict[str, Any]]:
    source = Path(path)
    payload = torch.load(source, map_location=device, weights_only=False)
    model = SharedMultiTLSQNetwork(
        int(payload["observation_size"]),
        int(payload["action_count"]),
        tuple(int(item) for item in payload.get("hidden_sizes", [128, 128])),
    )
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
