from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        horizon_steps: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.horizon_steps = horizon_steps
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon_steps * output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        out = self.head(h_n[-1])
        return out.view(x.shape[0], self.horizon_steps, self.output_size)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        horizon_steps: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.output_size = output_size
        self.input_proj = nn.Linear(input_size, d_model)
        self.positional = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon_steps * output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        z = self.positional(z)
        z = self.encoder(z)
        pooled = z[:, -1, :]
        out = self.head(pooled)
        return out.view(x.shape[0], self.horizon_steps, self.output_size)


class SpatioTemporalTransformerForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        horizon_steps: int,
        num_entities: int,
        entity_input_size: int | None = None,
        target_size: int = 3,
        time_feature_size: int = 2,
        d_model: int = 96,
        n_heads: int = 4,
        temporal_layers: int = 2,
        spatial_layers: int = 2,
        dropout: float = 0.1,
        graph_adjacency: torch.Tensor | None = None,
        graph_attention_bias: torch.Tensor | None = None,
        phase_id_feature_index: int | None = None,
        max_phase_id: int = 16,
        signal_state_feature_index: int | None = None,
        max_signal_state_id: int = 4,
    ):
        super().__init__()
        if num_entities <= 0:
            raise ValueError("num_entities must be positive")
        self.horizon_steps = horizon_steps
        self.output_size = output_size
        self.num_entities = num_entities
        self.time_feature_size = time_feature_size
        self.target_size = target_size
        self.graph_enabled = graph_adjacency is not None
        self.phase_id_feature_index = phase_id_feature_index
        self.signal_state_feature_index = signal_state_feature_index
        self.max_phase_id = int(max_phase_id)
        self.max_signal_state_id = int(max_signal_state_id)
        entity_body_size = input_size - time_feature_size
        if entity_body_size <= 0 or entity_body_size % num_entities != 0:
            raise ValueError(
                f"input_size={input_size} cannot be reshaped into "
                f"num_entities={num_entities} with time_feature_size={time_feature_size}"
            )
        self.entity_input_size = entity_input_size or entity_body_size // num_entities
        if self.entity_input_size * num_entities != entity_body_size:
            raise ValueError("entity_input_size does not match flattened entity feature size")
        if output_size != num_entities * target_size:
            raise ValueError(
                f"output_size={output_size} must equal num_entities * target_size "
                f"({num_entities} * {target_size})"
            )

        categorical_indices = {
            idx
            for idx in (phase_id_feature_index, signal_state_feature_index)
            if idx is not None and idx >= 0
        }
        for idx in categorical_indices:
            if idx >= self.entity_input_size:
                raise ValueError(f"categorical feature index {idx} is outside entity_input_size={self.entity_input_size}")
        self.categorical_feature_indices = sorted(categorical_indices)
        continuous_entity_input_size = self.entity_input_size - len(self.categorical_feature_indices)
        if continuous_entity_input_size <= 0:
            raise ValueError("V2 requires at least one continuous movement feature")

        self.entity_proj = nn.Linear(continuous_entity_input_size, d_model)
        self.time_proj = nn.Linear(time_feature_size, d_model) if time_feature_size > 0 else None
        self.entity_embedding = nn.Embedding(num_entities, d_model)
        self.phase_embedding = (
            nn.Embedding(self.max_phase_id + 2, d_model)
            if phase_id_feature_index is not None and phase_id_feature_index >= 0
            else None
        )
        self.signal_state_embedding = (
            nn.Embedding(self.max_signal_state_id + 2, d_model)
            if signal_state_feature_index is not None and signal_state_feature_index >= 0
            else None
        )
        self.temporal_positional = PositionalEncoding(d_model)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=spatial_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon_steps * target_size),
        )
        self.entity_output_bias = nn.Embedding(num_entities, horizon_steps * target_size)
        if graph_adjacency is not None:
            if graph_adjacency.shape != (num_entities, num_entities):
                raise ValueError(
                    "graph_adjacency shape must match (num_entities, num_entities), "
                    f"got {tuple(graph_adjacency.shape)} for {num_entities}"
                )
            self.register_buffer("graph_adjacency", graph_adjacency.float())
            self.graph_gate = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
        else:
            self.graph_adjacency = None
            self.graph_gate = None
        if graph_attention_bias is not None:
            if graph_attention_bias.shape != (num_entities, num_entities):
                raise ValueError(
                    "graph_attention_bias shape must match (num_entities, num_entities), "
                    f"got {tuple(graph_attention_bias.shape)} for {num_entities}"
                )
            self.register_buffer("graph_attention_bias", graph_attention_bias.float())
        else:
            self.graph_attention_bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, history_steps, _ = x.shape
        entity_flat = x[:, :, : self.num_entities * self.entity_input_size]
        entity_values = entity_flat.view(
            batch_size,
            history_steps,
            self.num_entities,
            self.entity_input_size,
        )
        continuous_values = self._continuous_entity_values(entity_values)
        z = self.entity_proj(continuous_values)

        if self.time_feature_size > 0 and self.time_proj is not None:
            time_features = x[:, :, -self.time_feature_size :]
            z = z + self.time_proj(time_features).unsqueeze(2)
        if self.phase_embedding is not None and self.phase_id_feature_index is not None:
            phase_ids = self._embedding_indices(
                entity_values[:, :, :, self.phase_id_feature_index],
                self.max_phase_id,
            )
            z = z + self.phase_embedding(phase_ids)
        if self.signal_state_embedding is not None and self.signal_state_feature_index is not None:
            signal_state_ids = self._embedding_indices(
                entity_values[:, :, :, self.signal_state_feature_index],
                self.max_signal_state_id,
            )
            z = z + self.signal_state_embedding(signal_state_ids)

        entity_ids = torch.arange(self.num_entities, device=x.device)
        z = z + self.entity_embedding(entity_ids).view(1, 1, self.num_entities, -1)
        z = z.permute(0, 2, 1, 3).reshape(batch_size * self.num_entities, history_steps, -1)
        z = self.temporal_positional(z)
        z = self.temporal_encoder(z)
        z_last = z[:, -1, :]
        z_mean = z.mean(dim=1)
        z = (0.7 * z_last + 0.3 * z_mean).view(batch_size, self.num_entities, -1)
        if self.graph_adjacency is not None and self.graph_gate is not None:
            z_graph = torch.einsum("ij,bjd->bid", self.graph_adjacency, z)
            z = z + torch.tanh(self.graph_gate) * z_graph
        spatial_mask = None
        if self.graph_attention_bias is not None:
            spatial_mask = self.graph_attention_bias.to(device=z.device, dtype=z.dtype)
        z = self.spatial_encoder(z, mask=spatial_mask)

        out = self.head(z)
        out = out + self.entity_output_bias(entity_ids).view(
            1,
            self.num_entities,
            self.horizon_steps * self.target_size,
        )
        out = out.view(
            batch_size,
            self.num_entities,
            self.horizon_steps,
            self.target_size,
        )
        return out.permute(0, 2, 1, 3).reshape(
            batch_size,
            self.horizon_steps,
            self.output_size,
        )

    def _continuous_entity_values(self, entity_values: torch.Tensor) -> torch.Tensor:
        if not self.categorical_feature_indices:
            return entity_values
        keep_indices = [
            idx
            for idx in range(self.entity_input_size)
            if idx not in self.categorical_feature_indices
        ]
        return entity_values[:, :, :, keep_indices]

    def _embedding_indices(self, raw_values: torch.Tensor, max_value: int) -> torch.Tensor:
        unknown_index = max_value + 1
        ids = torch.round(raw_values).long()
        ids = torch.where(ids < 0, torch.full_like(ids, unknown_index), ids)
        ids = torch.clamp(ids, min=0, max=unknown_index)
        return ids


def build_torch_model(
    kind: str,
    model_config: dict,
    entity_ids: list[str] | None = None,
) -> nn.Module:
    runtime_config = dict(model_config)
    if kind == "lstm":
        return LSTMForecaster(**runtime_config)
    if kind == "transformer_v1":
        return TransformerForecaster(**runtime_config)
    if kind == "transformer_v2":
        graph_enabled = bool(runtime_config.pop("graph_enabled", False))
        graph_path = runtime_config.pop("graph_path", "")
        graph_bias_enabled = bool(runtime_config.pop("graph_bias_enabled", False))
        runtime_config.pop("control_feature_scheme", None)
        runtime_config.pop("v2_experiment_variant", None)
        runtime_config.pop("phase_embedding_enabled", None)
        runtime_config.pop("queue_weight_enabled", None)
        graph_bias_scale = float(runtime_config.pop("graph_bias_scale", 1.0))
        if graph_enabled and graph_path:
            if not entity_ids:
                raise ValueError("entity_ids are required to load transformer_v2 graph_adjacency")
            from .graph_utils import load_movement_adjacency

            runtime_config["graph_adjacency"] = load_movement_adjacency(
                Path(graph_path),
                entity_ids,
            )
        if graph_bias_enabled and graph_path:
            if not entity_ids:
                raise ValueError("entity_ids are required to load transformer_v2 graph_attention_bias")
            from .graph_utils import load_movement_attention_bias

            runtime_config["graph_attention_bias"] = load_movement_attention_bias(
                Path(graph_path),
                entity_ids,
                scale=graph_bias_scale,
            )
        return SpatioTemporalTransformerForecaster(**runtime_config)
    raise ValueError(f"Unsupported torch model kind: {kind}")
