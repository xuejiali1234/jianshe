from __future__ import annotations

import math

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
    ):
        super().__init__()
        if num_entities <= 0:
            raise ValueError("num_entities must be positive")
        self.horizon_steps = horizon_steps
        self.output_size = output_size
        self.num_entities = num_entities
        self.time_feature_size = time_feature_size
        self.target_size = target_size
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

        self.entity_proj = nn.Linear(self.entity_input_size, d_model)
        self.time_proj = nn.Linear(time_feature_size, d_model) if time_feature_size > 0 else None
        self.entity_embedding = nn.Embedding(num_entities, d_model)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, history_steps, _ = x.shape
        entity_flat = x[:, :, : self.num_entities * self.entity_input_size]
        entity_values = entity_flat.view(
            batch_size,
            history_steps,
            self.num_entities,
            self.entity_input_size,
        )
        z = self.entity_proj(entity_values)

        if self.time_feature_size > 0 and self.time_proj is not None:
            time_features = x[:, :, -self.time_feature_size :]
            z = z + self.time_proj(time_features).unsqueeze(2)

        entity_ids = torch.arange(self.num_entities, device=x.device)
        z = z + self.entity_embedding(entity_ids).view(1, 1, self.num_entities, -1)
        z = z.permute(0, 2, 1, 3).reshape(batch_size * self.num_entities, history_steps, -1)
        z = self.temporal_positional(z)
        z = self.temporal_encoder(z)
        z = z[:, -1, :].view(batch_size, self.num_entities, -1)
        z = self.spatial_encoder(z)

        out = self.head(z).view(
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


def build_torch_model(kind: str, model_config: dict) -> nn.Module:
    if kind == "lstm":
        return LSTMForecaster(**model_config)
    if kind == "transformer_v1":
        return TransformerForecaster(**model_config)
    if kind == "transformer_v2":
        return SpatioTemporalTransformerForecaster(**model_config)
    raise ValueError(f"Unsupported torch model kind: {kind}")
