"""PyTorch LSTM models for remaining useful life prediction."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


REFERENCE_MODEL_CONFIG = {
    "input_size": 24,
    "lstm_hidden_sizes": [128, 64, 32],
    "attention_units": 64,
    "dense_sizes": [64, 32, 16],
    "lstm_dropout": 0.2,
    "dense_dropouts": [0.3, 0.2, 0.0],
}


class LSTMRegressor(nn.Module):
    """Small LSTM used for fast local smoke tests."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one RUL prediction for each sequence."""
        _validate_inputs(inputs, self.input_size)
        sequence_output, _ = self.lstm(inputs)
        return self.output_layer(sequence_output[:, -1, :]).squeeze(-1)


class AttentionPooling(nn.Module):
    """Learn a weighted combination of all LSTM time steps."""

    def __init__(self, input_size: int, attention_units: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_size, attention_units)
        self.context_vector = nn.Parameter(torch.empty(attention_units))
        nn.init.xavier_uniform_(self.context_vector.unsqueeze(0))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Pool a (batch, time, features) tensor into (batch, features)."""
        scores = torch.tanh(self.projection(sequence))
        scores = torch.matmul(scores, self.context_vector)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(sequence * weights, dim=1)


class ReferenceLSTMRegressor(nn.Module):
    """PyTorch equivalent of the upstream reference LSTM architecture."""

    def __init__(
        self,
        input_size: int = 24,
        lstm_hidden_sizes: list[int] | None = None,
        attention_units: int = 64,
        dense_sizes: list[int] | None = None,
        lstm_dropout: float = 0.2,
        dense_dropouts: list[float] | None = None,
    ) -> None:
        super().__init__()
        lstm_hidden_sizes = lstm_hidden_sizes or [128, 64, 32]
        dense_sizes = dense_sizes or [64, 32, 16]
        dense_dropouts = dense_dropouts or [0.3, 0.2, 0.0]

        if len(lstm_hidden_sizes) != 3:
            raise ValueError("Reference model requires three LSTM layers")
        if len(dense_sizes) != len(dense_dropouts):
            raise ValueError("Each dense layer requires a dropout value")

        self.input_size = input_size
        self.lstm_hidden_sizes = list(lstm_hidden_sizes)
        self.attention_units = attention_units
        self.dense_sizes = list(dense_sizes)
        self.lstm_dropout = lstm_dropout
        self.dense_dropouts = list(dense_dropouts)

        lstm_layers = []
        previous_size = input_size
        for hidden_size in self.lstm_hidden_sizes:
            lstm_layers.append(
                nn.LSTM(
                    input_size=previous_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                )
            )
            previous_size = hidden_size
        self.lstm_layers = nn.ModuleList(lstm_layers)
        self.lstm_output_dropout = nn.Dropout(lstm_dropout)
        self.attention = AttentionPooling(previous_size, attention_units)

        dense_modules = []
        previous_size = self.lstm_hidden_sizes[-1]
        for dense_size, dropout in zip(self.dense_sizes, self.dense_dropouts):
            dense_modules.extend([nn.Linear(previous_size, dense_size), nn.ReLU()])
            if dropout > 0:
                dense_modules.append(nn.Dropout(dropout))
            previous_size = dense_size
        self.dense_head = nn.Sequential(*dense_modules)
        self.output_layer = nn.Linear(previous_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict RUL using stacked LSTMs, attention, and dense layers."""
        _validate_inputs(inputs, self.input_size)
        output = inputs
        for lstm_layer in self.lstm_layers:
            output, _ = lstm_layer(output)
            output = self.lstm_output_dropout(output)
        context = self.attention(output)
        features = self.dense_head(context)
        return self.output_layer(features).squeeze(-1)

    def get_config(self) -> dict[str, object]:
        """Return constructor values required to rebuild this model."""
        return {
            "input_size": self.input_size,
            "lstm_hidden_sizes": self.lstm_hidden_sizes,
            "attention_units": self.attention_units,
            "dense_sizes": self.dense_sizes,
            "lstm_dropout": self.lstm_dropout,
            "dense_dropouts": self.dense_dropouts,
        }


def load_reference_model(
    artifact_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> ReferenceLSTMRegressor:
    """Rebuild a reference model and load its saved weights."""
    checkpoint = torch.load(
        artifact_path,
        map_location=map_location,
        weights_only=True,
    )
    model = ReferenceLSTMRegressor(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _validate_inputs(inputs: torch.Tensor, input_size: int) -> None:
    if inputs.ndim != 3:
        raise ValueError(
            "inputs must have shape (batch, sequence_length, features)"
        )
    if inputs.shape[-1] != input_size:
        raise ValueError(
            f"Expected {input_size} input features, "
            f"but received {inputs.shape[-1]}"
        )
