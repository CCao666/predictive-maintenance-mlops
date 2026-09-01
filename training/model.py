"""LSTM model used for remaining useful life prediction."""

from pathlib import Path

import torch
from torch import nn


class AttentionPooling(nn.Module):
    """Combine a sequence into one context vector using learned weights."""

    def __init__(self, input_size: int, attention_units: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_size, attention_units)
        self.context_vector = nn.Parameter(torch.empty(attention_units))
        nn.init.xavier_uniform_(self.context_vector.unsqueeze(0))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        scores = torch.tanh(self.projection(sequence)) @ self.context_vector
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(sequence * weights, dim=1)


class ReferenceLSTMRegressor(nn.Module):
    """PyTorch version of the upstream LSTM + attention architecture."""

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
        self.input_size = input_size
        self.lstm_hidden_sizes = lstm_hidden_sizes or [128, 64, 32]
        self.attention_units = attention_units
        self.dense_sizes = dense_sizes or [64, 32, 16]
        self.lstm_dropout = lstm_dropout
        self.dense_dropouts = dense_dropouts or [0.3, 0.2, 0.0]

        if len(self.lstm_hidden_sizes) != 3:
            raise ValueError("Reference model requires three LSTM layers")
        if len(self.dense_sizes) != len(self.dense_dropouts):
            raise ValueError("Each dense layer requires a dropout value")

        self.lstm_layers = nn.ModuleList()
        layer_input_size = input_size
        for hidden_size in self.lstm_hidden_sizes:
            self.lstm_layers.append(
                nn.LSTM(layer_input_size, hidden_size, batch_first=True)
            )
            layer_input_size = hidden_size

        self.lstm_output_dropout = nn.Dropout(lstm_dropout)
        self.attention = AttentionPooling(layer_input_size, attention_units)
        self.dense_head = self._build_dense_head()
        self.output_layer = nn.Linear(self.dense_sizes[-1], 1)

    def _build_dense_head(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        input_size = self.lstm_hidden_sizes[-1]

        for output_size, dropout in zip(self.dense_sizes, self.dense_dropouts):
            layers.extend([nn.Linear(input_size, output_size), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            input_size = output_size

        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(inputs)
        output = inputs
        for layer in self.lstm_layers:
            output, _ = layer(output)
            output = self.lstm_output_dropout(output)

        context = self.attention(output)
        return self.output_layer(self.dense_head(context)).squeeze(-1)

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        if inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape (batch, sequence_length, features)"
            )
        if inputs.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, got {inputs.shape[-1]}"
            )

    def get_config(self) -> dict[str, object]:
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
    """Load a model checkpoint produced by the training pipeline."""
    checkpoint = torch.load(
        artifact_path,
        map_location=map_location,
        weights_only=True,
    )
    model = ReferenceLSTMRegressor(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
