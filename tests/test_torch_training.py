"""Lightweight tests for the PyTorch dataset and LSTM model."""

import numpy as np
import pytest
import torch

from training.model import (
    ReferenceLSTMRegressor,
    load_reference_model,
)
from training.torch_dataset import RULSequenceDataset


def test_rul_sequence_dataset_returns_expected_shapes():
    sequences = np.zeros((5, 30, 24), dtype=np.float32)
    targets = np.arange(5, dtype=np.float32)
    dataset = RULSequenceDataset(sequences, targets)

    sequence, target = dataset[0]

    assert len(dataset) == 5
    assert sequence.shape == (30, 24)
    assert target.shape == ()
    assert sequence.dtype == torch.float32
    assert target.dtype == torch.float32


def test_dataset_rejects_mismatched_sample_counts():
    sequences = np.zeros((5, 30, 24), dtype=np.float32)
    targets = np.zeros(4, dtype=np.float32)

    with pytest.raises(ValueError):
        RULSequenceDataset(sequences, targets)


def test_reference_model_matches_upstream_architecture():
    model = ReferenceLSTMRegressor()

    assert [layer.hidden_size for layer in model.lstm_layers] == [128, 64, 32]
    assert model.attention.projection.out_features == 64

    dense_sizes = [
        layer.out_features
        for layer in model.dense_head
        if isinstance(layer, torch.nn.Linear)
    ]
    assert dense_sizes == [64, 32, 16]


def test_reference_model_returns_one_prediction_per_sequence():
    model = ReferenceLSTMRegressor()
    inputs = torch.zeros((4, 50, 24), dtype=torch.float32)

    predictions = model(inputs)

    assert predictions.shape == (4,)


def test_reference_model_rejects_wrong_feature_count():
    model = ReferenceLSTMRegressor()
    inputs = torch.zeros((4, 50, 23), dtype=torch.float32)

    with pytest.raises(ValueError):
        model(inputs)


def test_saved_reference_model_reloads_with_identical_predictions(tmp_path):
    model = ReferenceLSTMRegressor()
    model.eval()
    inputs = torch.randn((2, 50, 24), dtype=torch.float32)
    artifact_path = tmp_path / "reference_lstm_model.pt"
    torch.save(
        {
            "model_class": "ReferenceLSTMRegressor",
            "model_config": model.get_config(),
            "state_dict": model.state_dict(),
        },
        artifact_path,
    )

    loaded_model = load_reference_model(artifact_path)

    with torch.no_grad():
        expected = model(inputs)
        actual = loaded_model(inputs)

    assert torch.equal(expected, actual)
