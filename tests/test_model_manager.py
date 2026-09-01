"""Unit tests for inference preprocessing and prediction."""

import numpy as np
import torch
from torch import nn

from inference.model_manager import ModelManager
from training.dataset import FEATURE_COLUMNS


class IdentityScaler:
    def transform(self, values):
        return values.to_numpy(dtype=np.float32)


class ConstantModel(nn.Module):
    def forward(self, inputs):
        assert inputs.shape == (1, 50, 24)
        return torch.tensor([150.0])


def test_predict_prepares_expected_tensor_and_caps_rul():
    manager = ModelManager()
    manager.model = ConstantModel()
    manager.scaler = IdentityScaler()
    sequence = [{name: 1.0 for name in FEATURE_COLUMNS} for _ in range(50)]

    prediction = manager.predict(sequence)

    assert prediction == 125.0


def test_model_info_contains_serving_contract():
    manager = ModelManager(
        model_name="test-model",
        model_alias="candidate",
    )
    manager.version = "3"
    manager.run_id = "run-123"

    assert manager.info() == {
        "name": "test-model",
        "version": "3",
        "alias": "candidate",
        "run_id": "run-123",
        "sequence_length": 50,
        "feature_count": 24,
    }
