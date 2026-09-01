"""Load the champion model and run predictions."""

import os
from typing import Any

import joblib
import mlflow
import pandas as pd
import torch
from mlflow.tracking import MlflowClient

from training.dataset import FEATURE_COLUMNS
from training.mlflow_tracking import DEFAULT_TRACKING_URI


class ModelManager:
    def __init__(
        self,
        tracking_uri: str | None = None,
        model_name: str | None = None,
        model_alias: str | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI",
            DEFAULT_TRACKING_URI,
        )
        self.model_name = model_name or os.getenv(
            "MODEL_NAME",
            "predictive-maintenance-rul",
        )
        self.model_alias = model_alias or os.getenv("MODEL_ALIAS", "champion")

        self.model: Any | None = None
        self.scaler: Any | None = None
        self.version = ""
        self.run_id = ""
        self.sequence_length = 50
        self.max_rul = 125

    @property
    def ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    def load(self) -> None:
        """Load the aliased model and scaler from the same MLflow run."""
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        version = client.get_model_version_by_alias(
            self.model_name,
            self.model_alias,
        )

        self.model = mlflow.pytorch.load_model(
            f"models:/{self.model_name}@{self.model_alias}"
        )
        self.model.eval()

        scaler_path = mlflow.artifacts.download_artifacts(
            artifact_uri=(
                f"runs:/{version.run_id}/"
                "model_bundle/robust_scaler.joblib"
            )
        )
        self.scaler = joblib.load(scaler_path)
        self.version = str(version.version)
        self.run_id = version.run_id

        params = client.get_run(self.run_id).data.params
        self.sequence_length = int(params.get("sequence_length", 50))
        self.max_rul = int(params.get("max_rul", 125))

    def predict(self, sequence: list[Any]) -> float:
        """Scale one sensor sequence and return predicted RUL."""
        if not self.ready:
            raise RuntimeError("Model is not loaded")
        if len(sequence) != self.sequence_length:
            raise ValueError(
                f"Expected {self.sequence_length} readings, got {len(sequence)}"
            )

        rows = [
            reading.model_dump() if hasattr(reading, "model_dump") else reading
            for reading in sequence
        ]
        features = pd.DataFrame(rows)[FEATURE_COLUMNS]
        scaled = self.scaler.transform(features)
        inputs = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(inputs).item()

        return min(max(prediction, 0.0), float(self.max_rul))

    def info(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "version": self.version,
            "alias": self.model_alias,
            "run_id": self.run_id,
            "sequence_length": self.sequence_length,
            "feature_count": len(FEATURE_COLUMNS),
        }
