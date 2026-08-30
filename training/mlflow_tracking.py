"""MLflow experiment tracking for RUL model training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_NAME = "predictive-maintenance-rul"
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
DEFAULT_ARTIFACT_LOCATION = (PROJECT_ROOT / "mlruns").as_uri()


class MLflowExperimentTracker:
    """Small wrapper around one active MLflow training run."""

    def __init__(
        self,
        tracking_uri: str = DEFAULT_TRACKING_URI,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        artifact_location: str = DEFAULT_ARTIFACT_LOCATION,
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.experiment_id = self._get_or_create_experiment()
        self.run_id: str | None = None

    def _get_or_create_experiment(self) -> str:
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is not None:
            return experiment.experiment_id
        return self.client.create_experiment(
            self.experiment_name,
            artifact_location=self.artifact_location,
        )

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Start a run and return its stable MLflow run ID."""
        if self.run_id is not None:
            raise RuntimeError("An MLflow run is already active")
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags,
        )
        self.run_id = run.info.run_id
        return self.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """Log immutable run configuration values."""
        self._require_active_run()
        mlflow.log_params(params)

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        """Log numeric training metrics for one epoch."""
        self._require_active_run()
        mlflow.log_metrics(
            {
                "train_rmse": float(metrics["train_rmse"]),
                "validation_mae": float(metrics["validation_mae"]),
                "validation_rmse": float(metrics["validation_rmse"]),
                "learning_rate": float(metrics["learning_rate"]),
            },
            step=step,
        )

    def log_training_result(
        self,
        metadata: dict[str, Any],
        artifact_paths: dict[str, Path],
        model: Any | None = None,
    ) -> None:
        """Log best metrics, full metadata, and deployable artifacts."""
        self._require_active_run()
        mlflow.log_metrics(
            {
                "best_validation_rmse": float(
                    metadata["best_validation_rmse"]
                ),
                "best_epoch": float(metadata["best_epoch"]),
                "completed_epochs": float(
                    metadata["training"]["completed_epochs"]
                ),
            }
        )
        mlflow.log_dict(metadata, "metadata/training_metrics.json")
        for artifact_path in artifact_paths.values():
            mlflow.log_artifact(str(artifact_path), artifact_path="model_bundle")
        if model is not None:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                serialization_format="pickle",
                metadata={
                    "sequence_length": metadata["preprocessing"][
                        "sequence_length"
                    ],
                    "max_rul": metadata["preprocessing"]["max_rul"],
                    "feature_count": len(
                        metadata["preprocessing"]["feature_columns"]
                    ),
                },
            )

    def end_run(self, status: str = "FINISHED") -> None:
        """Finish the active run with an MLflow status."""
        if self.run_id is None:
            return
        mlflow.end_run(status=status)
        self.run_id = None

    def _require_active_run(self) -> None:
        if self.run_id is None:
            raise RuntimeError("Start an MLflow run before logging")
