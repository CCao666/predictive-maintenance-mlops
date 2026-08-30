"""Register the best MLflow tuning run and assign a model alias."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from training.mlflow_tracking import DEFAULT_TRACKING_URI
from training.train_lstm import ARTIFACTS_DIRECTORY


DEFAULT_REGISTERED_MODEL_NAME = "predictive-maintenance-rul"


def register_selected_model(
    selection_path: Path,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    alias: str = "champion",
) -> str:
    """Register the selected run's native MLflow model and set its alias."""
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    run_id = selection["best_run_id"]
    mlflow.set_tracking_uri(tracking_uri)
    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=model_name,
        tags={
            "selection_metric": selection["selection_metric"],
            "validation_rmse": str(selection["best_validation_rmse"]),
        },
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=model_version.version,
    )
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="source_run_id",
        value=run_id,
    )
    print(f"Registered model: {model_name} version {model_version.version}")
    print(f"Alias: {alias}")
    print(f"Source run: {run_id}")
    return model_version.version


def parse_args() -> argparse.Namespace:
    """Parse registry options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-path",
        type=Path,
        default=ARTIFACTS_DIRECTORY / "parameter_selection.json",
    )
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--model-name", default=DEFAULT_REGISTERED_MODEL_NAME)
    parser.add_argument("--alias", default="champion")
    return parser.parse_args()


def main() -> None:
    """Register the selected model."""
    args = parse_args()
    register_selected_model(
        selection_path=args.selection_path,
        tracking_uri=args.tracking_uri,
        model_name=args.model_name,
        alias=args.alias,
    )


if __name__ == "__main__":
    main()
