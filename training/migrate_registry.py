"""Copy an existing champion model into the containerized MLflow Registry."""

import argparse
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from training.mlflow_tracking import (
    DEFAULT_EXPERIMENT_NAME,
    LOCAL_TRACKING_URI,
)
from training.register_model import DEFAULT_REGISTERED_MODEL_NAME


def matching_version(
    client: MlflowClient,
    model_name: str,
    source_run_id: str,
):
    """Return an already migrated version, if one exists."""
    for version in client.search_model_versions(f"name = '{model_name}'"):
        if version.tags.get("migration_source_run_id") == source_run_id:
            return version
    return None


def migrate_champion(
    source_tracking_uri: str,
    destination_tracking_uri: str,
    model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    alias: str = "champion",
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> str:
    """Copy model, run metadata, and bundle artifacts to another Registry."""
    source_client = MlflowClient(tracking_uri=source_tracking_uri)
    source_version = source_client.get_model_version_by_alias(model_name, alias)
    source_run = source_client.get_run(source_version.run_id)

    destination_client = MlflowClient(tracking_uri=destination_tracking_uri)
    existing = matching_version(
        destination_client,
        model_name,
        source_version.run_id,
    )
    if existing is not None:
        destination_client.set_registered_model_alias(
            model_name,
            alias,
            existing.version,
        )
        print(f"Model already migrated as version {existing.version}")
        return str(existing.version)

    mlflow.set_tracking_uri(source_tracking_uri)
    model = mlflow.pytorch.load_model(
        f"models:/{model_name}@{alias}",
    )
    bundle_directory = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=(
                f"runs:/{source_version.run_id}/model_bundle"
            )
        )
    )

    experiment = destination_client.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else destination_client.create_experiment(experiment_name)
    )

    mlflow.set_tracking_uri(destination_tracking_uri)
    source_name = source_run.data.tags.get("mlflow.runName", "champion")
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"migrated-{source_name}",
        tags={
            "migration_source_run_id": source_version.run_id,
            "migration_source_tracking_uri": source_tracking_uri,
        },
    ) as destination_run:
        mlflow.log_params(source_run.data.params)
        mlflow.log_metrics(source_run.data.metrics)
        mlflow.log_artifacts(str(bundle_directory), artifact_path="model_bundle")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            serialization_format="pickle",
        )

    registered = mlflow.register_model(
        model_uri=f"runs:/{destination_run.info.run_id}/model",
        name=model_name,
        tags={
            "migration_source_run_id": source_version.run_id,
            "source_model_version": str(source_version.version),
        },
    )
    destination_client.set_registered_model_alias(
        model_name,
        alias,
        registered.version,
    )
    print(f"Migrated model: {model_name} version {registered.version}")
    print(f"Alias: {alias}")
    print(f"Destination run: {destination_run.info.run_id}")
    return str(registered.version)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tracking-uri", default=LOCAL_TRACKING_URI)
    parser.add_argument(
        "--destination-tracking-uri",
        default="http://127.0.0.1:5050",
    )
    parser.add_argument("--model-name", default=DEFAULT_REGISTERED_MODEL_NAME)
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    migrate_champion(
        source_tracking_uri=args.source_tracking_uri,
        destination_tracking_uri=args.destination_tracking_uri,
        model_name=args.model_name,
        alias=args.alias,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
