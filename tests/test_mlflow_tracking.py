"""Integration test for the local MLflow tracking wrapper."""

import mlflow
from mlflow.tracking import MlflowClient

from training.mlflow_tracking import MLflowExperimentTracker
from training.model import LSTMRegressor


def test_mlflow_tracker_records_params_metrics_and_artifact(tmp_path):
    database_path = tmp_path / "mlflow-test.db"
    artifact_directory = tmp_path / "mlruns"
    tracking_uri = f"sqlite:///{database_path}"
    tracker = MLflowExperimentTracker(
        tracking_uri=tracking_uri,
        experiment_name="test-rul-experiment",
        artifact_location=artifact_directory.as_uri(),
    )
    run_id = tracker.start_run(run_name="tracking-smoke-test")
    tracker.log_params({"sequence_length": 50, "max_rul": 125})
    tracker.log_epoch(
        {
            "train_rmse": 20.0,
            "validation_mae": 12.0,
            "validation_rmse": 15.0,
            "learning_rate": 0.001,
        },
        step=1,
    )
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"test-model")
    metadata = {
        "best_validation_rmse": 15.0,
        "best_epoch": 1,
        "training": {"completed_epochs": 1},
        "preprocessing": {
            "sequence_length": 3,
            "max_rul": 125,
            "feature_columns": ["sensor_1", "sensor_2"],
        },
    }
    model = LSTMRegressor(input_size=2, hidden_size=4, num_layers=1)
    tracker.log_training_result(metadata, {"model": artifact}, model=model)
    tracker.end_run()

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    artifacts = client.list_artifacts(run_id, path="model_bundle")

    assert run.info.status == "FINISHED"
    assert run.data.params["sequence_length"] == "50"
    assert run.data.metrics["validation_rmse"] == 15.0
    assert run.data.metrics["best_validation_rmse"] == 15.0
    assert [item.path for item in artifacts] == ["model_bundle/model.pt"]

    mlflow.set_tracking_uri(tracking_uri)
    version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name="test-rul-model",
    )
    client.set_registered_model_alias(
        "test-rul-model",
        "champion",
        version.version,
    )
    champion = client.get_model_version_by_alias(
        "test-rul-model",
        "champion",
    )

    assert champion.version == version.version
