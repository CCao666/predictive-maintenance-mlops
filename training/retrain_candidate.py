"""Approve, train, compare, and promote drift-triggered model candidates."""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from psycopg.types.json import Jsonb

from inference.model_manager import ModelManager
from streaming.database import connect_with_retry
from training.dataset import COLUMN_NAMES, load_cmapss
from training.evaluate_test import (
    build_test_sequences,
    load_test_targets,
    regression_metrics,
)
from training.mlflow_tracking import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_URI,
)
from training.register_model import DEFAULT_REGISTERED_MODEL_NAME
from training.train_lstm import TrainingConfig, run_tracked_training


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/cmapss"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/retraining"


def candidate_decision(
    affected_candidate: dict[str, float],
    affected_champion: dict[str, float],
    fd001_candidate: dict[str, float],
    fd001_champion: dict[str, float],
    *,
    min_affected_rmse_improvement: float = 0.05,
    max_fd001_rmse_degradation: float = 0.03,
) -> dict[str, float | bool]:
    """Require drift improvement without regressing the FD001 baseline."""
    if affected_champion["rmse"] <= 0 or fd001_champion["rmse"] <= 0:
        raise ValueError("Champion RMSE must be greater than zero")

    affected_improvement = (
        affected_champion["rmse"] - affected_candidate["rmse"]
    ) / affected_champion["rmse"]
    fd001_degradation = (
        fd001_candidate["rmse"] - fd001_champion["rmse"]
    ) / fd001_champion["rmse"]

    affected_nasa_safe = (
        affected_candidate["nasa_score"] <= affected_champion["nasa_score"]
    )
    affected_overestimates_safe = (
        affected_candidate["severe_overestimate_count"]
        <= affected_champion["severe_overestimate_count"]
    )
    fd001_nasa_safe = (
        fd001_candidate["nasa_score"] <= fd001_champion["nasa_score"]
    )
    fd001_overestimates_safe = (
        fd001_candidate["severe_overestimate_count"]
        <= fd001_champion["severe_overestimate_count"]
    )
    eligible = (
        affected_improvement >= min_affected_rmse_improvement
        and affected_nasa_safe
        and affected_overestimates_safe
        and fd001_degradation <= max_fd001_rmse_degradation
        and fd001_nasa_safe
        and fd001_overestimates_safe
    )
    return {
        "eligible": eligible,
        "affected_rmse_improvement": float(affected_improvement),
        "required_affected_rmse_improvement": min_affected_rmse_improvement,
        "affected_nasa_score_not_worse": affected_nasa_safe,
        "affected_severe_overestimates_not_worse": affected_overestimates_safe,
        "fd001_rmse_degradation": float(fd001_degradation),
        "max_fd001_rmse_degradation": max_fd001_rmse_degradation,
        "fd001_rmse_within_limit": (
            fd001_degradation <= max_fd001_rmse_degradation
        ),
        "fd001_nasa_score_not_worse": fd001_nasa_safe,
        "fd001_severe_overestimates_not_worse": fd001_overestimates_safe,
    }


def build_combined_training_file(
    dataset_id: str,
    data_dir: Path,
    output_path: Path,
) -> Path:
    """Combine FD001 with the shifted dataset using unique engine IDs."""
    baseline = load_cmapss(data_dir / "train_FD001.txt")
    if dataset_id == "FD001":
        combined = baseline
    else:
        shifted = load_cmapss(data_dir / f"train_{dataset_id}.txt").copy()
        shifted["unit_id"] += int(baseline["unit_id"].max())
        combined = pd.concat(
            [baseline, shifted],
            ignore_index=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined[COLUMN_NAMES].to_csv(
        output_path,
        sep=" ",
        header=False,
        index=False,
    )
    return output_path


def evaluate_model(
    model,
    scaler,
    dataset_id: str,
    data_dir: Path,
    sequence_length: int,
    max_rul: int,
) -> dict[str, float]:
    dataframe = load_cmapss(data_dir / f"test_{dataset_id}.txt")
    engine_ids, sequences = build_test_sequences(
        dataframe,
        scaler,
        sequence_length,
    )
    targets = load_test_targets(
        data_dir / f"RUL_{dataset_id}.txt",
        engine_ids,
    )
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(sequences)).cpu().numpy()
    predictions = np.clip(predictions, 0, max_rul)
    return regression_metrics(targets, predictions)


def ensure_request_schema(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            "ALTER TABLE retraining_requests "
            "ADD COLUMN IF NOT EXISTS candidate_run_id TEXT"
        )
        cursor.execute(
            "ALTER TABLE retraining_requests "
            "ADD COLUMN IF NOT EXISTS candidate_version TEXT"
        )
        cursor.execute(
            "ALTER TABLE retraining_requests "
            "ADD COLUMN IF NOT EXISTS comparison JSONB"
        )
        cursor.execute(
            "ALTER TABLE retraining_requests "
            "ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ"
        )
    connection.commit()


def approve_request(request_id: int) -> None:
    connection = connect_with_retry()
    try:
        ensure_request_schema(connection)
        with connection.cursor() as cursor:
            row = cursor.execute(
                """
                UPDATE retraining_requests
                SET status = 'approved'
                WHERE id = %s AND status = 'pending_review'
                RETURNING dataset_id
                """,
                (request_id,),
            ).fetchone()
        connection.commit()
        if not row:
            raise ValueError("Request must exist and be pending_review")
        print(f"Approved request {request_id} for {row[0]}")
    finally:
        connection.close()


def train_request(
    request_id: int,
    *,
    data_dir: Path,
    output_dir: Path,
    epochs: int,
    tracking_uri: str,
    experiment_name: str,
) -> dict:
    connection = connect_with_retry()
    ensure_request_schema(connection)
    try:
        request = connection.execute(
            "SELECT dataset_id, status FROM retraining_requests WHERE id = %s",
            (request_id,),
        ).fetchone()
        if not request or request[1] != "approved":
            raise ValueError("Request must exist and be approved")
        dataset_id = request[0]
        connection.execute(
            "UPDATE retraining_requests SET status = 'running' WHERE id = %s",
            (request_id,),
        )
        connection.commit()

        request_dir = output_dir / f"request_{request_id}"
        training_path = build_combined_training_file(
            dataset_id,
            data_dir,
            request_dir / f"train_FD001_{dataset_id}.txt",
        )
        config = TrainingConfig(data_path=training_path, epochs=epochs)
        run_id = run_tracked_training(
            config,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=f"retrain-{dataset_id}-request-{request_id}",
            tags={
                "dataset": f"NASA_CMAPSS_FD001+{dataset_id}",
                "retraining_request_id": str(request_id),
                "candidate": "true",
            },
        )
        if run_id is None:
            raise RuntimeError("Retraining must produce an MLflow run")

        mlflow.set_tracking_uri(tracking_uri)
        candidate_model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
        scaler_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/model_bundle/robust_scaler.joblib"
        )
        candidate_scaler = joblib.load(scaler_path)
        affected_candidate_metrics = evaluate_model(
            candidate_model,
            candidate_scaler,
            dataset_id,
            data_dir,
            config.sequence_length,
            config.max_rul,
        )

        champion = ModelManager(tracking_uri=tracking_uri)
        champion.load()
        affected_champion_metrics = evaluate_model(
            champion.model,
            champion.scaler,
            dataset_id,
            data_dir,
            champion.sequence_length,
            champion.max_rul,
        )

        if dataset_id == "FD001":
            fd001_candidate_metrics = affected_candidate_metrics
            fd001_champion_metrics = affected_champion_metrics
        else:
            fd001_candidate_metrics = evaluate_model(
                candidate_model,
                candidate_scaler,
                "FD001",
                data_dir,
                config.sequence_length,
                config.max_rul,
            )
            fd001_champion_metrics = evaluate_model(
                champion.model,
                champion.scaler,
                "FD001",
                data_dir,
                champion.sequence_length,
                champion.max_rul,
            )

        decision = candidate_decision(
            affected_candidate_metrics,
            affected_champion_metrics,
            fd001_candidate_metrics,
            fd001_champion_metrics,
        )
        comparison = {
            "affected_dataset": {
                "dataset_id": dataset_id,
                "candidate": affected_candidate_metrics,
                "champion": affected_champion_metrics,
            },
            "fd001_regression": {
                "dataset_id": "FD001",
                "candidate": fd001_candidate_metrics,
                "champion": fd001_champion_metrics,
            },
            "decision": decision,
        }

        client = MlflowClient(tracking_uri=tracking_uri)
        for name, value in affected_candidate_metrics.items():
            client.log_metric(run_id, f"retrain_affected_{name}", value)
        for name, value in fd001_candidate_metrics.items():
            client.log_metric(run_id, f"retrain_fd001_{name}", value)
        client.log_dict(run_id, comparison, "retraining/comparison.json")

        version = None
        status = "rejected"
        if decision["eligible"]:
            model_version = mlflow.register_model(
                f"runs:/{run_id}/model",
                DEFAULT_REGISTERED_MODEL_NAME,
            )
            version = str(model_version.version)
            client.set_registered_model_alias(
                DEFAULT_REGISTERED_MODEL_NAME,
                "challenger",
                version,
            )
            client.set_model_version_tag(
                DEFAULT_REGISTERED_MODEL_NAME,
                version,
                "promotion_eligible",
                "true",
            )
            status = "candidate_ready"

        connection.execute(
            """
            UPDATE retraining_requests
            SET status = %s, candidate_run_id = %s, candidate_version = %s,
                comparison = %s, completed_at = NOW()
            WHERE id = %s
            """,
            (status, run_id, version, Jsonb(comparison), request_id),
        )
        connection.commit()
        print(json.dumps(comparison, indent=2))
        print(f"Request status: {status}")
        return comparison
    except Exception:
        connection.rollback()
        connection.execute(
            """
            UPDATE retraining_requests
            SET status = 'failed', completed_at = NOW()
            WHERE id = %s AND status = 'running'
            """,
            (request_id,),
        )
        connection.commit()
        raise
    finally:
        connection.close()


def promote_request(
    request_id: int,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> None:
    connection = connect_with_retry()
    try:
        ensure_request_schema(connection)
        row = connection.execute(
            """
            SELECT candidate_version FROM retraining_requests
            WHERE id = %s AND status = 'candidate_ready'
            """,
            (request_id,),
        ).fetchone()
        if not row or not row[0]:
            raise ValueError("Request has no promotion-eligible candidate")

        client = MlflowClient(tracking_uri=tracking_uri)
        client.set_registered_model_alias(
            DEFAULT_REGISTERED_MODEL_NAME,
            "champion",
            row[0],
        )
        connection.execute(
            """
            UPDATE retraining_requests
            SET status = 'promoted', completed_at = NOW()
            WHERE id = %s
            """,
            (request_id,),
        )
        connection.commit()
        print(f"Promoted model version {row[0]} to champion")
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("approve", "promote"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("request_id", type=int)
        subparser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("request_id", type=int)
    train_parser.add_argument("--epochs", type=int, default=60)
    train_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    train_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    train_parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    train_parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "approve":
        approve_request(args.request_id)
    elif args.command == "train":
        train_request(
            args.request_id,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
        )
    else:
        promote_request(args.request_id, args.tracking_uri)


if __name__ == "__main__":
    main()
