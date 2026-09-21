"""Evaluate the registered champion on the official C-MAPSS FD001 test set."""

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from inference.model_manager import ModelManager
from training.dataset import FEATURE_COLUMNS, load_cmapss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_PATH = PROJECT_ROOT / "data/cmapss/test_FD001.txt"
DEFAULT_RUL_PATH = PROJECT_ROOT / "data/cmapss/RUL_FD001.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/test_evaluation"


def load_test_targets(path: str | Path, engine_ids: list[int]) -> np.ndarray:
    targets = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0]
    if len(targets) != len(engine_ids):
        raise ValueError(
            f"Expected {len(engine_ids)} RUL targets, found {len(targets)}"
        )
    return targets.to_numpy(dtype=np.float32)


def build_test_sequences(
    dataframe: pd.DataFrame,
    scaler,
    sequence_length: int,
) -> tuple[list[int], np.ndarray]:
    """Build one final sequence per engine using upstream zero padding."""
    engine_ids: list[int] = []
    sequences: list[np.ndarray] = []

    for engine_id, engine in dataframe.groupby("unit_id", sort=True):
        values = (
            engine.sort_values("time_cycle")[FEATURE_COLUMNS]
            .to_numpy(dtype=np.float32)[-sequence_length:]
        )
        if len(values) < sequence_length:
            padding = np.zeros(
                (sequence_length - len(values), len(FEATURE_COLUMNS)),
                dtype=np.float32,
            )
            values = np.vstack([padding, values])

        scaled = scaler.transform(
            pd.DataFrame(values, columns=FEATURE_COLUMNS)
        ).astype(np.float32)
        engine_ids.append(int(engine_id))
        sequences.append(scaled)

    return engine_ids, np.stack(sequences)


def nasa_scores(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Return the upstream PHM08 asymmetric penalty for each engine."""
    errors = predictions - targets
    return np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1,
        np.exp(errors / 10.0) - 1,
    )


def regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    errors = predictions - targets
    return {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(mean_squared_error(targets, predictions) ** 0.5),
        "r2": float(r2_score(targets, predictions)),
        "nasa_score": float(nasa_scores(targets, predictions).sum()),
        "mean_error": float(errors.mean()),
        "within_10_cycles_pct": float(np.mean(np.abs(errors) <= 10) * 100),
        "within_20_cycles_pct": float(np.mean(np.abs(errors) <= 20) * 100),
        "overestimate_pct": float(np.mean(errors > 0) * 100),
        "severe_overestimate_count": int(np.sum(errors > 20)),
    }


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    true_critical = targets <= 30
    predicted_critical = predictions <= 30
    true_warning = targets <= 60
    predicted_warning = predictions <= 60

    critical_tp = int(np.sum(true_critical & predicted_critical))
    predicted_critical_count = int(predicted_critical.sum())
    true_critical_count = int(true_critical.sum())

    return {
        "critical_recall": critical_tp / true_critical_count,
        "critical_precision": (
            critical_tp / predicted_critical_count
            if predicted_critical_count
            else 0.0
        ),
        "missed_critical_engines": int(
            np.sum(true_critical & ~predicted_critical)
        ),
        "false_critical_alerts": int(
            np.sum(~true_critical & predicted_critical)
        ),
        "warning_recall": float(
            np.sum(true_warning & predicted_warning) / true_warning.sum()
        ),
    }


def health_status(rul: float) -> str:
    if rul <= 10:
        return "imminent"
    if rul <= 30:
        return "critical"
    if rul <= 60:
        return "warning"
    return "healthy"


def evaluate(
    test_path: str | Path = DEFAULT_TEST_PATH,
    rul_path: str | Path = DEFAULT_RUL_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    log_to_mlflow: bool = False,
) -> dict:
    manager = ModelManager()
    manager.load()

    dataframe = load_cmapss(test_path)
    engine_ids, sequences = build_test_sequences(
        dataframe,
        manager.scaler,
        manager.sequence_length,
    )
    official_targets = load_test_targets(rul_path, engine_ids)
    capped_targets = np.minimum(official_targets, manager.max_rul)

    with torch.no_grad():
        predictions = manager.model(torch.from_numpy(sequences)).cpu().numpy()
    predictions = np.clip(predictions, 0, manager.max_rul)

    official_scores = nasa_scores(official_targets, predictions)
    metrics = {
        "model": manager.info(),
        "engines": len(engine_ids),
        "padding": "left_zero_before_scaling",
        "official": regression_metrics(official_targets, predictions),
        "capped": regression_metrics(capped_targets, predictions),
        "business": classification_metrics(official_targets, predictions),
    }

    results = pd.DataFrame({
        "engine_id": engine_ids,
        "true_rul": official_targets,
        "capped_true_rul": capped_targets,
        "predicted_rul": predictions,
        "error": predictions - official_targets,
        "absolute_error": np.abs(predictions - official_targets),
        "nasa_score": official_scores,
        "true_status": [health_status(value) for value in official_targets],
        "predicted_status": [health_status(value) for value in predictions],
    })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    results.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if log_to_mlflow:
        client = mlflow.MlflowClient(tracking_uri=manager.tracking_uri)
        flat_metrics = {
            f"test_{group}_{name}": value
            for group in ("official", "capped", "business")
            for name, value in metrics[group].items()
        }
        for name, value in flat_metrics.items():
            client.log_metric(manager.run_id, name, value)
        client.log_artifact(
            manager.run_id,
            str(predictions_path),
            artifact_path="test_evaluation",
        )
        client.log_artifact(
            manager.run_id,
            str(metrics_path),
            artifact_path="test_evaluation",
        )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--rul-path", type=Path, default=DEFAULT_RUL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-to-mlflow", action="store_true")
    args = parser.parse_args()

    metrics = evaluate(
        args.test_path,
        args.rul_path,
        args.output_dir,
        args.log_to_mlflow,
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
