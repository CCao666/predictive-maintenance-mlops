"""Compare a small upstream-derived LSTM training parameter set in MLflow."""

import argparse
import json

from mlflow.tracking import MlflowClient

from training.mlflow_tracking import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_URI,
)
from training.train_lstm import (
    ARTIFACTS_DIRECTORY,
    TrainingConfig,
    run_tracked_training,
)


TRIALS = [
    {
        "name": "reference-lr-0.001-batch-32",
        "learning_rate": 0.001,
        "batch_size": 32,
    },
    {
        "name": "reference-lr-0.0001-batch-32",
        "learning_rate": 0.0001,
        "batch_size": 32,
    },
    {
        "name": "reference-lr-0.001-batch-64",
        "learning_rate": 0.001,
        "batch_size": 64,
    },
]


def run_parameter_selection(
    *,
    epochs: int,
    tracking_uri: str,
    experiment_name: str,
) -> dict[str, object]:
    """Train all fixed-architecture trials and select the lowest RMSE run."""
    client = MlflowClient(tracking_uri=tracking_uri)
    results = []

    for index, trial in enumerate(TRIALS, start=1):
        print(f"Starting tuning trial {index}/{len(TRIALS)}: {trial['name']}")
        config = TrainingConfig(
            epochs=epochs,
            batch_size=trial["batch_size"],
            learning_rate=trial["learning_rate"],
        )
        run_id = run_tracked_training(
            config,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=trial["name"],
            tags={
                "tuning": "true",
                "tuning_trial": str(index),
                "architecture": "upstream_reference",
            },
        )
        run = client.get_run(run_id)
        results.append(
            {
                **trial,
                "run_id": run_id,
                "best_validation_rmse": run.data.metrics[
                    "best_validation_rmse"
                ],
                "best_epoch": int(run.data.metrics["best_epoch"]),
            }
        )

    best = min(results, key=lambda result: result["best_validation_rmse"])
    selection = {
        "selection_metric": "best_validation_rmse",
        "best_run_id": best["run_id"],
        "best_params": {
            "learning_rate": best["learning_rate"],
            "batch_size": best["batch_size"],
        },
        "best_validation_rmse": best["best_validation_rmse"],
        "trials": results,
    }
    ARTIFACTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACTS_DIRECTORY / "parameter_selection.json"
    output_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Best run: {selection['best_run_id']}")
    print(f"Best validation RMSE: {selection['best_validation_rmse']:.4f}")
    print(f"Saved selection: {output_path}")
    return selection


def parse_args() -> argparse.Namespace:
    """Parse tuning options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    return parser.parse_args()


def main() -> None:
    """Run the parameter selection experiment."""
    args = parse_args()
    run_parameter_selection(
        epochs=args.epochs,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
