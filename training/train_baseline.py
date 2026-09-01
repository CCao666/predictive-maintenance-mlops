"""Train and evaluate the Random Forest RUL baseline."""

import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from training.dataset import (
    FEATURE_COLUMNS,
    add_train_rul,
    load_cmapss,
)
from training.split import split_by_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data/cmapss/train_FD001.txt"
ARTIFACTS_DIRECTORY = PROJECT_ROOT / "artifacts"


def train_baseline(
    data_path: Path = DEFAULT_DATA_PATH,
) -> tuple[RandomForestRegressor, dict[str, float]]:
    """Train and evaluate a Random Forest RUL baseline."""
    dataframe = add_train_rul(load_cmapss(data_path))

    train_df, validation_df = split_by_engine(
        dataframe=dataframe,
        validation_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(train_df[FEATURE_COLUMNS], train_df["rul"])
    predictions = model.predict(validation_df[FEATURE_COLUMNS])
    targets = validation_df["rul"]

    metrics = {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(mean_squared_error(targets, predictions) ** 0.5),
        "training_engines": int(train_df["unit_id"].nunique()),
        "validation_engines": int(validation_df["unit_id"].nunique()),
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
    }

    return model, metrics


def save_artifacts(
    model: RandomForestRegressor,
    metrics: dict[str, float],
) -> None:
    """Save the trained model and evaluation metrics."""
    ARTIFACTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIRECTORY / "random_forest.joblib"
    metrics_path = ARTIFACTS_DIRECTORY / "baseline_metrics.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


def main() -> None:
    """Run baseline training from the command line."""
    model, metrics = train_baseline()

    print(f"Training engines: {metrics['training_engines']}")
    print(f"Validation engines: {metrics['validation_engines']}")
    print(f"Validation MAE: {metrics['mae']:.2f}")
    print(f"Validation RMSE: {metrics['rmse']:.2f}")

    save_artifacts(model, metrics)


if __name__ == "__main__":
    main()
