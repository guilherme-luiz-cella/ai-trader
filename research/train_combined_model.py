"""Train a supervised model from the combined price + news dataset.

This script expects research/build_datasets.py to have produced a CSV like:
- research/data_sets/aapl_training_dataset.csv

It uses all numeric feature columns except the target and timestamp/date fields.
Outputs:
- a trained model (.joblib)
- evaluation metrics (.json)
- a copy of the dataset used for training (.csv)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_DIR = Path(os.getenv("TRAIN_OUTPUT_DIR", BASE_DIR / "artifacts"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = Path(
    os.getenv(
        "COMBINED_DATASET_PATH",
        str(BASE_DIR / "data_sets" / ("multi_symbol_training_dataset.csv" if "," in os.getenv("DATA_SYMBOLS", "") else f"{os.getenv('DATA_SYMBOL', 'AAPL').lower()}_training_dataset.csv")),
    )
)
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "target")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))


@dataclass(frozen=True)
class TrainingResult:
    samples: int
    feature_count: int
    train_accuracy: float
    test_accuracy: float
    model_path: str
    metrics_path: str
    dataset_path: str


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def prepare_features(data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    frame = data.copy()
    if target_column not in frame.columns:
        raise ValueError(f"Target column not found: {target_column}")

    excluded_columns = {target_column, "timestamp", "date", "symbol", "future_return", "target_label", "target_return_threshold", "target_downside_threshold"}
    feature_columns = [
        column
        for column in frame.columns
        if column not in excluded_columns and pd.api.types.is_numeric_dtype(frame[column])
    ]

    if not feature_columns:
        raise ValueError("No numeric feature columns found in dataset")

    features = frame[feature_columns].copy()
    labels = frame[target_column].astype(int)
    return features, labels


def chronological_split(features: pd.DataFrame, labels: pd.Series, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_index = int(len(features) * (1 - test_size))
    if split_index <= 0 or split_index >= len(features):
        raise ValueError("Not enough rows to split train/test")

    x_train = features.iloc[:split_index].copy()
    x_test = features.iloc[split_index:].copy()
    y_train = labels.iloc[:split_index].copy()
    y_test = labels.iloc[split_index:].copy()
    return x_train, x_test, y_train, y_test


def train_model(x_train: pd.DataFrame, y_train: pd.Series):
    gradient_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=400,
        min_samples_leaf=16,
        l2_regularization=0.15,
        random_state=42,
    )
    forest_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    )
    ensemble = VotingClassifier(
        estimators=[
            ("hist_gb", gradient_model),
            ("rf", forest_model),
        ],
        voting="soft",
    )
    calibrated_model = CalibratedClassifierCV(estimator=ensemble, method="sigmoid", cv=3)
    calibrated_model.fit(x_train, y_train)
    return calibrated_model


def evaluate_model(model, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    test_probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "train_accuracy": float(accuracy_score(y_train, train_predictions)),
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "confusion_matrix": confusion_matrix(y_test, test_predictions).tolist(),
        "classification_report": classification_report(y_test, test_predictions, output_dict=True, zero_division=0),
        "mean_test_probability": float(pd.Series(test_probabilities).mean()),
    }


def save_outputs(model, metrics: Dict[str, object], dataset: pd.DataFrame, symbol: str) -> TrainingResult:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = OUTPUT_DIR / f"{symbol}_combined_{stamp}_model.joblib"
    metrics_path = OUTPUT_DIR / f"{symbol}_combined_{stamp}_metrics.json"
    dataset_path = OUTPUT_DIR / f"{symbol}_combined_{stamp}_dataset.csv"

    joblib.dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    dataset.to_csv(dataset_path, index=False)

    return TrainingResult(
        samples=len(dataset),
        feature_count=len([column for column in dataset.columns if column not in {"timestamp", "date", "target"}]),
        train_accuracy=metrics["train_accuracy"],
        test_accuracy=metrics["test_accuracy"],
        model_path=str(model_path),
        metrics_path=str(metrics_path),
        dataset_path=str(dataset_path),
    )


def main() -> None:
    dataset = load_dataset(DATASET_PATH)
    symbol = DATASET_PATH.stem.replace("_training_dataset", "")
    features, labels = prepare_features(dataset, TARGET_COLUMN)
    x_train, x_test, y_train, y_test = chronological_split(features, labels, TEST_SIZE)

    model = train_model(x_train, y_train)
    metrics = evaluate_model(model, x_train, y_train, x_test, y_test)
    result = save_outputs(model, metrics, dataset, symbol)

    print("Training complete")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Samples: {result.samples}")
    print(f"Features: {result.feature_count}")
    print(f"Train accuracy: {result.train_accuracy:.4f}")
    print(f"Test accuracy: {result.test_accuracy:.4f}")
    print(f"Model saved to: {result.model_path}")
    print(f"Metrics saved to: {result.metrics_path}")
    print(f"Dataset copy saved to: {result.dataset_path}")


if __name__ == "__main__":
    main()
