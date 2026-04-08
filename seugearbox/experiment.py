from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from hho import HarrisHawkOptimization

from .constants import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_LABEL_COLUMN,
    FALLBACK_FEATURE_COLUMNS,
    FALLBACK_LABEL_COLUMN,
)


@dataclass(frozen=True)
class ExperimentResult:
    best_params: tuple[int, int, float]
    best_score: float
    final_accuracy: float
    report: str


def resolve_columns(columns: Sequence[str]) -> tuple[list[str], str]:
    if set(DEFAULT_FEATURE_COLUMNS).issubset(columns) and DEFAULT_LABEL_COLUMN in columns:
        return list(DEFAULT_FEATURE_COLUMNS), DEFAULT_LABEL_COLUMN
    if set(FALLBACK_FEATURE_COLUMNS).issubset(columns) and FALLBACK_LABEL_COLUMN in columns:
        return list(FALLBACK_FEATURE_COLUMNS), FALLBACK_LABEL_COLUMN
    raise ValueError("Could not find the expected feature columns in the CSV.")


def build_classifier(hidden1: int, hidden2: int, learning_rate: float, seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(hidden1, hidden2),
        activation="relu",
        solver="adam",
        learning_rate_init=learning_rate,
        max_iter=2000,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
    )


def clamp_params(params: Sequence[float]) -> tuple[int, int, float]:
    hidden1 = max(10, min(int(params[0]), 200))
    hidden2 = max(5, min(int(params[1]), 200))
    learning_rate = round(max(0.0001, min(float(params[2]), 0.1)), 9)
    return hidden1, hidden2, learning_rate


def run_experiment(
    data_path: Path,
    output_dir: Path,
    hawks: int = 20,
    iterations: int = 50,
    test_size: float = 0.3,
    seed: int = 42,
    verbose: bool = True,
) -> ExperimentResult:
    dataframe = pd.read_csv(data_path)
    dataframe = dataframe.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    feature_columns, label_column = resolve_columns(dataframe.columns.tolist())
    features = dataframe[feature_columns]
    labels = dataframe[label_column]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    def fitness_function(raw_params: Sequence[float]) -> tuple[float, tuple[int, int, float]]:
        hidden1, hidden2, learning_rate = clamp_params(raw_params)
        classifier = build_classifier(hidden1, hidden2, learning_rate, seed)
        classifier.fit(x_train_scaled, y_train)
        predictions = classifier.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        return -accuracy, (hidden1, hidden2, learning_rate)

    optimizer = HarrisHawkOptimization(
        fitness_function=fitness_function,
        n_hawks=hawks,
        dim=3,
        max_iter=iterations,
        lb=[50, 50, 0.001],
        ub=[200, 200, 0.1],
        seed=seed,
    )

    _, best_score, best_params = optimizer.optimize(verbose=verbose)
    hidden1, hidden2, learning_rate = best_params

    final_model = build_classifier(hidden1, hidden2, learning_rate, seed)
    final_model.fit(x_train_scaled, y_train)
    predictions = final_model.predict(x_test_scaled)
    final_accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    save_experiment_artifacts(
        output_dir=output_dir,
        result=ExperimentResult(best_params, best_score, final_accuracy, report),
        confusion=matrix,
    )
    return ExperimentResult(best_params, best_score, final_accuracy, report)


def save_experiment_artifacts(
    output_dir: Path,
    result: ExperimentResult,
    confusion,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classification_report.txt").write_text(result.report, encoding="utf-8")
    pd.DataFrame(confusion).to_csv(output_dir / "confusion_matrix.csv", index=False)
    payload = asdict(result)
    payload["best_params"] = list(result.best_params)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
