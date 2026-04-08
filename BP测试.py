from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from hho import HarrisHawkOptimization


DEFAULT_FEATURE_COLUMNS = [
    "整流平均值",
    "均方根",
    "低频能量",
    "低频奇异值特征",
    "高频能量",
    "频带能量",
    "平均值",
    "均方频率",
    "重心频率",
]
DEFAULT_LABEL_COLUMN = "故障类别"

FALLBACK_FEATURE_COLUMNS = [
    "鏁存祦骞冲潎鍊?",
    "鍧囨柟鏍?",
    "浣庨鑳介噺",
    "浣庨濂囧紓鍊肩壒寰?",
    "楂橀鑳介噺",
    "棰戝甫鑳介噺",
    "骞冲潎鍊?",
    "鍧囨柟棰戠巼",
    "閲嶅績棰戠巼",
]
FALLBACK_LABEL_COLUMN = "鏁呴殰绫诲埆"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an HHO-optimized BP neural network for gearbox fault diagnosis.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("features_reconstructed_simplified_df.csv"),
        help="Path to the prepared feature CSV.",
    )
    parser.add_argument("--hawks", type=int, default=20, help="Number of hawks in HHO.")
    parser.add_argument("--iters", type=int, default=50, help="Number of HHO iterations.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a confusion matrix heatmap after training.",
    )
    return parser.parse_args()


def resolve_columns(columns: Sequence[str]) -> tuple[list[str], str]:
    if set(DEFAULT_FEATURE_COLUMNS).issubset(columns) and DEFAULT_LABEL_COLUMN in columns:
        return list(DEFAULT_FEATURE_COLUMNS), DEFAULT_LABEL_COLUMN
    if set(FALLBACK_FEATURE_COLUMNS).issubset(columns) and FALLBACK_LABEL_COLUMN in columns:
        return list(FALLBACK_FEATURE_COLUMNS), FALLBACK_LABEL_COLUMN
    raise ValueError(
        "Could not find the expected feature columns in the CSV. "
        "Please export the notebook output with the documented schema."
    )


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


def main() -> None:
    args = parse_args()
    dataframe = pd.read_csv(args.data)
    dataframe = dataframe.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    feature_columns, label_column = resolve_columns(dataframe.columns.tolist())
    features = dataframe[feature_columns]
    labels = dataframe[label_column]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    def fitness_function(raw_params: Sequence[float]) -> tuple[float, tuple[int, int, float]]:
        hidden1, hidden2, learning_rate = clamp_params(raw_params)
        classifier = build_classifier(hidden1, hidden2, learning_rate, args.seed)
        classifier.fit(x_train_scaled, y_train)
        predictions = classifier.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        return -accuracy, (hidden1, hidden2, learning_rate)

    optimizer = HarrisHawkOptimization(
        fitness_function=fitness_function,
        n_hawks=args.hawks,
        dim=3,
        max_iter=args.iters,
        lb=[50, 50, 0.001],
        ub=[200, 200, 0.1],
        seed=args.seed,
    )

    _, best_accuracy, best_params = optimizer.optimize(verbose=True)
    hidden1, hidden2, learning_rate = best_params

    final_model = build_classifier(hidden1, hidden2, learning_rate, args.seed)
    final_model.fit(x_train_scaled, y_train)
    predictions = final_model.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\n=== Optimization Complete ===")
    print(
        "Best parameters: "
        f"hidden1={hidden1}, hidden2={hidden2}, learning_rate={learning_rate}"
    )
    print(f"Best score found by HHO: {best_accuracy:.4f}")
    print(f"Final test accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(report)

    if args.plot:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=final_model.classes_,
            yticklabels=final_model.classes_,
        )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("HHO-BP Fault Classification Confusion Matrix")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
