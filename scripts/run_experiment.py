from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seugearbox.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the HHO-BP gearbox fault diagnosis experiment.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("artifacts/data/features_reconstructed_simplified_df.csv"),
        help="Path to the feature CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments/default"),
        help="Directory to store metrics and reports.",
    )
    parser.add_argument("--hawks", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        data_path=args.data,
        output_dir=args.output_dir,
        hawks=args.hawks,
        iterations=args.iters,
        test_size=args.test_size,
        seed=args.seed,
        verbose=True,
    )
    print("Best parameters:", result.best_params)
    print(f"Best score found by HHO: {result.best_score:.4f}")
    print(f"Final test accuracy: {result.final_accuracy:.4f}")


if __name__ == "__main__":
    main()
