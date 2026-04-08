from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seugearbox.features import build_feature_dataframe, rank_feature_importance, save_dataframe
from seugearbox.preprocessing import RawSignalConfig, load_raw_signal_matrix, segment_signal_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare reproducible gearbox research datasets from raw CSV signals.",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with raw CSV files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data"),
        help="Directory for generated research artifacts.",
    )
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--step-size", type=int, default=1000)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--level", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_signals = load_raw_signal_matrix(RawSignalConfig(input_dir=args.input_dir))
    segmented = segment_signal_frame(
        raw_signals=raw_signals,
        window_size=args.window_size,
        step_size=args.step_size,
    )
    features = build_feature_dataframe(
        segmented_frame=segmented,
        wavelet=args.wavelet,
        level=args.level,
    )
    importance = rank_feature_importance(features)

    save_dataframe(raw_signals, args.output_dir / "data_1.csv")
    save_dataframe(segmented, args.output_dir / "data_2.csv")
    save_dataframe(features, args.output_dir / "features_reconstructed_simplified_df.csv")
    save_dataframe(importance, args.output_dir / "feature_importance.csv")

    print(f"Saved raw signals to {args.output_dir / 'data_1.csv'}")
    print(f"Saved segmented windows to {args.output_dir / 'data_2.csv'}")
    print(f"Saved extracted features to {args.output_dir / 'features_reconstructed_simplified_df.csv'}")
    print(f"Saved feature importance ranking to {args.output_dir / 'feature_importance.csv'}")


if __name__ == "__main__":
    main()
