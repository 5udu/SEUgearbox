from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .constants import EXPECTED_RAW_FILES


@dataclass(frozen=True)
class RawSignalConfig:
    input_dir: Path
    file_names: Sequence[str] = tuple(EXPECTED_RAW_FILES)
    row_start: int = 1015
    row_stop: int = 202063
    signal_column: int = 2
    max_columns: int = 8
    truncate_length: int = 201048


def load_raw_signal_matrix(config: RawSignalConfig) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for file_name in config.file_names:
        file_path = config.input_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Missing raw signal file: {file_path}")

        try:
            data = pd.read_csv(file_path, sep="\t", header=None, low_memory=False)
        except Exception:
            data = pd.read_csv(file_path, header=None, low_memory=False)

        trimmed = data.iloc[config.row_start : config.row_stop, : config.max_columns]
        if trimmed.shape[1] <= config.signal_column:
            raise ValueError(f"{file_path} does not contain signal column {config.signal_column}")

        signal = trimmed.iloc[:, config.signal_column].reset_index(drop=True)
        columns[file_name.replace(".csv", "")] = signal.iloc[: config.truncate_length]

    return pd.DataFrame(columns)


def segment_signal_frame(
    raw_signals: pd.DataFrame,
    window_size: int = 2048,
    step_size: int = 1000,
    label_column: str = "labels",
) -> pd.DataFrame:
    if raw_signals.empty:
        raise ValueError("raw_signals must not be empty")
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive")

    segments: list[np.ndarray] = []
    labels: list[int] = []

    for label, (_, column) in enumerate(raw_signals.items()):
        values = column.to_numpy(dtype=np.float64)
        num_samples = (len(values) - window_size) // step_size + 1
        if num_samples <= 0:
            raise ValueError("window_size is larger than the available signal length")

        for index in range(num_samples):
            start = index * step_size
            end = start + window_size
            segments.append(values[start:end])
            labels.append(label)

    segmented = pd.DataFrame(np.asarray(segments))
    segmented[label_column] = np.asarray(labels, dtype=int)
    return segmented
