from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from numpy.linalg import svd
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier

from .constants import FEATURE_COLUMNS


def extract_time_features(
    reconstructed_signal: np.ndarray,
    c_a: np.ndarray,
    c_d: list[np.ndarray],
) -> list[float]:
    reconstructed_signal = np.asarray(reconstructed_signal, dtype=np.float64)
    c_a = np.asarray(c_a, dtype=np.float64)
    c_d = [np.asarray(d, dtype=np.float64) for d in c_d]

    max_val = np.max(reconstructed_signal)
    min_val = np.min(reconstructed_signal)
    mean_val = np.mean(reconstructed_signal)
    peak_val = np.max(np.abs(reconstructed_signal))
    peak_to_peak = max_val - min_val
    kurt = kurtosis(reconstructed_signal)
    skewness = skew(reconstructed_signal)
    rms = np.sqrt(np.mean(reconstructed_signal**2))
    rectified_mean = np.mean(np.abs(reconstructed_signal))
    waveform_factor = rms / rectified_mean if rectified_mean != 0 else 0.0
    peak_factor = peak_val / rms if rms != 0 else 0.0
    sqrt_abs_mean = np.mean(np.sqrt(np.abs(reconstructed_signal)))
    margin_factor = peak_val / (sqrt_abs_mean**2) if sqrt_abs_mean != 0 else 0.0
    energy_high = sum(np.sum(d**2) for d in c_d)
    energy_low = np.sum(c_a**2)

    return [
        max_val,
        min_val,
        mean_val,
        peak_val,
        peak_to_peak,
        kurt,
        skewness,
        rms,
        rectified_mean,
        waveform_factor,
        peak_factor,
        margin_factor,
        energy_high,
        energy_low,
    ]


def extract_frequency_features(reconstructed_signal: np.ndarray, fs: float = 1.0) -> list[float]:
    signal = np.asarray(reconstructed_signal, dtype=np.float64)
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), 256))
    psd = psd.astype(np.float64)
    sum_psd = np.sum(psd)
    if sum_psd == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    centroid_freq = np.sum(freqs * psd) / sum_psd
    mean_square_freq = np.sum((freqs**2) * psd) / sum_psd
    freq_variance = mean_square_freq - centroid_freq**2
    freq_band_energy = sum_psd
    freq_std = np.sqrt(freq_variance) if freq_variance >= 0 else 0.0
    return [centroid_freq, mean_square_freq, freq_variance, freq_band_energy, freq_std]


def extract_svd_features(c_a: np.ndarray, c_d: list[np.ndarray]) -> list[float]:
    c_a = np.asarray(c_a, dtype=np.float64).ravel()
    c_d_flat = [np.asarray(d, dtype=np.float64).ravel() for d in c_d]
    high_freq_components = np.concatenate(c_d_flat) if c_d_flat else np.array([], dtype=np.float64)

    low_freq_svd = 0.0
    if c_a.size:
        _, singular_values, _ = svd(c_a.reshape(-1, 1), full_matrices=False)
        if len(singular_values):
            low_freq_svd = float(singular_values[0])

    high_freq_svd = 0.0
    if high_freq_components.size:
        _, singular_values, _ = svd(high_freq_components.reshape(-1, 1), full_matrices=False)
        if len(singular_values):
            high_freq_svd = float(singular_values[0])

    return [high_freq_svd, low_freq_svd]


def build_feature_dataframe(
    segmented_frame: pd.DataFrame,
    label_column: str = "labels",
    wavelet: str = "db4",
    level: int = 4,
) -> pd.DataFrame:
    if label_column not in segmented_frame.columns:
        raise ValueError(f"Missing label column: {label_column}")

    labels = segmented_frame[label_column].to_numpy()
    samples = segmented_frame.drop(columns=[label_column]).to_numpy(dtype=np.float64)
    features_matrix: list[list[float]] = []

    for sample in samples:
        coeffs = pywt.wavedec(sample.ravel(), wavelet, level=level)
        c_a = coeffs[0]
        c_d = coeffs[1:]
        reconstructed = pywt.waverec(coeffs, wavelet)[: len(sample)]

        time_features = extract_time_features(reconstructed, c_a, c_d)
        freq_features = extract_frequency_features(reconstructed)
        svd_features = extract_svd_features(c_a, c_d)
        features_matrix.append(time_features + freq_features + svd_features)

    features_df = pd.DataFrame(features_matrix, columns=FEATURE_COLUMNS)
    features_df["故障类别"] = labels
    return features_df


def rank_feature_importance(
    features_df: pd.DataFrame,
    label_column: str = "故障类别",
    random_state: int = 42,
) -> pd.DataFrame:
    features = features_df.drop(columns=[label_column])
    labels = features_df[label_column]
    classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)
    classifier.fit(features, labels)
    return (
        pd.DataFrame(
            {"Feature": features.columns, "Importance": classifier.feature_importances_}
        )
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )


def save_dataframe(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False, encoding="utf-8-sig")
