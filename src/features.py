"""Windowing + handcrafted features.

Each 4 s window yields one feature vector: per-channel relative band powers
(delta/theta/alpha/beta/gamma) from a Welch PSD estimate, plus simple
time-domain stats (line length, variance, zero-crossings). Labels come from
seizure intervals: a window is positive if any of its samples overlap a
seizure (fractional threshold is configurable).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


def _window_features(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Feature vector for one window of shape (n_channels, n_samples)."""
    n_channels = x.shape[0]
    freqs, psd = welch(x, fs=sfreq, nperseg=min(int(sfreq * 2), x.shape[1]))
    band_powers = []
    for lo, hi in BANDS.values():
        band_powers.append(psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=1))
    band_powers = np.asarray(band_powers).T  # (channels, bands)
    total = band_powers.sum(axis=1, keepdims=True)
    relative = band_powers / np.maximum(total, 1e-12)

    line_length = np.abs(np.diff(x, axis=1)).sum(axis=1)
    variance = x.var(axis=1)
    zero_cross = ((x[:, :-1] * x[:, 1:]) < 0).sum(axis=1)

    return np.concatenate(
        [relative.ravel(), line_length, variance, zero_cross]
    )


def extract_features(
    data: np.ndarray,
    sfreq: float,
    intervals: list[tuple[float, float]],
    window_s: float = 4.0,
    positive_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice (n_channels, n_samples) into non-overlapping windows.

    Returns (X, y): X is (n_windows, n_features), y is 0/1 per window.
    A window is positive if the fraction of its duration inside a seizure
    interval exceeds positive_fraction.
    """
    n_samples = data.shape[1]
    win = int(window_s * sfreq)
    n_windows = n_samples // win
    feats, labels = [], []
    for w in range(n_windows):
        seg = data[:, w * win : (w + 1) * win]
        feats.append(_window_features(seg, sfreq))
        win_start, win_end = w * window_s, (w + 1) * window_s
        overlap = sum(
            max(0.0, min(win_end, off) - max(win_start, on)) for on, off in intervals
        )
        labels.append(1 if overlap / window_s >= positive_fraction else 0)
    return np.asarray(feats), np.asarray(labels, dtype=int)
