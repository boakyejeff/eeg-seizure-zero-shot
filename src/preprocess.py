"""Preprocessing: bandpass, notch, resample.

Standard scalp-EEG cleaning for seizure work: keep the 0.5-40 Hz band
(where most ictal power lives), remove 60 Hz mains interference, and
downsample to 128 Hz to halve the data volume before feature extraction.
"""

from __future__ import annotations

import mne


def preprocess(
    raw: mne.io.Raw,
    l_freq: float = 0.5,
    h_freq: float = 40.0,
    notch: float = 60.0,
    target_sfreq: float = 128.0,
) -> mne.io.Raw:
    """Bandpass + notch filter and resample. Operates in place on a copy."""
    raw = raw.copy()
    raw.filter(l_freq, h_freq, verbose="ERROR")
    raw.notch_filter(notch, verbose="ERROR")
    raw.resample(target_sfreq, verbose="ERROR")
    return raw
