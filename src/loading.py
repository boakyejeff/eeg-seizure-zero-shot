"""I/O utilities: EDF loading, CHB-MIT summary parsing, seizure annotation parsing.

CHB-MIT channels are bipolar montages (e.g. "FP1-F7"); the summary file lists
the exact channel order per patient. Seizure onsets come from the
`<file>.edf.seizures` sidecar files.
"""

from __future__ import annotations

import re
from pathlib import Path

import mne
import numpy as np


def read_summary(summary_path: str | Path) -> dict:
    """Parse a chbXX-summary.txt into a dict of metadata.

    Returns a dict with keys: sampling_rate, channels (list of bipolar
    channel names), and files (dict mapping edf filename -> dict with
    seizure intervals in seconds).
    """
    summary_path = Path(summary_path)
    text = summary_path.read_text()

    m = re.search(r"Data Sampling Rate:\s*([\d.]+)\s*Hz", text)
    sampling_rate = float(m.group(1)) if m else 256.0

    channels = re.findall(r"Channel \d+:\s*(.+)", text)

    files: dict[str, dict] = {}
    # Blocks look like: "File Name: chb01_03.edf\nFile Start Time: ...\n...
    # Number of Seizures in File: 1\nSeizure Start Time: 362 seconds\n..."
    blocks = re.split(r"\n(?=File Name:)", text)
    for block in blocks:
        name_m = re.search(r"File Name:\s*(\S+\.edf)", block)
        if not name_m:
            continue
        fname = name_m.group(1)
        starts = [float(x) for x in re.findall(r"Seizure Start Time:\s*([\d.]+)\s*seconds", block)]
        ends = [float(x) for x in re.findall(r"Seizure End Time:\s*([\d.]+)\s*seconds", block)]
        intervals = list(zip(starts, ends))
        files[fname] = {"intervals": intervals}
    return {"sampling_rate": sampling_rate, "channels": channels, "files": files}


def read_seizure_annotations(seizures_path: str | Path) -> list[tuple[float, float]]:
    """Parse a <file>.edf.seizures annotation file -> [(onset_s, offset_s), ...]."""
    text = Path(seizures_path).read_text()
    starts = [float(x) for x in re.findall(r"Seizure Start Time:\s*([\d.]+)\s*seconds", text)]
    ends = [float(x) for x in re.findall(r"Seizure End Time:\s*([\d.]+)\s*seconds", text)]
    return list(zip(starts, ends))


def load_edf(edf_path: str | Path, expected_channels: list[str] | None = None) -> mne.io.Raw:
    """Load a CHB-MIT EDF file with MNE, preloading into memory.

    If expected_channels is given (from the summary file), verify the
    EDF channel set matches; channel order follows the EDF file itself.
    """
    edf_path = Path(edf_path)
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    if expected_channels is not None:
        edf_names = set(raw.ch_names)
        missing = [c for c in expected_channels if c not in edf_names]
        if missing:
            raise ValueError(f"{edf_path.name}: missing channels vs summary: {missing}")
        raw.pick_channels(expected_channels, verbose="ERROR")
    return raw


def seizure_mask(n_samples: int, sfreq: float, intervals: list[tuple[float, float]]) -> np.ndarray:
    """Boolean mask of seizure samples given (onset, offset) intervals in seconds."""
    mask = np.zeros(n_samples, dtype=bool)
    for onset, offset in intervals:
        start = max(0, int(round(onset * sfreq)))
        stop = min(n_samples, int(round(offset * sfreq)))
        mask[start:stop] = True
    return mask
