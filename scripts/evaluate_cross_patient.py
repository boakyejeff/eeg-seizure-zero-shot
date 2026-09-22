#!/usr/bin/env python3
"""Cross-patient (zero-shot) seizure detection on CHB-MIT.

Usage:
    python scripts/evaluate_cross_patient.py --data-dir data/raw --model logreg

Pipeline per EDF file:
    load -> bandpass/notch/resample -> 4 s windows -> band-power features
Then leave-one-patient-out over patients defined in config.json:
    train on all patients except one, test on the held-out patient.

Report metrics: AUC, sensitivity, specificity, false alarms/hour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval import cross_patient_eval, print_report  # noqa: E402
from features import extract_features  # noqa: E402
from loading import load_edf, read_seizure_annotations, read_summary  # noqa: E402
from preprocess import preprocess  # noqa: E402


def load_patient(patient_dir: Path, summary_path: Path, window_s: float):
    """Load all EDF files for one patient -> (X, y) windowed dataset."""
    meta = read_summary(summary_path)
    Xs, ys = [], []
    for edf_path in sorted(patient_dir.glob("*.edf")):
        # Labels come from the summary file (plain text). The .edf.seizures
        # sidecar files are binary WFDB annotations; the summary mirrors
        # their seizure intervals in readable form.
        intervals = meta["files"][edf_path.name]["intervals"]
        print(f"  {edf_path.name}: {len(intervals)} seizure(s), loading...", flush=True)
        raw = load_edf(edf_path, expected_channels=meta["channels"])
        raw = preprocess(raw)
        data = raw.get_data()
        X, y = extract_features(data, raw.info["sfreq"], intervals, window_s=window_s)
        Xs.append(X)
        ys.append(y)
        print(f"    {X.shape[0]} windows, {y.sum()} seizure windows", flush=True)
    return np.vstack(Xs), np.concatenate(ys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", default="logreg", choices=["logreg", "rf"])
    parser.add_argument("--window-s", type=float, default=4.0)
    parser.add_argument("--out", default="results/cross_patient_results.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config = json.loads((data_dir / args.config).read_text()
                        if (data_dir / args.config).exists()
                        else Path(args.config).read_text())

    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for patient_id, patient_cfg in config["patients"].items():
        print(f"patient {patient_id}")
        summary_path = data_dir / patient_cfg["summary"]
        patient_path = data_dir / patient_cfg["files_dir"]
        datasets[patient_id] = load_patient(patient_path, summary_path, args.window_s)

    print("\nleave-one-patient-out evaluation:")
    results = cross_patient_eval(datasets, model_name=args.model, window_s=args.window_s)
    print_report(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
