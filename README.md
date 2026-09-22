# EEG Seizure Detection, Zero-Shot Across Patients

Cross-patient (zero-shot) seizure detection on the **CHB-MIT scalp EEG
database**: train a classifier on one patient's EEG and evaluate it on a
**held-out patient's** EEG — no patient-specific tuning allowed.

## Why this matters

Most published seizure-detection results train *and* test on the same
patients, which hides a hard truth: EEG is wildly patient-specific, and a
model trained on patient A often fails on patient B. The novel angle here is
a clean **cross-patient evaluation harness**: leave-one-patient-out, so the
reported AUC / sensitivity / false-alarm-rate numbers reflect true
generalization to unseen patients — the thing that actually matters for a
deployable seizure monitor.

## Pipeline

```
EDF load (mne) → bandpass 0.5–40 Hz + 60 Hz notch → resample 128 Hz
→ 4 s non-overlapping windows → per-channel band powers
   (δ/θ/α/β/γ) + line length / variance / zero-crossings
→ logistic regression or random forest (class-balanced)
→ train on patient A, evaluate on patient B (zero-shot)
```

- **Preprocessing:** `src/preprocess.py`
- **Features:** `src/features.py` (handcrafted, no deep learning)
- **Baseline models:** `src/model.py` (sklearn)
- **Centerpiece:** `scripts/evaluate_cross_patient.py` — generalizes to
  leave-one-patient-out as you add patients to `config.json`

## Quickstart

```bash
git clone <this-repo> && cd eeg-seizure-zero-shot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/download_data.sh   # ~96 MB dev sample, public, no login
python scripts/evaluate_cross_patient.py --data-dir data/raw --model logreg
```

The smoke run trains on `chb01` (2 one-hour files), validates on the held-out
`chb01` file, and tests **zero-shot on `chb02`** — results land in
`results/cross_patient_results.json`.

## Dataset & citation

Data: **CHB-MIT Scalp EEG Database** (Shoeb et al.), PhysioNet
[https://physionet.org/content/chbmit/1.0.0/](https://physionet.org/content/chbmit/1.0.0/),
23 subjects, 664 EDF recordings, 198 annotated seizures.

Citation: Shoeb, A. *Application of machine learning to epileptic seizure
onset detection and treatment*. PhD thesis, MIT, 2009. Goldberger et al.,
PhysioBank, *Circulation* 101(23), 2000.

**Licensing:** CHB-MIT is publicly available for non-commercial research use
under the [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/about/licenses/);
the public CHB-MIT dataset itself is open-access — see the
[project page](https://physionet.org/content/chbmit/1.0.0/) for the exact
terms and attribution requirements. Use for research only.

## Repo layout

```
src/           loading, preprocess, features, model, eval
scripts/       evaluate_cross_patient.py (main), download_data.sh
config.json    patient -> files mapping (add patients here)
data/raw/      gitignored EDF downloads
results/       gitignored JSON metrics
```

## What the smoke run showed

See `BUILD-NOTES.md` for the actual numbers from the end-to-end run on the
dev sample (train on `chb01`, test zero-shot on `chb02`).
