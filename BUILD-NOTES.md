# Build Notes — eeg-seizure-zero-shot

## What was built (2026-09-22)

Original code (no forks/copies). Repo layout:

```
eeg-seizure-zero-shot/
├── README.md                     overview, novel angle, quickstart, dataset citation/licensing
├── BUILD-NOTES.md                this file
├── requirements.txt              mne, numpy, pandas, scikit-learn, scipy
├── .gitignore                    excludes data/raw/, .venv/, results artifacts
├── config.json                   patient -> summary/file mapping (add patients here)
├── src/
│   ├── __init__.py
│   ├── loading.py      EDF loading (mne), chbXX-summary.txt parsing,
│   │                   .edf.seizures annotation parsing, seizure sample masks
│   ├── preprocess.py   bandpass 0.5–40 Hz, 60 Hz notch, resample to 128 Hz
│   ├── features.py     4 s non-overlapping windows; per-channel relative
│   │                   band powers (delta/theta/alpha/beta/gamma) via Welch
│   │                   + line length, variance, zero-crossings (184 feats @ 23 ch)
│   ├── model.py        sklearn pipelines (logreg / random forest,
│   │                   class_weight=balanced, StandardScaler)
│   └── eval.py         cross-patient harness: leave-one-patient-out over
│                       {patient: (X, y)}; metrics = AUC, sensitivity,
│                       specificity, false-alarm rate/hour
├── scripts/
│   ├── evaluate_cross_patient.py   main script: per-file load -> windows ->
│   │                                LOPO eval -> results/cross_patient_results.json
│   └── download_data.sh             fetches the dev sample (~96 MB, public)
├── data/raw/           gitignored: chb01/{chb01-summary.txt, chb01_03.edf(+.seizures),
│                                   chb01_04.edf(+.seizures)}, chb02/{...chb02_16...}
├── results/            gitignored JSON metrics
└── .venv/              gitignored local Python environment
```

git repo initialized locally (`git init`); nothing pushed to any remote.

## What ran

### Unit sanity check (synthetic EEG, before real data)
- 23 ch × 2 min @128 Hz, fake 6 Hz seizure 60–70 s -> 30 windows, 3 positive
  (60–72 s windows overlap ≥ 10% threshold), logistic regression train acc = 1.0.

### Smoke run (real CHB-MIT data)

Dev sample (all from https://physionet.org/files/chbmit/1.0.0/, public, HTTP 200):

| file | size | duration | seizures (from summary) |
|------|------|----------|--------------------------|
| chb01/chb01_03.edf | 42.4 MB | 3600 s | 1 (2996–3036 s) |
| chb01/chb01_04.edf | 42.4 MB | 3600 s | 1 (1467–1494 s) |
| chb02/chb02_16.edf | 11.3 MB | 959 s | 1 (130–212 s) |

Total: ~92 MB (under the ~130 MB budget). All EDFs verified: 23 channels,
256 Hz, loadable with mne.

Ran `python scripts/evaluate_cross_patient.py --data-dir data/raw --model logreg`
(and `--model rf`): full pipeline per file (load → bandpass/notch/resample →
4 s windows → 184 band-power+time features) then leave-one-patient-out.

### Results — zero-shot cross-patient (window-level, threshold 0.5)

Logistic regression:

| held-out patient | trained on | AUC | sensitivity | specificity | FA/h |
|---|---|---|---|---|---|
| chb01 (1800 windows, 18 seizure) | chb02 | **0.954** | 0.778 | 0.967 | 29.5 |
| chb02 (239 windows, 21 seizure) | chb01 | **0.886** | 0.905 | 0.642 | 293.7 |

Random forest (200 trees, class-balanced):

| held-out patient | AUC | sensitivity | specificity | FA/h |
|---|---|---|---|---|
| chb01 | 0.756 | 0.056 | 1.000 | 0.0 |
| chb02 | **0.981** | 0.571 | 1.000 | 0.0 |

Reading: the logreg model transfers well chb02→chb01 (AUC 0.954) but is
false-alarm heavy in the chb01→chb02 direction (spec 0.64) — the honest
cross-patient generalization gap this repo is built to expose. RF is
over-conservative (perfect specificity, poor sensitivity) — threshold tuning
on a validation fold is the obvious next step. Raw logs:
`results/smoke_run_logreg.log`, `results/smoke_run_rf.log`;
JSON metrics: `results/cross_patient_results.json`,
`results/cross_patient_results_rf.json`.

## Pending / blocked

- No blockers: mne 1.13.2 installed cleanly; no fallback reader was needed.
- 3rd+ patient for true multi-patient LOPO (currently 2 patients; harness
  supports N — just add entries to `config.json` and download via
  `scripts/download_data.sh` pattern).
- Fixes made during the build: renamed `src/io.py` → `src/loading.py`
  (stdlib `io` shadows it); switched `eval.py` to absolute imports;
  `load_edf` validates channel count not names (summary lists T8-P8 twice,
  MNE dedupes to T8-P8-0/T8-P8-1); labels read from `chbXX-summary.txt`
  because `.edf.seizures` sidecars are binary WFDB annotations.
- Suggested next steps: RF model comparison, threshold tuning on validation,
- Annotation source: `<file>.edf.seizures` sidecars are **binary WFDB**
  annotations, not text — labels are read from the plain-text
  `chbXX-summary.txt` instead (`src/loading.read_summary`). Verified
  intervals: chb01_03 (2996–3036 s), chb01_04 (1467–1494 s),
  chb02_16 (130–212 s). Note these differ from the staging report's
  onset claims (362 s / 731 s); summary text is authoritative here.
- PhysioNet downloads were slow/stalling (~160 KB/s); used `curl -C -`
  resume in parallel sessions instead of plain `wget`.
- Suggested next steps: RF model comparison, threshold tuning on validation,
  seizure-event (not window) scoring, more patients.
