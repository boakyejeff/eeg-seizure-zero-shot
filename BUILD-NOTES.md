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
*(pending — fill in after download completes)*

Dev sample (all from https://physionet.org/files/chbmit/1.0.0/, public, HTTP 200):

| file | size | seizures |
|------|------|----------|
| chb01_03.edf | ~42.4 MB | 1 (onset 362 s per staging) |
| chb01_04.edf | ~42.4 MB | 1 (onset 731 s) |
| chb02_16.edf | ~11.3 MB | ≥1 (in RECORDS-WITH-SEIZURES) |

Plan: train on chb01 (03+04), evaluate zero-shot on chb02_16 via
`cross_patient_eval` (2 patients -> train on chb01, test on chb02; also the
reverse LOPO fold).

### Results

*(fill in)*

## Pending / blocked

- (pending download) — downloads in progress at build time; ~96 MB total,
  under the ~130 MB budget.
- 3rd+ patient for true LOPO (currently 2 patients; harness supports N).
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
