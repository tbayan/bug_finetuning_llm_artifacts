# Baseline Models (Local-only)

This folder contains the classical ML baseline artifacts (pickled models + feature artifacts) used by CyberTriage at runtime.

These files are treated as local runtime assets and are typically excluded from Git commits (see the repository `.gitignore`).

## Required files (expected names)

- `config.json`
- `tfidf.pkl`
- `lr.pkl`
- `svm.pkl`
- `rf.pkl`
- `nb.pkl`
- `xgb.pkl`
- `ensemble.pkl`

## Google Drive download links

The baseline model artifacts are stored in Google Drive here:

- Baseline models folder: https://drive.google.com/drive/folders/1W59o9fKTFAbfb09XlMoq6Xt6AoATqUyg?usp=sharing

## After downloading

Place the downloaded files directly into this folder (`models/baseline/`) so the filenames match exactly.

Quick check:
```bash
ls -la
```

