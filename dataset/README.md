# Dataset Files (Local-only)

This folder stores the *prepared* dataset split files used by the project at runtime.

Most dataset files are large and are usually excluded from GitHub commits (see the repository `.gitignore`).

## Expected split files

The project expects these split files to exist with these exact names:

- `train.csv` (baseline training)
- `test.csv` (baseline evaluation)
- `train.jsonl` (LLM fine-tuning/instruction format)
- `test.jsonl` (LLM evaluation / holdout set)

Detailed usage is documented in:
- `DATASET_GUIDE.md`

## Google Drive (recommended)

Dataset files are stored in Google Drive here:

- Dataset folder: https://drive.google.com/drive/folders/1Hh2dVq1PnKy2Mpxzky96a-ZEbU1DyInP?usp=sharing

After downloading, place the files directly into this folder (`dataset/`) so the filenames match exactly.

