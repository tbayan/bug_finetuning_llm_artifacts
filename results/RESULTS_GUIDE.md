# Results Guide

This directory stores generated evaluation outputs and local runtime state.

## Typical files

- `baseline_comparison.csv` - baseline model metrics table
- `llama_zeroshot_metrics.csv`, `llama_zeroshot_predictions.csv`
- `llama_finetuned_metrics.csv`, `llama_finetuned_predictions.csv`
- `qwen_zeroshot_metrics.csv`, `qwen_zeroshot_predictions.csv`
- `qwen_finetuned_metrics.csv`, `qwen_finetuned_predictions.csv`
- `history_entries.json` - persisted ingestion history (backend local storage)

## How files are generated

- Baselines:
  - `python scripts/evaluate_baselines.py`
- LLM evaluations:
  - `python scripts/evaluate_llama_zeroshot.py --batch-size 50`
  - `python scripts/evaluate_llama_finetuned.py --batch-size 50`
  - `python scripts/evaluate_qwen_zeroshot.py --batch-size 50`
  - `python scripts/evaluate_qwen_finetuned.py --batch-size 50`
- History persistence:
  - Written automatically by backend API endpoints under `api/server.py`

## Reproducibility note

These are generated outputs and local runtime files. They are usually excluded from Git and reproduced per environment.
