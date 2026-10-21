# Dataset Guide

This directory stores local dataset files used for training and evaluation.

## Expected files (local, usually not committed)

- `train.csv` - Training split for baseline and fine-tuning workflows
- `test.csv` - Holdout split for baseline evaluation
- `train.jsonl` - Instruction-tuning format for LLM training
- `test.jsonl` - Instruction-tuning format for LLM evaluation

Additional source files may exist for experiments (for example large raw exports).

## How files are used

- Baseline evaluation scripts read `test.csv`
  - `scripts/evaluate_baselines.py`
- LLM evaluation scripts read `test.jsonl`
  - `scripts/evaluate_llama_zeroshot.py`
  - `scripts/evaluate_llama_finetuned.py`
  - `scripts/evaluate_qwen_zeroshot.py`
  - `scripts/evaluate_qwen_finetuned.py`

## Related documentation

- LLM JSONL generation logic is documented in `context /LLM_TRAINING_LOGIC.md`.

## Reproducibility note

This folder is treated as local data storage. If you clone the repository without data files, place prepared split files here with the names above.
