# Models Guide

This directory stores local model artifacts used during inference and evaluation.

## Structure

- `baseline/`
  - Classical ML assets (for example `tfidf.pkl`, `lr.pkl`, `svm.pkl`, `rf.pkl`, `nb.pkl`, `xgb.pkl`, `ensemble.pkl`)
  - `config.json` with label ordering and metadata
- `llm/`
  - GGUF models loaded by fine-tuned engines
  - Typical filenames:
    - `llama_finetuned.gguf`
    - `qwen_finetuned.gguf`

## Related model assets outside this folder

- `codebert_model/` at project root stores CodeBERT tokenizer/config/weights files.

## How to regenerate baseline artifacts

```bash
python scripts/train_baseline.py
```

This writes baseline artifacts to `models/baseline/`.

## Reproducibility and source control

Large binary model files are local runtime assets and are typically excluded from GitHub commits.
