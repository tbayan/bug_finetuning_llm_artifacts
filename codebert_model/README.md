# CodeBERT Model Files (Local-only)

This folder stores the CodeBERT model files required for the `codebert` engine.

Most weights are treated as local runtime artifacts and are typically excluded from Git commits (see `.gitignore`).

## Required files (expected names)

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors`
- `training_args.bin`

## Google Drive download links

The CodeBERT model files are stored in Google Drive here:

- CodeBERT folder: https://drive.google.com/drive/folders/1WqrYaTk9AW-z8xzVYYRrqauur9EG_V6q?usp=sharing

## After downloading

Place the downloaded files directly into this folder (`codebert_model/`) so the filenames match exactly.

Quick check:
```bash
ls -la
```

