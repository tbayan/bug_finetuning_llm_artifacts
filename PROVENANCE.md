# Provenance

This folder was assembled on 2026-10-21 from the working copy of
`https://github.com/asset8k/bug-triage-app` at commit `b47c1d95e1e589c5f78efcc2b4fc40260c83623d`
(2026-04-02 16:46:13 +0500), plus files that were never committed there
(dataset splits, raw dump, predictions, metric logs, model files) and the arXiv manuscript.

## Changes made in this copy

- `utils/llm_metrics.py` moved to `records/legacy_scoring/` (see the README there).
- `api/server.py`: the evaluation screen no longer pads LLM metrics; the original file is kept as `records/legacy_scoring/server_original.py`.
- `screenshots/04-batch-evaluation-llms.png` (padded numbers) moved to `records/legacy_scoring/`.
- Thesis-only material (thesis text, thesis figure scripts, planning notes) was not copied.

Files written for this release are new: `README.md`, `PROVENANCE.md`, `requirements-experiments.txt`, `.gitignore`, `paper/` (manuscript and analysis script), `records/legacy_scoring/README.md` and `notes/`. Every other file is a byte-identical copy of its source.

## Commit history of the source repository

```
b47c1d9 2026-04-02 Asset  Add README screenshots for CyberTriage UI
9961bc4 2026-04-02 Asset  Document dataset location in Google Drive
fe3f549 2026-04-02 Asset  Remove Streamlit dependency and config
e8b66ff 2026-04-02 Asset  Document model artifact locations (Google Drive)
b2d5fbd 2026-04-02 Asset  Deployable CyberTriage thesis codebase
```

## SHA-256 of files larger than 1 MB

Files marked * are not in git (over GitHub's 100 MB limit); check downloads against these values.

| SHA-256 | File |
|---|---|
| `b2abb073e25b878c08694cc690a76591fb20b5aeb873a0184ba8246fcfe956ac` | `codebert_model/model.safetensors` * |
| `ca59632112d18bffd237dbfa1f5b5c35d92b30d0587165e473cea7f763d4ad62` | `codebert_model/tokenizer.json` |
| `19a3f8b2eedb171741f4f4fd920dc8ca833aee51ee106ff1c20f384558215087` | `dataset/bug_reports_train.csv` * |
| `221801139a7c9ce156617f5761dddd0c408daf8d7df1aa244b73b6cebbb4336e` | `dataset/test.csv` |
| `947747424c5b226dd8db0643e336fa685d741f1739e9cfa1ae1efe3c0fe2c247` | `dataset/test.jsonl` |
| `c983fe2be838e27aa040eac30ee89c5889324eb5b1e716c0cecac0e4c473f3ed` | `dataset/train.csv` |
| `c2eba8c481571f5721e8eae3302fc31f764f4050c4640059c5b69f88b58d4525` | `dataset/train.jsonl` |
| `d0adfc87ad123ca9abc49bdf87fa4a65cc4edd512ca895dc7af318da172978dd` | `models/baseline/ensemble.pkl` * |
| `bf202984a5f997941f3a91869769186c711d94928d094faafac2a94f4902f0ed` | `models/baseline/nb.pkl` |
| `35da4e8de337c17b16572361fbb9019906189099cb045fa5d569bba9167eb030` | `models/baseline/rf.pkl` * |
| `acbf1a400c19dc0dd5fc8bddffa98a207dd8ea3a2f47176536df84edf757bdbf` | `models/baseline/xgb.pkl` |
| `4e5e32859ed3a47843ae9c635e4c284c0ce6fb742359ed269db742f576c477b8` | `models/llm/llama_finetuned.gguf` * |
| `3c295af220b9053a59641646e7701200c8a49030d675b12e55841f0fdf0aa3b2` | `models/llm/qwen_finetuned.gguf` * |
