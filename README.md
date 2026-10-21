# Bug severity classification with fine-tuned open LLMs: code, data records and paper

This repository holds everything behind the preprint *Bug Severity Classification with
Fine-Tuned Open LLMs: A Paired Comparison with TF-IDF and CodeBERT Baselines on Eclipse
Reports* (Kanatov, Bayan, Boranbayev): the data preparation, training and evaluation
code, the stored per-report predictions and logs, the analysis script that produces
every number, table and figure in the paper, the manuscript itself, and CyberTriage, the
web tool described in Section 6. Where each file came from is recorded in
[PROVENANCE.md](PROVENANCE.md).

## Layout

| Path | Contents |
|---|---|
| `paper/` | Manuscript (`main.tex`, `references.bib`, Elsevier CAS class files), generated `tables/` and `figs/`, `Makefile` |
| `paper/analysis/rebuild_results.py` | The script behind every reported number |
| `paper/outputs/` | Its outputs: `results.json` (all values), `split_ids.csv` (bug ID, split, label, resolution, shared-subset and duplicate flags), `test_predictions.csv` (per-report labels of all 11 configurations) |
| `dataset/` | Preparation scripts; the raw dump and the 16,000 / 4,000 split are rebuilt locally (below) |
| `results/` | Logged metrics and wall-clock times (`*_metrics.csv`, `baseline_comparison.csv`) |
| `colab_scripts/` | QLoRA and CodeBERT training notebooks, with their training logs |
| `scripts/`, `src/`, `utils/` | Baseline training, evaluation scripts and inference engines |
| `models/`, `codebert_model/` | Trained models (large files not in git, see below) |
| `api/`, `frontend/`, `run.sh` | CyberTriage back end (FastAPI) and front end (React) |
| `records/legacy_scoring/` | The scoring routine that padded the thesis-era LLM scores, kept only as a record (see its README) |

## Reproduce the paper's numbers

1. Download the Hugging Face dataset `AndressaStefany/bug-reports` (train split) and save it
   as `dataset/bug_reports_train.csv`, for example with
   `datasets.load_dataset("AndressaStefany/bug-reports", split="train").to_pandas().to_csv(..., index=False)`.
   The file used for the paper is listed with its SHA-256 in PROVENANCE.md. A fresh export
   may differ byte-wise, so check the content instead: the script reports the label counts
   of Table 2 and compares the split it rebuilds with `dataset/train.csv` and `test.csv`.
2. Install the Python packages in `requirements-experiments.txt`.
3. From the repository root run `python paper/analysis/rebuild_results.py`. It rebuilds the
   split from the raw dump with `dataset/1_prep_balance_split_csv.py`, checks it against
   `dataset/train.csv` and `test.csv` when present, recomputes every metric, interval and
   test, and rewrites `paper/tables/`, `paper/figs/` and `paper/outputs/`. With
   `--rescore` it also re-runs the TF-IDF classifiers and CodeBERT, which needs the model
   files.
4. `make -C paper` compiles the manuscript; `make -C paper arxiv` builds the arXiv bundle.

Running step 3 from only the files tracked in git plus the downloaded dump regenerates
all tables and figures byte for byte.

## Continue or extend the experiments

The repository is self-contained for new runs:

| Step | Code | Needs |
|---|---|---|
| Rebuild the dataset and split | `dataset/1_prep_balance_split_csv.py`, then `dataset/2_convert_to_jsonl.py` for the LLM records | the Hugging Face dump |
| Train and evaluate TF-IDF baselines | `scripts/train_baseline.py`, `scripts/evaluate_baselines.py` | CPU |
| Fine-tune CodeBERT | `colab_scripts/codebert_colab_finetune.ipynb` | one GPU |
| QLoRA fine-tuning of Llama 3 / Qwen 2.5 | `colab_scripts/llama_tuning_script.ipynb`, `qwen_tuning_script.ipynb` (16-bit variant: `llama_training_unquantized.ipynb`) | one GPU (A100 40 GB used) |
| Evaluate LLMs | `scripts/evaluate_{llama,qwen}_{zeroshot,finetuned}.py` | Ollama (zero-shot), llama-cpp-python and the GGUF files (fine-tuned) |
| Recompute the paper's numbers | `paper/analysis/rebuild_results.py` | the files above |

A check run in October 2026: `2_convert_to_jsonl.py` regenerated `train.jsonl` byte for byte,
and Naive Bayes retrained from scratch with `train_baseline.py` chose the same setting
(alpha = 10) and scored macro-F1 0.4452 on the test split against 0.4439 for the stored
model; the small gap comes from library versions. The stored models reproduce the
logged scores exactly. When extending the study, keep the raw LLM responses, record the
Ollama version and model digests, and use a validation split for every tuning decision;
the paper lists these as gaps of the present runs.

## Files not in git

Files with report text are kept out of git because Eclipse reports contain names and
e-mail addresses: the split (`dataset/train.csv`, `test.csv`, `*.jsonl`), the raw LLM
prediction files with 200-character report snippets (`results/*_predictions.csv`), and
the app's local history. The split is rebuilt from the dump in step 3, and the per-report
labels of every model are in `paper/outputs/test_predictions.csv`.

The trained model files are not redistributed: the two fine-tuned GGUF files (Q4_K_M,
4.9 and 4.7 GB), `codebert_model/model.safetensors`, `models/baseline/rf.pkl` and
`models/baseline/ensemble.pkl`. They are also too large for GitHub. Their SHA-256 values
are listed in PROVENANCE.md, and every number in the paper can be recomputed from the
released per-report predictions without them. The raw dump is public on Hugging Face
(step 1 above).

## Hardware

Everything except training runs on an ordinary laptop without a GPU: rebuilding the
dataset and split, training and evaluating the TF-IDF baselines, CodeBERT inference and
the analysis script. Fine-tuned GGUF models run on a laptop with llama-cpp-python (the
original runs used an Apple M1 Pro; plan for at least 8 GB of RAM), and the zero-shot runs
need Ollama. QLoRA and CodeBERT training need one GPU; the notebooks run on Google Colab
(an A100 40 GB was used).

## Licences

The code is under the MIT licence (`LICENSE`). The Hugging Face dataset states no licence;
its reports come from the public Eclipse Bugzilla. The fine-tuned weights are not
redistributed; anyone who shares fine-tuned Llama 3 weights must follow the Meta Llama 3
Community Licence. Qwen 2.5 7B is under Apache 2.0 and CodeBERT under the MIT licence.

## What was not recorded

The raw LLM responses were not stored, only the parsed labels. The Ollama version, model
digests and context setting of the zero-shot runs, and peak GPU memory during training,
were not logged. The paper states these gaps.
