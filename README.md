# Bug severity classification with fine-tuned open LLMs

Code and data records for the study *Bug Severity Classification with Fine-Tuned
Open LLMs: A Paired Comparison with TF-IDF and CodeBERT Baselines on Eclipse Reports*
(Kanatov, Bayan, Fazli, Kappassova, Boranbayev). The preprint will be linked here once it is on arXiv.

The study is still under way: new experiments for a journal version are being added, so the
`main` branch will move ahead of the preprint. Each version of the paper gets a git tag from
which its results can be reproduced.

## What the study does

Bug reports carry a severity label (Critical, Major, Minor, Trivial) that decides what gets
fixed first. We test whether open LLMs fine-tuned on one GPU predict this label better than
classic text classifiers. Eleven models are compared on the **same 2,000 Eclipse bug
reports**: six TF-IDF classifiers, a fine-tuned CodeBERT, and Llama 3 8B and Qwen 2.5 7B,
each used zero-shot and after QLoRA fine-tuning.

| Model | Macro-F1 (95% interval) |
|---|---|
| TF-IDF classifiers (best of six) | 0.489 (0.467–0.510) |
| CodeBERT, fine-tuned | 0.518 (0.497–0.539) |
| Llama 3 8B, zero-shot | 0.305 (0.286–0.325) |
| Llama 3 8B, QLoRA fine-tuned | **0.526** (0.504–0.548) |
| Qwen 2.5 7B, QLoRA fine-tuned | 0.485 (0.463–0.506) |

Fine-tuned Llama 3 and CodeBERT cannot be told apart on this test set, and CodeBERT is about
27 times faster per report. The repository also contains **CyberTriage**, a web tool that
classifies a report and exports it to Jira (Section 6 of the paper).

## Reproduce the paper's results in three steps

No GPU is needed. Tested with Python 3.12 on Linux, from a fresh clone.

**1. Get the code and install the packages**

```bash
git clone https://github.com/tbayan/bug_finetuning_llm_artifacts.git
cd bug_finetuning_llm_artifacts
python3 -m venv .venv
source .venv/bin/activate
pip install -r analysis/requirements.txt
```

**2. Download the data** (public Hugging Face dataset, about 465 MB)

```bash
python -c "from datasets import load_dataset; load_dataset('AndressaStefany/bug-reports', split='train').to_pandas().to_csv('dataset/bug_reports_train.csv', index=False)"
sha256sum dataset/bug_reports_train.csv
# expected: 19a3f8b2eedb171741f4f4fd920dc8ca833aee51ee106ff1c20f384558215087
```

**3. Rebuild every number, table and figure**

```bash
python analysis/rebuild_results.py
```

The script rebuilds the 20,000-report sample and the train/test split from the raw data,
recomputes every metric, interval and test, and rewrites `outputs/`, including the paper's
tables (`outputs/tables/`) and figures (`outputs/figs/`). At the end it prints its checks: the two metric checks should say `True`
for every model. To confirm you got exactly the paper's results:

```bash
git diff --stat outputs     # prints nothing if everything matches
```

## Run or extend the experiments

Install the full set of packages with `pip install -r requirements-experiments.txt`.

| Step | Code | Hardware |
|---|---|---|
| Build the dataset and split | `dataset/1_prep_balance_split_csv.py`, then `dataset/2_convert_to_jsonl.py` for the LLM training records | laptop |
| Train and test the TF-IDF classifiers | `scripts/train_baseline.py`, `scripts/evaluate_baselines.py` | laptop |
| Fine-tune CodeBERT | `colab_scripts/codebert_colab_finetune.ipynb` | one GPU (Google Colab) |
| Fine-tune Llama 3 / Qwen 2.5 with QLoRA | `colab_scripts/llama_tuning_script.ipynb`, `colab_scripts/qwen_tuning_script.ipynb` | one GPU (an A100 40 GB was used) |
| Run the LLMs | `scripts/evaluate_{llama,qwen}_{zeroshot,finetuned}.py` | laptop with at least 8 GB RAM; [Ollama](https://ollama.com) for zero-shot, llama-cpp-python for fine-tuned models |
| Start CyberTriage | `./run.sh` (starts the back end in `api/` and the front end in `frontend/`) | laptop |

The notebooks keep their training logs, so the exact settings of every run can be read from
them. When you run new experiments, store the raw model outputs and use a validation split
for tuning; the paper lists both as gaps of the original runs.

## What is not included, and why

- **Trained model files** (fine-tuned GGUF files for Llama 3 and Qwen 2.5, CodeBERT weights,
  two large TF-IDF models). They are not redistributed and are too large for GitHub. You do
  not need them to reproduce the paper, because every per-report prediction is released in
  `outputs/test_predictions.csv`. Their checksums are in `PROVENANCE.md`.
- **Report text** (the split files and raw prediction files). Eclipse bug reports contain
  names and e-mail addresses, so they are rebuilt from the public data in step 3 instead of
  being stored here.

## Repository layout

| Path | Contents |
|---|---|
| `analysis/` | `rebuild_results.py`, which computes every number, table and figure of the paper |
| `outputs/` | every computed value (`results.json`), the bug IDs of the split (`split_ids.csv`), each model's prediction per test report (`test_predictions.csv`), and the paper's tables and figures |
| `dataset/` | scripts that build the dataset and the split |
| `colab_scripts/` | training notebooks with their logs |
| `scripts/`, `src/`, `utils/` | training and evaluation scripts, model wrappers |
| `results/` | metrics and run times logged by the original evaluation runs |
| `models/`, `codebert_model/` | small model files and configurations (large files not included) |
| `api/`, `frontend/` | CyberTriage web tool |
| `records/legacy_scoring/` | the routine that inflated the LLM scores in the earlier master's thesis, kept only as a record (see its README) |
| `PROVENANCE.md` | where every file came from, with checksums |

## History

The study started from Asset Kanatov's master's thesis at Nazarbayev University (2026); he
and the co-authors wrote the first draft together. The thesis code is at
[asset8k/bug-triage-app](https://github.com/asset8k/bug-triage-app), kept unchanged; this
repository was built from it (see `PROVENANCE.md`). The work has changed considerably since:
every result was recomputed from the stored records and new analyses were added after an
internal review. Where this repository and the thesis differ, the values here hold; the
thesis's LLM scores were inflated by a scoring routine (see `records/legacy_scoring/`).

The first author has since taken up other work. With his written agreement, the co-authors
at the School of Computing and Artificial Intelligence, Nazarbayev University, continue the
study: the preprint, the new experiments and the revisions for a journal version.

## Licence

Code: MIT (`LICENSE`). Data: the public Hugging Face dataset `AndressaStefany/bug-reports`
(Eclipse Bugzilla reports). Base models: Meta Llama 3 Community Licence, Apache 2.0 for
Qwen 2.5, MIT for CodeBERT.
