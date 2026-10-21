#!/usr/bin/env python3
"""
Rebuild every number, table and figure in the arXiv manuscript from the raw
records already stored in this repository. Nothing is retrained; the frozen
TF-IDF classifiers and CodeBERT are re-scored on the original test split so
that every model can be compared on the same report IDs.

Run from the repository root:
    python paper/analysis/rebuild_results.py            # reuse cached re-scores
    python paper/analysis/rebuild_results.py --rescore  # re-run TF-IDF + CodeBERT

Inputs
  dataset/bug_reports_train.csv     raw Hugging Face dump (AndressaStefany/bug-reports)
  dataset/1_prep_balance_split_csv.py  the original preparation script (imported, not re-implemented)
  dataset/train.csv, test.csv       the 16,000 / 4,000 split used in the thesis
  dataset/test.jsonl                LLM test inputs
  models/baseline/*.pkl             frozen TF-IDF classifiers
  codebert_model/                   frozen CodeBERT classifier
  results/*_predictions.csv         per-report LLM predictions (2,000 reports each)
  results/*_metrics.csv, results/baseline_comparison.csv   logged metrics and wall-clock times
  colab_scripts/*.ipynb             training logs (QLoRA and CodeBERT)

Outputs (under paper/)
  outputs/   JSON and CSV records behind every reported value
  tables/    LaTeX tabular bodies included by main.tex
  figs/      PDF figures included by main.tex

Every value is a measurement on stored records unless the record says
"estimate" (prevalence re-weighting), which is computed, not observed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# pandas 3 stores strings in Arrow arrays when pyarrow is installed (it comes with `datasets`);
# use the Python-backed storage so results do not depend on which packages are present.
pd.set_option("mode.string_storage", "python")

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
OUT = PAPER / "outputs"
TAB = PAPER / "tables"
FIG = PAPER / "figs"
for d in (OUT, TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

LABELS = ["Critical", "Major", "Minor", "Trivial"]
L2I = {c: i for i, c in enumerate(LABELS)}
RANDOM_STATE = 42
N_BOOT = 10_000
BOOT_SEED = 2026

# (key, display name, family). Order is the order used in every table and figure.
MODELS = [
    ("nb", "Naive Bayes", "TF-IDF"),
    ("lr", "Logistic regression", "TF-IDF"),
    ("svm", "Linear SVM", "TF-IDF"),
    ("rf", "Random forest", "TF-IDF"),
    ("xgb", "XGBoost", "TF-IDF"),
    ("ensemble", "Soft-voting ensemble", "TF-IDF"),
    ("codebert", "CodeBERT", "Encoder"),
    ("llama_zs", "Llama 3 8B", "Zero-shot"),
    ("qwen_zs", "Qwen 2.5 7B", "Zero-shot"),
    ("llama_ft", "Llama 3 8B", "Fine-tuned"),
    ("qwen_ft", "Qwen 2.5 7B", "Fine-tuned"),
]
KEYS = [m[0] for m in MODELS]
NAME = {m[0]: m[1] for m in MODELS}
FAMILY = {m[0]: m[2] for m in MODELS}
LLM_FILES = {
    "llama_zs": "llama_zeroshot",
    "qwen_zs": "qwen_zeroshot",
    "llama_ft": "llama_finetuned",
    "qwen_ft": "qwen_finetuned",
}
TFIDF_KEYS = ["nb", "lr", "svm", "rf", "xgb", "ensemble"]
BASELINE_CSV_NAME = {
    "lr": "Logistic Regression", "svm": "SVM", "rf": "rf", "nb": "nb",
    "xgb": "xgb", "ensemble": "ensemble", "codebert": "CodeBERT",
}
# Planned pairwise comparisons, chosen from the research questions (RQ1: LLM vs. encoder and TF-IDF; RQ2: FT vs. ZS).
PAIRS = [
    ("llama_ft", "codebert"), ("qwen_ft", "codebert"), ("llama_ft", "qwen_ft"),
    ("llama_ft", "llama_zs"), ("qwen_ft", "qwen_zs"), ("codebert", "ensemble"),
    ("llama_ft", "ensemble"), ("qwen_ft", "ensemble"),
    ("llama_zs", "ensemble"), ("qwen_zs", "ensemble"),
]


def label(key: str) -> str:
    return f"{NAME[key]} ({FAMILY[key].lower()})" if FAMILY[key] in ("Zero-shot", "Fine-tuned") else NAME[key]


def dump(obj, name: str) -> None:
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# --------------------------------------------------------------------------
# A. Dataset provenance and the split, rebuilt from the raw dump
# --------------------------------------------------------------------------
def load_prep_module():
    spec = importlib.util.spec_from_file_location("prep", ROOT / "dataset" / "1_prep_balance_split_csv.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rebuild_dataset() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    prep = load_prep_module()
    raw = pd.read_csv(ROOT / "dataset" / "bug_reports_train.csv")
    rec: dict = {"raw_rows": len(raw), "raw_columns": list(raw.columns)}
    rec["raw_label_counts"] = raw["bug_severity"].value_counts().to_dict()

    key = ["bug_id", "description", "bug_severity", "bug_status", "resolution"]
    dup = raw.duplicated(subset=key, keep="first")
    uniq = raw.loc[~dup]
    rec["exact_duplicate_rows"] = int(dup.sum())
    rec["unique_bug_ids"] = int(raw["bug_id"].nunique())
    rec["unique_reports"] = len(uniq)
    rec["unique_label_counts"] = uniq["bug_severity"].value_counts().to_dict()
    rec["unique_status_counts"] = uniq["bug_status"].value_counts().to_dict()
    rec["unique_resolution_counts"] = uniq["resolution"].value_counts().to_dict()
    rec["bug_id_min"], rec["bug_id_max"] = int(raw["bug_id"].min()), int(raw["bug_id"].max())
    low = uniq["description"].astype(str).str.lower()
    rec["unique_desc_mentions"] = {k: int(low.str.contains(k, regex=False).sum())
                                   for k in ("eclipse", "mozilla", "firefox", "thunderbird")}

    # The preparation exactly as executed by dataset/1_prep_balance_split_csv.py.
    df = raw[["bug_id", "description", "bug_severity", "bug_status", "resolution"]].copy()
    df["description"] = df["description"].astype(str)
    df["bug_severity"] = df["bug_severity"].astype(str)
    sev = df["bug_severity"].str.lower().str.strip()
    rec["rows_after_dropping_normal_enhancement"] = int((~sev.isin(prep.DROP_SEVERITIES)).sum())
    pool = prep.clean_and_map(df)
    rec["pool_rows"] = len(pool)
    rec["pool_rows_by_class"] = pool["mapped_severity"].value_counts().reindex(LABELS).to_dict()
    pool_u = pool.drop_duplicates(subset=key)
    rec["pool_unique_by_class"] = pool_u["mapped_severity"].value_counts().reindex(LABELS).to_dict()
    rec["pool_unique_total"] = len(pool_u)
    rec["pool_unique_share"] = {c: rec["pool_unique_by_class"][c] / len(pool_u) for c in LABELS}
    # Identical report texts filed under different bug IDs, and how many of them disagree on the label.
    txt = pool_u.assign(d=pool_u["description"].str.strip()).groupby("d").agg(
        n=("mapped_severity", "size"), k=("mapped_severity", "nunique"), ids=("bug_id", "nunique"))
    shared_txt = txt[txt["ids"] > 1]
    rec["pool_texts_shared_by_several_ids"] = {"texts": len(shared_txt), "reports": int(shared_txt["n"].sum())}
    rec["pool_texts_with_conflicting_labels"] = {"texts": int((shared_txt["k"] > 1).sum()),
                                                 "reports": int(shared_txt.loc[shared_txt["k"] > 1, "n"].sum())}

    balanced = prep.balance(pool)
    train, test = train_test_split(balanced, test_size=0.2, random_state=RANDOM_STATE,
                                   stratify=balanced["mapped_severity"])
    # The split files are not in the public repository; when present, the rebuilt split must match them.
    if (ROOT / "dataset" / "train.csv").exists() and (ROOT / "dataset" / "test.csv").exists():
        tr0, te0 = pd.read_csv(ROOT / "dataset" / "train.csv"), pd.read_csv(ROOT / "dataset" / "test.csv")
        same = lambda a, b: bool((a["description"].values == b["description"].values).all()
                                 and (a["mapped_severity"].values == b["mapped_severity"].values).all())
        rec["split_reproduces_train_csv"] = same(train, tr0)
        rec["split_reproduces_test_csv"] = same(test, te0)
        assert rec["split_reproduces_train_csv"] and rec["split_reproduces_test_csv"], "split did not reproduce"
    else:
        rec["split_reproduces_train_csv"] = rec["split_reproduces_test_csv"] = "not checked (split files absent)"

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    both = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
    rec["sample_rows"] = len(both)
    rec["sample_by_class"] = both["mapped_severity"].value_counts().reindex(LABELS).to_dict()
    rec["sample_duplicate_bug_ids"] = int(both["bug_id"].duplicated().sum())
    rec["sample_duplicate_descriptions"] = int(both["description"].duplicated().sum())
    train_desc = set(train["description"])
    test["dup_desc_in_train"] = test["description"].isin(train_desc)
    test["bug_id_in_train"] = test["bug_id"].isin(set(train["bug_id"]))
    rec["sample_distinct_bug_ids"] = int(both["bug_id"].nunique())
    rec["test_rows_repeating_test_bug_id"] = int(test["bug_id"].duplicated().sum())
    rec["test_desc_in_train"] = int(test["dup_desc_in_train"].sum())
    rec["test_bug_id_in_train"] = int(test["bug_id_in_train"].sum())
    rec["test_desc_in_train_by_class"] = test.loc[test["dup_desc_in_train"], "mapped_severity"].value_counts().reindex(LABELS, fill_value=0).to_dict()
    rec["sample_resolution_counts"] = both["resolution"].value_counts().to_dict()
    rec["sample_fixed_share"] = float((both["resolution"] == "FIXED").mean())
    # Text added after filing: a "NOTES:" section or dated initials such as "KM (10/2/2001 5:55:18 PM)".
    notes = both["description"].str.contains(r"\bNOTES:|\b[A-Z]{2,3} \(\d{1,2}/\d{1,2}/\d{4} \d", regex=True)
    rec["sample_desc_with_appended_notes"] = int(notes.sum())
    lens = both["description"].str.len()
    rec["desc_chars_median"] = float(lens.median())
    rec["desc_chars_p90"] = float(lens.quantile(0.9))
    rec["desc_over_4000_chars_share"] = float((lens > 4000).mean())
    rec["train_by_class"] = train["mapped_severity"].value_counts().reindex(LABELS).to_dict()
    rec["test_by_class"] = test["mapped_severity"].value_counts().reindex(LABELS).to_dict()

    ids = pd.concat([
        train.assign(split="train", row=np.arange(len(train)))[["split", "row", "bug_id", "mapped_severity", "resolution"]],
        test.assign(split="test", row=np.arange(len(test)))[["split", "row", "bug_id", "mapped_severity", "resolution", "dup_desc_in_train"]],
    ], ignore_index=True)
    return train, test, rec, ids


# --------------------------------------------------------------------------
# B. Per-report predictions on the 4,000-report test split
# --------------------------------------------------------------------------
def rescore_tfidf(test: pd.DataFrame) -> dict[str, np.ndarray]:
    mdir = ROOT / "models" / "baseline"
    vec = joblib.load(mdir / "tfidf.pkl")
    X = vec.transform(test["description"].astype(str).values)
    out = {}
    for k in TFIDF_KEYS:
        pred = np.asarray(joblib.load(mdir / f"{k}.pkl").predict(X))
        out[k] = pred.astype(int) if pred.dtype.kind in "iu" else np.array([L2I[str(p)] for p in pred])
        print(f"  re-scored {k}")
    return out


def rescore_codebert(test: pd.DataFrame) -> tuple[np.ndarray, dict]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(max(1, torch.get_num_threads()))
    mdir = ROOT / "codebert_model"
    tok = AutoTokenizer.from_pretrained(str(mdir))
    model = AutoModelForSequenceClassification.from_pretrained(str(mdir)).eval()
    texts = [str(t).strip() for t in test["description"].values]
    n_tok = np.array([len(tok(t, truncation=False)["input_ids"]) for t in texts])
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], truncation=True, padding=True, max_length=512, return_tensors="pt")
            preds.append(model(**enc).logits.argmax(-1).numpy())
            if (i // 32) % 25 == 0:
                print(f"  CodeBERT {i}/{len(texts)}")
    info = {
        "codebert_params": int(sum(p.numel() for p in model.parameters())),
        "codebert_test_tokens_median": float(np.median(n_tok)),
        "codebert_test_over_512_share": float((n_tok > 512).mean()),
    }
    return np.concatenate(preds).astype(int), info


def load_llm_predictions(n_test: int, released: pd.DataFrame | None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # The subsample drawn by scripts/evaluate_*.py (same generator, same seed, same call).
    idx = np.random.default_rng(RANDOM_STATE).choice(n_test, size=2000, replace=False)
    have_raw = (ROOT / "dataset" / "test.jsonl").exists() and all(
        (ROOT / "results" / f"{stem}_predictions.csv").exists() for stem in LLM_FILES.values())
    if not have_raw:
        # Public repository: the raw prediction files hold report text and are not released;
        # use the per-report labels in outputs/test_predictions.csv instead.
        assert released is not None, "need results/*_predictions.csv or outputs/test_predictions.csv"
        assert set(np.flatnonzero(released["in_llm_subset"].values)) == set(idx)
        return idx, {k: np.array([L2I[v] if v else -1 for v in released[k].fillna("")]) for k in LLM_FILES}
    rows = [json.loads(l) for l in open(ROOT / "dataset" / "test.jsonl", encoding="utf-8")]
    inputs = [r["input"].strip() for r in rows]
    assert len(inputs) == n_test
    out = {}
    for key, stem in LLM_FILES.items():
        p = pd.read_csv(ROOT / "results" / f"{stem}_predictions.csv", keep_default_na=False)
        assert len(p) == len(idx)
        assert all(inputs[i][:200] == s for i, s in zip(idx, p["text_snippet"])), f"{key}: snippet mismatch"
        assert all(json.loads(rows[i]["output"])["severity"] == t for i, t in zip(idx, p["true_severity"]))
        full = np.full(n_test, -1, dtype=int)
        full[idx] = [L2I[s] for s in p["predicted_severity"]]
        out[key] = full
    return idx, out


# --------------------------------------------------------------------------
# C. Metrics, uncertainty and tests
# --------------------------------------------------------------------------
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(4), zero_division=0)
    return {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, p)),
        "macro_precision": float(pr.mean()), "macro_recall": float(rc.mean()), "macro_f1": float(f1.mean()),
        "per_class": {c: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])}
                      for i, c in enumerate(LABELS)},
        "predicted_counts": {c: int((p == i).sum()) for i, c in enumerate(LABELS)},
        "confusion": confusion_matrix(y, p, labels=range(4)).tolist(),
    }


def boot_counts(y: np.ndarray, p: np.ndarray, idx: np.ndarray) -> dict:
    yt, yp = y[idx], p[idx]
    tp = np.stack([((yt == c) & (yp == c)).sum(1) for c in range(4)], 1)
    fp = np.stack([((yt != c) & (yp == c)).sum(1) for c in range(4)], 1)
    fn = np.stack([((yt == c) & (yp != c)).sum(1) for c in range(4)], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        rec_crit = tp[:, 0] / (tp[:, 0] + fn[:, 0])
    return {"macro_f1": f1.mean(1), "acc": (yt == yp).mean(1), "critical_recall": rec_crit}


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m, adj, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


# --------------------------------------------------------------------------
# D. Training logs (Colab notebooks)
# --------------------------------------------------------------------------
def parse_training_logs() -> dict:
    out = {}
    for key, nb in (("llama_ft", "llama_tuning_script.ipynb"), ("qwen_ft", "qwen_tuning_script.ipynb"),
                    ("llama_16bit_lora", "llama_training_unquantized.ipynb")):
        cells = json.load(open(ROOT / "colab_scripts" / nb, encoding="utf-8"))["cells"]
        src = "".join(cells[1]["source"])
        text = ""
        html = ""
        for o in cells[1]["outputs"]:
            text += "".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", ""))
            html += "".join(o.get("data", {}).get("text/html", ""))
        plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
        losses = [float(v) for v in re.findall(r"(?<= )\d+ (\d+\.\d+)(?= )", plain.split("Training Loss", 1)[1])]
        rt = re.search(r"\[(\d+)/(\d+) ([\d:]+), Epoch", plain)
        tp = re.search(r"Trainable parameters = ([\d,]+) of ([\d,]+)", text)
        ver = re.search(r"Unsloth ([\d.]+): Fast \w+ patching\. Transformers: ([\d.]+)", text)
        torch_v = re.search(r"Torch: ([\w.+]+)\.", text)
        gpu = re.search(r"(NVIDIA [\w-]+)\. Num GPUs", text)
        grab = lambda pat: (re.search(pat, src).group(1) if re.search(pat, src) else None)
        steps = len(losses)
        out[key] = {
            "notebook": nb, "base_checkpoint": grab(r'model_name = "([^"]+)"'),
            "load_in_4bit": grab(r"load_in_4bit = (\w+)"), "rank": grab(r"r = (\d+),"),
            "lora_alpha": grab(r"lora_alpha = (\d+)"), "lora_dropout": grab(r"lora_dropout = ([\d.]+)"),
            "target_modules": re.findall(r'"(\w+_proj)"', src), "learning_rate": grab(r"learning_rate\s+= ([\de.-]+)"),
            "optimizer": grab(r'optim\s+= "(\w+)"'), "scheduler": grab(r'lr_scheduler_type\s+= "(\w+)"'),
            "warmup_ratio": grab(r"warmup_ratio\s+= ([\d.]+)"), "weight_decay": grab(r"weight_decay\s+= ([\d.]+)"),
            "per_device_batch": grab(r"per_device_train_batch_size = (\d+)"),
            "grad_accum": grab(r"gradient_accumulation_steps = (\d+)"), "epochs": grab(r"NUM_EPOCHS\s+= (\d+)"),
            "max_seq_length": grab(r"max_seq_length = (\d+)"), "seed": grab(r"seed\s+= (\d+)"),
            "export_quantization": grab(r'quantization_method = "(\w+)"'),
            "completion_only_loss": ("train_on_responses_only" in src) or ("DataCollatorForCompletionOnlyLM" in src),
            "trainable_params": int(tp.group(1).replace(",", "")) if tp else None,
            "total_params": int(tp.group(2).replace(",", "")) if tp else None,
            "steps_logged": steps, "wall_clock": rt.group(3) if rt else None,
            "unsloth": ver.group(1) if ver else None, "transformers": ver.group(2) if ver else None,
            "torch": torch_v.group(1) if torch_v else None, "gpu": gpu.group(1) if gpu else None,
            "loss_mean_first_100": float(np.mean(losses[:100])) if steps else None,
            "loss_mean_last_100": float(np.mean(losses[-100:])) if steps else None,
            "loss_mean_by_epoch": [float(np.mean(c)) for c in np.array_split(losses, 3)] if steps else None,
        }
    cb = json.load(open(ROOT / "colab_scripts" / "codebert_colab_finetune.ipynb", encoding="utf-8"))["cells"][0]
    html = "".join("".join(o.get("data", {}).get("text/html", "")) for o in cb["outputs"])
    plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    rows = re.findall(r" (\d) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", plain)
    out["codebert"] = {
        "wall_clock": re.search(r"\[3000/3000 ([\d:]+),", plain).group(1),
        "per_epoch_eval_on_test": [{"epoch": int(r[0]), "train_loss": float(r[1]), "eval_loss": float(r[2]),
                                    "accuracy": float(r[3]), "macro_f1": float(r[4])} for r in rows],
        "selection": "load_best_model_at_end=True, metric_for_best_model=f1_macro, eval_dataset=test split",
    }
    return out


def gguf_metadata() -> dict:
    """Architecture and quantisation type read from the deployed GGUF headers (files are not in git)."""
    import struct
    sizes = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    want = ("general.file_type", "block_count", "embedding_length", "feed_forward_length",
            "attention.head_count", "attention.head_count_kv")
    out = {}
    for key, name in (("llama_ft", "llama_finetuned.gguf"), ("qwen_ft", "qwen_finetuned.gguf")):
        path = ROOT / "models" / "llm" / name
        if not path.exists():
            out[key] = "file not present"
            continue
        with open(path, "rb") as f:
            assert f.read(4) == b"GGUF"
            f.read(4)
            _, n_kv = struct.unpack("<QQ", f.read(16))
            rstr = lambda: f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")

            def rval(t):
                if t == 8:
                    return rstr()
                if t == 9:
                    at, n = struct.unpack("<IQ", f.read(12))
                    for _ in range(n):
                        rstr() if at == 8 else f.read(struct.calcsize(sizes[at]))
                    return None
                return struct.unpack(sizes[t], f.read(struct.calcsize(sizes[t])))[0]

            rec = {}
            for _ in range(n_kv):
                k = rstr()
                v = rval(struct.unpack("<I", f.read(4))[0])
                if k.endswith(want):
                    rec[k.split(".", 1)[1] if not k.startswith("general") else k] = v
        rec["file_type_name"] = {15: "Q4_K_M"}.get(rec.get("general.file_type"), "other")
        out[key] = rec
    return out


def target_token_counts() -> dict:
    """Length of the trained Stage-1 answer under each base tokenizer (needs the public tokenizer files)."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return {}
    out = {}
    for key, name in (("llama_ft", "unsloth/llama-3-8b"), ("qwen_ft", "Qwen/Qwen2.5-7B")):
        try:
            tok = AutoTokenizer.from_pretrained(name)
        except Exception as e:  # offline: keep the record explicit rather than silent
            out[key] = f"tokenizer unavailable: {e.__class__.__name__}"
            continue
        out[key] = {c: len(tok(json.dumps({"severity": c}), add_special_tokens=False)["input_ids"]) for c in LABELS}
    return out


# --------------------------------------------------------------------------
# E. LaTeX tables
# --------------------------------------------------------------------------
def f3(x: float) -> str:
    return f"{x:.3f}"


def write(name: str, text: str) -> None:
    (TAB / name).write_text(text, encoding="utf-8")


def tables(R: dict) -> None:
    D = R["dataset"]
    raw_r, raw_u = D["raw_label_counts"], D["unique_label_counts"]
    lines = [
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}} lrrlrr @{}}",
        r"\toprule",
        r"Label in the dump & Rows & Unique reports & Mapped class & Pool rows & Sampled (train / test) \\",
        r"\midrule",
    ]
    pool = D["pool_rows_by_class"]
    tr, te = D["train_by_class"], D["test_by_class"]
    rowspec = [("blocker", "Critical"), ("critical", "Critical"), ("major", "Major"), ("minor", "Minor"),
               ("trivial", "Trivial")]
    shown = set()
    for lab, cls in rowspec:
        first = cls not in shown
        shown.add(cls)
        lines.append(f"\\texttt{{{lab}}} & {raw_r[lab]:,} & {raw_u[lab]:,} & {cls if first else ''} & "
                     f"{f'{pool[cls]:,}' if first else ''} & {f'{tr[cls]:,} / {te[cls]:,}' if first else ''} \\\\")
    for lab in ("normal", "enhancement"):
        lines.append(f"\\texttt{{{lab}}} & {raw_r[lab]:,} & {raw_u[lab]:,} & excluded & --- & --- \\\\")
    lines += [r"\midrule",
              f"Total & {D['raw_rows']:,} & {D['unique_reports']:,} & & {D['pool_rows']:,} & "
              f"{sum(tr.values()):,} / {sum(te.values()):,} \\\\",
              r"\bottomrule", r"\end{tabular*}"]
    write("dataset.tex", "\n".join(lines) + "\n")

    S = R["shared"]
    full = R["full4000"]
    lines = [r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}} llccccc @{}}", r"\toprule",
             r"Family & Model & Acc. & Macro-P & Macro-R & Macro-F1 [95\% CI] & F1, dedup. \\",
             r"\midrule"]
    best = max(S[k]["macro_f1"] for k in KEYS)
    prev = None
    for k in KEYS:
        m, ci = S[k], R["boot_ci"][k]["macro_f1"]
        fam = FAMILY[k] if FAMILY[k] != prev else ""
        if prev is not None and FAMILY[k] != prev:
            lines.append(r"\midrule")
        prev = FAMILY[k]
        f1s = f3(m["macro_f1"])
        if abs(m["macro_f1"] - best) < 1e-12:
            f1s = r"\textbf{" + f1s + "}"
        lines.append(f"{fam} & {NAME[k]} & {f3(m['accuracy'])} & {f3(m['macro_precision'])} & "
                     f"{f3(m['macro_recall'])} & {f1s} [{f3(ci[0])}, {f3(ci[1])}] & {f3(R['no_dup'][k]['macro_f1'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    write("main_results.tex", "\n".join(lines) + "\n")

    lines = [r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}} lcccccc @{}}", r"\toprule",
             r"Model & Accuracy & Macro-P & Macro-R & Macro-F1 & Logged F1 & Logged time (s) \\", r"\midrule"]
    for k in TFIDF_KEYS + ["codebert"]:
        m, lg = full[k], R["logged_baselines"][k]
        lines.append(f"{NAME[k]} & {f3(m['accuracy'])} & {f3(m['macro_precision'])} & {f3(m['macro_recall'])} & "
                     f"{f3(m['macro_f1'])} & {lg['macro_f1']:.4f} & {lg['duration_sec']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    write("full_test.tex", "\n".join(lines) + "\n")

    show = ["ensemble", "codebert", "llama_zs", "qwen_zs", "llama_ft", "qwen_ft"]
    lines = [r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}} l cc cc cc cc c @{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Critical} & \multicolumn{2}{c}{Major} & \multicolumn{2}{c}{Minor} & "
             r"\multicolumn{2}{c}{Trivial} & Critical recall \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"Model & P & R & P & R & P & R & P & R & 95\% CI \\", r"\midrule"]
    for k in show:
        pc = S[k]["per_class"]
        ci = R["boot_ci"][k]["critical_recall"]
        cells = " & ".join(f"{f3(pc[c]['precision'])} & {f3(pc[c]['recall'])}" for c in LABELS)
        lines.append(f"{label(k)} & {cells} & [{f3(ci[0])}, {f3(ci[1])}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    write("per_class.tex", "\n".join(lines) + "\n")

    lines = [r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}} l c c c c @{}}", r"\toprule",
             r"Comparison (A vs.\ B) & $\Delta$ macro-F1 [95\% CI] & $p_{\text{Holm}}$, bootstrap & "
             r"$b$ / $c$ & $p_{\text{Holm}}$, McNemar \\", r"\midrule"]
    floor = len(R["pairwise"]) / N_BOOT
    SHORT = {"llama_ft": "Llama 3 FT", "qwen_ft": "Qwen 2.5 FT", "llama_zs": "Llama 3 ZS",
             "qwen_zs": "Qwen 2.5 ZS", "codebert": "CodeBERT", "ensemble": "TF-IDF ensemble"}
    # Bootstrap p-values cannot go below 1/N_BOOT; after Holm that floor is m/N_BOOT.
    fmt_p = lambda p, lo=None: (f"$\\le${lo:.3f}" if lo is not None and p <= lo + 1e-12
                                else ("$<$0.001" if p < 0.001 else f"{p:.3f}"))
    sg = lambda x: f"${x:+.3f}$"
    for r in R["pairwise"]:
        lines.append(f"{SHORT[r['a']]} vs.\\ {SHORT[r['b']]} & {sg(r['delta_f1'])} [{sg(r['ci'][0])}, {sg(r['ci'][1])}] & "
                     f"{fmt_p(r['p_boot_holm'], floor)} & {r['b_only_a_correct']} / {r['c_only_b_correct']} & "
                     f"{fmt_p(r['p_mcnemar_holm'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    write("pairwise.tex", "\n".join(lines) + "\n")

    T = R["training"]
    L, Q, C = T["llama_ft"], T["qwen_ft"], T["codebert"]
    cbp = R["codebert_inputs"]["codebert_params"]
    both = lambda s: f"\\multicolumn{{2}}{{>{{\\raggedright\\arraybackslash}}p{{8.6cm}}}}{{{s}}}"
    rows = [
        ("Initial checkpoint", r"\texttt{microsoft/codebert-base}", f"\\texttt{{{L['base_checkpoint']}}}",
         f"\\texttt{{{Q['base_checkpoint']}}}"),
        ("Trainable / total parameters", f"{cbp:,} / {cbp:,}", f"{L['trainable_params']:,} / {L['total_params']:,}",
         f"{Q['trainable_params']:,} / {Q['total_params']:,}"),
        ("Adaptation", "full fine-tuning, classification head",
         both(f"LoRA $r={L['rank']}$, $\\alpha={L['lora_alpha']}$, dropout {L['lora_dropout']}, on q, k, v, o, gate, up, down")),
        ("Precision in training", "fp32 weights, fp16 mixed precision",
         both("block projections frozen in NF4 with double quantisation (bitsandbytes); embeddings, norms and LM head in bf16; bf16 compute")),
        ("Objective", "cross-entropy on 4 labels", both("next-token cross-entropy on the whole sequence (prompt not masked)")),
        ("Optimiser, learning rate", "AdamW, $2\\times10^{-5}$, linear decay, no warm-up",
         both(f"{L['optimizer'].replace('_', chr(92) + '_')}, $1\\times10^{{-4}}$, cosine, 10\\% warm-up")),
        ("Weight decay", "0.01", both(L["weight_decay"])),
        ("Batch, epochs, steps", "16, 3, 3,000", both(f"{int(L['per_device_batch']) * int(L['grad_accum'])} "
                                                      f"($2\\times4$ accumulation), {L['epochs']}, {L['steps_logged']:,}")),
        ("Input limit", "512 tokens", both(f"4,000 characters; {L['max_seq_length']}-token context")),
        ("Seed", "42", both(L["seed"])),
        ("Hardware, software", "Colab GPU (type not logged)",
         both(f"A100-SXM4-40GB; Unsloth {L['unsloth']}, Transformers {L['transformers'].rstrip('.')}, PyTorch {L['torch']}")),
        ("Training wall clock", f"0:{C['wall_clock']}", L["wall_clock"], Q["wall_clock"]),
        ("Mean loss, epochs 1 / 2 / 3", "not logged per epoch",
         " / ".join(f"{v:.3f}" for v in L["loss_mean_by_epoch"]), " / ".join(f"{v:.3f}" for v in Q["loss_mean_by_epoch"])),
        ("Deployed artefact", "PyTorch weights (fp32)", both("LoRA merged into 16-bit weights, GGUF Q4\\_K\\_M (llama.cpp)")),
    ]
    rl = r">{\raggedright\arraybackslash}"
    lines = [r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}} " + rl + "p{2.6cm} " + rl + "p{3.5cm} " + rl + "p{4.2cm} " + rl + r"p{4.2cm} @{}}", r"\toprule",
             r" & CodeBERT & Llama 3 8B (QLoRA) & Qwen 2.5 7B (QLoRA) \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    write("training.tex", "\n".join(lines) + "\n")

    lat = R["latency"]
    tf = [lat[k]["ms_per_report"] for k in TFIDF_KEYS]
    rows = [
        ("TF-IDF classifiers (six)", "4,000", f"{min(lat[k]['total_s'] for k in TFIDF_KEYS):.2f}--{max(lat[k]['total_s'] for k in TFIDF_KEYS):.2f}",
         f"{min(tf):.2f}--{max(tf):.2f} ms", "load model, vectorise, predict"),
        ("CodeBERT", "4,000", f"{lat['codebert']['total_s']:.1f}", f"{lat['codebert']['ms_per_report']:.0f} ms",
         "per-report tokenise and forward pass"),
    ]
    for k in ("llama_zs", "qwen_zs", "llama_ft", "qwen_ft"):
        cover = ("HTTP call to local Ollama, 150-token cap, 50 ms pause" if k.endswith("_zs")
                 else "llama-cpp-python call, 32-token cap")
        rows.append((label(k), f"{lat[k]['n']:,}", f"{lat[k]['total_s']:,.0f}", f"{lat[k]['s_per_report']:.2f} s", cover))
    lines = [r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}} lrrr>{\raggedright\arraybackslash}p{4.6cm} @{}}", r"\toprule",
             r"Model & Reports & Total (s) & Per report & Timed loop covers \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular*}"]
    write("latency.tex", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# F. Figures (one style for the whole paper)
# --------------------------------------------------------------------------
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GREYS = ["#ffffff", "#e9e9e9", "#cfcfcf", "#a8a8a8", "#7f7f7f", "#585858", "#333333"]


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150,
    })


def fig_f1(R: dict) -> None:
    import matplotlib.pyplot as plt
    style()
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    header = {"TF-IDF": "TF-IDF classifiers", "Encoder": "Encoder", "Zero-shot": "Zero-shot LLMs",
              "Fine-tuned": "QLoRA-fine-tuned LLMs"}
    ys, labels, bold = [], [], []
    y, prev = 0.0, None
    for k in KEYS:
        if FAMILY[k] != prev:
            if prev is not None:
                y += 0.4
            ys.append(y); labels.append(header[FAMILY[k]]); bold.append(True)
            y += 1
            prev = FAMILY[k]
        m, (lo, hi) = R["shared"][k]["macro_f1"], R["boot_ci"][k]["macro_f1"]
        ax.plot([lo, hi], [y, y], color="#555555", lw=1.4, solid_capstyle="butt", zorder=2)
        ax.scatter([m], [y], s=26, color="black", zorder=3)
        ax.text(hi + 0.006, y, f"{m:.3f}", va="center", ha="left", fontsize=7, color="black")
        ys.append(y); labels.append("   " + NAME[k]); bold.append(False)
        y += 1
    ax.set_yticks(ys, labels)
    for t, b in zip(ax.get_yticklabels(), bold):
        t.set_fontweight("bold" if b else "normal")
        t.set_color("black")
    ax.set_ylim(ys[-1] + 0.7, -0.7)
    ax.axvline(0.25, color="#777777", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(0.254, ys[-1] + 0.25, "uniform random guess (0.25)", fontsize=6.8, color="#444444", va="center", ha="left")
    ax.set_xlim(0.2, 0.62)
    ax.set_xlabel("Macro-F1 on the 2,000 shared test reports, with 95% bootstrap interval")
    ax.grid(axis="x", color="#e6e6e6", lw=0.6, zorder=0)
    ax.tick_params(axis="y", length=0)
    fig.savefig(FIG / "fig_macro_f1.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def fig_confusion(R: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    style()
    cmap = LinearSegmentedColormap.from_list("greys", GREYS)
    show = ["ensemble", "codebert", "llama_zs", "qwen_zs", "llama_ft", "qwen_ft"]
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.7), sharex=True, sharey=True)
    for ax, k in zip(axes.flat, show):
        cm = np.array(R["shared"][k]["confusion"], dtype=float)
        rn = cm / cm.sum(1, keepdims=True)
        ax.imshow(rn, cmap=cmap, vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=7.6,
                        color="#fcfcfb" if rn[i, j] > 0.5 else INK, fontweight="bold" if i == j else "normal")
        ax.set_title(label(k).replace(" (", "\n("), color=INK, pad=4)
        ax.set_xticks(range(4), [c[:4] + "." if len(c) > 5 else c for c in LABELS])
        ax.set_yticks(range(4), LABELS)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        ax.tick_params(which="minor", length=0)
    for ax in axes[1]:
        ax.set_xlabel("Predicted")
    for ax in axes[:, 0]:
        ax.set_ylabel("True")
    fig.subplots_adjust(wspace=0.12, hspace=0.42)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Share of the true class (row)", color=INK2)
    cb.outline.set_visible(False)
    fig.savefig(FIG / "fig_confusion.pdf", metadata={"CreationDate": None})
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true", help="re-run TF-IDF and CodeBERT inference")
    args = ap.parse_args()

    print("A. rebuilding dataset and split from the raw dump ...")
    train, test, D, ids = rebuild_dataset()
    y = test["mapped_severity"].map(L2I).values
    n = len(test)

    print("B. per-report predictions ...")
    cache = OUT / "test_predictions.csv"
    extra = {}
    released = pd.read_csv(cache, keep_default_na=False) if cache.exists() else None
    if args.rescore or released is None:
        preds = rescore_tfidf(test)
        preds["codebert"], extra = rescore_codebert(test)
        dump(extra, "codebert_inputs.json")
    else:
        assert (released["bug_id"].values == test["bug_id"].values).all()
        preds = {k: released[k].map(L2I).values for k in TFIDF_KEYS + ["codebert"]}
        extra = json.load(open(OUT / "codebert_inputs.json"))
    idx, llm = load_llm_predictions(n, released)
    preds.update(llm)
    shared = np.zeros(n, dtype=bool)
    shared[idx] = True
    ids.loc[ids["split"] == "test", "in_llm_subset"] = shared
    ids.to_csv(OUT / "split_ids.csv", index=False)
    tp = test[["bug_id", "mapped_severity"]].rename(columns={"mapped_severity": "true"}).copy()
    tp.insert(0, "test_row", np.arange(n))
    tp["in_llm_subset"] = shared
    tp["dup_desc_in_train"] = test["dup_desc_in_train"].values
    for k in KEYS:
        tp[k] = [LABELS[v] if v >= 0 else "" for v in preds[k]]
    tp.to_csv(cache, index=False)

    print("C. metrics, intervals and tests ...")
    R: dict = {"dataset": D, "codebert_inputs": extra}
    logged = pd.read_csv(ROOT / "results" / "baseline_comparison.csv").set_index("model")
    R["logged_baselines"] = {k: logged.loc[BASELINE_CSV_NAME[k]].to_dict() for k in TFIDF_KEYS + ["codebert"]}
    R["full4000"] = {k: metrics(y, preds[k]) for k in TFIDF_KEYS + ["codebert"]}
    R["full4000_matches_logged"] = {
        k: bool(abs(R["full4000"][k]["macro_f1"] - R["logged_baselines"][k]["macro_f1"]) < 6e-4
                and abs(R["full4000"][k]["accuracy"] - R["logged_baselines"][k]["accuracy"]) < 6e-4)
        for k in TFIDF_KEYS + ["codebert"]}
    ys = y[shared]
    R["shared_class_counts"] = {c: int((ys == i).sum()) for i, c in enumerate(LABELS)}
    R["shared"] = {k: metrics(ys, preds[k][shared]) for k in KEYS}
    logged_llm = {k: pd.read_csv(ROOT / "results" / f"{v}_metrics.csv").iloc[0].to_dict() for k, v in LLM_FILES.items()}
    R["logged_llm"] = logged_llm
    R["shared_llm_matches_logged"] = {k: bool(abs(R["shared"][k]["macro_f1"] - logged_llm[k]["macro_f1"]) < 6e-4)
                                      for k in LLM_FILES}

    # If a triager reviewed every report predicted Major or above: share of Critical reports
    # covered, and share of all reports in that queue (balanced set).
    R["review_major_or_above"] = {
        k: {"critical_covered": (R["shared"][k]["confusion"][0][0] + R["shared"][k]["confusion"][0][1])
                                / R["shared"][k]["per_class"]["Critical"]["support"],
            "queue_share": (R["shared"][k]["predicted_counts"]["Critical"]
                            + R["shared"][k]["predicted_counts"]["Major"]) / int(shared.sum())}
        for k in KEYS}

    keep = shared & ~test["dup_desc_in_train"].values
    R["no_dup_n"] = int(keep.sum())
    R["no_dup"] = {k: metrics(y[keep], preds[k][keep]) for k in KEYS}
    # Shared reports whose text (and label) also appear in training: how often does each model reproduce them?
    seen = shared & test["dup_desc_in_train"].values
    R["accuracy_seen_vs_unseen"] = {k: {"seen": float((preds[k][seen] == y[seen]).mean()),
                                        "unseen": float((preds[k][keep] == y[keep]).mean()),
                                        "n_seen": int(seen.sum())} for k in KEYS}
    R["shared_repeated_descriptions"] = int(test.loc[shared, "description"].duplicated().sum())
    # scripts/train_baseline.py fits the TF-IDF models on 80% of the training split; how many of the
    # seen shared reports were in that fitting set?
    from sklearn.preprocessing import LabelEncoder
    y_tr = LabelEncoder().fit_transform(train["mapped_severity"].values)
    fit_part, _, _, _ = train_test_split(train["description"].astype(str).values, y_tr, test_size=0.2,
                                         stratify=y_tr, random_state=RANDOM_STATE)
    R["seen_in_tfidf_fit_set"] = int(test.loc[seen, "description"].isin(set(fit_part)).sum())

    rng = np.random.default_rng(BOOT_SEED)
    bidx = rng.integers(0, len(ys), size=(N_BOOT, len(ys)))
    B = {k: boot_counts(ys, preds[k][shared], bidx) for k in KEYS}
    q = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    R["boot_ci"] = {k: {"macro_f1": q(B[k]["macro_f1"]), "critical_recall": q(B[k]["critical_recall"])} for k in KEYS}
    rows = []
    for a, b in PAIRS:
        d = B[a]["macro_f1"] - B[b]["macro_f1"]
        p_boot = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        ca, cb_ = preds[a][shared] == ys, preds[b][shared] == ys
        nb_, nc_ = int((ca & ~cb_).sum()), int((~ca & cb_).sum())
        rows.append({"a": a, "b": b, "delta_f1": R["shared"][a]["macro_f1"] - R["shared"][b]["macro_f1"],
                     "ci": q(d), "p_boot": max(p_boot, 1 / N_BOOT), "b_only_a_correct": nb_, "c_only_b_correct": nc_,
                     "delta_acc": R["shared"][a]["accuracy"] - R["shared"][b]["accuracy"],
                     "p_mcnemar": float(binomtest(nb_, nb_ + nc_, 0.5).pvalue)})
    for col in ("p_boot", "p_mcnemar"):
        for r, v in zip(rows, holm([r[col] for r in rows])):
            r[col + "_holm"] = v
    R["pairwise"] = rows

    # Estimate (not a measurement): metrics if the four classes appeared at the prevalence of the
    # cleaned pool instead of 25% each, assuming per-class behaviour carries over.
    prior = np.array([D["pool_unique_by_class"][c] for c in LABELS], dtype=float)
    prior /= prior.sum()

    def reweight(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        w = cm / cm.sum(-1, keepdims=True) * prior[:, None]
        d = np.diagonal(w, axis1=-2, axis2=-1)
        col = w.sum(-2)
        prec = np.divide(d, col, out=np.zeros_like(d), where=col > 0)
        rec = d / w.sum(-1)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(d), where=(prec + rec) > 0)
        return prec, rec, f1.mean(-1), d.sum(-1)

    est = {}
    for k in KEYS:
        prec, rec, mf1, acc = reweight(np.array(R["shared"][k]["confusion"], dtype=float))
        est[k] = {"macro_f1": float(mf1), "accuracy": float(acc),
                  "precision": dict(zip(LABELS, prec.tolist())), "recall": dict(zip(LABELS, rec.tolist()))}
    # Uncertainty of the re-weighted gap between fine-tuned Llama 3 and CodeBERT (same resamples as above).
    boot_rw = {}
    for k in ("llama_ft", "codebert"):
        yt, yp = ys[bidx], preds[k][shared][bidx]
        cms = np.stack([np.stack([((yt == i) & (yp == j)).sum(1) for j in range(4)], -1) for i in range(4)], -2)
        boot_rw[k] = reweight(cms.astype(float))[2]
    diff = boot_rw["llama_ft"] - boot_rw["codebert"]
    # Review queue "predicted Major or above" if the classes arrived at the pool's mix.
    for k in KEYS:
        cm = np.array(R["shared"][k]["confusion"], dtype=float)
        R["review_major_or_above"][k]["queue_share_pool_mix"] = float(
            (prior[:, None] * cm / cm.sum(1, keepdims=True))[:, :2].sum())
    R["prevalence_estimate"] = {"prior": dict(zip(LABELS, prior.tolist())), "models": est,
                                "llama_ft_minus_codebert": {"estimate": est["llama_ft"]["macro_f1"] - est["codebert"]["macro_f1"],
                                                            "ci": q(diff)}}

    lat = {}
    for k in TFIDF_KEYS + ["codebert"]:
        lg = R["logged_baselines"][k]
        lat[k] = {"n": int(lg["n_val"]), "total_s": float(lg["duration_sec"]),
                  "ms_per_report": 1000 * float(lg["duration_sec"]) / float(lg["n_val"])}
    for k, lg in logged_llm.items():
        lat[k] = {"n": int(lg["n_val"]), "total_s": float(lg["duration_sec"]),
                  "s_per_report": float(lg["duration_sec"]) / float(lg["n_val"])}
    R["latency"] = lat
    R["training"] = parse_training_logs()
    R["stage1_target_tokens"] = target_token_counts()
    R["deployed_gguf"] = gguf_metadata()
    R["bootstrap"] = {"resamples": N_BOOT, "seed": BOOT_SEED, "unit": "report, paired across models"}
    dump(R, "results.json")

    print("D. tables and figures ...")
    tables(R)
    fig_f1(R)
    fig_confusion(R)

    print("\nChecks")
    print("  split reproduces train/test CSV:", D["split_reproduces_train_csv"], D["split_reproduces_test_csv"])
    print("  re-scored 4,000-report metrics match logged CSV:", R["full4000_matches_logged"])
    print("  LLM metrics on shared subset match logged CSV:", R["shared_llm_matches_logged"])
    print("\nShared-subset macro-F1")
    for k in KEYS:
        print(f"  {label(k):40s} {R['shared'][k]['macro_f1']:.4f}  CI {R['boot_ci'][k]['macro_f1']}")


if __name__ == "__main__":
    main()
