"""
Evaluate all baseline models on the validation set.
Replicates training preprocessing exactly to avoid data leakage:
undersample Minor (if > MINOR_CAP) before split, same split (test_size, random_state),
TF-IDF transform only (no extra cleaning). Saves results/baseline_comparison.csv.

CLI: optional --model lr|svm|rf|nb|xgb|ensemble|codebert (run single model);
     optional --batch-size N (subsample validation set to N).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.metrics import compute_metrics

# New, isolated test CSV for baselines (already cleaned, mapped, balanced, and split).
# Prepared via dataset/1_prep_balance_split_csv.py
TEST_CSV = ROOT / "dataset" / "test.csv"
MODELS_DIR = ROOT / "models" / "baseline"
CONFIG_PATH = MODELS_DIR / "config.json"
RESULTS_DIR = ROOT / "results"
COMPARISON_CSV = RESULTS_DIR / "baseline_comparison.csv"

RANDOM_STATE = 42

MODEL_IDS = ("lr", "svm", "rf", "nb", "xgb", "ensemble")
ALL_MODEL_IDS = MODEL_IDS + ("codebert",)
MODEL_LABELS = {
    "lr": "Logistic Regression",
    "svm": "SVM",
    "rf": "Random Forest",
    "nb": "Naive Bayes",
    "xgb": "XGBoost",
    "ensemble": "Voting Ensemble",
    "codebert": "CodeBERT",
}
MODEL_NAME_TO_ID = {
    "lr": "lr",
    "logistic regression": "lr",
    "svm": "svm",
    "rf": "rf",
    "random forest": "rf",
    "nb": "nb",
    "naive bayes": "nb",
    "xgb": "xgb",
    "xgboost": "xgb",
    "ensemble": "ensemble",
    "voting ensemble": "ensemble",
    "codebert": "codebert",
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate baseline models.")
    p.add_argument("--model", choices=ALL_MODEL_IDS, default=None, help="Run only this model (default: all)")
    p.add_argument("--batch-size", type=int, default=None, help="Subsample validation set to N (default: use full set)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_models = [args.model] if args.model else list(ALL_MODEL_IDS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not TEST_CSV.exists():
        print(
            "New test CSV not found.\n"
            "Run dataset/1_prep_balance_split_csv.py first to create dataset/test.csv.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not CONFIG_PATH.exists():
        print("Run scripts/train_baseline.py first (config.json required).", file=sys.stderr)
        sys.exit(1)

    # Load test data (already balanced and split; no further undersampling or split).
    df = pd.read_csv(TEST_CSV)
    df = df.dropna(subset=["description", "mapped_severity"])
    X_val = df["description"].astype(str).values
    y_str = df["mapped_severity"].values

    # Label encoding: use config order so evaluation labels match training
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    labels_ordered = config["labels"]
    label_to_int = {c: i for i, c in enumerate(labels_ordered)}
    y_val = np.array([label_to_int.get(s, 0) for s in y_str])

    # Optional subsample (batch size) from the fixed test set.
    n_val_full = len(y_val)
    if args.batch_size is not None:
        if args.batch_size < n_val_full:
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(n_val_full, size=args.batch_size, replace=False)
            X_val = X_val[idx]
            y_val = y_val[idx]
            print(f"Subsampled test set to {len(y_val)} rows (batch-size={args.batch_size}).")
        elif args.batch_size > n_val_full:
            print(f"Requested batch-size {args.batch_size}; test set has {n_val_full} samples. Using full test set.")

    # Feature extraction for classic baselines only (CodeBERT uses raw text).
    vec = None
    X_val_tf = None
    vec_duration = 0.0
    classic_requested = any(mid in MODEL_IDS for mid in run_models)
    if classic_requested:
        vec = joblib.load(MODELS_DIR / "tfidf.pkl")
        t_vec_start = time.perf_counter()
        X_val_tf = vec.transform(X_val)
        vec_duration = time.perf_counter() - t_vec_start

    n_val = len(y_val)
    label_list = list(range(len(labels_ordered)))

    # Report which models will be evaluated (have .pkl) vs skipped
    have_pkl = [m for m in run_models if m in MODEL_IDS and (MODELS_DIR / f"{m}.pkl").exists()]
    missing_pkl = [m for m in run_models if m in MODEL_IDS and not (MODELS_DIR / f"{m}.pkl").exists()]
    if "codebert" in run_models:
        print("Evaluating (1): codebert")
    if have_pkl:
        print(f"Evaluating ({len(have_pkl)}): {', '.join(have_pkl)}")
    if missing_pkl:
        print(f"Skipped (no .pkl): {', '.join(missing_pkl)}")

    # Load existing rows so single-model runs overwrite prior values instead of duplicating.
    existing = {}
    if COMPARISON_CSV.exists():
        try:
            existing_df = pd.read_csv(COMPARISON_CSV)
            for _, r in existing_df.iterrows():
                model_raw = str(r.get("model", "")).strip()
                model_key = MODEL_NAME_TO_ID.get(model_raw.lower())
                if model_key in ALL_MODEL_IDS:
                    existing[model_key] = r.to_dict()
        except Exception:
            pass

    rows_by_id = dict(existing)
    for mid in run_models:
        if mid == "codebert":
            try:
                from src.engines.codebert import CodeBERTEngine
            except Exception as e:
                print(f"  Skip codebert (import error): {e}")
                rows_by_id.pop("codebert", None)
                continue

            engine = CodeBERTEngine()
            y_pred = []
            t_model_start = time.perf_counter()
            for x in X_val:
                try:
                    out = engine.predict(str(x))
                    pred = out.get("severity", labels_ordered[0])
                except Exception:
                    pred = labels_ordered[0]
                y_pred.append(label_to_int.get(str(pred), 0))
            y_pred = np.array(y_pred, dtype=int)
            m = compute_metrics(y_val, y_pred, labels=label_list)
            duration_sec = round(time.perf_counter() - t_model_start, 2)
            rows_by_id[mid] = {
                "model": MODEL_LABELS[mid],
                "n_val": n_val,
                "accuracy": round(m["accuracy"], 4),
                "macro_f1": round(m["macro_f1"], 4),
                "macro_precision": round(m["macro_precision"], 4),
                "macro_recall": round(m["macro_recall"], 4),
                "duration_sec": duration_sec,
            }
            print(f"  {mid}: F1={m['macro_f1']:.4f} duration={duration_sec}s")
            continue

        pkl_path = MODELS_DIR / f"{mid}.pkl"
        if not pkl_path.exists():
            print(f"  Skip {mid} (no {mid}.pkl)")
            rows_by_id.pop(mid, None)  # Do not keep stale results from previous runs
            continue
        # Measure total duration as: TF-IDF transform time + model inference time.
        t_model_start = time.perf_counter()
        clf = joblib.load(pkl_path)
        y_pred = clf.predict(X_val_tf)
        y_pred = np.asarray(y_pred)
        # Models may output ints (xgb, ensemble, or retrained lr/svm/rf/nb) or strings (older .pkl)
        if y_pred.dtype.kind in ("i", "u") or np.issubdtype(y_pred.dtype, np.integer):
            y_pred = y_pred.astype(int)
        else:
            y_pred = np.array([label_to_int.get(str(p), 0) for p in y_pred])
        m = compute_metrics(y_val, y_pred, labels=label_list)
        infer_duration = time.perf_counter() - t_model_start
        duration_sec = round(vec_duration + infer_duration, 2)
        rows_by_id[mid] = {
            "model": MODEL_LABELS[mid],
            "n_val": n_val,
            "accuracy": round(m["accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "macro_precision": round(m["macro_precision"], 4),
            "macro_recall": round(m["macro_recall"], 4),
            "duration_sec": duration_sec,
        }
        print(f"  {mid}: F1={m['macro_f1']:.4f} duration={duration_sec}s")

    # Preserve order: all MODEL_IDS that we have, then any extras from existing
    order = [m for m in ALL_MODEL_IDS if m in rows_by_id]
    for k in rows_by_id:
        if k not in order:
            order.append(k)
    rows = [rows_by_id[m] for m in order]

    if not rows:
        print("No model .pkl files found. Train models first.", file=sys.stderr)
        sys.exit(1)

    tab = pd.DataFrame(rows)
    tab.to_csv(COMPARISON_CSV, index=False)
    print(f"\nWrote {COMPARISON_CSV}")
    print(tab.to_string(index=False))
    if "macro_f1" in tab.columns:
        best = tab.loc[tab["macro_f1"].idxmax(), "model"]
        print(f"\nBest by macro F1: {best}")


if __name__ == "__main__":
    main()
