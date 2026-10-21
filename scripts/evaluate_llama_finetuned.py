"""
Evaluate fine-tuned Llama 3 (GGUF) on the holdout test set.
Uses dataset/test.jsonl only (from dataset/2_convert_to_jsonl.py); optional --batch-size to subsample.
Writes results/llama_finetuned_metrics.csv for the LLM Results Table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.metrics import compute_metrics

TEST_JSONL = ROOT / "dataset" / "test.jsonl"
CONFIG_PATH = ROOT / "models" / "baseline" / "config.json"
RESULTS_DIR = ROOT / "results"
METRICS_CSV = RESULTS_DIR / "llama_finetuned_metrics.csv"
PREDICTIONS_CSV = RESULTS_DIR / "llama_finetuned_predictions.csv"

RANDOM_STATE = 42
DEFAULT_BATCH_SIZE = 50


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate fine-tuned Llama 3 GGUF model.")
    p.add_argument("--batch-size", type=int, default=None, help=f"Subsample size (default: {DEFAULT_BATCH_SIZE})")
    p.add_argument("--debug", action="store_true", help="Print raw model generation to the terminal in real-time.")
    return p.parse_args()


def _load_test_jsonl(path: Path) -> tuple[list[str], list[str]]:
    """Load bug texts and true severity labels from test.jsonl. Returns (texts, severity_strings)."""
    texts = []
    labels = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = (obj.get("input") or "").strip()
            out_raw = obj.get("output") or "{}"
            try:
                out_obj = json.loads(out_raw) if isinstance(out_raw, str) else out_raw
                sev = (out_obj.get("severity") or "Minor").strip()
            except (json.JSONDecodeError, TypeError):
                sev = "Minor"
            if text:
                texts.append(text)
                labels.append(sev)
    return texts, labels


def main() -> None:
    args = parse_args()
    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZE
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not TEST_JSONL.exists():
        print(
            "New test JSONL not found.\n"
            "Run dataset/2_convert_to_jsonl.py first to create dataset/test.jsonl.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not CONFIG_PATH.exists():
        print("Run scripts/train_baseline.py first (config.json required).", file=sys.stderr)
        sys.exit(1)

    # Load test set only (holdout from convert_to_jsonl; no train data).
    X_all, y_str_all = _load_test_jsonl(TEST_JSONL)
    if not X_all:
        print("test.jsonl has no valid examples.", file=sys.stderr)
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    labels_ordered = config["labels"]
    label_to_int = {c: i for i, c in enumerate(labels_ordered)}

    rng = np.random.default_rng(RANDOM_STATE)
    n_take = min(batch_size, len(X_all))
    idx = rng.choice(len(X_all), size=n_take, replace=False)
    X_sub = [X_all[i] for i in idx]
    y_sub_str = [y_str_all[i] for i in idx]
    y_val_int = np.array([label_to_int.get(s, 0) for s in y_sub_str])
    print(f"Loaded test.jsonl: {len(X_all)} holdout examples; evaluating on {n_take} (batch-size={batch_size}).")

    # Import engine and run predictions
    try:
        from src.engines.llama_finetuned import FinetunedEngine
    except Exception as e:
        print(f"Failed to import FinetunedEngine: {e}", file=sys.stderr)
        sys.exit(1)
    engine = FinetunedEngine()
    y_pred_str = []
    pred_rows: list[dict] = []
    t0 = time.perf_counter()
    for i, text in enumerate(X_sub):
        try:
            out = engine.predict(text)
            pred_sev = out.get("severity", "Minor")
            y_pred_str.append(pred_sev)
        except Exception as e:
            print(f"  Row {i+1}/{n_take}: {e}")
            y_pred_str.append("Minor")
            pred_sev = "Minor"

        if args.debug:
            print(f"\n{'-'*60}")
            print(f"Row {i+1}/{n_take} | True: {y_sub_str[i]} | Predicted: {pred_sev}")
            print(f"{'-'*60}")

        # Record per-example predictions for error analysis.
        pred_rows.append(
            {
                "text_snippet": text[:200],
                "true_severity": y_sub_str[i],
                "predicted_severity": pred_sev,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  Completed {i+1}/{n_take}...")
    duration_sec = round(time.perf_counter() - t0, 2)
    y_pred_int = np.array([label_to_int.get(p, 0) for p in y_pred_str])
    m = compute_metrics(y_val_int, y_pred_int, labels=list(range(len(labels_ordered))))

    # Save detailed predictions for analysis.
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Saved {PREDICTIONS_CSV}")

    metrics_row = {
        "model": "Llama 3 8B (Fine-tuned)",
        "n_val": n_take,
        "accuracy": round(m["accuracy"], 4),
        "macro_f1": round(m["macro_f1"], 4),
        "macro_precision": round(m["macro_precision"], 4),
        "macro_recall": round(m["macro_recall"], 4),
        "duration_sec": duration_sec,
    }
    pd.DataFrame([metrics_row]).to_csv(METRICS_CSV, index=False)
    print(f"Saved {METRICS_CSV} (duration={duration_sec}s)")
    print(
        f"\nLlama 3 8B (Fine-tuned) — Accuracy: {m['accuracy']:.4f} | Macro F1: {m['macro_f1']:.4f} | Duration: {duration_sec:.1f}s"
    )


if __name__ == "__main__":
    main()
