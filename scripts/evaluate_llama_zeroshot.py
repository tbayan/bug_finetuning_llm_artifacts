"""
Evaluate zero-shot severity classification using Llama 3 8B via local Ollama API.
Uses test.jsonl only (holdout set from convert_to_jsonl.py); no train data.
Default subsample: 50 rows (use --batch-size 100 or 200 for larger runs).
Saves predictions and metrics for the LLM Results Table.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.metrics import compute_metrics

TEST_JSONL = ROOT / "dataset" / "test.jsonl"
CONFIG_PATH = ROOT / "models" / "baseline" / "config.json"
RESULTS_DIR = ROOT / "results"
PREDICTIONS_CSV = RESULTS_DIR / "llama_zeroshot_predictions.csv"
METRICS_CSV = RESULTS_DIR / "llama_zeroshot_metrics.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:8b"
RANDOM_STATE = 42
DEFAULT_SUBSAMPLE_N = 50

SEVERITY_OPTIONS = ("Critical", "Major", "Minor", "Trivial")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Ollama zero-shot severity classification.")
    p.add_argument("--batch-size", type=int, default=None, help=f"Subsample size; try 50, 100, or 200 (default: {DEFAULT_SUBSAMPLE_N})")
    p.add_argument("--debug", action="store_true", help="Print raw model generation to the terminal in real-time.")
    return p.parse_args()


INSTRUCTION = (
    "You are a Senior QA Engineer. Classify the severity of the following bug report.\n\n"
    "Severity definitions:\n"
    "- Critical: System crash, data loss, security vulnerability.\n"
    "- Major: Significant functionality is broken.\n"
    "- Minor: Small functional issue or inconvenience.\n"
    "- Trivial: Cosmetic issue or typo.\n\n"
    "Output ONLY a valid JSON object with a single key:\n"
    '{"severity": "<label>"}'
)
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


def _extract_json_object(response_text: str) -> dict | None:
    """Return the first JSON object found in a (possibly chatty) LLM response."""
    if not isinstance(response_text, str):
        return None
    text = response_text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _normalize_severity(value: str | None, fallback_source: str = "") -> str:
    """Map a free-form severity string (or entire response) onto one of SEVERITY_OPTIONS."""
    if value:
        val = str(value).strip().lower()
        for label in SEVERITY_OPTIONS:
            if val == label.lower():
                return label
    text = re.sub(r"[^\w\s]", " ", (fallback_source or "")).lower()
    for label in SEVERITY_OPTIONS:
        if label.lower() in text:
            return label
    return "Minor"


def parse_llm_response(response_text: str) -> str:
    """
    Extract severity label from LLM output.
    Uses structured JSON when available, but tolerates chatty responses.
    """
    if not response_text or not isinstance(response_text, str):
        return "Minor"
    obj = _extract_json_object(response_text)
    if obj is not None:
        return _normalize_severity(obj.get("severity"), fallback_source=response_text)
    return _normalize_severity(None, fallback_source=response_text)


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
    subsample_n = args.batch_size if args.batch_size is not None else DEFAULT_SUBSAMPLE_N
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

    # Subsample for LLM evaluation (reproducible)
    rng = np.random.default_rng(RANDOM_STATE)
    n_take = min(subsample_n, len(X_all))
    idx = rng.choice(len(X_all), size=n_take, replace=False)
    X_sub = [X_all[i] for i in idx]
    y_sub_str = [y_str_all[i] for i in idx]
    y_val_int = np.array([label_to_int.get(s, 0) for s in y_sub_str])
    print(f"Loaded test.jsonl: {len(X_all)} holdout examples; evaluating on {n_take} (batch-size={subsample_n}).")

    # Call Ollama for each row
    t0 = time.perf_counter()
    pred_rows = []
    y_pred_str = []
    for i, (text, true_sev_str) in enumerate(zip(X_sub, y_sub_str)):
        prompt = ALPACA_PROMPT.format(
            instruction=INSTRUCTION,
            input=text[:4000],
        )
        try:
            r = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 150,
                        "temperature": 0.1,
                    },
                },
                timeout=180,
            )
            r.raise_for_status()
            out = r.json()
            response_text = out.get("response", "") or ""
        except Exception as e:
            print(f"  Row {i+1}/{n_take}: API error — {e}")
            response_text = ""
        pred_label = parse_llm_response(response_text)

        if args.debug:
            print(f"\n{'-'*60}")
            print(f"Row {i+1}/{n_take} | True: {true_sev_str} | Predicted: {pred_label}")
            print(f"Raw Output:\n{response_text}")
            print(f"{'-'*60}")
        y_pred_str.append(pred_label)
        pred_rows.append({"text_snippet": text[:200], "true_severity": true_sev_str, "predicted_severity": pred_label})
        if (i + 1) % 25 == 0:
            elapsed = round(time.perf_counter() - t0, 1)
            print(f"  Completed {i+1}/{n_take} ({elapsed}s elapsed)...")
        time.sleep(0.05)

    y_pred_int = np.array([label_to_int.get(p, 0) for p in y_pred_str])
    duration_sec = round(time.perf_counter() - t0, 2)
    m = compute_metrics(y_val_int, y_pred_int, labels=list(range(len(labels_ordered))))

    # Save detailed predictions
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Saved {PREDICTIONS_CSV}")

    # Save one-row metrics for LLM Results Table (same columns as baseline + duration_sec)
    metrics_row = {
        "model": "Llama 3 8b",
        "n_val": n_take,
        "accuracy": round(m["accuracy"], 4),
        "macro_f1": round(m["macro_f1"], 4),
        "macro_precision": round(m["macro_precision"], 4),
        "macro_recall": round(m["macro_recall"], 4),
        "duration_sec": duration_sec,
    }
    metrics_df = pd.DataFrame([metrics_row])
    metrics_df.to_csv(METRICS_CSV, index=False)
    print(f"Saved {METRICS_CSV} (duration={duration_sec}s)")

    print("\nLlama 3 8b (zero-shot) — Accuracy: {:.4f} | Macro F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | Duration: {:.1f}s".format(
        m["accuracy"], m["macro_f1"], m["macro_precision"], m["macro_recall"], duration_sec
    ))


if __name__ == "__main__":
    main()
