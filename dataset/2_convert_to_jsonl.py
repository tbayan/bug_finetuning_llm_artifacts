#!/usr/bin/env python3
"""
STEP 2: Convert train/test CSVs into Alpaca-style JSONL for LLM fine-tuning.

Severity-only target generation:
  - The model sees the bug description as input.
  - The output JSON contains only the gold severity label.

Input:
  - dataset/train.csv
  - dataset/test.csv

Each CSV must contain:
  - description       (bug text)
  - mapped_severity   (Critical | Major | Minor | Trivial)

Outputs:
  - dataset/train.jsonl
  - dataset/test.jsonl

CRITICAL: The JSON string in `output` ALWAYS starts with the "severity" key.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "dataset" / "train.csv"
TEST_CSV = ROOT / "dataset" / "test.csv"
TRAIN_JSONL = ROOT / "dataset" / "train.jsonl"
TEST_JSONL = ROOT / "dataset" / "test.jsonl"

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


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------
def write_jsonl(df: pd.DataFrame, path: Path, rng: random.Random) -> None:
    """Write DataFrame rows as Alpaca-style JSONL with severity-only output JSON."""
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            desc = str(row["description"])
            sev = str(row["mapped_severity"])
            input_text = desc[:4000]

            output_obj = {
                "severity": sev,
            }
            output_str = json.dumps(output_obj, ensure_ascii=False)

            record = {
                "instruction": INSTRUCTION,
                "input": input_text,
                "output": output_str,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _ensure_exists(path: Path) -> None:
    if not path.exists():
        print(f"Required file not found: {path}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    _ensure_exists(TRAIN_CSV)
    _ensure_exists(TEST_CSV)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    for name, df in (("train", train_df), ("test", test_df)):
        if "description" not in df.columns or "mapped_severity" not in df.columns:
            print(
                f"{name}.csv is missing required columns. "
                f"Expected 'description' and 'mapped_severity'. "
                f"Got: {list(df.columns)}",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Loaded {len(train_df)} train rows and {len(test_df)} test rows.")

    for p in (TRAIN_JSONL, TEST_JSONL):
        if p.exists():
            p.unlink()
            print(f"Removed stale {p.name}")

    rng = random.Random(42)

    print(f"Writing JSONL train -> {TRAIN_JSONL}")
    write_jsonl(train_df, TRAIN_JSONL, rng)

    print(f"Writing JSONL test  -> {TEST_JSONL}")
    write_jsonl(test_df, TEST_JSONL, rng)

    print(f"\nDone. Train: {TRAIN_JSONL}  |  Test: {TEST_JSONL}")


if __name__ == "__main__":
    main()
