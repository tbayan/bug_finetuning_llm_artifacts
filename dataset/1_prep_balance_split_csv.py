#!/usr/bin/env python3
"""
STEP 1: Download, clean, map, balance, and split the new bug-report dataset.

Dataset: AndressaStefany/bug-reports (Hugging Face, train split)

Pipeline:
  1) Load dataset via `datasets.load_dataset`.
  2) Keep only `description` (bug text) and `bug_severity`.
  3) CLEANING & MAPPING:
       - Drop rows where bug_severity in {"normal", "enhancement"}.
       - Map severities to a strict 4-point rubric:
           blocker, critical -> Critical
           major             -> Major
           minor             -> Minor
           trivial           -> Trivial
       - Drop rows with empty / null descriptions.
  4) BALANCING:
       - Random undersampling to exactly 5,000 rows per class
         (Critical/Major/Minor/Trivial), total 20,000 rows.
         random_state = 42.
  5) SPLITTING & OUTPUT:
       - 80/20 train/test split (random_state=42, stratified by mapped class).
       - Save:
           dataset/train.csv  (16,000 rows)
           dataset/test.csv   (4,000 rows)
       - Print class distributions for both train and test.

All outputs stay inside `dataset/` to keep this experiment isolated.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "dataset"
TRAIN_CSV = OUT_DIR / "train.csv"
TEST_CSV = OUT_DIR / "test.csv"

SEVERITY_MAP: Dict[str, str] = {
    "blocker": "Critical",
    "critical": "Critical",
    "major": "Major",
    "minor": "Minor",
    "trivial": "Trivial",
}

DROP_SEVERITIES = {"normal", "enhancement"}
TARGET_CLASSES: List[str] = ["Critical", "Major", "Minor", "Trivial"]
TARGET_PER_CLASS = 5_000
TOTAL_TARGET = TARGET_PER_CLASS * len(TARGET_CLASSES)
RANDOM_STATE = 42


def load_raw_df() -> pd.DataFrame:
    """Load HuggingFace dataset into a DataFrame with needed columns."""
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "datasets library not installed.\n"
            "Install it with:\n"
            "  pip install datasets\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Loading HuggingFace dataset AndressaStefany/bug-reports (train split)...")
    ds = load_dataset("AndressaStefany/bug-reports", split="train")
    df = ds.to_pandas()

    if "description" not in df.columns or "bug_severity" not in df.columns:
        print(
            f"Expected columns 'description' and 'bug_severity' in dataset. "
            f"Got: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    df = df[["description", "bug_severity"]].copy()
    df["description"] = df["description"].astype(str)
    df["bug_severity"] = df["bug_severity"].astype(str)
    return df


def clean_and_map(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out unwanted severities, map to 4 classes, drop bad descriptions."""
    sev = df["bug_severity"].str.lower().str.strip()

    # Drop rows with severities we don't want
    mask_drop = sev.isin(DROP_SEVERITIES)
    df = df.loc[~mask_drop].copy()

    # Map to 4-point rubric
    df["mapped_severity"] = sev.map(SEVERITY_MAP)
    df = df.dropna(subset=["mapped_severity"]).copy()

    # Drop empty / null descriptions
    desc = df["description"].astype(str).str.strip()
    df = df.loc[desc != ""].copy()
    return df


def balance(df: pd.DataFrame) -> pd.DataFrame:
    """Randomly undersample to 5,000 rows per mapped class."""
    counts = df["mapped_severity"].value_counts()
    print("Counts after cleaning/mapping:")
    print(counts.to_string())

    for cls in TARGET_CLASSES:
        if counts.get(cls, 0) < TARGET_PER_CLASS:
            print(
                f"Not enough examples for class '{cls}': "
                f"have {counts.get(cls, 0)}, need {TARGET_PER_CLASS}.",
                file=sys.stderr,
            )
            sys.exit(1)

    rng = np.random.default_rng(RANDOM_STATE)
    parts = []
    for cls in TARGET_CLASSES:
        cls_df = df[df["mapped_severity"] == cls]
        idx = rng.choice(len(cls_df), size=TARGET_PER_CLASS, replace=False)
        parts.append(cls_df.iloc[idx])

    balanced = pd.concat(parts, ignore_index=True)
    # Shuffle once more for randomness
    balanced = balanced.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    assert len(balanced) == TOTAL_TARGET
    return balanced


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_df()
    print(f"Raw rows: {len(df_raw)}")

    df_clean = clean_and_map(df_raw)
    print(f"Rows after cleaning & mapping: {len(df_clean)}")

    df_balanced = balance(df_clean)
    print("\nBalanced class distribution (all 20,000 rows):")
    print(df_balanced["mapped_severity"].value_counts().to_string())
    print(f"\nTotal rows in balanced set: {len(df_balanced)} (expected {TOTAL_TARGET})")

    # 80/20 split, stratified by mapped_severity
    train_df, test_df = train_test_split(
        df_balanced,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df_balanced["mapped_severity"],
    )

    print(f"\nTrain size: {len(train_df)}")
    print("Train class distribution:")
    print(train_df["mapped_severity"].value_counts().to_string())

    print(f"\nTest size:  {len(test_df)}")
    print("Test class distribution:")
    print(test_df["mapped_severity"].value_counts().to_string())

    print(f"\nSaving train CSV to {TRAIN_CSV} ...")
    train_df.to_csv(TRAIN_CSV, index=False)
    print(f"Saving test CSV to  {TEST_CSV} ...")
    test_df.to_csv(TEST_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()

