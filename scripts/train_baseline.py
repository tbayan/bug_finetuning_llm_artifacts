"""
Train baseline models: TF-IDF + LR, SVM, RF, NB, XGBoost, Voting Ensemble.
SCIENTIFIC VERSION: Label encoding, strategic undersampling, GridSearchCV, best estimator saved.
Train one at a time: python scripts/train_baseline.py -m xgb  or  -m ensemble
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# New, isolated dataset for baselines (already cleaned, mapped, balanced, and split).
# Prepared via dataset/1_prep_balance_split_csv.py
DATA_CSV = ROOT / "dataset" / "train.csv"
MODELS_DIR = ROOT / "models" / "baseline"

TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE = (1, 3)
TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_IDS = ("lr", "svm", "rf", "nb", "xgb", "ensemble")

# (display_name, base_estimator, param_grid) for GridSearchCV. Empty param_grid = fit once (no grid search).
GRID_CONFIG = {
    "lr": (
        "Logistic Regression",
        LogisticRegression(
            solver="saga",
            max_iter=2000,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        {"C": [0.1, 1.0, 5.0, 10.0]},
    ),
    "svm": (
        "Linear SVM",
        LinearSVC(
            loss="squared_hinge",
            dual=False,
            intercept_scaling=1,
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
        {"C": [0.01, 0.1, 1.0, 5.0]},
    ),
    "rf": (
        "Random Forest",
        RandomForestClassifier(
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        {
            "n_estimators": [300, 500],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", "log2"],
        },
    ),
    "nb": (
        "Naive Bayes",
        MultinomialNB(fit_prior=False),
        {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    ),
    "xgb": (
        "XGBoost",
        XGBClassifier(eval_metric="mlogloss", n_jobs=-1, random_state=RANDOM_STATE),
        {
            "n_estimators": [100, 200],
            "max_depth": [6, 10],
            "colsample_bytree": [0.5, 0.8],
        },
    ),
    "ensemble": (
        "Voting Ensemble",
        VotingClassifier(
            voting="soft",
            estimators=[
                ("lr", LogisticRegression(solver="saga", max_iter=2000, n_jobs=-1, random_state=RANDOM_STATE)),
                (
                    "svm",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(
                            C=1.0,
                            loss="squared_hinge",
                            dual=False,
                            intercept_scaling=1,
                            max_iter=3000,
                            random_state=RANDOM_STATE,
                        ),
                        cv=5,
                    ),
                ),
                ("rf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)),
                ("nb", MultinomialNB(fit_prior=False, alpha=0.1)),
            ],
        ),
        {
            "weights": [
                (1, 1, 1, 1),
                (2, 2, 1, 1),
                (1, 2, 1, 1),
                (2, 1, 1, 1),
            ],
        },
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline models with GridSearch (optionally one at a time).")
    p.add_argument(
        "-m",
        "--model",
        choices=MODEL_IDS,
        action="append",
        dest="models",
        help="Train only this model (can be repeated). Omit to train all.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    to_train = args.models if args.models else list(MODEL_IDS)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_CSV.exists():
        print(
            "New training CSV not found.\n"
            "Run dataset/1_prep_balance_split_csv.py first to create dataset/train.csv.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading data from {DATA_CSV}...")
    df = pd.read_csv(DATA_CSV)
    # New dataset uses 'description' as text and 'mapped_severity' as 4-class label.
    df = df.dropna(subset=["description", "mapped_severity"])

    X = df["description"].astype(str).values
    y_str = df["mapped_severity"].values

    # Label encoding: XGBoost (and consistent eval) use integer labels 0,1,2,3
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    classes = list(le.classes_)  # order matches 0,1,2,3 for mapping back
    print(f"Label encoding: {classes} -> 0..{len(classes)-1}. Saved to config.")

    print(f"Training on {len(df)} rows. TF-IDF max_features={TFIDF_MAX_FEATURES}")

    stratify = y if len(y) >= 25 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=stratify, random_state=RANDOM_STATE
    )

    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
        max_df=0.75,
        min_df=5,
        stop_words="english",
    )
    print("Vectorizing text...")
    X_train_tf = vec.fit_transform(X_train)
    X_val_tf = vec.transform(X_val)

    joblib.dump(vec, MODELS_DIR / "tfidf.pkl")
    print("Saved tfidf.pkl")

    config = {
        "labels": classes,
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "tfidf_ngram_range": list(TFIDF_NGRAM_RANGE),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }
    with open(MODELS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Saved config.json (labels = class names for index -> string mapping)")

    def _format_duration(secs: float) -> str:
        if secs < 60:
            return f"{secs:.0f}s"
        m, s = divmod(int(secs), 60)
        return f"{m}m {s}s" if m else f"{s}s"

    n_models = len(to_train)
    loop_start = time.perf_counter()

    for i, mid in enumerate(to_train, start=1):
        name, base_estimator, param_grid = GRID_CONFIG[mid]
        pct = int(100 * (i - 1) / n_models)
        if param_grid:
            print(f"\n[{i}/{n_models}] ({pct}%) Grid search for {name} (cv=10, scoring=macro F1)...")
        else:
            print(f"\n[{i}/{n_models}] ({pct}%) Training {name} (single fit, no grid search)...")
        model_start = time.perf_counter()

        if param_grid:
            gs = GridSearchCV(
                base_estimator,
                param_grid,
                cv=10,
                scoring="f1_macro",
                n_jobs=-1,
                verbose=2,
                refit=True,
            )
            gs.fit(X_train_tf, y_train)
            best_estimator = gs.best_estimator_
            print(f"  Best params: {gs.best_params_}")
        else:
            # Empty param_grid (e.g. ensemble): fit once, no grid search
            best_estimator = base_estimator
            best_estimator.fit(X_train_tf, y_train)
            print(f"  (No grid search; using fixed params)")

        y_pred = best_estimator.predict(X_val_tf)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_val, y_pred)
        macro_rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
        print(f"  ✅ {name}: Macro F1={macro_f1:.4f} | Accuracy={acc:.4f} | Recall={macro_rec:.4f}")

        joblib.dump(best_estimator, MODELS_DIR / f"{mid}.pkl")
        print(f"  Saved to {mid}.pkl")

        model_elapsed = time.perf_counter() - model_start
        total_elapsed = time.perf_counter() - loop_start
        remaining_models = n_models - i
        if remaining_models > 0 and i > 0:
            est_remaining_sec = (total_elapsed / i) * remaining_models
            print(f"  Done in {_format_duration(model_elapsed)}. Elapsed: {_format_duration(total_elapsed)} | Est. remaining: ~{_format_duration(est_remaining_sec)}")
        else:
            print(f"  Done in {_format_duration(model_elapsed)}. Elapsed: {_format_duration(total_elapsed)}")

    total = time.perf_counter() - loop_start
    print(f"\nAll done. Total time: {_format_duration(total)}")


if __name__ == "__main__":
    main()
