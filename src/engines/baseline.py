"""
Baseline inference: TF-IDF + LR / SVM / RF / NB.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.cleaning import clean_text

from .base_engine import BaseEngine

MODELS_DIR = ROOT / "models" / "baseline"
VALID_IDS = ("lr", "svm", "rf", "nb", "xgb", "ensemble")
DISPLAY_NAMES = {
    "lr": "Baseline — Logistic Regression",
    "svm": "Baseline — SVM",
    "rf": "Baseline — Random Forest",
    "nb": "Baseline — Naive Bayes",
    "xgb": "Baseline — XGBoost",
    "ensemble": "Baseline — Voting Ensemble",
}


class BaselineEngine(BaseEngine):
    def __init__(self, model_id: str) -> None:
        if model_id not in VALID_IDS:
            raise ValueError(f"model_id must be one of {VALID_IDS}")
        self._model_id = model_id
        self._vec = joblib.load(MODELS_DIR / "tfidf.pkl")
        self._clf = joblib.load(MODELS_DIR / f"{model_id}.pkl")
        with open(MODELS_DIR / "config.json") as f:
            self._config = json.load(f)
        self._classes = self._config["labels"]

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def display_name(self) -> str:
        return DISPLAY_NAMES[self._model_id]

    def predict(self, text: str) -> dict[str, Any]:
        raw = str(text).strip() if text else ""
        if not raw:
            return {
                "severity": self._classes[0],
                "confidence": 0.0,
                "probabilities": {},
                "model": self._model_id,
            }
        cleaned = clean_text(raw)
        if not cleaned:
            return {
                "severity": self._classes[0],
                "confidence": 0.0,
                "probabilities": {},
                "model": self._model_id,
            }
        X = self._vec.transform([cleaned])
        pred = self._clf.predict(X)[0]
        # xgb/ensemble (and retrained lr/svm/rf/nb) output ints; map to class name via config["labels"]
        if isinstance(pred, (int, np.integer)):
            pred = self._classes[int(pred)] if 0 <= int(pred) < len(self._classes) else self._classes[0]
        else:
            pred = str(pred)
        probs = {}
        confidence = 0.0
        if hasattr(self._clf, "predict_proba"):
            try:
                proba = self._clf.predict_proba(X)[0]
                for i, p in enumerate(proba):
                    name = self._classes[i] if i < len(self._classes) else str(i)
                    probs[name] = float(p)
                idx = self._classes.index(pred) if pred in self._classes else 0
                confidence = float(proba[idx])
            except Exception:
                confidence = 1.0
                probs = {c: 0.0 for c in self._classes}
                if pred in self._classes:
                    probs[pred] = 1.0
        else:
            confidence = 1.0
            probs = {c: 0.0 for c in self._classes}
            if pred in self._classes:
                probs[pred] = 1.0
        return {
            "severity": pred,
            "confidence": confidence,
            "probabilities": probs,
            "model": self._model_id,
        }
