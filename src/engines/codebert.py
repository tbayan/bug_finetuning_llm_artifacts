"""
Local CodeBERT inference engine for severity classification.
Loads a fine-tuned Hugging Face model from codebert_model/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .base_engine import BaseEngine

MODEL_DIR = ROOT / "codebert_model"
DEFAULT_LABELS = ["Critical", "Major", "Minor", "Trivial"]


class CodeBERTEngine(BaseEngine):
    def __init__(self, model_id: str = "codebert") -> None:
        self._model_id = model_id
        self._tokenizer = None
        self._model = None
        self._labels = list(DEFAULT_LABELS)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def display_name(self) -> str:
        return "CodeBERT"

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
        if not (MODEL_DIR / "config.json").exists():
            raise FileNotFoundError(f"Missing config.json in: {MODEL_DIR}")

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise RuntimeError(
                "transformers and torch are required for CodeBERT inference. "
                "Install with: pip install transformers torch"
            ) from None

        # Read labels from model config if available.
        try:
            with open(MODEL_DIR / "config.json", encoding="utf-8") as f:
                cfg = json.load(f)
            id2label = cfg.get("id2label") or {}
            if id2label:
                ordered = []
                for i in range(len(id2label)):
                    ordered.append(str(id2label.get(str(i), DEFAULT_LABELS[i] if i < len(DEFAULT_LABELS) else i)))
                if ordered:
                    self._labels = ordered
        except Exception:
            self._labels = list(DEFAULT_LABELS)

        self._tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        self._model.eval()
        self._torch = torch

    def predict(self, text: str) -> dict[str, Any]:
        raw = str(text).strip() if text else ""
        if not raw:
            return {
                "severity": self._labels[0],
                "confidence": 0.0,
                "probabilities": {},
                "model": self._model_id,
            }

        self._load()
        inputs = self._tokenizer(
            raw,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )

        with self._torch.no_grad():
            out = self._model(**inputs)
            probs = self._torch.softmax(out.logits, dim=-1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        severity = self._labels[pred_idx] if pred_idx < len(self._labels) else self._labels[0]
        prob_map = {
            self._labels[i] if i < len(self._labels) else str(i): float(p)
            for i, p in enumerate(probs)
        }
        confidence = float(probs[pred_idx]) if pred_idx < len(probs) else 0.0

        return {
            "severity": severity,
            "confidence": confidence,
            "probabilities": prob_map,
            "model": self._model_id,
        }
