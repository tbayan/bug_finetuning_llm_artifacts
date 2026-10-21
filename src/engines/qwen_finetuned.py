"""
Fine-tuned Qwen 2.5 7B (GGUF) severity classification via llama-cpp-python.
Uses the same Alpaca-style prompt as training (train.jsonl / qwen_final_tuning_script.ipynb)
with a severity-only JSON target (classification, no generated text fields).
Model is loaded once on first predict(). Expects qwen_finetuned.gguf in models/llm/.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .base_engine import BaseEngine

MODEL_PATH = ROOT / "models" / "llm" / "qwen_finetuned.gguf"

SEVERITY_OPTIONS = ("Critical", "Major", "Minor", "Trivial")

# Must match the instruction in dataset/2_convert_to_jsonl.py and train.jsonl
# so the model sees the same context it was trained on.
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

# Alpaca format: same as in qwen_final_tuning_script.ipynb / train.jsonl.
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


def _extract_json_object(response_text: str) -> dict[str, Any] | None:
    """Return the first JSON object found in a (possibly chatty) LLM response."""
    if not isinstance(response_text, str):
        return None
    text = response_text.strip()
    if not text:
        return None

    # Fast path: whole string is JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback: grab substring between first '{' and last '}' and try to parse
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


def _parse_structured_response(response_text: str) -> dict[str, str]:
    """
    Parse the LLM response into structured fields.
    Primary target is a severity-only JSON: {"severity": "<label>"}.
    """
    obj = _extract_json_object(response_text)
    if obj is not None:
        raw_sev = obj.get("severity")
        severity = _normalize_severity(raw_sev, fallback_source=response_text)
    else:
        severity = _normalize_severity(None, fallback_source=response_text)

    return {
        "severity": severity,
    }


def _parse_response(response_text: str) -> str:
    """Backward-compatible helper: extract severity only."""
    return _parse_structured_response(response_text)["severity"]


class QwenFinetunedEngine(BaseEngine):
    """Severity prediction using a fine-tuned Qwen 2.5 7B GGUF model via llama-cpp."""

    def __init__(self, model_id: str = "qwen_finetuned") -> None:
        self._model_id = model_id
        self._llm = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def display_name(self) -> str:
        return "Qwen 2.5 7B (Fine-tuned)"

    def _load_model(self) -> None:
        if self._llm is not None:
            return
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"GGUF model not found: {MODEL_PATH}. "
                "Place qwen_finetuned.gguf in the models/llm/ directory."
            )
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is required for the fine-tuned Qwen model. "
                "Install with: pip install llama-cpp-python"
            ) from None

        self._llm = Llama(
            model_path=str(MODEL_PATH),
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
        )

    def predict(self, text: str) -> dict[str, Any]:
        raw = str(text).strip() if text else ""
        if not raw:
            return {
                "severity": "Minor",
                "reasoning": "",
                "summary": "",
                "description": "",
                "confidence": 0.0,
                "model": self._model_id,
            }
        self._load_model()

        prompt = ALPACA_PROMPT.format(
            instruction=INSTRUCTION,
            input=raw[:4000],
        )
        try:
            output = self._llm(
                prompt,
                max_tokens=32,  # Optimized for severity-only output
                stop=["<|im_end|>", "<|endoftext|>"],  # Qwen 2.5 specific stop tokens
                temperature=0.1,
                repeat_penalty=1.15,
                echo=False,
            )
            response_text = output["choices"][0]["text"].strip()
            finish = output["choices"][0].get("finish_reason", "unknown")
        except Exception as e:
            raise RuntimeError(f"Qwen fine-tuned model inference error: {e}")

        print(f"[QwenFinetunedEngine] finish_reason={finish} len={len(response_text)}")
        print(f"[QwenFinetunedEngine] raw response (first 600 chars): {response_text[:600]}")

        parsed = _parse_structured_response(response_text)

        severity = parsed["severity"]

        return {
            "severity": severity,
            "reasoning": "",
            "summary": "",
            "description": "",
            "confidence": 1.0,
            "model": self._model_id,
        }

