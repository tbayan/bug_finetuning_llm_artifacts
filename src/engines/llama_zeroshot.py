"""
Ollama inference: Llama 3 8B via local Ollama API for severity classification.
Uses a reasoning-based JSON prompt; model returns a JSON string with:
severity → reasoning → summary → description (same order as fine-tuned model).
Implements BaseEngine for the model selector.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .base_engine import BaseEngine

CONFIG_PATH = ROOT / "models" / "baseline" / "config.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:8b"

SEVERITY_OPTIONS = ("Critical", "Major", "Minor", "Trivial")

# Structured JSON response schema (order: severity first for comparison with fine-tuned)
JSON_RESPONSE_SCHEMA = {
    "severity": "Critical | Major | Minor | Trivial",
    "reasoning": "Brief explanation of why this severity was chosen",
    "summary": "One-line summary of the bug",
    "description": "Short description of the issue",
}

# Must match the INSTRUCTION used in dataset/2_convert_to_jsonl.py and test.jsonl
# so zero-shot Ollama sees the same task description as the fine-tuned model.
INSTRUCTION = (
    "You are a Senior QA Engineer performing bug triage. "
    "Analyze the following software bug report and produce a structured JSON analysis.\n\n"
    "Severity definitions:\n"
    "- Critical: System crash, data loss, security vulnerability, or complete feature failure with no workaround.\n"
    "- Major: Significant functionality is broken or degraded, but partial workarounds may exist.\n"
    "- Minor: A small functional issue or inconvenience; core features work and easy workarounds exist.\n"
    "- Trivial: Cosmetic issue, typo, or negligible impact on functionality.\n\n"
    "Provide a concise, paraphrased summary and a technical description limited to 3 sentences.\n\n"
    "Output ONLY a valid JSON object with these keys in this exact order:\n"
    '{"severity": "<label>", "reasoning": "<logic_analysis>", '
    '"summary": "<one_sentence_title>", "description": "<condensed_technical_description>"}'
)

# Alpaca-style wrapper: identical shape to the fine-tuned engine and training prompt.
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


def _parse_llm_structured_response(response_text: str) -> dict[str, str]:
    """
    Parse the LLM response into structured fields.
    Returns dict with keys: severity, reasoning, summary, description.
    """
    reasoning = ""
    summary = ""
    description = ""

    obj = _extract_json_object(response_text)
    if obj is not None:
        reasoning = str(obj.get("reasoning") or "").strip()
        summary = str(obj.get("summary") or "").strip()
        description = str(obj.get("description") or "").strip()
        severity = _normalize_severity(obj.get("severity"), fallback_source=response_text)
    else:
        severity = _normalize_severity(None, fallback_source=response_text)

    return {
        "severity": severity,
        "reasoning": reasoning,
        "summary": summary,
        "description": description,
    }


def _parse_llm_response(response_text: str) -> str:
    """Backward-compatible helper: extract severity only."""
    return _parse_llm_structured_response(response_text)["severity"]


class OllamaEngine(BaseEngine):
    """Severity prediction via local Ollama (Llama 3 8B)."""

    def __init__(self, model_id: str = "ollama") -> None:
        self._model_id = model_id
        self._classes = ["Critical", "Major", "Minor", "Trivial"]
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                self._classes = json.load(f).get("labels", self._classes)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def display_name(self) -> str:
        return "Llama 3 8B (Zero)"

    def predict(self, text: str) -> dict[str, Any]:
        raw = str(text).strip() if text else ""
        if not raw:
            return {
                "severity": self._classes[0],
                "reasoning": "",
                "summary": "",
                "description": "",
                "confidence": 0.0,
                "probabilities": {},
                "model": self._model_id,
            }

        # Use the exact same severity instruction as in dataset/test.jsonl,
        # wrapped in the same Alpaca-style prompt as the fine-tuned engine.
        prompt = ALPACA_PROMPT.format(
            instruction=INSTRUCTION,
            input=raw[:4000],
        )
        try:
            r = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=120,
            )
            r.raise_for_status()
            out = r.json()
            response_text = out.get("response", "") or ""
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot reach Ollama at localhost:11434. Start Ollama and run: ollama run llama3:8b"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama request timed out. Try again or use a baseline model.")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")

        parsed = _parse_llm_structured_response(response_text)
        return {
            "severity": parsed["severity"],
            "reasoning": parsed["reasoning"],
            "summary": parsed["summary"],
            "description": parsed["description"],
            "confidence": 1.0,
            "probabilities": {},
            "model": self._model_id,
        }
