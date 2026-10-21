"""
Model registry: single source of truth for available models.
list_models() -> [(model_id, display_name), ...]
get_engine(model_id) -> BaseEngine

To add a new model: implement BaseEngine, then register it below.
No changes to app.py or the Analyze flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_engine import BaseEngine

from .baseline import BaselineEngine
from .codebert import CodeBERTEngine
from .llama_zeroshot import OllamaEngine
from .llama_finetuned import FinetunedEngine
from .qwen_zeroshot import QwenEngine
from .qwen_finetuned import QwenFinetunedEngine

# (model_id, factory callable that returns BaseEngine)
_REGISTRY: list[tuple[str, str, type]] = []


def _register_baselines() -> None:
    for mid in ("lr", "svm", "rf", "nb", "xgb", "ensemble"):
        name = {
            "lr": "Logistic Regression",
            "svm": "SVM",
            "rf": "Random Forest",
            "nb": "Naive Bayes",
            "xgb": "XGBoost",
            "ensemble": "Voting Ensemble",
        }[mid]
        _REGISTRY.append((mid, name, BaselineEngine))
    _REGISTRY.append(("codebert", "CodeBERT", CodeBERTEngine))
    _REGISTRY.append(("ollama", "Llama 3 8B (Zero)", OllamaEngine))
    _REGISTRY.append(("finetuned", "Llama 3 8B (Fine-tuned)", FinetunedEngine))
    _REGISTRY.append(("qwen", "Qwen 2.5 7B (Zero)", QwenEngine))
    _REGISTRY.append(("qwen_finetuned", "Qwen 2.5 7B (Fine-tuned)", QwenFinetunedEngine))


def list_models() -> list[tuple[str, str]]:
    """Returns [(model_id, display_name), ...] for all registered models."""
    if not _REGISTRY:
        _register_baselines()
    return [(mid, name) for mid, name, _ in _REGISTRY]


_ENGINE_CACHE: dict[str, "BaseEngine"] = {}


def get_engine(model_id: str) -> "BaseEngine":
    """Return the engine for model_id. Uses a cache to avoid reloading."""
    if not _REGISTRY:
        _register_baselines()
    by_id = {mid: (name, fac) for mid, name, fac in _REGISTRY}
    if model_id not in by_id:
        raise KeyError(f"Unknown model_id: {model_id}. Available: {[m[0] for m in _REGISTRY]}")
    if model_id not in _ENGINE_CACHE:
        _ENGINE_CACHE[model_id] = by_id[model_id][1](model_id)
    return _ENGINE_CACHE[model_id]


def register(model_id: str, display_name: str, factory: type) -> None:
    """
    Register a new model. Example:

        from src.engines.llm import LLMEngine
        register("llm", "LLM (Llama 3)", LLMEngine)

    The factory must accept (model_id: str) and return a BaseEngine.
    For LLMEngine you may use a lambda or partial if it has no model_id arg.
    """
    _REGISTRY.append((model_id, display_name, factory))
    _ENGINE_CACHE.pop(model_id, None)
