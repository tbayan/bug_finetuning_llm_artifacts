"""
Abstract base interface for severity prediction engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEngine(ABC):
    """Interface for bug severity models. Each engine has model_id and display_name."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique id (e.g. 'lr', 'svm', 'llm')."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for the UI."""
        ...

    @abstractmethod
    def predict(self, text: str) -> dict[str, Any]:
        """
        Predict severity for a bug report. At minimum returns:
        - severity: str
        - confidence: float (if available)
        - model: str (model_id)
        Optionally: probabilities, root_cause, fix_suggestion, etc.
        """
        ...
