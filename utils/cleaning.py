"""
Text preprocessing for bug reports.
Used in data cleaning, inference, and training.
"""

from __future__ import annotations

import re
from typing import Optional

# Optional stopwords; keep configurable. None = no stopword removal.
DEFAULT_STOPWORDS: Optional[set[str]] = None  # Set via enable_stopwords() if needed.

_url_pat = re.compile(r"https?://\S+")
_email_pat = re.compile(r"\S+@\S+\.\S+")
_ws_pat = re.compile(r"\s+")


def clean_text(
    raw: Optional[str],
    *,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    normalize_whitespace: bool = True,
    stopwords: Optional[set[str]] = None,
) -> str:
    """
    Normalize bug report text: lowercase, strip, optional URL/email removal, optional stopwords.
    Handles None and empty input safely.
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    if lowercase:
        s = s.lower()
    if remove_urls:
        s = _url_pat.sub(" ", s)
    if remove_emails:
        s = _email_pat.sub(" ", s)
    if normalize_whitespace:
        s = _ws_pat.sub(" ", s).strip()
    sw = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    if sw:
        tokens = s.split()
        s = " ".join(t for t in tokens if t not in sw)
    return s.strip()
