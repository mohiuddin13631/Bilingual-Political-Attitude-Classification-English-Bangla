"""Lightweight, dependency-free tokenization and language detection.

For the linear and BiLSTM models we use a regex word tokenizer that is aware of
both Latin and Bengali scripts. Transformer models use their own subword
tokenizer (from HuggingFace) and bypass this module.
"""

from __future__ import annotations

import re
from typing import List

# A token is a run of Latin letters+digits+apostrophe, a run of Bengali
# characters, or a single sentiment punctuation mark.
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[ঀ-৿]+|[!?।]")

# Bengali Unicode block for script-based language detection.
_BENGALI_RE = re.compile(r"[ঀ-৿]")


def detect_language(text: str) -> str:
    """Heuristically detect ``bn`` vs ``en`` by Bengali-character ratio."""
    if not text:
        return "en"
    bengali = len(_BENGALI_RE.findall(text))
    letters = len(re.findall(r"[A-Za-z]", text))
    total = bengali + letters
    if total == 0:
        return "en"
    return "bn" if bengali / total >= 0.3 else "en"


class WhitespaceTokenizer:
    """Regex word tokenizer for EN + BN.

    Despite the name it is script-aware rather than pure whitespace: Bengali
    text is not always whitespace-delimited at clause boundaries, and we want
    punctuation such as ``!`` and ``।`` (Bengali danda) as standalone tokens so
    negation scoping can see clause ends.
    """

    def tokenize(self, text: str) -> List[str]:
        return _TOKEN_RE.findall(text or "")

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)
