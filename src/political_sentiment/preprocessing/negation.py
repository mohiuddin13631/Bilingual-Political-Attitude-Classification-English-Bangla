"""Negation scope marking for English and Bangla.

Negation is one of the hardest phenomena for bag-of-words sentiment models:
"not good" should not look like "good". Following the classic Das & Chen /
Pang-Lee heuristic, we append a ``_NEG`` suffix to tokens that fall inside a
negation scope, so the linear model can learn a distinct weight for
``good_NEG``. Transformers handle negation contextually and do not use this.

English scope: from a negation cue up to the next clause-ending punctuation or
a fixed window. Bangla often marks negation *after* the predicate (নয়, না, নেই),
so we also mark a backward window before those cues.
"""

from __future__ import annotations

import re
from typing import List

_EN_NEG_CUES = {
    "not", "no", "never", "none", "cannot", "can't", "won't", "don't",
    "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "n't",
    "without", "hardly", "barely",
}
_BN_NEG_CUES = {"না", "নয়", "নেই", "নাই", "কখনো", "কখনোই", "মোটেও"}

_CLAUSE_END = re.compile(r"[।!?.,;:]")


def _is_clause_end(token: str) -> bool:
    return bool(_CLAUSE_END.fullmatch(token)) or token in {"but", "কিন্তু"}


def mark_negation(tokens: List[str], window: int = 4, suffix: str = "_NEG") -> List[str]:
    """Return a copy of ``tokens`` with negated tokens suffixed.

    Parameters
    ----------
    tokens:
        Whitespace/tokenizer output.
    window:
        Maximum number of tokens affected by a single cue (forward for English
        cues, and additionally backward for Bangla post-predicate cues).
    """
    out = list(tokens)
    n = len(out)

    for i, tok in enumerate(out):
        low = tok.lower()

        # Forward scope (English-style pre-negation, and Bangla না/কখনো).
        if low in _EN_NEG_CUES or tok in _BN_NEG_CUES:
            for j in range(i + 1, min(n, i + 1 + window)):
                if _is_clause_end(out[j]):
                    break
                if out[j].strip():
                    out[j] = out[j] + suffix

        # Backward scope for Bangla post-predicate cues (নয়/নেই/নাই).
        if tok in {"নয়", "নেই", "নাই"}:
            for j in range(i - 1, max(-1, i - 1 - window), -1):
                if _is_clause_end(out[j]):
                    break
                if out[j].strip() and not out[j].endswith(suffix):
                    out[j] = out[j] + suffix

    return out
