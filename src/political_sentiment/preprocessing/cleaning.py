"""Language-aware text cleaning for social-media political text.

The cleaner is intentionally conservative for transformer models (which benefit
from natural casing/punctuation) and more aggressive for the linear/BiLSTM path.
Behaviour is toggled through :class:`TextCleaner` flags, so a single class serves
every model with model-appropriate settings.

Handles both English (Latin) and Bangla (Bengali script). Emoji and URLs are
normalised rather than dropped, because in political discourse an emoji or a
shared link often *carries* sentiment.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Precompiled patterns ------------------------------------------------------- #
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")  # loooove -> loove
# Keep Latin letters, Bengali block (U+0980–U+09FF), digits, sentiment
# punctuation, and the angle brackets used by our <url>/<user>/<emoji>
# placeholder tokens. Strip everything else.
_KEEP_RE = re.compile(r"[^0-9A-Za-zঀ-৿\s!?.,'’<>]+")
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF]",
    flags=re.UNICODE,
)


@dataclass
class TextCleaner:
    """Configurable text cleaner.

    Parameters
    ----------
    lowercase:
        Lowercase Latin text (no-op for Bengali). Disable for cased
        transformers.
    strip_urls, strip_mentions:
        Replace URLs / @mentions with placeholder tokens.
    split_hashtags:
        Turn ``#SaveClimate`` into the word ``SaveClimate`` (kept, not dropped).
    collapse_repeats:
        Collapse elongated characters ("soooo" -> "soo").
    demojize:
        Replace emoji with a neutral ``<emoji>`` token so they are not lost but
        do not explode the vocabulary.
    keep_punctuation:
        Retain sentiment-bearing punctuation (! ?). When False, strip it.
    """

    lowercase: bool = True
    strip_urls: bool = True
    strip_mentions: bool = True
    split_hashtags: bool = True
    collapse_repeats: bool = True
    demojize: bool = True
    keep_punctuation: bool = True

    def __call__(self, text: str) -> str:
        return self.clean(text)

    def clean(self, text: str) -> str:
        if text is None:
            return ""
        text = unicodedata.normalize("NFKC", str(text))

        if self.strip_urls:
            text = _URL_RE.sub(" <url> ", text)
        if self.strip_mentions:
            text = _MENTION_RE.sub(" <user> ", text)
        if self.split_hashtags:
            text = _HASHTAG_RE.sub(r" \1 ", text)
        if self.demojize:
            text = _EMOJI_RE.sub(" <emoji> ", text)
        if self.collapse_repeats:
            text = _REPEAT_CHAR_RE.sub(r"\1\1", text)

        text = _KEEP_RE.sub(" ", text)
        if not self.keep_punctuation:
            text = re.sub(r"[!?.,'’]", " ", text)
        if self.lowercase:
            # Bengali has no case; .lower() only affects Latin characters.
            text = text.lower()

        text = _WHITESPACE_RE.sub(" ", text).strip()
        return text


def clean_text(text: str, **kwargs: object) -> str:
    """Convenience one-shot cleaner with default settings."""
    return TextCleaner(**kwargs).clean(text)  # type: ignore[arg-type]
