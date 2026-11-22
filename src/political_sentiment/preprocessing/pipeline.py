"""Composable preprocessing pipeline.

Ties cleaning, tokenization and negation-marking into one object with a
scikit-learn-style interface. Two factory presets are provided:

* :meth:`PreprocessingPipeline.for_linear` — aggressive: lowercase, strip
  punctuation, negation marking. Output joined back to a string for TF-IDF.
* :meth:`PreprocessingPipeline.for_neural` — light: keep casing/punctuation,
  return the cleaned string for a subword tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .cleaning import TextCleaner
from .negation import mark_negation
from .tokenizer import WhitespaceTokenizer


@dataclass
class PreprocessConfig:
    """Flags controlling the pipeline's behaviour."""

    lowercase: bool = True
    keep_punctuation: bool = False
    apply_negation: bool = True
    negation_window: int = 4
    demojize: bool = True


@dataclass
class PreprocessingPipeline:
    """Turn raw text into either a normalised string or a token list."""

    config: PreprocessConfig = field(default_factory=PreprocessConfig)

    def __post_init__(self) -> None:
        self._cleaner = TextCleaner(
            lowercase=self.config.lowercase,
            keep_punctuation=self.config.keep_punctuation,
            demojize=self.config.demojize,
        )
        self._tokenizer = WhitespaceTokenizer()

    # -- core steps ---------------------------------------------------------
    def tokenize(self, text: str) -> List[str]:
        """Full pipeline returning a token list (clean -> tokenize -> negate)."""
        cleaned = self._cleaner.clean(text)
        tokens = self._tokenizer.tokenize(cleaned)
        if self.config.apply_negation:
            tokens = mark_negation(tokens, window=self.config.negation_window)
        return tokens

    def transform(self, text: str) -> str:
        """Full pipeline returning a normalised, space-joined string."""
        return " ".join(self.tokenize(text))

    def transform_batch(self, texts: List[str]) -> List[str]:
        return [self.transform(t) for t in texts]

    # -- presets ------------------------------------------------------------
    @classmethod
    def for_linear(cls) -> "PreprocessingPipeline":
        """Preset for TF-IDF / BiLSTM: aggressive normalisation + negation."""
        return cls(
            PreprocessConfig(
                lowercase=True,
                keep_punctuation=False,
                apply_negation=True,
            )
        )

    @classmethod
    def for_neural(cls) -> "PreprocessingPipeline":
        """Preset for transformers: light cleaning, preserve case/punctuation."""
        return cls(
            PreprocessConfig(
                lowercase=False,
                keep_punctuation=True,
                apply_negation=False,
            )
        )
