"""Vocabulary construction and numericalization for the BiLSTM model.

Kept dependency-free (pure Python / no torch) so it can be unit-tested without
a deep-learning stack installed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class Vocabulary:
    """Maps tokens to integer ids with frequency- and size-based pruning."""

    def __init__(self, stoi: Dict[str, int]) -> None:
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}

    def __len__(self) -> int:
        return len(self.stoi)

    @property
    def pad_index(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_index(self) -> int:
        return self.stoi[UNK_TOKEN]

    @classmethod
    def build(
        cls,
        tokenized_texts: Sequence[Sequence[str]],
        max_size: int = 30000,
        min_freq: int = 2,
    ) -> "Vocabulary":
        """Build a vocabulary from tokenized texts.

        ``<pad>`` is always index 0 and ``<unk>`` index 1, so padding masks and
        embedding padding_idx line up.
        """
        counter: Counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        stoi = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for token, freq in counter.most_common():
            if freq < min_freq:
                break
            if len(stoi) >= max_size:
                break
            if token not in stoi:
                stoi[token] = len(stoi)
        return cls(stoi)

    def encode(self, tokens: Sequence[str], max_len: int) -> List[int]:
        """Numericalize and pad/truncate a token list to ``max_len``."""
        ids = [self.stoi.get(tok, self.unk_index) for tok in tokens[:max_len]]
        if len(ids) < max_len:
            ids += [self.pad_index] * (max_len - len(ids))
        return ids

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.stoi, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        stoi = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(stoi)
