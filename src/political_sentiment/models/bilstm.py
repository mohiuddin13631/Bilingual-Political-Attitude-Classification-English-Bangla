"""BiLSTM classifier with additive (Bahdanau-style) attention — PyTorch.

Architecture (report Section IV.B)::

    tokens -> Embedding -> BiLSTM -> Additive Attention pool -> Dropout -> Linear -> softmax

The attention layer produces a weighted sum of the LSTM hidden states, giving
both a fixed-size document representation *and* per-token attention weights that
serve as a cheap, faithful explanation of which words drove the prediction
(surfaced in the dashboard and the explainability report).

This module imports torch at top level and is therefore imported lazily by the
model registry, so the TF-IDF baseline path never requires PyTorch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .. import LABEL2ID, LABELS
from ..preprocessing import PreprocessingPipeline
from ..utils.logging_utils import get_logger
from .base import SentimentModel
from .vocab import Vocabulary

logger = get_logger(__name__)


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve a torch device from a preference string."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


class _AdditiveAttention(nn.Module):
    """Additive attention over a sequence of encoder states.

    Given encoder outputs ``H`` of shape ``(B, T, D)`` and a boolean pad mask,
    computes scalar scores ``v^T tanh(W H)``, masks padding, softmaxes over time
    and returns the context vector plus the attention weights.
    """

    def __init__(self, hidden_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.project = nn.Linear(hidden_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self, encoder_states: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encoder_states: (B, T, H); mask: (B, T) with True at real tokens
        scores = self.score(torch.tanh(self.project(encoder_states))).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), encoder_states).squeeze(1)  # (B, H)
        return context, weights


class BiLSTMAttentionNet(nn.Module):
    """The neural network module (see :class:`BiLSTMAttentionModel` for training)."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        attention_dim: int = 96,
        num_classes: int = 3,
        dropout: float = 0.4,
        pad_index: int = 0,
    ) -> None:
        super().__init__()
        self.pad_index = pad_index
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        directions = 2 if bidirectional else 1
        self.attention = _AdditiveAttention(hidden_dim * directions, attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * directions, num_classes)

    def forward(
        self, input_ids: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        mask = input_ids != self.pad_index  # (B, T)
        lengths = mask.sum(dim=1).clamp(min=1).cpu()

        embedded = self.embedding(input_ids)  # (B, T, E)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        encoder_states, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=input_ids.size(1)
        )
        context, weights = self.attention(encoder_states, mask)
        logits = self.classifier(self.dropout(context))
        if return_attention:
            return logits, weights
        return logits


class _TextDataset(Dataset):
    """Encodes tokenized text into padded id tensors on the fly."""

    def __init__(self, encoded: List[List[int]], labels: List[int]) -> None:
        self.encoded = encoded
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.encoded[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class BiLSTMAttentionModel(SentimentModel):
    """Trainer/wrapper around :class:`BiLSTMAttentionNet` implementing the
    shared :class:`SentimentModel` interface."""

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        attention_dim: int = 96,
        dropout: float = 0.4,
        max_len: int = 64,
        max_vocab: int = 30000,
        min_freq: int = 2,
        epochs: int = 8,
        batch_size: int = 64,
        lr: float = 2e-3,
        weight_decay: float = 1e-5,
        grad_clip: float = 5.0,
        early_stopping_patience: int = 3,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        self.hparams = dict(
            embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            bidirectional=bidirectional, attention_dim=attention_dim, dropout=dropout,
            max_len=max_len, max_vocab=max_vocab, min_freq=min_freq,
        )
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.patience = early_stopping_patience
        self.device = resolve_device(device)
        self.seed = seed
        self.labels = list(LABELS)

        self._pre = PreprocessingPipeline.for_linear()
        self.vocab: Vocabulary | None = None
        self.net: BiLSTMAttentionNet | None = None

    # -- config constructor -------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "BiLSTMAttentionModel":  # noqa: ANN001
        m, t = cfg.model, cfg.train
        return cls(
            embedding_dim=m.embedding_dim, hidden_dim=m.hidden_dim,
            num_layers=m.num_layers, bidirectional=m.bidirectional,
            attention_dim=m.attention_dim, dropout=m.dropout,
            max_len=m.vocab.max_len, max_vocab=m.vocab.max_size, min_freq=m.vocab.min_freq,
            epochs=t.epochs, batch_size=t.batch_size, lr=t.lr,
            weight_decay=t.weight_decay, grad_clip=t.grad_clip,
            early_stopping_patience=t.early_stopping_patience, device=t.device,
            seed=cfg.seed,
        )

    # -- helpers ------------------------------------------------------------
    def _encode(self, texts: Sequence[str]) -> List[List[int]]:
        assert self.vocab is not None
        max_len = self.hparams["max_len"]
        return [self.vocab.encode(self._pre.tokenize(t), max_len) for t in texts]

    def _make_loader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        encoded = self._encode(df["text"].tolist())
        labels = df["label"].map(LABEL2ID).tolist()
        ds = _TextDataset(encoded, labels)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    # -- training -----------------------------------------------------------
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "BiLSTMAttentionModel":
        torch.manual_seed(self.seed)
        tokenized = [self._pre.tokenize(t) for t in train_df["text"].tolist()]
        self.vocab = Vocabulary.build(
            tokenized, max_size=self.hparams["max_vocab"], min_freq=self.hparams["min_freq"]
        )
        self.net = BiLSTMAttentionNet(
            vocab_size=len(self.vocab),
            embedding_dim=self.hparams["embedding_dim"],
            hidden_dim=self.hparams["hidden_dim"],
            num_layers=self.hparams["num_layers"],
            bidirectional=self.hparams["bidirectional"],
            attention_dim=self.hparams["attention_dim"],
            num_classes=len(self.labels),
            dropout=self.hparams["dropout"],
            pad_index=self.vocab.pad_index,
        ).to(self.device)

        train_loader = self._make_loader(train_df, shuffle=True)
        val_loader = self._make_loader(val_df, shuffle=False) if val_df is not None else None

        # class weights counter imbalance
        counts = train_df["label"].map(LABEL2ID).value_counts().sort_index()
        weights = torch.tensor(
            (counts.sum() / (len(counts) * counts)).values, dtype=torch.float32
        ).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            running = 0.0
            for input_ids, target in train_loader:
                input_ids, target = input_ids.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                logits = self.net(input_ids)
                loss = criterion(logits, target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                optimizer.step()
                running += loss.item() * input_ids.size(0)
            train_loss = running / len(train_loader.dataset)

            val_loss = self._eval_loss(val_loader, criterion) if val_loader else train_loss
            logger.info("epoch %d | train_loss=%.4f | val_loss=%.4f", epoch, train_loss, val_loss)

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info("early stopping at epoch %d", epoch)
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def _eval_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        assert self.net is not None
        self.net.eval()
        total = 0.0
        with torch.no_grad():
            for input_ids, target in loader:
                input_ids, target = input_ids.to(self.device), target.to(self.device)
                total += criterion(self.net(input_ids), target).item() * input_ids.size(0)
        return total / len(loader.dataset)

    # -- inference ----------------------------------------------------------
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        assert self.net is not None and self.vocab is not None
        self.net.eval()
        encoded = self._encode(list(texts))
        probs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(encoded), self.batch_size):
                batch = torch.tensor(encoded[start : start + self.batch_size], dtype=torch.long)
                logits = self.net(batch.to(self.device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs) if probs else np.empty((0, len(self.labels)))

    def attention_weights(self, text: str) -> Tuple[List[str], np.ndarray]:
        """Return tokens and their attention weights for a single text."""
        assert self.net is not None and self.vocab is not None
        self.net.eval()
        tokens = self._pre.tokenize(text)
        ids = torch.tensor([self.vocab.encode(tokens, self.hparams["max_len"])], dtype=torch.long)
        with torch.no_grad():
            _, weights = self.net(ids.to(self.device), return_attention=True)
        w = weights.squeeze(0).cpu().numpy()[: len(tokens)]
        return tokens, w

    # -- persistence --------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        assert self.net is not None and self.vocab is not None
        torch.save(self.net.state_dict(), directory / "model.pt")
        self.vocab.save(directory / "vocab.json")
        meta = {"model_type": "bilstm_attention", "labels": self.labels, "hparams": self.hparams}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "BiLSTMAttentionModel":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        model = cls(**{k: meta["hparams"][k] for k in (
            "embedding_dim", "hidden_dim", "num_layers", "bidirectional",
            "attention_dim", "dropout", "max_len", "max_vocab", "min_freq")})
        model.labels = meta["labels"]
        model.vocab = Vocabulary.load(directory / "vocab.json")
        model.net = BiLSTMAttentionNet(
            vocab_size=len(model.vocab),
            embedding_dim=model.hparams["embedding_dim"],
            hidden_dim=model.hparams["hidden_dim"],
            num_layers=model.hparams["num_layers"],
            bidirectional=model.hparams["bidirectional"],
            attention_dim=model.hparams["attention_dim"],
            num_classes=len(model.labels),
            dropout=model.hparams["dropout"],
            pad_index=model.vocab.pad_index,
        )
        model.net.load_state_dict(torch.load(directory / "model.pt", map_location=model.device))
        model.net.to(model.device)
        return model
