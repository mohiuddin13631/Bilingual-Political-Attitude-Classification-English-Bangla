"""Fine-tuned multilingual transformer classifier — PyTorch + HuggingFace.

Default backbone: ``distilbert-base-multilingual-cased`` — a distilled, 6-layer
multilingual BERT that covers both English and Bangla and fine-tunes quickly on
CPU/modest GPU (report Section IV.C). Swap ``pretrained_name`` in the config to
``xlm-roberta-base`` for stronger multilingual performance on GPU hardware.

Imported lazily by the registry so the baseline path needs neither torch nor
transformers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .. import ID2LABEL, LABEL2ID, LABELS
from ..preprocessing import PreprocessingPipeline
from ..utils.logging_utils import get_logger
from .base import SentimentModel
from .bilstm import resolve_device

logger = get_logger(__name__)


class _EncodedDataset(Dataset):
    """Tokenized dataset yielding HuggingFace-style tensor batches."""

    def __init__(self, encodings: dict, labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class TransformerModel(SentimentModel):
    """Fine-tune and serve a HuggingFace sequence-classification transformer."""

    def __init__(
        self,
        pretrained_name: str = "distilbert-base-multilingual-cased",
        max_len: int = 64,
        dropout: float = 0.2,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        grad_clip: float = 1.0,
        early_stopping_patience: int = 2,
        fp16: bool = False,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        self.pretrained_name = pretrained_name
        self.max_len = max_len
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.grad_clip = grad_clip
        self.patience = early_stopping_patience
        self.fp16 = fp16
        self.device = resolve_device(device)
        self.seed = seed
        self.labels = list(LABELS)

        # Light cleaning only — the subword tokenizer prefers natural text.
        self._pre = PreprocessingPipeline.for_neural()
        self.tokenizer = None
        self.model = None

    @classmethod
    def from_config(cls, cfg) -> "TransformerModel":  # noqa: ANN001
        m, t = cfg.model, cfg.train
        return cls(
            pretrained_name=m.pretrained_name, max_len=m.max_len, dropout=m.dropout,
            epochs=t.epochs, batch_size=t.batch_size, lr=t.lr,
            weight_decay=t.weight_decay, warmup_ratio=t.warmup_ratio,
            grad_clip=t.grad_clip, early_stopping_patience=t.early_stopping_patience,
            fp16=t.fp16, device=t.device, seed=cfg.seed,
        )

    # -- helpers ------------------------------------------------------------
    def _tokenize(self, texts: Sequence[str]) -> dict:
        assert self.tokenizer is not None
        cleaned = [self._pre.transform(t) for t in texts]
        return dict(
            self.tokenizer(
                cleaned,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
            )
        )

    # -- training -----------------------------------------------------------
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "TransformerModel":
        torch.manual_seed(self.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_name)
        config = AutoConfig.from_pretrained(
            self.pretrained_name,
            num_labels=len(self.labels),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        for dropout_field in ("hidden_dropout_prob", "classifier_dropout", "seq_classif_dropout"):
            if hasattr(config, dropout_field):
                setattr(config, dropout_field, self.dropout)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_name,
            config=config,
        ).to(self.device)

        train_ds = _EncodedDataset(
            self._tokenize(train_df["text"].tolist()),
            train_df["label"].map(LABEL2ID).tolist(),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = None
        if val_df is not None:
            val_ds = _EncodedDataset(
                self._tokenize(val_df["text"].tolist()),
                val_df["label"].map(LABEL2ID).tolist(),
            )
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self.fp16 and self.device.type == "cuda")

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running = 0.0
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                with torch.autocast(
                    device_type=self.device.type, enabled=self.fp16 and self.device.type == "cuda"
                ):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running += loss.item() * batch["labels"].size(0)
            train_loss = running / len(train_loader.dataset)

            val_loss = self._eval_loss(val_loader) if val_loader else train_loss
            logger.info("epoch %d | train_loss=%.4f | val_loss=%.4f", epoch, train_loss, val_loss)

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info("early stopping at epoch %d", epoch)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _eval_loss(self, loader: DataLoader) -> float:
        assert self.model is not None
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                total += self.model(**batch).loss.item() * batch["labels"].size(0)
        return total / len(loader.dataset)

    # -- inference ----------------------------------------------------------
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        enc = self._tokenize(list(texts))
        n = len(texts)
        probs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                sl = slice(start, start + self.batch_size)
                batch = {k: torch.tensor(v[sl]).to(self.device) for k, v in enc.items()}
                logits = self.model(**batch).logits
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs) if probs else np.empty((0, len(self.labels)))

    # -- persistence --------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        assert self.model is not None and self.tokenizer is not None
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        meta = {
            "model_type": "transformer",
            "pretrained_name": self.pretrained_name,
            "labels": self.labels,
            "max_len": self.max_len,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "TransformerModel":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        model = cls(pretrained_name=meta["pretrained_name"], max_len=meta["max_len"])
        model.labels = meta["labels"]
        model.tokenizer = AutoTokenizer.from_pretrained(directory)
        model.model = AutoModelForSequenceClassification.from_pretrained(directory).to(model.device)
        return model
