"""Bilingual Political Attitude Classification (English & Bangla).

A modular, reproducible research codebase comparing a TF-IDF linear baseline,
a BiLSTM-with-attention network, and a fine-tuned multilingual transformer for
sentiment classification of political social-media text (English + Bangla).
"""

__version__ = "0.1.0"

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
