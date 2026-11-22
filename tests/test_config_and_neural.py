"""Tests for config loading, vocabulary, and (torch-gated) neural models."""

from __future__ import annotations

import pytest

from political_sentiment.models.vocab import PAD_TOKEN, UNK_TOKEN, Vocabulary
from political_sentiment.utils.config import Config, load_config


def test_config_inheritance_and_override(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("seed: 42\ntrain:\n  lr: 0.1\n  epochs: 3\n")
    exp = tmp_path / "exp.yaml"
    exp.write_text("experiment_name: x\ntrain:\n  lr: 0.5\n")
    cfg = load_config(exp, base_path=base)
    assert cfg.seed == 42               # inherited
    assert cfg.train.lr == 0.5          # overridden
    assert cfg.train.epochs == 3        # deep-merged, kept from base
    assert cfg.experiment_name == "x"


def test_config_get_default():
    cfg = Config({"a": {"b": 1}})
    assert cfg.a.b == 1
    assert cfg.get("missing", 99) == 99


def test_config_yaml_roundtrip(tmp_path):
    cfg = Config({"seed": 7, "nested": {"x": [1, 2]}})
    path = tmp_path / "out.yaml"
    cfg.to_yaml(path)
    reloaded = load_config(path, base_path=None)
    assert reloaded.seed == 7
    assert reloaded.nested.x == [1, 2]


def test_vocab_build_and_encode():
    texts = [["good", "plan"], ["good", "policy"], ["bad", "plan"]]
    vocab = Vocabulary.build(texts, min_freq=1)
    assert vocab.stoi[PAD_TOKEN] == 0
    assert vocab.stoi[UNK_TOKEN] == 1
    enc = vocab.encode(["good", "unseen"], max_len=4)
    assert len(enc) == 4
    assert enc[0] == vocab.stoi["good"]
    assert enc[1] == vocab.unk_index
    assert enc[2] == vocab.pad_index


def test_vocab_min_freq_pruning():
    texts = [["a", "b"], ["a"]]  # 'b' appears once
    vocab = Vocabulary.build(texts, min_freq=2)
    assert "a" in vocab.stoi
    assert "b" not in vocab.stoi


def test_vocab_save_load(tmp_path):
    vocab = Vocabulary.build([["x", "y"], ["x"]], min_freq=1)
    vocab.save(tmp_path / "v.json")
    reloaded = Vocabulary.load(tmp_path / "v.json")
    assert reloaded.stoi == vocab.stoi


# ------------------------- torch-gated neural tests ------------------------ #

def test_bilstm_end_to_end(small_corpus, torch_available):
    if not torch_available:
        pytest.skip("PyTorch not installed; skipping BiLSTM training test.")
    from political_sentiment.models.bilstm import BiLSTMAttentionModel

    model = BiLSTMAttentionModel(epochs=1, batch_size=32, hidden_dim=32, embedding_dim=32)
    model.fit(small_corpus.iloc[:200], small_corpus.iloc[200:260])
    proba = model.predict_proba(["a good plan", "a disaster"])
    assert proba.shape == (2, 3)
    tokens, weights = model.attention_weights("a good plan")
    assert len(tokens) == len(weights)
