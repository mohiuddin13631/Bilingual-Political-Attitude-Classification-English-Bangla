"""Tests for cleaning, tokenization and negation handling."""

from __future__ import annotations

from political_sentiment.preprocessing import (
    PreprocessingPipeline,
    clean_text,
    detect_language,
    mark_negation,
)
from political_sentiment.preprocessing.tokenizer import WhitespaceTokenizer


def test_clean_text_strips_urls_and_mentions():
    out = clean_text("Check https://x.com @leader now!!!", lowercase=True)
    assert "http" not in out
    assert "@leader" not in out
    assert "<url>" in out and "<user>" in out


def test_clean_text_preserves_bangla():
    text = "এই নীতি ভালো"
    out = clean_text(text)
    assert "নীতি" in out and "ভালো" in out


def test_collapse_repeats():
    assert "soo" in clean_text("soooo good")


def test_detect_language():
    assert detect_language("this is english text") == "en"
    assert detect_language("এই একটি বাংলা বাক্য") == "bn"
    assert detect_language("") == "en"


def test_tokenizer_splits_scripts_and_punct():
    toks = WhitespaceTokenizer().tokenize("good plan! ভালো পরিকল্পনা।")
    assert "good" in toks and "!" in toks and "।" in toks


def test_english_negation_scope():
    toks = ["the", "plan", "is", "not", "good", "at", "all"]
    marked = mark_negation(toks, window=3)
    assert "good_NEG" in marked
    assert "the" not in [t for t in marked if t.endswith("_NEG")]  # before cue untouched


def test_negation_stops_at_clause_end():
    toks = ["not", "good", ".", "great"]
    marked = mark_negation(toks)
    assert "good_NEG" in marked
    assert "great" in marked and "great_NEG" not in marked


def test_bangla_post_negation_backward_scope():
    # "মোটেও ভালো নয়" — নয় negates the preceding predicate.
    toks = ["মোটেও", "ভালো", "নয়"]
    marked = mark_negation(toks, window=3)
    assert any(t.endswith("_NEG") for t in marked)


def test_pipeline_presets_differ():
    linear = PreprocessingPipeline.for_linear()
    neural = PreprocessingPipeline.for_neural()
    text = "This is NOT Great!"
    # Linear lowercases + negation-marks; neural preserves case, no _NEG.
    assert linear.transform(text) != neural.transform(text)
    assert "_NEG" in linear.transform(text)
    assert "_NEG" not in neural.transform(text)
