"""Streamlit dashboard: real-time political sentiment demo.

Run with::

    streamlit run src/political_sentiment/dashboard/app.py

Features
--------
* Single-text and batch classification with probability bars.
* Optional token-level explanation (BiLSTM attention, or LIME for any model)
  when the backing libraries are installed.
* Works against whichever model was trained (transformer > BiLSTM > baseline),
  or against a running FastAPI service via its URL.
"""

from __future__ import annotations

import os
from typing import List

import pandas as pd
import streamlit as st

# Streamlit runs this file as a script; make the package importable either way.
try:
    from political_sentiment.inference import SentimentService
except ImportError:  # pragma: no cover - path shim for `streamlit run src/...`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from political_sentiment.inference import SentimentService

EXAMPLES = [
    "PartyA's plan on the economy is a real step forward.",
    "Honestly the immigration policy from PartyB is a broken promise.",
    "Oh great, another climate promise from PartyC. Sure.",
    "এই স্বাস্থ্যসেবা নীতি মোটেও ভালো নয়।",
    "অর্থনীতি নিয়ে PartyA যা করল তা সত্যিই আশাব্যঞ্জক।",
]

_COLOR = {"negative": "#d1495b", "neutral": "#8d99ae", "positive": "#2a9d8f"}


@st.cache_resource(show_spinner="Loading model…")
def get_service() -> SentimentService:
    """Load the trained model once and cache it across reruns."""
    artifacts_dir = os.environ.get("PS_ARTIFACTS_DIR", "artifacts")
    return SentimentService.from_artifacts(artifacts_dir)


def _render_prediction(result: dict) -> None:
    label = result["label"]
    st.markdown(
        f"**Prediction:** <span style='color:{_COLOR.get(label, '#333')};"
        f"font-weight:700'>{label.upper()}</span> "
        f"(confidence {result['confidence']:.2f})",
        unsafe_allow_html=True,
    )
    proba = result["probabilities"]
    st.bar_chart(pd.Series(proba, name="probability"))


def main() -> None:
    st.set_page_config(
        page_title="Bilingual Political Attitude Classification (English & Bangla)",
        page_icon="🗳️",
        layout="wide",
    )
    st.title("Bilingual Political Attitude Classification (English & Bangla)")
    st.caption(
        "Classify political attitudes expressed in English and Bangla text as "
        "negative, neutral, or positive sentiment. Research demo; see the report "
        "for methodology and bias analysis."
    )

    try:
        service = get_service()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.header("About")
    st.sidebar.write(f"**Model:** `{type(service.model).__name__}`")
    st.sidebar.write(f"**Labels:** {', '.join(service.labels)}")
    st.sidebar.info(
        "Predicted sentiment is about the *text*, not an endorsement. "
        "Party/topic metadata is used only for offline bias auditing."
    )

    tab_single, tab_batch = st.tabs(["Single text", "Batch"])

    with tab_single:
        choice = st.selectbox("Try an example, or type your own below:", [""] + EXAMPLES)
        text = st.text_area("Text", value=choice, height=120, placeholder="Enter a political post…")
        if st.button("Analyze", type="primary") and text.strip():
            result = service.predict([text])[0]
            _render_prediction(result)

            with st.expander("Why this prediction? (token explanation)"):
                _render_explanation(service, text)

    with tab_batch:
        raw = st.text_area(
            "One text per line", height=160,
            value="\n".join(EXAMPLES),
        )
        if st.button("Analyze batch"):
            texts: List[str] = [ln for ln in raw.splitlines() if ln.strip()]
            results = service.predict(texts)
            df = pd.DataFrame(
                [
                    {"text": r["text"], "label": r["label"], "confidence": r["confidence"]}
                    for r in results
                ]
            )
            st.dataframe(df, use_container_width=True)
            st.bar_chart(df["label"].value_counts())


def _render_explanation(service: SentimentService, text: str) -> None:
    """Best-effort token explanation, degrading gracefully if libs are absent."""
    model = service.model
    # Prefer built-in attention (BiLSTM) — faithful and dependency-free.
    if hasattr(model, "attention_weights"):
        try:
            tokens, weights = model.attention_weights(text)  # type: ignore[attr-defined]
            st.write("Attention weights (BiLSTM):")
            st.dataframe(pd.DataFrame({"token": tokens, "weight": weights.round(3)}))
            return
        except Exception as exc:  # pragma: no cover
            st.caption(f"Attention unavailable: {exc}")
    # Fall back to LIME if installed.
    try:
        from political_sentiment.explainability import explain_text

        exp = explain_text(model, text, method="lime", num_features=8, num_samples=500)
        st.write(f"LIME contributions toward **{exp['predicted_class']}**:")
        st.dataframe(pd.DataFrame(exp["contributions"], columns=["token", "weight"]))
    except Exception as exc:  # pragma: no cover
        st.caption(f"Install `lime` or `shap` for token explanations. ({exc})")


if __name__ == "__main__":
    main()
