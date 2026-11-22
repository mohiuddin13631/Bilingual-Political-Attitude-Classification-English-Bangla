# Bilingual Political Attitude Classification (English & Bangla)

**A multilingual (English–Bangla) study comparing linear, recurrent, and transformer models, with bias, explainability, and regulatory analysis**

*Team Scube · 2026*

---

### Task scope

The rebranded project uses **political attitude** to mean document-level sentiment
(negative, neutral, positive) in English and Bangla. It does not implement
separate target-conditioned stance or political-affiliation classification.
The experiments described below retain their original sentiment task and labels;
the title change does not constitute new experimental evidence.

### Abstract

Political discourse on social media shapes public opinion, elections, and the
information environment, making automated analysis of political sentiment both
valuable and hazardous. This paper presents an end-to-end, reproducible system
for classifying the sentiment of political social-media posts into *negative*,
*neutral*, and *positive* classes across two languages — English and Bangla — and
systematically compares three model families: a TF-IDF linear baseline
(Logistic Regression / Linear SVM), a bidirectional LSTM with additive
attention, and a fine-tuned multilingual transformer (DistilBERT-multilingual /
XLM-RoBERTa). Beyond predictive accuracy, we treat *responsible deployment* as a
first-class concern: we quantify model bias across political parties, topics, and
languages; we attach model-agnostic explanations (attention, LIME, SHAP) to
individual predictions; and we analyse the system's obligations under the EU
General Data Protection Regulation (GDPR) and the EU AI Act. On a deliberately
hard, reproducible corpus (with negation, sarcasm, contrastive framing, and
annotator-style label noise setting an accuracy ceiling near 0.94), the linear
baselines reach ≈0.92 accuracy and macro-F1, providing a strong reference point;
the deep models are designed to close the remaining gap, particularly on
cross-lingual negation. We release the full codebase — data pipeline, models,
evaluation, fairness and explainability modules, unit tests, and a FastAPI +
Streamlit demo — with strict reproducibility controls.

**Keywords:** political sentiment analysis, social media, BERT, BiLSTM,
attention, multilingual NLP, Bangla, algorithmic bias, explainable AI, EU AI Act,
GDPR.

---

## I. Introduction and Motivation

### A. Why political sentiment analysis matters

Social media platforms have become a primary arena for political expression,
mobilisation, and persuasion. The automated measurement of *political sentiment*
— whether a post expresses approval, disapproval, or a neutral stance toward a
party, policy, or figure — underpins several socially important applications:

1. **Public-opinion tracking.** Continuous, large-scale sentiment signals
   complement traditional polling, offering higher temporal resolution and
   reach into populations that polls under-sample.
2. **Election analysis and forecasting.** Aggregate sentiment trends have been
   studied as leading indicators of electoral outcomes, though with well-known
   caveats about representativeness and manipulation.
3. **Polarisation measurement.** Tracking the distribution and intensity of
   sentiment across communities helps researchers quantify affective
   polarisation and its evolution.
4. **Misinformation and coordinated-behaviour detection.** Sudden, unnatural
   shifts in sentiment around a topic can flag influence operations or
   inauthentic amplification.

### B. Why it is hard — and risky

Political text is among the most difficult domains for sentiment analysis.
Sarcasm and irony are pervasive ("*Oh great, another broken promise*"); negation
routinely flips polarity ("*not good at all*"); posts are short, code-mixed, and
laden with named entities and hashtags; and the same words carry opposite valence
depending on the author's allegiance. Multilingual settings compound the
difficulty: lower-resource languages such as **Bangla** — the seventh most spoken
language in the world — have far fewer labelled resources and pretrained tools
than English, yet host vast, politically active online communities.

The risks are equally salient. A model that systematically assigns more negative
sentiment to one party than the ground truth warrants can distort the very public
opinion it purports to measure, and — if used in content moderation or
recommendation — can materially disadvantage a political group. For this reason
we treat **bias auditing, explainability, and regulatory compliance not as
add-ons but as core deliverables.**

### C. Contributions

- A **modular, reproducible, production-quality codebase** implementing and
  comparing three model families under a single interface, in PyTorch.
- A **multilingual (English + Bangla)** preprocessing and modelling pipeline with
  script-aware tokenization and bilingual negation handling.
- A **reproducible synthetic corpus generator** that injects the hard phenomena
  of political text (negation, sarcasm, contrast, label noise), plus documented
  loaders for real public datasets.
- A **responsible-AI layer**: bias analysis across parties/topics/languages,
  attention/LIME/SHAP explanations, and a GDPR / EU AI Act treatment.
- A **deployable demo**: FastAPI inference service and Streamlit dashboard.

---

## II. Related Work

### A. Lexicon and classical machine-learning approaches

Early sentiment analysis relied on sentiment lexicons (e.g. rule-based systems
such as VADER, tuned for social-media text) and on bag-of-words features fed to
linear classifiers. TF-IDF with Logistic Regression or Support Vector Machines
remains a remarkably strong, fast, and interpretable baseline, and is the
reference every neural model must justify itself against. The classic negation
heuristic of appending a `_NEG` marker to tokens within a negation scope
(Pang, Lee, and Vaithyanathan; Das and Chen) substantially improves these models
and is adopted here.

### B. Deep sequence models: CNNs and LSTMs

Convolutional networks for sentence classification (Kim, 2014) and recurrent
architectures — particularly the Long Short-Term Memory network (Hochreiter and
Schmidhuber, 1997) in its bidirectional form — brought distributed word
representations and word-order sensitivity to sentiment analysis. The
attention mechanism (Bahdanau, Cho, and Bengio, 2015), originally for machine
translation, was widely adopted to pool sequence representations and, as a
by-product, to expose which tokens a model attends to. Our BiLSTM-with-attention
model follows this lineage.

### C. Transformers and transfer learning

The transformer (Vaswani et al., 2017) and BERT (Devlin et al., 2019)
established pre-train-then-fine-tune as the dominant paradigm. For efficiency,
DistilBERT (Sanh et al., 2019) retains most of BERT's accuracy at ~40% fewer
parameters. For multilingual and cross-lingual settings, multilingual BERT and
XLM-RoBERTa (Conneau et al., 2020) enable a single model to serve many
languages — essential for an English–Bangla system, since Bangla-specific
pretrained models are comparatively scarce.

### D. Aspect-based and cross-lingual political NLP

Aspect-Based Sentiment Analysis (ABSA) refines sentiment to specific targets
(e.g. a party or a policy within one post) rather than a single document label —
directly relevant to political text, where a post may praise one actor while
criticising another. Cross-lingual political NLP and stance detection (e.g. the
SemEval stance-detection tasks) study how models transfer across languages and
how stance differs from sentiment. Our party/topic metadata and contrastive
examples are a step toward target-aware evaluation, and the bias analysis is
framed per party and per topic.

### E. Bias and explainability

Work on fairness in NLP documents how models encode and amplify social and
political asymmetries, and how sentiment systems can exhibit differential error
rates across groups. For interpretability, LIME (Ribeiro, Singh, and Guestrin,
2016) fits a local sparse linear surrogate around individual predictions, and
SHAP (Lundberg and Lee, 2017) provides additive, theoretically-grounded feature
attributions. We use both, plus the model's own attention, as complementary
lenses.

---

## III. Dataset

### A. Design rationale: synthetic corpus with real-data hooks

Reproducibility and licensing are in tension for this domain. The most relevant
real corpora — political tweets and Reddit threads — now sit behind restrictive
platform terms that forbid redistribution, and public political-sentiment
datasets are fragmented across shared tasks with heterogeneous licenses. To keep
the project **fully runnable and reproducible offline**, the default data source
is a **synthetic generator**; the codebase additionally ships **documented
loaders** (`data/loaders.py`) so a practitioner can drop in a real dataset via
CSV or the HuggingFace Hub with a one-line config change.

### B. Recommended real public datasets and licensing notes

| Dataset | Language(s) | Content | Licensing note |
|---|---|---|---|
| SemEval-2016 Task 6 (Stance) / Task 4 (Sentiment) | English | Tweets, stance + sentiment | Research use; accept task terms; tweet IDs (rehydrate) |
| GermEval / TASS | German / Spanish | Sentiment shared tasks | Research use; registration |
| SentNoB / Bangla sentiment sets | Bangla | Social-media comments | Public (CC-BY variants); verify per release |
| ParlSpeech / EU-Parliament debates | Multiple EU | Long-form political speech | CC-BY; redistributable |

Where raw text cannot be redistributed (e.g. tweet corpora shared as IDs), users
must rehydrate under the platform's terms. The loaders normalise any label space
onto the canonical three classes.

### C. Synthetic generator

The generator (`data/synthetic.py`) fills bilingual sentence templates from
curated sentiment lexicons, attaching *party* and *topic* metadata for the bias
analysis. Crucially, it injects the phenomena that make political sentiment hard,
so the task does not saturate:

- **Negation** (~18%): polarity-flipping cues, including Bangla's
  post-predicate markers (নয়/নেই).
- **Sarcasm** (~12% of negatives): positive-surface words in a negative frame,
  labelled by intended polarity.
- **Contrastive / mixed polarity** (~22%): a concession clause of opposite
  polarity followed by the dominant clause carrying the gold label.
- **Hedging** (~25%): softeners ("somewhat", "arguably", "কিছুটা").
- **Label noise** (~6%): annotator-style disagreement, setting a realistic
  accuracy ceiling near **0.94**.

The default corpus is 6,000 examples, balanced across three classes and ~50/50
across English and Bangla, split 70/15/15 (train/val/test) with stratification by
label and a fixed seed. Exploratory analysis (`notebooks/01_eda.ipynb`) confirms
class balance, language mix, and the presence of the hard phenomena.

---

## IV. Methodology

### A. Framework choice: PyTorch

We implement all neural models in **PyTorch**. Rationale: (i) the custom additive
attention and per-token weight extraction are expressed naturally and
transparently; (ii) the HuggingFace `transformers` ecosystem — the de-facto
standard for fine-tuning multilingual models — is PyTorch-first; (iii)
interpretability tooling (LIME/SHAP, captum-style attribution) integrates
cleanly; and (iv) PyTorch dominates recent European academic NLP, easing
extension into a thesis. All models expose the same `SentimentModel` interface
(`fit`, `predict_proba`, `predict`, `save`, `load`), so training, evaluation,
explainability, and serving code is model-agnostic.

### B. Preprocessing pipeline

Preprocessing (`preprocessing/`) is language-aware and model-appropriate:

- **Cleaning**: Unicode NFKC normalisation; URLs/mentions replaced with
  placeholder tokens (`<url>`, `<user>`); hashtags split and retained; emoji
  mapped to `<emoji>`; character elongation collapsed. Latin and the Bengali
  Unicode block (U+0980–U+09FF) are preserved; sentiment punctuation (`!`, `?`,
  Bengali danda `।`) is kept as tokens so negation scoping can detect clause
  boundaries.
- **Tokenization**: a script-aware regex tokenizer for the linear/BiLSTM path;
  the transformer uses its own subword tokenizer.
- **Negation handling**: `_NEG` suffixing within a bounded scope, forward for
  English/Bangla pre-cues and *backward* for Bangla post-predicate cues.
- **Presets**: an *aggressive* preset (lowercase, strip punctuation, negation
  marking) for the linear and BiLSTM models, and a *light* preset (preserve case
  and punctuation, no `_NEG`) for the transformer, which handles negation
  contextually.

### C. Models

**1) Baseline — TF-IDF + linear classifier.** Word n-gram (1–2) TF-IDF features
(sublinear term frequency, `min_df=2`, up to 50k features, with `_NEG` tokens
preserved) feed either multinomial Logistic Regression or a calibrated Linear
SVM (calibration exposes probabilities for ROC-AUC). Class weights counter
residual imbalance. The model reports its most informative n-grams per class,
giving immediate interpretability.

**2) BiLSTM + additive attention.** Tokens → embedding → (bi)LSTM → additive
(Bahdanau-style) attention pooling → dropout → linear → softmax. Attention
computes masked, softmax-normalised weights over time steps, yielding both a
fixed-size document vector and **per-token attention weights** that serve as a
faithful, built-in explanation. Trained with Adam, gradient clipping, class
weights, and early stopping on validation loss.

**3) Transformer — fine-tuned multilingual model.** `distilbert-base-
multilingual-cased` by default (covers English and Bangla, fine-tunes quickly),
swappable to `xlm-roberta-base` for stronger multilingual capacity on GPU
hardware. Fine-tuned with AdamW, linear warmup/decay, gradient clipping, mixed
precision (optional), and early stopping. The sequence-classification head maps
the pooled representation to three classes.

All hyper-parameters live in versioned YAML configs (`configs/`), deep-merged
onto a shared base config.

---

## V. Experiments and Results

### A. Protocol

Models are trained on the training split, selected on validation loss (early
stopping), and evaluated **once** on the held-out test split. All RNGs are seeded
from the config; the resolved config, logs, and metrics are written alongside
each run's artifacts. Metrics: accuracy, macro-F1, weighted-F1, macro
precision/recall, and one-vs-rest macro ROC-AUC. We additionally report the
confusion matrix, ROC curves, and a qualitative error analysis.

### B. Headline results

The two linear baselines were trained and evaluated in the reference environment.
The deep models reproduce with `make bilstm` / `make transformer` on a
torch-enabled machine; expected ranges below are grounded in the corpus's
label-noise ceiling (≈0.94) and in the literature for comparable multilingual
short-text tasks.

| Model | Accuracy | Macro-F1 | Precision (macro) | Recall (macro) | ROC-AUC (OvR) |
|---|:---:|:---:|:---:|:---:|:---:|
| TF-IDF + Logistic Regression | **0.921** | **0.921** | 0.922 | 0.921 | 0.951 |
| TF-IDF + Linear SVM | 0.921 | 0.921 | 0.922 | 0.921 | 0.948 |
| BiLSTM + Attention *(expected)* | ~0.92–0.94 | ~0.92–0.94 | — | — | ~0.95–0.97 |
| Transformer (DistilBERT-mBERT) *(expected)* | ~0.93–0.95 | ~0.93–0.95 | — | — | ~0.96–0.98 |

*Measured values are exact; italicised deep-model rows are reproducible
expectations, not claims. The comparison figure is written to
`artifacts/comparison/model_comparison.png`.*

The baselines are strong precisely because much of the signal is lexical. The
value of the deep models is concentrated in the **hard slices** — negation,
sarcasm, contrast, and cross-lingual transfer — where context and subword
information matter most; this is where the transformer is expected to separate
from the baseline rather than on aggregate accuracy alone.

### C. Confusion matrix

On the baseline, the **negative** class is hardest (row-normalised recall ≈0.88),
and its errors flow mainly toward **positive** — exactly the confusion induced by
negation ("*not good*") and sarcasm ("*oh great … another broken promise*").
Neutral and positive are recovered at ≈0.93 and ≈0.95. This structure motivates
the deep models and the negation-aware preprocessing.

### D. Qualitative error analysis

The most-confident errors (`artifacts/*/eval/top_errors.csv`) cluster into three
recognisable groups:

1. **Cross-lingual negation.** The baseline over-generalises the Bangla
   construction "মোটেও X নয়" ("not X at all"): having seen it mostly negate
   *negative* words (yielding positive posts), it mislabels "মোটেও ভালো নয়"
   ("not good at all") as positive. A contextual model is expected to resolve
   this.
2. **Sarcasm.** Positive-surface sarcasm remains the residual hard case for all
   bag-of-words features.
3. **Contrastive posts.** Mixed-polarity posts where the dominant clause is
   short are occasionally dominated by the longer concession clause.

Error rate is modestly higher on **Bangla (≈8.4%)** than **English (≈7.3%)** —
an expected lower-resource gap and an explicit fairness concern (§VI).

---

## VI. Bias and Fairness Analysis

Political sentiment models are unusually prone to harmful bias: systematic,
party-correlated errors can distort measured opinion and, downstream,
disadvantage a political group. The `fairness/` module quantifies three
complementary lenses (`artifacts/*/eval/fairness_*.csv`).

### A. Group performance parity

We compute accuracy, macro-F1, and ROC-AUC **per party**, **per topic**, and
**per language**, and summarise disparity as the max–min gap and standard
deviation of macro-F1. On the baseline, the **party F1 gap is small (≈0.017)**,
indicating the model does not serve one party markedly worse than another *on
this corpus*. The **language gap** is the more material finding: Bangla trails
English, consistent with the error analysis.

### B. Predicted-sentiment distribution (favourability bias)

Parity of *error* is necessary but not sufficient. A model can be equally
accurate yet still shift the *sentiment mix* it assigns to a party away from the
gold distribution — the favourability bias that matters most politically.
`sentiment_distribution_by_group` reports, per party and class, the gold share,
the predicted share, and their difference (`pred_minus_gold`). Large positive or
negative deltas for a specific party are the red flag to investigate before any
deployment.

### C. Interpretation and caveats

These are **descriptive diagnostics, not a fairness certificate**. On synthetic
data, party is independent of sentiment by construction, so small gaps are
expected; on real data, part of any measured gap reflects genuine differences in
how parties are discussed, and disentangling *model* bias from *world* bias
requires careful, target-aware annotation. The tooling is designed to make that
audit routine and repeatable.

---

## VII. Reproducibility

Reproducibility is treated as an engineering requirement:

- **Seed control.** `seed_everything` seeds Python, NumPy, and PyTorch (incl.
  CUDA) and requests deterministic cuDNN kernels; called in every entry point.
- **Logged configs.** The fully-resolved config is serialised next to each run's
  artifacts, together with run logs and metrics JSON.
- **Deterministic data.** The synthetic generator and the stratified splits are
  seed-controlled and reproduce byte-for-byte.
- **Pinned environment.** `requirements.txt` pins compatible version ranges; the
  package is installable via `pip install -e .`.
- **Tests.** A unit-test suite covers preprocessing, data, the baseline, metrics,
  fairness, config, and vocabulary; neural tests run where PyTorch is present and
  skip cleanly otherwise.

---

## VIII. Explainability

Explainability is both a scientific and, increasingly, a legal requirement
(§IX). The `explainability/` module offers three model-agnostic or model-native
methods over the shared `predict_proba` interface:

- **Attention (BiLSTM, native).** Per-token attention weights show which words
  drove a prediction — faithful and free of a surrogate model.
- **LIME.** Perturbs the input by removing words and fits a local sparse linear
  surrogate, yielding per-word contributions toward the predicted class.
- **SHAP.** Additive feature attributions via a text partition masker, suitable
  for global and local analysis.

These are surfaced in the Streamlit dashboard (per-prediction token explanations)
and are the mechanism by which an operator can justify an individual
classification — the "explanation of a specific decision" that regulators expect
for high-stakes, opinion-shaping systems.

---

## IX. Ethics, GDPR, and the EU AI Act

### A. Data protection (GDPR)

Political social-media text is **special-category personal data** under GDPR
Art. 9 (data revealing political opinions), attracting heightened protection.
Design implications, reflected in this project:

- **Lawful basis and minimisation.** Processing political opinions requires an
  Art. 9 condition (e.g. data manifestly made public by the data subject, or
  explicit consent) and strict data minimisation. The shipped pipeline uses
  **synthetic data by default**, avoiding personal data entirely; the real-data
  loaders should be used only with a documented lawful basis.
- **Purpose limitation and storage.** Sentiment labels should serve the stated
  research/analytics purpose only; raw text retention should be minimised and
  time-bounded. Party/topic metadata here is used **solely for offline bias
  auditing**, never as a model feature.
- **Data-subject rights.** Rehydration-based datasets (tweet IDs) naturally honour
  deletion, since removed posts cannot be re-fetched.
- **Anonymisation.** Mentions and user handles are stripped in preprocessing.

### B. The EU AI Act

Under the EU AI Act's risk-based framework, a political-sentiment system's
classification depends on use. Analytics and academic research are typically
lower-risk, but deployments that **influence political processes, election
information, or content visibility** can approach high-risk or trigger specific
obligations, and systems interacting with people or generating/curating content
carry **transparency duties**. The Act's emphasis on **risk management, data
governance, transparency, human oversight, and technical documentation** maps
directly onto this project's fairness auditing, explainability, reproducibility
logging, and documentation. In particular:

- **Transparency.** Users of any deployed demo should be told they are seeing
  automated sentiment estimates, not endorsements — reflected in the dashboard's
  disclaimer.
- **Human oversight.** The system is positioned as decision-support with
  per-prediction explanations, not autonomous action.
- **Documentation.** This report, the logged configs, and the evaluation bundles
  constitute the technical documentation trail such regimes expect.

### C. Broader ethical considerations

Sentiment about politics is not an endorsement; aggregate signals can be gamed by
coordinated behaviour; and deploying such a model at scale can itself influence
the discourse it measures. We therefore recommend transparency about method and
uncertainty, continuous bias monitoring, and human review for any consequential
use.

*This section is an engineering-oriented summary, not legal advice; a
deployment should obtain a jurisdiction-specific legal assessment.*

---

## X. Deployment

The system ships with a real-time demo:

- **FastAPI service** (`api/`): a typed `/predict` endpoint (Pydantic
  request/response schemas), a `/health` probe, and auto-generated OpenAPI docs.
  The model is discovered from `artifacts/` at startup (transformer → BiLSTM →
  baseline) and loaded once.
- **Streamlit dashboard** (`dashboard/`): single-text and batch classification
  with probability bars and optional token-level explanations; a sidebar states
  the transparency disclaimer.

Both reuse a single `SentimentService`, so serving logic is defined once and is
model-agnostic. The design is container-ready and scales horizontally behind a
load balancer for production use.

---

## XI. Limitations and Future Work

- **Synthetic data.** The default corpus, while deliberately hard, is simpler and
  more separable than real political text; absolute numbers will differ on real
  data, and the favourability-bias analysis is only meaningful once real,
  target-annotated data is used.
- **Two languages, coarse labels.** We model three-class document sentiment in
  English and Bangla. Natural extensions are **aspect-based** sentiment (per
  target within a post), finer-grained or intensity labels, and additional
  languages via XLM-R.
- **Sarcasm and code-mixing.** Both remain hard; dedicated sarcasm modelling and
  code-mixed (Banglish) handling are promising directions.
- **Bias mitigation.** We *measure* bias; future work should add mitigation
  (reweighting, adversarial debiasing, counterfactual data augmentation) and
  target-aware fairness metrics.
- **Robustness.** Adversarial and distribution-shift evaluation (e.g. across time
  and events) would strengthen deployment readiness.

---

## XII. Conclusion

We presented a complete, reproducible, and responsibly-engineered system for
multilingual (English–Bangla) political sentiment analysis on social media,
comparing linear, recurrent-attention, and transformer models under a unified
interface. On a deliberately hard, reproducible corpus, strong linear baselines
reach ≈0.92 accuracy and macro-F1, and the deep models are positioned to improve
the hard, context-dependent and cross-lingual cases. Equally important, the
project operationalises **bias auditing, explainability, and GDPR / EU AI Act
compliance** as first-class components, with a deployable FastAPI + Streamlit
demo. The codebase is structured to be extended directly into a thesis chapter or
a conference submission, and to be re-run on real public datasets with a
single-line configuration change.

---

## References

1. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," *NAACL-HLT*, 2019.
2. A. Vaswani et al., "Attention Is All You Need," *NeurIPS*, 2017.
3. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, 9(8), 1997.
4. D. Bahdanau, K. Cho, and Y. Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate," *ICLR*, 2015.
5. Y. Kim, "Convolutional Neural Networks for Sentence Classification," *EMNLP*, 2014.
6. V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT," *arXiv:1910.01108*, 2019.
7. A. Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)," *ACL*, 2020.
8. M. T. Ribeiro, S. Singh, and C. Guestrin, "\"Why Should I Trust You?\": Explaining the Predictions of Any Classifier (LIME)," *KDD*, 2016.
9. S. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions (SHAP)," *NeurIPS*, 2017.
10. B. Pang, L. Lee, and S. Vaithyanathan, "Thumbs up? Sentiment Classification using Machine Learning Techniques," *EMNLP*, 2002.
11. C. J. Hutto and E. Gilbert, "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text," *ICWSM*, 2014.
12. S. Mohammad, S. Kiritchenko, P. Sobhani, X. Zhu, and C. Cherry, "SemEval-2016 Task 6: Detecting Stance in Tweets," *SemEval*, 2016.
13. A. Go, R. Bhayani, and L. Huang, "Twitter Sentiment Classification using Distant Supervision (Sentiment140)," Stanford, 2009.
14. T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," *EMNLP: System Demonstrations*, 2020.
15. European Parliament and Council, "Regulation (EU) 2016/679 (General Data Protection Regulation)," 2016.
16. European Parliament and Council, "Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (AI Act)," 2024.
17. T. Bolukbasi, K.-W. Chang, J. Zou, V. Saligrama, and A. Kalai, "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings," *NeurIPS*, 2016.
18. M. Hardt, E. Price, and N. Srebro, "Equality of Opportunity in Supervised Learning," *NeurIPS*, 2016.

---

*Reproduce all measured results with `make data && make baseline && make evaluate CONFIG=configs/baseline_tfidf.yaml`; train the deep models with `make bilstm` and `make transformer` on a torch-enabled machine; compare with `make compare`.*
