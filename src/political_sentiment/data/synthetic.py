"""Synthetic political-sentiment corpus generator (English + Bangla).

Why synthetic?
--------------
Real political social-media corpora (Twitter/X, Reddit) now sit behind
restrictive API terms and cannot be redistributed with a research repo (see
report Section IX, Ethics & GDPR). To keep the project *fully runnable and
reproducible offline*, we generate a labelled corpus from curated sentiment
lexicons and sentence templates, in two languages, with party/topic metadata
for the fairness analysis. Documented hooks in :mod:`.loaders` let a user swap
in a real public dataset.

The generator deliberately injects two phenomena that stress sentiment models:

* **Negation** — e.g. "not good", "কখনোই ভালো নয়" — which flips polarity.
* **Sarcasm** — positive-sounding words in a negative frame ("oh great, another
  broken promise"), labelled by the *intended* polarity, not surface words.

Labels remain internally consistent: the polarity of the chosen template and
lexicon, after negation/sarcasm transforms, determines the gold label.
"""

from __future__ import annotations

import random
import uuid
from typing import Dict, List, Sequence

from .schema import SentimentRecord

# --------------------------------------------------------------------------- #
# Sentiment lexicons (adjective/phrase banks) per language and polarity.
# --------------------------------------------------------------------------- #
_LEXICON: Dict[str, Dict[str, List[str]]] = {
    "en": {
        "positive": [
            "excellent", "a real step forward", "genuinely promising",
            "impressive", "well handled", "a huge win", "encouraging",
            "exactly what we needed", "responsible and fair", "inspiring",
        ],
        "negative": [
            "a disaster", "deeply disappointing", "corrupt", "a broken promise",
            "reckless", "out of touch", "an embarrassment", "harmful",
            "poorly thought out", "a betrayal of voters",
        ],
        "neutral": [
            "under discussion", "on the agenda", "being reviewed",
            "reported today", "scheduled for a vote", "unchanged this quarter",
            "the subject of a new report", "still being debated",
        ],
    },
    "bn": {
        "positive": [
            "চমৎকার", "সত্যিই আশাব্যঞ্জক", "প্রশংসনীয়", "একটি বড় সাফল্য",
            "দায়িত্বশীল", "অনুপ্রেরণাদায়ক", "খুবই কার্যকর", "সঠিক পদক্ষেপ",
        ],
        "negative": [
            "একটি বিপর্যয়", "চরম হতাশাজনক", "দুর্নীতিগ্রস্ত", "ভাঙা প্রতিশ্রুতি",
            "বেপরোয়া", "জনবিচ্ছিন্ন", "ক্ষতিকর", "ভোটারদের সাথে বিশ্বাসঘাতকতা",
        ],
        "neutral": [
            "আলোচনায় রয়েছে", "পর্যালোচনা করা হচ্ছে", "আজ রিপোর্ট করা হয়েছে",
            "ভোটের জন্য নির্ধারিত", "এই প্রান্তিকে অপরিবর্তিত", "বিতর্কের বিষয়",
        ],
    },
}

# Hedges / intensifiers that weaken or complicate polarity (make the task
# realistically hard rather than trivially separable).
_HEDGE: Dict[str, List[str]] = {
    "en": ["kind of", "somewhat", "arguably", "to be fair", "i guess", "more or less"],
    "bn": ["অনেকটা", "কিছুটা", "সম্ভবত", "একরকম", "মোটামুটি"],
}

# Contrastive connectors used to build genuinely ambiguous, mixed-polarity posts
# whose gold label is the *dominant* (second) clause.
_CONTRAST: Dict[str, str] = {"en": "but", "bn": "কিন্তু"}

# Sentence templates. "{party}" / "{topic}" / "{phrase}" are slot markers.
_TEMPLATES: Dict[str, List[str]] = {
    "en": [
        "{party}'s stance on {topic} is {phrase}.",
        "Honestly, the {topic} plan from {party} is {phrase}.",
        "What {party} did about {topic} today was {phrase}.",
        "The new {topic} policy is {phrase}, and voters are noticing.",
        "{party} on {topic}: {phrase}, if you ask me.",
        "Reading about {party} and {topic} — {phrase}.",
    ],
    "bn": [
        "{topic} নিয়ে {party}-এর অবস্থান {phrase}।",
        "{party}-এর {topic} পরিকল্পনা {phrase}।",
        "আজ {topic} নিয়ে {party} যা করল তা {phrase}।",
        "নতুন {topic} নীতি {phrase}, আর ভোটাররা তা লক্ষ্য করছে।",
        "{party} এবং {topic} প্রসঙ্গে — {phrase}।",
    ],
}

# Negation frames flip positive<->negative surface polarity.
_NEGATION: Dict[str, str] = {
    "en": "not {phrase}",
    "bn": "মোটেও {phrase} নয়",
}

# Sarcasm frames wrap positive-sounding words in a negative intent.
_SARCASM: Dict[str, List[str]] = {
    "en": [
        "Oh {phrase}, another {topic} promise from {party}. Sure.",
        "Wow, {party} being {phrase} on {topic} again. Shocking.",
    ],
    "bn": [
        "বাহ, {topic} নিয়ে {party} আবার {phrase}। দারুণ।",
        "{party} তো {topic}-এ {phrase}! সত্যিই অসাধারণ।",
    ],
}


class SyntheticCorpusGenerator:
    """Generate a balanced, reproducible political-sentiment corpus.

    Parameters
    ----------
    languages, parties, topics, labels:
        Vocabularies used to fill templates and metadata.
    seed:
        RNG seed for reproducibility.
    sarcasm_rate, negation_rate:
        Fraction of eligible examples that receive each transform.
    """

    def __init__(
        self,
        languages: Sequence[str] = ("en", "bn"),
        parties: Sequence[str] = ("PartyA", "PartyB", "PartyC", "Independent"),
        topics: Sequence[str] = ("economy", "healthcare", "immigration", "climate"),
        labels: Sequence[str] = ("negative", "neutral", "positive"),
        seed: int = 42,
        sarcasm_rate: float = 0.12,
        negation_rate: float = 0.18,
        hedge_rate: float = 0.25,
        contrast_rate: float = 0.22,
        label_noise: float = 0.06,
    ) -> None:
        self.languages = list(languages)
        self.parties = list(parties)
        self.topics = list(topics)
        self.labels = list(labels)
        self.sarcasm_rate = sarcasm_rate
        self.negation_rate = negation_rate
        # hedge_rate: fraction of examples softened with a hedge (harder signal)
        # contrast_rate: fraction built as mixed-polarity contrastive posts
        # label_noise: fraction of gold labels randomly corrupted, emulating the
        #   annotator disagreement inherent in real political sentiment data.
        self.hedge_rate = hedge_rate
        self.contrast_rate = contrast_rate
        self.label_noise = label_noise
        self._rng = random.Random(seed)

    # -- topic surface forms -------------------------------------------------
    _TOPIC_BN = {
        "economy": "অর্থনীতি", "healthcare": "স্বাস্থ্যসেবা",
        "immigration": "অভিবাসন", "climate": "জলবায়ু",
        "security": "নিরাপত্তা", "education": "শিক্ষা",
    }

    def _topic_surface(self, topic: str, lang: str) -> str:
        return self._TOPIC_BN.get(topic, topic) if lang == "bn" else topic

    def _one(self, lang: str, label: str) -> SentimentRecord:
        """Produce a single record for a given language and gold label."""
        party = self._rng.choice(self.parties)
        topic = self._rng.choice(self.topics)
        topic_surface = self._topic_surface(topic, lang)

        use_sarcasm = label == "negative" and self._rng.random() < self.sarcasm_rate
        use_contrast = (
            not use_sarcasm
            and label != "neutral"
            and self._rng.random() < self.contrast_rate
        )
        use_negation = (
            label in ("positive", "negative")
            and not use_sarcasm
            and not use_contrast
            and self._rng.random() < self.negation_rate
        )

        if use_sarcasm:
            # Sarcasm: pull a positive-sounding phrase but keep the negative label.
            phrase = self._rng.choice(_LEXICON[lang]["positive"])
            frame = self._rng.choice(_SARCASM[lang])
            text = frame.format(phrase=phrase, topic=topic_surface, party=party)
        elif use_contrast:
            # Mixed-polarity: a concession clause of the opposite polarity, then
            # the dominant clause carrying the gold label. Genuinely ambiguous.
            opposite = "negative" if label == "positive" else "positive"
            concession = self._rng.choice(_LEXICON[lang][opposite])
            dominant = self._rng.choice(_LEXICON[lang][label])
            template = self._rng.choice(_TEMPLATES[lang])
            first = template.format(party=party, topic=topic_surface, phrase=concession)
            # strip trailing punctuation before the contrast connector
            first = first.rstrip("।. ")
            text = f"{first} {_CONTRAST[lang]} {dominant}."
        else:
            if use_negation:
                # Negate: surface lexicon is the *opposite* polarity, then negated.
                surface_polarity = "negative" if label == "positive" else "positive"
                base_phrase = self._rng.choice(_LEXICON[lang][surface_polarity])
                phrase = _NEGATION[lang].format(phrase=base_phrase)
            else:
                phrase = self._rng.choice(_LEXICON[lang][label])
            if self._rng.random() < self.hedge_rate:
                phrase = f"{self._rng.choice(_HEDGE[lang])} {phrase}"
            template = self._rng.choice(_TEMPLATES[lang])
            text = template.format(party=party, topic=topic_surface, phrase=phrase)

        # Inject annotator-style label noise: the *observed* label may differ
        # from the generating polarity (kept separate so text stays coherent).
        observed = label
        if self._rng.random() < self.label_noise:
            observed = self._rng.choice([lbl for lbl in self.labels if lbl != label])

        return SentimentRecord(
            id=str(uuid.UUID(int=self._rng.getrandbits(128))),
            text=text,
            label=observed,
            language=lang,
            party=party,
            topic=topic,
            source="synthetic",
        )

    def generate(self, n_samples: int, balanced: bool = True) -> List[SentimentRecord]:
        """Generate ``n_samples`` records.

        With ``balanced=True`` the label distribution is (near) uniform; the
        language distribution is always near-uniform across ``self.languages``.
        """
        records: List[SentimentRecord] = []
        if balanced:
            per_cell = max(1, n_samples // (len(self.languages) * len(self.labels)))
            for lang in self.languages:
                for label in self.labels:
                    for _ in range(per_cell):
                        records.append(self._one(lang, label))
        else:
            # Natural-ish skew: more negative/positive than neutral.
            weights = {"negative": 0.4, "positive": 0.4, "neutral": 0.2}
            for _ in range(n_samples):
                lang = self._rng.choice(self.languages)
                label = self._rng.choices(
                    self.labels, weights=[weights.get(x, 1.0) for x in self.labels]
                )[0]
                records.append(self._one(lang, label))

        self._rng.shuffle(records)
        return records
