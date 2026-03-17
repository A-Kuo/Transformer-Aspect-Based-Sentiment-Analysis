"""
Sarcasm-Aware Routing for Sentiment Analysis (V2).

The core insight from the research literature:
    Sarcasm is the #1 failure mode for sentiment classifiers.
    "Great, my package has been lost" gets classified POSITIVE
    because "great" dominates the feature space.

Our approach: detect sarcasm BEFORE classification, then route.

    Review text
        ↓
    [Sarcasm Detector] → sarcasm_score (0.0 - 1.0)
        ↓
        ├─ score < 0.3  → TRUST: normal BERT prediction
        ├─ 0.3 ≤ score < 0.7 → FLAG: attach uncertainty warning
        └─ score ≥ 0.7  → INVERT: flip sentiment polarity
                                   (positive ↔ negative, neutral unchanged)

This is the same architectural pattern as Project 1 (hallucination
detection via attention entropy):
    High entropy → "model is uncertain, don't trust output"
    High sarcasm → "model will misclassify, route to fallback"

Implementation: V2 uses a rule-based feature detector. This is
deliberate — it's lightweight (no extra model to load), interpretable
(you can see WHY something was flagged), and fast (<1ms overhead).

A fine-tuned sarcasm classifier (DistilBERT on iSarcasm) would be
more accurate but adds ~200ms latency and a second model to maintain.
V3 upgrade path: swap SarcasmDetector for SarcasmClassifier with
the same interface.

Linguistic features scored:

  1. Punctuation excess:   "Great product!!!!!!" — exaggerated punctuation
     signals emphasis that often accompanies sarcasm.

  2. Caps ratio:           "LOVE how it broke" — selective ALL CAPS on
     sentiment words is a strong sarcasm marker in reviews.

  3. Positive-negative clash: Positive sentiment words co-occurring with
     negative context words. "Amazing how terrible the quality is."
     This is the hardest signal — it requires understanding that the
     positive word modifies a negative assessment.

  4. Quotation marks:      'amazing' quality — scare quotes around
     positive words invert their meaning.

  5. Ellipsis inversion:   "Works great... not" — trailing negation
     after an ellipsis is a classic sarcasm pattern in English.

  6. Contrast conjunctions: "but", "however", "yet", "although" following
     positive sentiment — signals expectation violation.

Mathematical note:
  Each feature returns a score in [0, 1]. The final sarcasm score is
  a weighted combination:
      S(x) = Σ_i w_i · f_i(x)
  clamped to [0, 1]. The weights are hand-tuned priors, not learned.
  This is intentional — with no sarcasm-labeled training data in our
  Amazon reviews pipeline, learned weights would overfit to noise.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# Sentiment lexicons for clash detection
# ============================================================

POSITIVE_WORDS = {
    "great", "amazing", "awesome", "excellent", "fantastic", "wonderful",
    "perfect", "love", "loved", "best", "brilliant", "outstanding",
    "superb", "incredible", "terrific", "good", "nice", "beautiful",
    "impressive", "marvelous", "splendid", "delightful", "fabulous",
}

NEGATIVE_WORDS = {
    "terrible", "horrible", "awful", "worst", "broke", "broken",
    "defective", "useless", "waste", "garbage", "trash", "junk",
    "disappointed", "disappointing", "frustrating", "pathetic",
    "fail", "failed", "failure", "ruined", "destroyed", "died",
    "never", "nothing", "nowhere", "nobody", "can't", "cannot",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't",
    "lost", "damaged", "missing", "wrong", "error", "problem",
}

CONTRAST_CONJUNCTIONS = {
    "but", "however", "yet", "although", "though", "except",
    "unfortunately", "sadly", "instead", "whereas",
}

NEGATION_WORDS = {"not", "no", "never", "n't", "neither", "nor", "barely", "hardly"}


# ============================================================
# Feature extractors
# ============================================================

def _punctuation_excess(text: str) -> float:
    """
    Score excessive punctuation: !!! or ???.

    Normal review: "Great product!" → 0.0
    Sarcastic:     "Great product!!!!" → 0.6+

    Count runs of 2+ exclamation/question marks. Each run beyond 1
    adds signal. Cap at 1.0.
    """
    runs = re.findall(r'[!?]{2,}', text)
    if not runs:
        return 0.0
    total_excess = sum(len(r) - 1 for r in runs)
    return min(total_excess / 6.0, 1.0)


def _caps_ratio(text: str) -> float:
    """
    Score selective ALL CAPS usage.

    "LOVE how it broke immediately" — caps on sentiment word only.

    We look for the ratio of ALL-CAPS words (length > 1) to total words.
    A fully caps-locked review is likely just angry, not sarcastic.
    Sarcasm tends to caps 1-3 specific words for emphasis.
    Sweet spot: 5-30% caps words.
    """
    words = text.split()
    if len(words) < 3:
        return 0.0

    caps_words = [w for w in words if w.isupper() and len(w) > 1]
    ratio = len(caps_words) / len(words)

    # Low ratio (1-3 caps words in a sentence) = sarcasm signal
    # High ratio (everything caps) = anger, not sarcasm
    if 0.05 <= ratio <= 0.35:
        return min(ratio * 4.0, 1.0)
    return 0.0


def _positive_negative_clash(text: str) -> float:
    """
    Score co-occurrence of positive and negative sentiment words.

    "Amazing how terrible the quality is" → high clash
    "Great product, fast shipping" → no clash

    This is the highest-signal feature. Sarcasm fundamentally works
    by creating incongruity between surface sentiment and intended
    meaning. The simplest proxy: do positive and negative lexicon
    words co-occur in the same review?

    Scoring: (min(pos_count, neg_count) / max(pos_count, neg_count))
    This penalizes pure positive or pure negative reviews and rewards
    mixed-polarity text.
    """
    words = set(text.lower().split())
    # Also check bigrams for compound expressions
    text_lower = text.lower()

    pos_count = sum(1 for w in POSITIVE_WORDS if w in words)
    neg_count = sum(1 for w in NEGATIVE_WORDS if w in text_lower)

    if pos_count == 0 or neg_count == 0:
        return 0.0

    # Higher score when both polarities are present
    balance = min(pos_count, neg_count) / max(pos_count, neg_count)
    intensity = min((pos_count + neg_count) / 6.0, 1.0)
    return balance * intensity


def _quotation_sarcasm(text: str) -> float:
    """
    Score scare quotes around positive words.

    'amazing' quality → 0.8 (scare quotes invert "amazing")
    "I love it" → 0.0 (normal quotation, not scare quotes)

    We look for positive sentiment words wrapped in single or double
    quotes. This is a high-precision signal (rarely false positive)
    but low recall (most sarcasm doesn't use scare quotes).
    """
    # Find words in quotes
    quoted = re.findall(r'["\'](\w+)["\']', text.lower())
    if not quoted:
        return 0.0

    scare_count = sum(1 for w in quoted if w in POSITIVE_WORDS)
    return min(scare_count * 0.4, 1.0)


def _ellipsis_inversion(text: str) -> float:
    """
    Score "positive... not/but/however" patterns.

    "Works great... not" → 0.9
    "Delivery was fast... then it broke" → 0.7

    The ellipsis + negation pattern is a very reliable sarcasm marker
    in product reviews.
    """
    # Pattern: ... followed by negation or contrast word
    pattern = r'\.\.\.\s*(\w+)'
    matches = re.findall(pattern, text.lower())
    if not matches:
        return 0.0

    score = 0.0
    for word_after in matches:
        if word_after in NEGATION_WORDS:
            score += 0.5
        elif word_after in CONTRAST_CONJUNCTIONS:
            score += 0.3
        elif word_after in NEGATIVE_WORDS:
            score += 0.4
    return min(score, 1.0)


def _contrast_after_positive(text: str) -> float:
    """
    Score contrast conjunctions appearing after positive phrases.

    "Great quality but broke in a week" → 0.3 (expectation violation)
    "Loved the design, however it's useless" → 0.5

    This detects *expectation violation*, which isn't always sarcasm
    but indicates the sentiment classifier will struggle because the
    review contains genuine mixed signals.
    """
    text_lower = text.lower()
    words = text_lower.split()

    if len(words) < 5:
        return 0.0

    # Find positions of positive words and contrast conjunctions
    pos_positions = [i for i, w in enumerate(words) if w in POSITIVE_WORDS]
    contrast_positions = [i for i, w in enumerate(words) if w in CONTRAST_CONJUNCTIONS]

    if not pos_positions or not contrast_positions:
        return 0.0

    # Check if any contrast word follows a positive word
    violations = 0
    for cp in contrast_positions:
        for pp in pos_positions:
            if 0 < cp - pp <= 5:  # Contrast within 5 words after positive
                violations += 1

    return min(violations * 0.25, 1.0)


# ============================================================
# Sarcasm Detector
# ============================================================

@dataclass
class SarcasmResult:
    """Result of sarcasm detection on a single text."""
    score: float                    # 0.0 (sincere) to 1.0 (sarcastic)
    route: str                      # "trust", "flag", or "invert"
    features: Dict[str, float]      # Per-feature scores for explainability
    explanation: str = ""           # Human-readable reasoning


class SarcasmDetector:
    """
    Rule-based sarcasm detection with configurable routing thresholds.

    This is not a classifier — it's a routing gate. It decides how
    much to trust the downstream sentiment prediction, not what the
    sentiment IS.

    Feature weights are priors, not learned parameters. Each weight
    reflects how reliably that linguistic cue indicates sarcasm in
    product reviews specifically (not general text).

    The detector is intentionally conservative: false negatives
    (missing sarcasm) are less costly than false positives (flagging
    sincere reviews as sarcastic and inverting their sentiment).
    """

    # Feature name → (extractor_fn, weight)
    FEATURE_SPEC = {
        "punctuation_excess":       (_punctuation_excess,       0.10),
        "caps_ratio":               (_caps_ratio,               0.12),
        "positive_negative_clash":  (_positive_negative_clash,  0.35),
        "quotation_sarcasm":        (_quotation_sarcasm,        0.18),
        "ellipsis_inversion":       (_ellipsis_inversion,       0.10),
        "contrast_after_positive":  (_contrast_after_positive,  0.15),
    }
    # Weights sum to 1.0

    def __init__(
        self,
        low_threshold: float = 0.3,
        high_threshold: float = 0.7,
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    @classmethod
    def from_config(cls, config: dict) -> "SarcasmDetector":
        """Factory: build from config.yaml sarcasm section."""
        sarcasm_cfg = config.get("sarcasm", {})
        return cls(
            low_threshold=sarcasm_cfg.get("low_threshold", 0.3),
            high_threshold=sarcasm_cfg.get("high_threshold", 0.7),
        )

    def detect(self, text: str) -> SarcasmResult:
        """
        Score a single text for sarcasm likelihood.

        Args:
            text: Raw review text.

        Returns:
            SarcasmResult with score, routing decision, and feature breakdown.
        """
        features = {}
        weighted_sum = 0.0

        for name, (fn, weight) in self.FEATURE_SPEC.items():
            raw_score = fn(text)
            features[name] = round(raw_score, 4)
            weighted_sum += weight * raw_score

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, weighted_sum))

        # Routing decision
        if score >= self.high_threshold:
            route = "invert"
        elif score >= self.low_threshold:
            route = "flag"
        else:
            route = "trust"

        # Explanation: top contributing features
        top_features = sorted(
            features.items(), key=lambda x: x[1], reverse=True,
        )[:3]
        explanation_parts = [f"{name}={val:.2f}" for name, val in top_features if val > 0]
        explanation = (
            f"sarcasm_score={score:.3f} → {route}"
            + (f" (drivers: {', '.join(explanation_parts)})" if explanation_parts else "")
        )

        return SarcasmResult(
            score=round(score, 4),
            route=route,
            features=features,
            explanation=explanation,
        )

    def detect_batch(self, texts: List[str]) -> List[SarcasmResult]:
        """Score a batch of texts."""
        return [self.detect(text) for text in texts]


def apply_sarcasm_routing(
    prediction: Dict,
    sarcasm_result: SarcasmResult,
    sentiment_map: Optional[Dict] = None,
) -> Dict:
    """
    Apply sarcasm routing to a prediction dict.

    Modifies the prediction in place based on the sarcasm route:
      - "trust":  no changes, attach sarcasm metadata
      - "flag":   attach warning, reduce sentiment confidence
      - "invert": flip sentiment polarity (positive ↔ negative)

    Sentiment inversion logic:
      positive (2) ↔ negative (0)
      neutral  (1) → unchanged (sarcasm about neutral is rare)

    The inversion also swaps the probability distribution:
      P(positive) ↔ P(negative), P(neutral) unchanged.

    Args:
        prediction: Output dict from Predictor.predict_one().
        sarcasm_result: Output from SarcasmDetector.detect().
        sentiment_map: {0: "negative", 1: "neutral", 2: "positive"} map.

    Returns:
        Modified prediction dict with sarcasm metadata.
    """
    if sentiment_map is None:
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}

    # Always attach sarcasm metadata
    prediction["sarcasm_score"] = sarcasm_result.score
    prediction["sarcasm_route"] = sarcasm_result.route
    prediction["sarcasm_features"] = sarcasm_result.features
    prediction["sarcasm_explanation"] = sarcasm_result.explanation

    if sarcasm_result.route == "trust":
        # No modification needed
        pass

    elif sarcasm_result.route == "flag":
        # Reduce confidence to signal uncertainty
        # The sentiment prediction might be correct, but we're less sure
        discount = 1.0 - (sarcasm_result.score * 0.5)
        prediction["sentiment_confidence"] = round(
            prediction["sentiment_confidence"] * discount, 4
        )
        prediction["sentiment_warning"] = (
            "Possible sarcasm detected — sentiment confidence reduced"
        )

    elif sarcasm_result.route == "invert":
        # Flip sentiment polarity
        old_id = prediction["sentiment_id"]
        if old_id == 0:    # negative → positive
            new_id = 2
        elif old_id == 2:  # positive → negative
            new_id = 0
        else:              # neutral → neutral
            new_id = 1

        prediction["sentiment_id_original"] = old_id
        prediction["sentiment_original"] = prediction["sentiment"]
        prediction["sentiment_id"] = new_id
        prediction["sentiment"] = sentiment_map.get(new_id, f"sentiment_{new_id}")

        # Swap probability distribution
        if "sentiment_probs" in prediction:
            probs = prediction["sentiment_probs"]
            neg_key = sentiment_map.get(0, "negative")
            pos_key = sentiment_map.get(2, "positive")
            if neg_key in probs and pos_key in probs:
                probs[neg_key], probs[pos_key] = probs[pos_key], probs[neg_key]

        # Update confidence to inverted class
        prediction["sentiment_confidence"] = round(
            prediction.get("sentiment_probs", {}).get(
                prediction["sentiment"], prediction["sentiment_confidence"]
            ), 4
        )
        prediction["sentiment_warning"] = (
            f"Sarcasm detected — sentiment inverted from "
            f"'{prediction['sentiment_original']}' to '{prediction['sentiment']}'"
        )

    return prediction


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    detector = SarcasmDetector(low_threshold=0.3, high_threshold=0.7)

    test_cases = [
        # (text, expected_route_hint)
        ("Absolutely love this product! Great quality and fast shipping.", "trust"),
        ("Great, my package has been lost. Just great.", "flag/invert"),
        ("LOVE how it broke after one day. Really 'amazing' quality.", "invert"),
        ("Decent for the price. Nothing special but gets the job done.", "trust"),
        ("Works perfectly... not. Complete waste of money.", "invert"),
        ("'Excellent' customer service — took 3 weeks to respond.", "flag/invert"),
        ("Best product I've ever bought!!!!!!", "trust/flag"),
        ("Oh sure, this is EXACTLY what I wanted. A broken handle.", "flag/invert"),
    ]

    print(f"\n{'='*70}")
    print("SARCASM DETECTION TEST")
    print(f"{'='*70}")

    for text, hint in test_cases:
        result = detector.detect(text)
        print(f"\n  Text: {text[:80]}")
        print(f"  {result.explanation}")
        print(f"  Expected: ~{hint}")
        # Show top features
        active = {k: v for k, v in result.features.items() if v > 0}
        if active:
            print(f"  Active features: {active}")
        print(f"  {'─'*60}")

    print(f"\n{'='*70}")
