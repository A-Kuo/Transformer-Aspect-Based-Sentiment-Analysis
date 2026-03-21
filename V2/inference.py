"""
Inference pipeline for BERT Aspect-Based Sentiment Analysis (V2).

Pipeline (V2 with sarcasm-aware routing):
    Raw text(s)
        ↓
    [Sarcasm Detector] → sarcasm_score (0-1)
        ↓
    ├─ score < 0.3  → TRUST path
    ├─ 0.3-0.7      → FLAG path (reduced confidence)
    └─ score ≥ 0.7  → INVERT path (flip sentiment)
        ↓
    [Tokenize] → input_ids, attention_mask
        ↓
    [Model.predict()] → logits → softmax → argmax
        ↓
    [Apply sarcasm routing] → modify prediction if needed
        ↓
    [Map labels] → human-readable aspect + sentiment + sarcasm metadata
        ↓
    Output: {text, aspect, sentiment, sarcasm_score, sarcasm_route, ...}

This follows the uncertainty-aware routing pattern:
    High attention entropy → "model uncertain, flag output"
    High sarcasm score    → "model will misclassify, route to fallback"
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

from src.data import load_config
from src.model import AspectSentimentModel, build_model
from src.train import load_checkpoint
from src.sarcasm import SarcasmDetector, apply_sarcasm_routing

logger = logging.getLogger(__name__)


class Predictor:
    """
    Production inference wrapper.

    Handles tokenization, batching, prediction, and label mapping
    in a single interface. Tracks latency for monitoring.

    Usage:
        predictor = Predictor.from_checkpoint("models/checkpoint_best.pt")
        result = predictor.predict_one("Great product but slow shipping")
        results = predictor.predict_batch(["review1", "review2", ...])
    """

    def __init__(
        self,
        model: AspectSentimentModel,
        tokenizer: AutoTokenizer,
        config: dict,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or next(model.parameters()).device
        self.max_length = config["model"]["max_seq_length"]

        # Label maps for human-readable output
        self.aspect_map = config.get("aspects", {})
        self.sentiment_map = config.get("sentiments", {})

        # V2: Sarcasm detector (rule-based, <1ms overhead)
        sarcasm_cfg = config.get("sarcasm", {})
        self.sarcasm_enabled = sarcasm_cfg.get("enabled", False)
        if self.sarcasm_enabled:
            self.sarcasm_detector = SarcasmDetector.from_config(config)
            logger.info(
                f"Sarcasm routing enabled "
                f"(thresholds: {self.sarcasm_detector.low_threshold}/"
                f"{self.sarcasm_detector.high_threshold})"
            )
        else:
            self.sarcasm_detector = None

        # Ensure model is in eval mode
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str = "config.yaml",
    ) -> "Predictor":
        """
        Factory: load a trained model from checkpoint.

        Args:
            checkpoint_path: Path to .pt file saved by Trainer.
            config_path: Path to config.yaml.

        Returns:
            Initialized Predictor ready for inference.
        """
        config = load_config(config_path)
        model = build_model(config)
        load_checkpoint(model, checkpoint_path)

        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

        return cls(model=model, tokenizer=tokenizer, config=config)

    @classmethod
    def from_pretrained(
        cls,
        config_path: str = "config.yaml",
    ) -> "Predictor":
        """
        Factory: build model from base pretrained weights (no fine-tuning).

        Useful for baseline comparison — how does untrained BERT perform
        on aspect/sentiment? (Answer: poorly, since the heads are random.)
        """
        config = load_config(config_path)
        model = build_model(config)
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        return cls(model=model, tokenizer=tokenizer, config=config)

    def predict_one(self, text: str) -> Dict:
        """
        Predict aspect + sentiment for a single review.

        Args:
            text: Raw review text.

        Returns:
            Dict with:
              - text: original input
              - aspect: predicted aspect name (e.g., "shipping")
              - aspect_id: integer label
              - aspect_confidence: float (0-1)
              - sentiment: predicted sentiment (e.g., "negative")
              - sentiment_id: integer label
              - sentiment_confidence: float (0-1)
              - latency_ms: wall-clock time for this prediction
        """
        results = self.predict_batch([text])
        return results[0]

    @torch.no_grad()
    def predict_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[Dict]:
        """
        Predict aspect + sentiment for a batch of reviews.

        For large inputs, splits into mini-batches to avoid OOM.

        Args:
            texts: List of review strings.
            batch_size: Max texts per forward pass. Defaults to config.

        Returns:
            List of prediction dicts (same format as predict_one).
        """
        if batch_size is None:
            batch_size = self.config["inference"]["batch_size"]

        start_time = time.perf_counter()
        all_results = []

        # V2: Run sarcasm detection on all texts upfront
        # This is <1ms for rule-based detector, so no batching needed
        sarcasm_results = None
        if self.sarcasm_enabled and self.sarcasm_detector:
            sarcasm_results = self.sarcasm_detector.detect_batch(texts)

        # Process in mini-batches
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]

            # Tokenize batch
            encoding = self.tokenizer(
                chunk,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            # Model forward
            preds = self.model.predict(input_ids, attention_mask)

            # Convert to result dicts
            for j in range(len(chunk)):
                aspect_id = preds["aspect_preds"][j].item()
                sentiment_id = preds["sentiment_preds"][j].item()

                result = {
                    "text": chunk[j],
                    "aspect": self.aspect_map.get(aspect_id, f"aspect_{aspect_id}"),
                    "aspect_id": aspect_id,
                    "aspect_confidence": round(
                        preds["aspect_confidence"][j].item(), 4
                    ),
                    "aspect_probs": {
                        self.aspect_map.get(k, f"aspect_{k}"): round(
                            preds["aspect_probs"][j][k].item(), 4
                        )
                        for k in range(self.config["model"]["num_aspect_labels"])
                    },
                    "sentiment": self.sentiment_map.get(
                        sentiment_id, f"sentiment_{sentiment_id}"
                    ),
                    "sentiment_id": sentiment_id,
                    "sentiment_confidence": round(
                        preds["sentiment_confidence"][j].item(), 4
                    ),
                    "sentiment_probs": {
                        self.sentiment_map.get(k, f"sentiment_{k}"): round(
                            preds["sentiment_probs"][j][k].item(), 4
                        )
                        for k in range(self.config["model"]["num_sentiment_labels"])
                    },
                }

                # V2: Apply sarcasm routing
                if sarcasm_results is not None:
                    global_idx = i + j
                    result = apply_sarcasm_routing(
                        result,
                        sarcasm_results[global_idx],
                        sentiment_map=self.sentiment_map,
                    )

                all_results.append(result)

        # Compute latency
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        per_item_ms = elapsed_ms / max(len(texts), 1)

        for r in all_results:
            r["latency_ms"] = round(per_item_ms, 2)

        return all_results


def format_prediction(pred: Dict, verbose: bool = False) -> str:
    """
    Pretty-format a prediction dict for display.

    Args:
        pred: Output from Predictor.predict_one().
        verbose: If True, show full probability distributions.

    Returns:
        Formatted string.
    """
    lines = [
        f"Text:      {pred['text'][:120]}{'...' if len(pred['text']) > 120 else ''}",
        f"Aspect:    {pred['aspect']} (confidence: {pred['aspect_confidence']:.1%})",
        f"Sentiment: {pred['sentiment']} (confidence: {pred['sentiment_confidence']:.1%})",
        f"Latency:   {pred['latency_ms']:.1f}ms",
    ]

    # V2: Sarcasm routing info
    if "sarcasm_score" in pred:
        route_icon = {"trust": "✓", "flag": "⚠", "invert": "⟲"}.get(pred["sarcasm_route"], "?")
        lines.append(
            f"Sarcasm:   {route_icon} {pred['sarcasm_route']} "
            f"(score: {pred['sarcasm_score']:.2f})"
        )
        if "sentiment_warning" in pred:
            lines.append(f"  Warning: {pred['sentiment_warning']}")

    if verbose:
        lines.append(f"  Aspect probs:    {pred['aspect_probs']}")
        lines.append(f"  Sentiment probs: {pred['sentiment_probs']}")
        if "sarcasm_features" in pred:
            active = {k: v for k, v in pred["sarcasm_features"].items() if v > 0}
            if active:
                lines.append(f"  Sarcasm features: {active}")

    return "\n".join(lines)


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    """Quick test with base (untrained) model."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    predictor = Predictor.from_pretrained()

    test_reviews = [
        "Absolutely love this product! Great quality and fast shipping.",
        "Terrible experience. Broke after one week. Customer service was unhelpful.",
        "Decent for the price. Nothing special but gets the job done.",
        "Easy to set up and use. Instructions were clear. Good value for money.",
        "Package arrived damaged. Took 3 weeks to get here. Very disappointed.",
    ]

    print(f"\n{'='*70}")
    print("Predictions (NOTE: untrained model — random head weights)")
    print(f"{'='*70}")

    results = predictor.predict_batch(test_reviews)
    for r in results:
        print(f"\n{format_prediction(r, verbose=True)}")
        print("-" * 70)

    # Throughput estimate
    import time
    n = 100
    big_batch = test_reviews * (n // len(test_reviews))
    t0 = time.perf_counter()
    _ = predictor.predict_batch(big_batch)
    elapsed = time.perf_counter() - t0
    throughput = len(big_batch) / elapsed
    print(f"\nThroughput: {throughput:.0f} reviews/sec ({len(big_batch)} reviews in {elapsed:.2f}s)")
    print(f"{'='*70}")
