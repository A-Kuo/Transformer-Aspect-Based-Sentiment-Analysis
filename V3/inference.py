"""
V3 inference: hybrid backbone + V2-parity sarcasm routing and optional span/quantum fields.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

from src.data import load_config
from src.sarcasm import SarcasmDetector, apply_sarcasm_routing
from src.train import load_checkpoint

from .model_hybrid import AspectSentimentHybridModel, build_model
from .span_extraction import decode_bio_spans

logger = logging.getLogger(__name__)


class Predictor:
    """Production inference wrapper for the V3 hybrid model."""

    def __init__(
        self,
        model: AspectSentimentHybridModel,
        tokenizer: AutoTokenizer,
        config: dict,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or next(model.parameters()).device
        self.max_length = config["model"]["max_seq_length"]

        self.aspect_map = config.get("aspects", {})
        self.sentiment_map = config.get("sentiments", {})

        sarcasm_cfg = config.get("sarcasm", {})
        self.sarcasm_enabled = sarcasm_cfg.get("enabled", False)
        if self.sarcasm_enabled:
            self.sarcasm_detector = SarcasmDetector.from_config(config)
            logger.info(
                "V3 sarcasm routing enabled (thresholds %s / %s)",
                self.sarcasm_detector.low_threshold,
                self.sarcasm_detector.high_threshold,
            )
        else:
            self.sarcasm_detector = None

        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str = "V3/config_hybrid.yaml",
    ) -> "Predictor":
        config = load_config(config_path)
        model = build_model(config)
        load_checkpoint(model, checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        return cls(model=model, tokenizer=tokenizer, config=config)

    @classmethod
    def from_pretrained(
        cls,
        config_path: str = "V3/config_hybrid.yaml",
    ) -> "Predictor":
        config = load_config(config_path)
        model = build_model(config)
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        return cls(model=model, tokenizer=tokenizer, config=config)

    def predict_one(self, text: str) -> Dict:
        return self.predict_batch([text])[0]

    @torch.no_grad()
    def predict_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[Dict]:
        if batch_size is None:
            batch_size = self.config["inference"]["batch_size"]

        start_time = time.perf_counter()
        all_results: List[Dict] = []

        sarcasm_results = None
        if self.sarcasm_enabled and self.sarcasm_detector:
            sarcasm_results = self.sarcasm_detector.detect_batch(texts)

        aspect_names = {int(k): v for k, v in self.aspect_map.items()}
        span_on = self.config["model"].get("span_extraction", False)

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]

            encoding = self.tokenizer(
                chunk,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            preds = self.model.predict(input_ids, attention_mask)

            for j in range(len(chunk)):
                aspect_id = preds["aspect_preds"][j].item()
                sentiment_id = preds["sentiment_preds"][j].item()

                result = {
                    "text": chunk[j],
                    "aspect": self.aspect_map.get(aspect_id, f"aspect_{aspect_id}"),
                    "aspect_id": aspect_id,
                    "aspect_confidence": round(preds["aspect_confidence"][j].item(), 4),
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
                    "sentiment_confidence": round(preds["sentiment_confidence"][j].item(), 4),
                    "sentiment_probs": {
                        self.sentiment_map.get(k, f"sentiment_{k}"): round(
                            preds["sentiment_probs"][j][k].item(), 4
                        )
                        for k in range(self.config["model"]["num_sentiment_labels"])
                    },
                    "quantum_entropy": round(preds["quantum_entropy"][j].item(), 4),
                    "quantum_interference": round(preds["quantum_interference"][j].item(), 4),
                }

                if span_on:
                    spans = decode_bio_spans(
                        preds["span_preds"][j],
                        attention_mask[j],
                        aspect_names=aspect_names,
                    )
                    result["aspect_spans"] = spans

                if sarcasm_results is not None:
                    global_idx = i + j
                    result = apply_sarcasm_routing(
                        result,
                        sarcasm_results[global_idx],
                        sentiment_map=self.sentiment_map,
                    )

                all_results.append(result)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        per_item_ms = elapsed_ms / max(len(texts), 1)

        for r in all_results:
            r["latency_ms"] = round(per_item_ms, 2)

        return all_results


def format_prediction(pred: Dict, verbose: bool = False) -> str:
    lines = [
        f"Text:      {pred['text'][:120]}{'...' if len(pred['text']) > 120 else ''}",
        f"Aspect:    {pred['aspect']} (confidence: {pred['aspect_confidence']:.1%})",
        f"Sentiment: {pred['sentiment']} (confidence: {pred['sentiment_confidence']:.1%})",
        f"Latency:   {pred['latency_ms']:.1f}ms",
    ]

    if "quantum_entropy" in pred:
        lines.append(
            f"Quantum:   entropy={pred['quantum_entropy']:.3f}, "
            f"interference={pred['quantum_interference']:.3f}"
        )

    if "aspect_spans" in pred and pred["aspect_spans"]:
        lines.append(f"Spans:     {pred['aspect_spans']}")

    if "sarcasm_score" in pred:
        route_icon = {"trust": "✓", "flag": "⚠", "invert": "⟲"}.get(
            pred["sarcasm_route"], "?"
        )
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
