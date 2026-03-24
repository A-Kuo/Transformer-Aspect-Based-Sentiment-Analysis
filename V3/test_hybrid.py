"""
Smoke tests for V3 hybrid backbone, span decode, and sarcasm routing.

Requires: transformers, torch (same as main tests). Optional: mamba-ssm on CUDA.
Run from repo root: pytest V3/test_hybrid.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from V3.model_hybrid import AspectSentimentHybridModel, BidirectionalSSMBlock
from V3.sarcasm import SarcasmDetector, apply_sarcasm_routing
from V3.span_extraction import bio_b, decode_bio_spans, label_id_to_aspect_id


@pytest.fixture(scope="module")
def hybrid_config():
    return {
        "model": {
            "name": "bert-base-uncased",
            "num_aspect_labels": 5,
            "num_sentiment_labels": 3,
            "num_span_labels": 11,
            "max_seq_length": 256,
            "dropout": 0.1,
            "span_extraction": True,
        },
        "ssm": {
            "d_state": 64,
            "d_conv": 4,
            "expand": 2,
            "num_early_attention": 4,
            "num_ssm": 6,
            "num_late_attention": 2,
        },
        "loss_weights": {"aspect": 0.2, "sentiment": 0.4, "span": 0.4},
        "paths": {
            "model_dir": "models_v3/",
            "results_dir": "results_v3/",
        },
        "training": {"batch_size": 2},
        "sarcasm": {"enabled": True, "low_threshold": 0.3, "high_threshold": 0.7},
        "aspects": {0: "quality", 1: "usability", 2: "value", 3: "shipping", 4: "customer_service"},
        "sentiments": {0: "negative", 1: "neutral", 2: "positive"},
    }


@pytest.fixture(scope="module")
def hybrid_model(hybrid_config):
    """One download of bert-base-uncased per test session."""
    m = AspectSentimentHybridModel(
        model_name=hybrid_config["model"]["name"],
        ssm_cfg=hybrid_config["ssm"],
        num_aspect_labels=hybrid_config["model"]["num_aspect_labels"],
        num_sentiment_labels=hybrid_config["model"]["num_sentiment_labels"],
        num_span_labels=hybrid_config["model"]["num_span_labels"],
        dropout=hybrid_config["model"]["dropout"],
        loss_weights=hybrid_config["loss_weights"],
    )
    return m.to("cpu")


def test_ssm_fallback_block_shapes():
    blk = BidirectionalSSMBlock(768, {"d_state": 16, "d_conv": 4, "expand": 2})
    x = torch.randn(2, 32, 768)
    mask = torch.ones(2, 32)
    y = blk(x, attention_mask=mask)
    assert y.shape == x.shape


def test_ssm_fallback_gradients():
    blk = BidirectionalSSMBlock(128, {"d_state": 16, "d_conv": 4, "expand": 2})
    x = torch.randn(2, 16, 128, requires_grad=True)
    y = blk(x, attention_mask=torch.ones(2, 16))
    y.sum().backward()
    assert x.grad is not None


def test_label_id_to_aspect_id():
    assert label_id_to_aspect_id(0) is None
    assert label_id_to_aspect_id(bio_b(2)) == 2


def test_decode_bio_spans_simple():
    # O O B-quality I-quality O  (padding masked off)
    t = torch.tensor([0, 0, 1, 2, 0, 0])
    m = torch.tensor([1, 1, 1, 1, 1, 0])
    spans = decode_bio_spans(t, m, aspect_names={0: "quality"})
    assert len(spans) == 1
    assert spans[0]["aspect_id"] == 0
    assert spans[0]["start_token"] == 2
    assert spans[0]["end_token"] == 4


def test_hybrid_forward_with_loss(hybrid_model):
    hybrid_model.train()
    B, T = 2, 48
    input_ids = torch.randint(0, 30522, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    aspect = torch.randint(0, 5, (B,))
    sent = torch.randint(0, 3, (B,))
    span = torch.randint(-100, 11, (B, T))
    span[:, 0] = -100
    out = hybrid_model(
        input_ids,
        attention_mask,
        aspect_labels=aspect,
        sentiment_labels=sent,
        span_labels=span,
    )
    assert "loss" in out
    assert out["loss"].dim() == 0
    out["loss"].backward()


def test_hybrid_predict(hybrid_model):
    hybrid_model.eval()
    B, T = 2, 48
    input_ids = torch.randint(0, 30522, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    p = hybrid_model.predict(input_ids, attention_mask)
    assert p["aspect_preds"].shape == (B,)
    assert p["span_preds"].shape == (B, T)
    assert p["quantum_entropy"].shape == (B,)


def test_freeze_backbone(hybrid_model):
    hybrid_model.unfreeze_all()
    hybrid_model.freeze_backbone(num_layers_to_freeze=10)
    trainable = sum(1 for p in hybrid_model.parameters() if p.requires_grad)
    assert trainable > 0


def test_sarcasm_routing_invert():
    cfg = {
        "sarcasm": {"enabled": True, "low_threshold": 0.3, "high_threshold": 0.7},
    }
    det = SarcasmDetector.from_config(cfg)
    # Exaggerated punctuation + clash often pushes score up
    text = 'Great product!!!! Totally "amazing" how terrible it is.'
    s = det.detect(text)
    result = {
        "text": text,
        "aspect": "quality",
        "aspect_id": 0,
        "aspect_confidence": 0.9,
        "aspect_probs": {},
        "sentiment": "positive",
        "sentiment_id": 2,
        "sentiment_confidence": 0.9,
        "sentiment_probs": {0: 0.05, 1: 0.05, 2: 0.9},
    }
    out = apply_sarcasm_routing(
        result,
        s,
        sentiment_map={0: "negative", 1: "neutral", 2: "positive"},
    )
    assert "sarcasm_route" in out
    assert "sarcasm_score" in out
