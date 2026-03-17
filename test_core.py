"""
Smoke tests for BERT Aspect-Based Sentiment Analysis pipeline.

Philosophy: These are NOT unit tests in the traditional sense. They're
integration smoke tests that verify the full pipeline works end-to-end
with real BERT weights on tiny data. Each test touches real tokenization,
real model forward passes, and real gradient computation.

Why smoke tests > unit tests here?
  - The interesting bugs in ML pipelines are shape mismatches, device
    mismatches, and silent numerical errors — all of which only surface
    when real tensors flow through real layers.
  - Mocking BERT's forward pass would test nothing useful.
  - These run in ~30s total on CPU, which is fast enough for CI.

Test hierarchy:
  1. Config loading
  2. Data pipeline (tokenization, batching, label assignment)
  3. Model architecture (forward pass, output shapes, loss computation)
  4. Backward pass (gradients flow to both heads AND backbone)
  5. Training step (optimizer updates weights)
  6. Inference (predictions have correct structure)
  7. Evaluation metrics (sklearn integration works)
  8. Checkpoint save/load roundtrip
"""

import os
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    load_config,
    assign_aspect,
    stars_to_sentiment,
    ASPECT_KEYWORDS,
    ReviewDataset,
)
from src.model import AspectSentimentModel, build_model
from src.inference import Predictor, format_prediction
from src.evaluate import (
    _compute_classification_metrics,
    _compute_latency_metrics,
    _compute_cross_task_metrics,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="session")
def config():
    """Load config once for all tests."""
    return load_config()


@pytest.fixture(scope="session")
def small_config(config):
    """
    Config with tiny sizes for fast testing.

    We override sample counts and batch size so tests run in seconds
    rather than minutes. The model architecture stays identical —
    we want to test the real BERT, just on fewer samples.
    """
    cfg = config.copy()
    cfg["data"] = config["data"].copy()
    cfg["training"] = config["training"].copy()
    cfg["inference"] = config["inference"].copy()

    cfg["data"]["train_samples"] = 16
    cfg["data"]["val_samples"] = 8
    cfg["data"]["test_samples"] = 8
    cfg["training"]["batch_size"] = 4
    cfg["training"]["epochs"] = 1
    cfg["inference"]["batch_size"] = 4
    return cfg


@pytest.fixture(scope="session")
def model(config):
    """Build model once on CPU for all tests."""
    model = build_model(config)
    model = model.to("cpu")
    model.eval()
    return model


@pytest.fixture(scope="session")
def tokenizer():
    """Load tokenizer once."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("bert-base-uncased")


# ============================================================
# 1. Config
# ============================================================

class TestConfig:
    """Verify config loads and has required keys."""

    def test_config_loads(self, config):
        assert isinstance(config, dict)

    def test_required_keys(self, config):
        required = ["model", "data", "training", "inference", "paths"]
        for key in required:
            assert key in config, f"Missing config key: {key}"

    def test_model_config(self, config):
        m = config["model"]
        assert m["name"] == "bert-base-uncased"
        assert m["num_aspect_labels"] == 5
        assert m["num_sentiment_labels"] == 3
        assert m["max_seq_length"] <= 512  # BERT's absolute max

    def test_label_maps_exist(self, config):
        assert len(config["aspects"]) == config["model"]["num_aspect_labels"]
        assert len(config["sentiments"]) == config["model"]["num_sentiment_labels"]


# ============================================================
# 2. Data Pipeline
# ============================================================

class TestDataPipeline:
    """Verify weak supervision and tokenization."""

    def test_aspect_keywords_complete(self):
        """All 5 aspects have keyword lists."""
        assert len(ASPECT_KEYWORDS) == 5
        for aspect, keywords in ASPECT_KEYWORDS.items():
            assert len(keywords) > 0, f"No keywords for {aspect}"

    def test_assign_aspect_quality(self):
        """Keywords for quality should trigger quality label."""
        text = "This product is very durable and well made"
        label = assign_aspect(text)
        assert label == 0  # quality

    def test_assign_aspect_shipping(self):
        text = "Delivery was extremely late and packaging was damaged"
        label = assign_aspect(text)
        assert label == 3  # shipping

    def test_assign_aspect_default(self):
        """No keyword matches → defaults to quality (0)."""
        text = "asdfghjkl random gibberish"
        label = assign_aspect(text)
        assert label == 0

    def test_stars_to_sentiment(self):
        """Star mapping: {1,2}→0, {3}→1, {4,5}→2."""
        assert stars_to_sentiment(1) == 0  # negative
        assert stars_to_sentiment(2) == 0
        assert stars_to_sentiment(3) == 1  # neutral
        assert stars_to_sentiment(4) == 2  # positive
        assert stars_to_sentiment(5) == 2

    def test_review_dataset_output_shape(self, config, tokenizer):
        """
        ReviewDataset returns tensors with correct shapes.

        input_ids shape: [max_seq_length]  (256)
        attention_mask:  [max_seq_length]  (256)
        aspect_label:    scalar
        sentiment_label: scalar
        """
        fake_data = [
            {"text": "Great quality product, very durable", "rating": 5.0},
            {"text": "Terrible shipping, arrived damaged", "rating": 1.0},
        ]
        ds = ReviewDataset(fake_data, tokenizer, max_length=config["model"]["max_seq_length"])

        assert len(ds) == 2

        sample = ds[0]
        max_len = config["model"]["max_seq_length"]
        assert sample["input_ids"].shape == (max_len,)
        assert sample["attention_mask"].shape == (max_len,)
        assert sample["aspect_label"].dim() == 0  # scalar
        assert sample["sentiment_label"].dim() == 0

    def test_label_ranges(self, config, tokenizer):
        """Labels must be in valid ranges for cross-entropy."""
        fake_data = [
            {"text": f"Review {i}", "rating": float(r)}
            for i, r in enumerate([1, 2, 3, 4, 5] * 2)
        ]
        ds = ReviewDataset(fake_data, tokenizer, max_length=256)

        for i in range(len(ds)):
            sample = ds[i]
            assert 0 <= sample["aspect_label"].item() < config["model"]["num_aspect_labels"]
            assert 0 <= sample["sentiment_label"].item() < config["model"]["num_sentiment_labels"]


# ============================================================
# 3. Model Architecture
# ============================================================

class TestModelArchitecture:
    """Verify model structure and forward pass."""

    def test_model_type(self, model):
        assert isinstance(model, AspectSentimentModel)

    def test_parameter_count(self, model):
        """BERT-base has ~110M params. Heads add <10K."""
        total = sum(p.numel() for p in model.parameters())
        assert total > 100_000_000  # >100M
        assert total < 120_000_000  # <120M

    def test_forward_output_shapes(self, model, config):
        """
        Forward pass produces correctly shaped logits and scalar losses.

        aspect_logits:    [batch, num_aspects] = [2, 5]
        sentiment_logits: [batch, num_sentiments] = [2, 3]
        loss:             scalar (joint loss)
        """
        batch_size = 2
        seq_len = config["model"]["max_seq_length"]
        n_aspects = config["model"]["num_aspect_labels"]
        n_sentiments = config["model"]["num_sentiment_labels"]

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        aspect_labels = torch.randint(0, n_aspects, (batch_size,))
        sentiment_labels = torch.randint(0, n_sentiments, (batch_size,))

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aspect_labels=aspect_labels,
            sentiment_labels=sentiment_labels,
        )

        assert outputs["aspect_logits"].shape == (batch_size, n_aspects)
        assert outputs["sentiment_logits"].shape == (batch_size, n_sentiments)
        assert outputs["loss"].dim() == 0
        assert outputs["aspect_loss"].dim() == 0
        assert outputs["sentiment_loss"].dim() == 0

    def test_loss_is_positive(self, model, config):
        """Cross-entropy loss should always be > 0."""
        batch_size = 2
        seq_len = config["model"]["max_seq_length"]

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        aspect_labels = torch.randint(0, config["model"]["num_aspect_labels"], (batch_size,))
        sentiment_labels = torch.randint(0, config["model"]["num_sentiment_labels"], (batch_size,))

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aspect_labels=aspect_labels,
            sentiment_labels=sentiment_labels,
        )

        assert outputs["loss"].item() > 0
        assert outputs["aspect_loss"].item() > 0
        assert outputs["sentiment_loss"].item() > 0

    def test_joint_loss_decomposition(self, model, config):
        """
        Joint loss = α * aspect_loss + (1-α) * sentiment_loss.
        With default α=0.5, joint = 0.5 * aspect + 0.5 * sentiment.
        """
        batch_size = 2
        seq_len = config["model"]["max_seq_length"]

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        aspect_labels = torch.randint(0, config["model"]["num_aspect_labels"], (batch_size,))
        sentiment_labels = torch.randint(0, config["model"]["num_sentiment_labels"], (batch_size,))

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aspect_labels=aspect_labels,
            sentiment_labels=sentiment_labels,
        )

        expected = 0.5 * outputs["aspect_loss"] + 0.5 * outputs["sentiment_loss"]
        assert torch.isclose(outputs["loss"], expected, atol=1e-5)


# ============================================================
# 4. Backward Pass
# ============================================================

class TestBackwardPass:
    """Verify gradients flow correctly through the network."""

    def test_gradients_reach_heads(self, config):
        """Both classification heads receive non-zero gradients."""
        model = build_model(config).to("cpu")
        model.train()

        batch_size = 2
        seq_len = config["model"]["max_seq_length"]

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        aspect_labels = torch.randint(0, config["model"]["num_aspect_labels"], (batch_size,))
        sentiment_labels = torch.randint(0, config["model"]["num_sentiment_labels"], (batch_size,))

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aspect_labels=aspect_labels,
            sentiment_labels=sentiment_labels,
        )
        outputs["loss"].backward()

        aspect_grad = model.aspect_head.weight.grad
        assert aspect_grad is not None
        assert aspect_grad.abs().sum() > 0

        sentiment_grad = model.sentiment_head.weight.grad
        assert sentiment_grad is not None
        assert sentiment_grad.abs().sum() > 0

    def test_gradients_reach_backbone(self, config):
        """
        Joint loss gradients flow back into BERT's encoder layers.
        Both tasks contribute gradient signal to the shared backbone.
        """
        model = build_model(config).to("cpu")
        model.train()

        batch_size = 2
        seq_len = config["model"]["max_seq_length"]

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        aspect_labels = torch.randint(0, config["model"]["num_aspect_labels"], (batch_size,))
        sentiment_labels = torch.randint(0, config["model"]["num_sentiment_labels"], (batch_size,))

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aspect_labels=aspect_labels,
            sentiment_labels=sentiment_labels,
        )
        outputs["loss"].backward()

        last_layer = model.backbone.encoder.layer[-1]
        for param in last_layer.parameters():
            if param.grad is not None:
                assert param.grad.abs().sum() > 0
                return

        pytest.fail("No gradients in last encoder layer")

    def test_freeze_blocks_gradients(self, config):
        """
        freeze_backbone(10) should block gradients in layers 0-9
        but allow them in layers 10-11 and both heads.
        """
        model = build_model(config).to("cpu")
        model.freeze_backbone(num_layers_to_freeze=10)
        model.train()

        batch_size = 2
        seq_len = config["model"]["max_seq_length"]

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        aspect_labels = torch.randint(0, config["model"]["num_aspect_labels"], (batch_size,))
        sentiment_labels = torch.randint(0, config["model"]["num_sentiment_labels"], (batch_size,))

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aspect_labels=aspect_labels,
            sentiment_labels=sentiment_labels,
        )
        outputs["loss"].backward()

        frozen_layer = model.backbone.encoder.layer[0]
        for param in frozen_layer.parameters():
            assert not param.requires_grad

        unfrozen_layer = model.backbone.encoder.layer[11]
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in unfrozen_layer.parameters()
        )
        assert has_grad, "Unfrozen layer 11 should have gradients"


# ============================================================
# 5. Training Step
# ============================================================

class TestTrainingStep:
    """Verify a single optimizer step changes weights."""

    def test_weights_update(self, config):
        """
        After one forward-backward-step, at least some weights must
        change. If they don't, training is broken.
        """
        model = build_model(config).to("cpu")
        model.train()

        w_before = model.aspect_head.weight.data.clone()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        batch_size = 2
        seq_len = config["model"]["max_seq_length"]

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        aspect_labels = torch.randint(0, config["model"]["num_aspect_labels"], (batch_size,))
        sentiment_labels = torch.randint(0, config["model"]["num_sentiment_labels"], (batch_size,))

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aspect_labels=aspect_labels,
            sentiment_labels=sentiment_labels,
        )

        optimizer.zero_grad()
        outputs["loss"].backward()
        optimizer.step()

        w_after = model.aspect_head.weight.data
        assert not torch.equal(w_before, w_after), "Weights didn't change after optimizer step"


# ============================================================
# 6. Inference
# ============================================================

class TestInference:
    """Verify prediction pipeline output structure."""

    def test_predict_one_structure(self, config):
        """predict_one() returns dict with all required fields."""
        predictor = Predictor.from_pretrained()
        result = predictor.predict_one("This is a great product with fast shipping")

        required_fields = [
            "text", "aspect", "aspect_id", "aspect_confidence",
            "aspect_probs", "sentiment", "sentiment_id",
            "sentiment_confidence", "sentiment_probs", "latency_ms",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_predict_one_ranges(self, config):
        """Predictions must be in valid label ranges."""
        predictor = Predictor.from_pretrained()
        result = predictor.predict_one("Decent value for the price")

        assert 0 <= result["aspect_id"] < config["model"]["num_aspect_labels"]
        assert 0 <= result["sentiment_id"] < config["model"]["num_sentiment_labels"]
        assert 0.0 <= result["aspect_confidence"] <= 1.0
        assert 0.0 <= result["sentiment_confidence"] <= 1.0

    def test_predict_batch_length(self, config):
        """Batch prediction returns one result per input."""
        predictor = Predictor.from_pretrained()
        texts = ["Review one", "Review two", "Review three"]
        results = predictor.predict_batch(texts)
        assert len(results) == 3

    def test_softmax_sums_to_one(self, config):
        """
        Probability distributions must sum to 1.

        softmax(z)_i = exp(z_i) / Σ_j exp(z_j)

        By construction, Σ_i softmax(z)_i = 1. If this test fails,
        something is very wrong with the output postprocessing.
        """
        predictor = Predictor.from_pretrained()
        result = predictor.predict_one("Amazing product")

        aspect_sum = sum(result["aspect_probs"].values())
        sentiment_sum = sum(result["sentiment_probs"].values())

        assert abs(aspect_sum - 1.0) < 1e-3, f"Aspect probs sum to {aspect_sum}"
        assert abs(sentiment_sum - 1.0) < 1e-3, f"Sentiment probs sum to {sentiment_sum}"

    def test_format_prediction(self, config):
        """format_prediction returns a non-empty string."""
        predictor = Predictor.from_pretrained()
        result = predictor.predict_one("Test review")
        formatted = format_prediction(result, verbose=True)
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert "Aspect:" in formatted
        assert "Sentiment:" in formatted


# ============================================================
# 7. Evaluation Metrics
# ============================================================

class TestEvaluationMetrics:
    """Verify sklearn metric computations with synthetic data."""

    def test_classification_metrics_perfect(self):
        """Perfect predictions → accuracy=1.0, F1=1.0."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        label_map = {0: "a", 1: "b", 2: "c"}

        metrics = _compute_classification_metrics(y_true, y_pred, label_map, "test")

        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["weighted_f1"] == 1.0

    def test_classification_metrics_random(self):
        """Random predictions → metrics between 0 and 1."""
        np.random.seed(42)
        y_true = np.random.randint(0, 3, size=100)
        y_pred = np.random.randint(0, 3, size=100)
        label_map = {0: "a", 1: "b", 2: "c"}

        metrics = _compute_classification_metrics(y_true, y_pred, label_map, "test")

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be n_classes × n_classes."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 2, 2, 1, 1])
        label_map = {0: "a", 1: "b", 2: "c"}

        metrics = _compute_classification_metrics(y_true, y_pred, label_map, "test")
        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape == (3, 3)

    def test_latency_metrics(self):
        """Latency percentiles are in correct order."""
        latencies = np.random.exponential(10, size=100)
        metrics = _compute_latency_metrics(latencies)

        assert metrics["p50_ms"] <= metrics["p95_ms"]
        assert metrics["p95_ms"] <= metrics["p99_ms"]

    def test_cross_task_metrics(self):
        """Per-aspect sentiment accuracy computes without error."""
        n = 50
        np.random.seed(42)
        aspect_labels = np.random.randint(0, 5, size=n)
        aspect_preds = np.random.randint(0, 5, size=n)
        sentiment_labels = np.random.randint(0, 3, size=n)
        sentiment_preds = np.random.randint(0, 3, size=n)
        aspect_map = {i: f"aspect_{i}" for i in range(5)}

        cross = _compute_cross_task_metrics(
            aspect_labels, aspect_preds,
            sentiment_labels, sentiment_preds,
            aspect_map,
        )

        assert isinstance(cross, dict)
        for name, vals in cross.items():
            assert 0.0 <= vals["sentiment_accuracy"] <= 1.0
            assert vals["num_samples"] > 0


# ============================================================
# 8. Checkpoint Save/Load
# ============================================================

class TestCheckpointing:
    """Verify checkpoint roundtrip preserves weights."""

    def test_save_load_roundtrip(self, config):
        """
        Save weights → load into fresh model → outputs must match.

        Validates that serialization preserves the full model state.
        If any tensor is corrupted or missing, the loaded model will
        produce different outputs.
        """
        model = build_model(config).to("cpu")
        model.eval()

        torch.manual_seed(42)
        seq_len = config["model"]["max_seq_length"]
        input_ids = torch.randint(0, 1000, (1, seq_len))
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)

        with torch.no_grad():
            orig_preds = model.predict(input_ids, attention_mask)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name
            torch.save({
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {},
                "scheduler_state_dict": {},
                "best_val_loss": 0.5,
                "config": config,
            }, tmp_path)

        try:
            loaded_model = build_model(config).to("cpu")
            checkpoint = torch.load(tmp_path, map_location="cpu", weights_only=False)
            loaded_model.load_state_dict(checkpoint["model_state_dict"])
            loaded_model.eval()

            with torch.no_grad():
                loaded_preds = loaded_model.predict(input_ids, attention_mask)

            assert torch.equal(
                orig_preds["aspect_preds"],
                loaded_preds["aspect_preds"],
            )
            assert torch.equal(
                orig_preds["sentiment_preds"],
                loaded_preds["sentiment_preds"],
            )
            assert torch.allclose(
                orig_preds["aspect_probs"],
                loaded_preds["aspect_probs"],
                atol=1e-6,
            )
        finally:
            os.unlink(tmp_path)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
