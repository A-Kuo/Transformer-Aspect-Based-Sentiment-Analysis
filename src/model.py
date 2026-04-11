"""
BERT Dual-Head Model for Aspect-Based Sentiment Analysis.

Architecture:
                    Review Text
                        ↓
              [BERT Encoder (12 layers)]
        Each layer: MultiHeadAttention → FFN → LayerNorm
        Attention: softmax(QK^T / √d_k) V  where d_k = 64
                        ↓
                [CLS] pooled output (768-dim)
                        ↓
                    [Dropout]
                        ↓
                ┌───────┴───────┐
                ↓               ↓
          [Aspect Head]   [Sentiment Head]
          Linear(768→5)   Linear(768→3)
                ↓               ↓
          5-class softmax  3-class softmax

Why dual heads sharing a backbone?

  Multi-task learning provides implicit regularization through the
  shared representation. The BERT backbone must learn features useful
  for BOTH tasks, which:

  1. Reduces overfitting — the shared layers can't memorize patterns
     specific to only one task.

  2. Improves sample efficiency — gradients from both heads flow back
     through the shared encoder, effectively doubling the training
     signal per example.

  3. Learns richer representations — aspect detection requires
     understanding WHAT is discussed (topic features), while sentiment
     requires understanding HOW it's discussed (opinion features).
     The shared backbone learns both.

  Mathematically, the joint loss is:
      L = α · L_aspect + (1 - α) · L_sentiment

  where α ∈ [0, 1] controls the task weighting. Default α = 0.5
  gives equal weight. If aspect detection is noisier (weak supervision),
  you might lower α to let the cleaner sentiment signal dominate
  backbone updates.

  Both L_aspect and L_sentiment are cross-entropy:
      L_CE = -Σ y_i · log(p_i)

  where y_i is the one-hot ground truth and p_i = softmax(z_i) is the
  predicted probability for class i.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

logger = logging.getLogger(__name__)


class AspectSentimentModel(nn.Module):
    """
    BERT backbone with dual classification heads.

    The [CLS] token's final hidden state is a 768-dim vector that
    encodes the entire input sequence via self-attention. We project
    this into two separate label spaces.

    Parameters:
        model_name:          HuggingFace model ID (e.g. 'bert-base-uncased')
        num_aspect_labels:   Number of aspect categories (default: 5)
        num_sentiment_labels: Number of sentiment classes (default: 3)
        dropout:             Dropout probability before classification heads
        loss_alpha:          Weight for aspect loss in joint loss (0-1)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_aspect_labels: int = 5,
        num_sentiment_labels: int = 3,
        dropout: float = 0.1,
        loss_alpha: float = 0.5,
    ):
        super().__init__()

        self.loss_alpha = loss_alpha
        self.num_aspect_labels = num_aspect_labels
        self.num_sentiment_labels = num_sentiment_labels

        # --- Shared BERT backbone ---
        # 12 transformer layers, 12 attention heads, 768 hidden dim
        # Total: ~110M parameters (all shared between tasks)
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base

        # --- Dropout before heads ---
        # Applied to [CLS] representation to prevent co-adaptation
        # of features between the two heads.
        self.dropout = nn.Dropout(dropout)

        # --- Aspect classification head ---
        # Linear: R^768 → R^5
        # Each output neuron learns a weight vector w_i ∈ R^768 and bias b_i
        # Score for aspect i: z_i = w_i^T · h_CLS + b_i
        # Probability: p_i = softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
        self.aspect_head = nn.Linear(hidden_size, num_aspect_labels)

        # --- Sentiment classification head ---
        # Linear: R^768 → R^3  (negative / neutral / positive)
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment_labels)

        # Initialize heads with small weights (Xavier uniform)
        # This prevents large initial logits that would create
        # overconfident softmax outputs before training.
        self._init_head(self.aspect_head)
        self._init_head(self.sentiment_head)

        # Log parameter counts
        self._log_params()

    def _init_head(self, layer: nn.Linear) -> None:
        """Xavier uniform initialization for classification heads."""
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

    def _log_params(self) -> None:
        """Log parameter counts: total, trainable, per-component."""
        backbone_params = sum(p.numel() for p in self.bert.parameters())
        aspect_params = sum(p.numel() for p in self.aspect_head.parameters())
        sentiment_params = sum(p.numel() for p in self.sentiment_head.parameters())
        total = backbone_params + aspect_params + sentiment_params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(
            f"Model parameters:\n"
            f"  BERT backbone:   {backbone_params:>12,} ({backbone_params / total:.1%})\n"
            f"  Aspect head:     {aspect_params:>12,} ({aspect_params / total:.1%})\n"
            f"  Sentiment head:  {sentiment_params:>12,} ({sentiment_params / total:.1%})\n"
            f"  Total:           {total:>12,}\n"
            f"  Trainable:       {trainable:>12,}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_labels: Optional[torch.Tensor] = None,
        sentiment_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BERT backbone → dual heads.

        Args:
            input_ids:        (B, seq_len) token IDs from BERT tokenizer
            attention_mask:    (B, seq_len) 1 for real tokens, 0 for padding
            aspect_labels:     (B,) ground truth aspect labels (optional, for loss)
            sentiment_labels:  (B,) ground truth sentiment labels (optional, for loss)

        Returns:
            Dict with keys:
              - aspect_logits:    (B, num_aspect_labels)  raw scores before softmax
              - sentiment_logits: (B, num_sentiment_labels)
              - loss:             scalar (only if labels provided)
              - aspect_loss:      scalar (only if labels provided)
              - sentiment_loss:   scalar (only if labels provided)

        Shape walkthrough for B=16, seq_len=256:
            input_ids:        (16, 256)
                ↓ BERT encoder
            last_hidden_state: (16, 256, 768)  — one 768-dim vector per token
                ↓ extract [CLS] (index 0)
            cls_output:        (16, 768)        — sequence-level representation
                ↓ dropout
            pooled:            (16, 768)
                ↓ aspect_head
            aspect_logits:     (16, 5)
                ↓ sentiment_head
            sentiment_logits:  (16, 3)
        """
        # --- BERT forward pass ---
        # Self-attention across all 12 layers computes contextualized
        # representations. The [CLS] token attends to all other tokens,
        # making it a natural sequence-level summary.
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Extract [CLS] token representation (first token)
        # Shape: (B, 768)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Dropout for regularization
        pooled = self.dropout(cls_output)

        # --- Dual heads ---
        aspect_logits = self.aspect_head(pooled)        # (B, 5)
        sentiment_logits = self.sentiment_head(pooled)  # (B, 3)

        result = {
            "aspect_logits": aspect_logits,
            "sentiment_logits": sentiment_logits,
        }

        # --- Compute joint loss if labels provided ---
        if aspect_labels is not None and sentiment_labels is not None:
            aspect_loss = F.cross_entropy(aspect_logits, aspect_labels)
            sentiment_loss = F.cross_entropy(sentiment_logits, sentiment_labels)

            # Joint loss: weighted combination
            # α controls how much the aspect task influences backbone updates
            joint_loss = (
                self.loss_alpha * aspect_loss
                + (1 - self.loss_alpha) * sentiment_loss
            )

            result["loss"] = joint_loss
            result["aspect_loss"] = aspect_loss.detach()
            result["sentiment_loss"] = sentiment_loss.detach()

        return result

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference-only forward pass (no loss computation).

        Returns predicted labels and confidence scores (softmax probs).

        Returns:
            Dict with:
              - aspect_preds:       (B,) predicted aspect labels
              - aspect_probs:       (B, num_aspect_labels) softmax probabilities
              - sentiment_preds:    (B,) predicted sentiment labels
              - sentiment_probs:    (B, num_sentiment_labels) softmax probabilities
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)

        aspect_probs = F.softmax(output["aspect_logits"], dim=-1)
        sentiment_probs = F.softmax(output["sentiment_logits"], dim=-1)

        return {
            "aspect_preds": aspect_probs.argmax(dim=-1),        # (B,)
            "aspect_probs": aspect_probs,                        # (B, 5)
            "aspect_confidence": aspect_probs.max(dim=-1).values,  # (B,)
            "sentiment_preds": sentiment_probs.argmax(dim=-1),  # (B,)
            "sentiment_probs": sentiment_probs,                  # (B, 3)
            "sentiment_confidence": sentiment_probs.max(dim=-1).values,  # (B,)
        }

    def freeze_backbone(self, num_layers_to_freeze: int = 10) -> None:
        """
        Freeze early BERT layers, only fine-tune top layers + heads.

        Gradual unfreezing strategy: early layers learn generic language
        features (syntax, morphology) that don't need updating for our
        task. Top layers learn task-specific semantics.

        Freezing 10 of 12 layers means:
          - ~85% of backbone params don't get gradient updates
          - ~5x faster training (less backward computation)
          - Reduces overfitting on small datasets

        This is especially useful for our prototype with only 5000
        training examples — full fine-tuning risks overfitting.

        Args:
            num_layers_to_freeze: How many of the 12 encoder layers to freeze.
                0 = full fine-tuning, 12 = frozen backbone (linear probe).
        """
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze specified number of encoder layers
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < num_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Froze {num_layers_to_freeze}/12 BERT layers. "
            f"Trainable: {trainable:,} / {total:,} "
            f"({trainable / total:.1%})"
        )

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (for full fine-tuning after warmup)."""
        for param in self.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Unfroze all parameters. Trainable: {trainable:,}")


def build_model(config: dict) -> AspectSentimentModel:
    """
    Factory function: build model from config.

    This is the main entry point. Call from train.py or main.py.

    Args:
        config: Loaded config dict.

    Returns:
        Initialized AspectSentimentModel on appropriate device.
    """
    model_cfg = config["model"]

    model = AspectSentimentModel(
        model_name=model_cfg["name"],
        num_aspect_labels=model_cfg["num_aspect_labels"],
        num_sentiment_labels=model_cfg["num_sentiment_labels"],
        dropout=model_cfg["dropout"],
        loss_alpha=0.5,  # Equal weight; tune in Phase 2
    )

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    logger.info(f"Model on device: {device}")

    return model


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    """Quick sanity check: build model, pass dummy input, verify shapes."""
    import sys
    sys.path.insert(0, ".")
    from src.data import load_config

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    config = load_config()
    model = build_model(config)

    # Dummy input: batch of 4, seq_len 256
    B, S = 4, config["model"]["max_seq_length"]
    device = next(model.parameters()).device

    dummy_ids = torch.randint(0, 30522, (B, S), device=device)  # 30522 = BERT vocab size
    dummy_mask = torch.ones(B, S, dtype=torch.long, device=device)
    dummy_aspects = torch.randint(0, 5, (B,), device=device)
    dummy_sentiments = torch.randint(0, 3, (B,), device=device)

    # Training forward (with loss)
    output = model(dummy_ids, dummy_mask, dummy_aspects, dummy_sentiments)
    print(f"\n{'='*60}")
    print("Training forward pass:")
    print(f"  aspect_logits:    {output['aspect_logits'].shape}")     # (4, 5)
    print(f"  sentiment_logits: {output['sentiment_logits'].shape}")  # (4, 3)
    print(f"  joint loss:       {output['loss'].item():.4f}")
    print(f"  aspect loss:      {output['aspect_loss'].item():.4f}")
    print(f"  sentiment loss:   {output['sentiment_loss'].item():.4f}")

    # Inference forward (no loss, with predictions)
    preds = model.predict(dummy_ids, dummy_mask)
    print("\nInference forward pass:")
    print(f"  aspect_preds:      {preds['aspect_preds']}")             # (4,)
    print(f"  aspect_confidence: {preds['aspect_confidence']}")        # (4,)
    print(f"  sentiment_preds:   {preds['sentiment_preds']}")          # (4,)
    print(f"  sentiment_conf:    {preds['sentiment_confidence']}")     # (4,)

    # Test freeze
    model.freeze_backbone(num_layers_to_freeze=10)
    print("\nAfter freezing 10 layers:")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print(f"{'='*60}")
