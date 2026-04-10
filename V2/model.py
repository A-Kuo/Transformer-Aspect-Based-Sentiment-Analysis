"""
BERT Multi-Head Model for Aspect-Based Sentiment Analysis (V2).

Architecture:
                    Review Text
                        ↓
              [BERT Encoder (12 layers)]
        Each layer: MultiHeadAttention → FFN → LayerNorm
        Attention: softmax(QK^T / √d_k) V  where d_k = 64
                        ↓
            ┌───── last_hidden_state ─────┐
            │      (B, seq_len, 768)      │
            │                             │
            │     [CLS] (index 0)    All tokens
            │      (B, 768)         (B, seq_len, 768)
            │         ↓                   ↓
            │     [Dropout]          [Dropout]
            │    ┌────┴────┐              ↓
            │    ↓         ↓        [Span Head]
            │ [Aspect]  [Sent.]   Linear(768→11)
            │ (768→5)   (768→3)        ↓
            │    ↓         ↓     11-class softmax
            │ 5-class  3-class    per token (BIO)
            │ softmax  softmax
            └─────────────────────────────┘

V2 addition: Token-level span extraction head.

  V1 assigned one aspect per review via [CLS].
  V2 additionally tags EACH TOKEN with BIO labels:
    O, B-quality, I-quality, B-usability, I-usability, ...

  This allows extracting multiple aspect spans per review:
    "Amazing camera but terrible battery life"
    → [O, B-quality, O, O, B-quality, I-quality]

  The span head uses the FULL token-level hidden states (B, T, 768),
  not just [CLS]. This is token classification — same architecture as
  NER (Named Entity Recognition), applied to aspect extraction.

  Joint loss becomes:
      L = α·L_aspect + β·L_sentiment + γ·L_span
  where L_span is per-token cross-entropy with -100 masking for
  special tokens ([CLS], [SEP], [PAD]).
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from .quantum_uncertainty import QuantumProjection

logger = logging.getLogger(__name__)


class AspectSentimentModel(nn.Module):
    """
    BERT backbone with triple classification heads (V2).

    Heads:
      1. Aspect head:    [CLS] → Linear(768→5)   — review-level aspect
      2. Sentiment head: [CLS] → Linear(768→3)   — review-level sentiment
      3. Span head:      tokens → Linear(768→11)  — token-level BIO tags (V2)

    Parameters:
        model_name:          HuggingFace model ID
        num_aspect_labels:   Number of aspect categories (default: 5)
        num_sentiment_labels: Number of sentiment classes (default: 3)
        num_span_labels:     Number of BIO tags (default: 11 = 1 + 5*2)
        dropout:             Dropout probability before classification heads
        loss_weights:        Dict with keys 'aspect', 'sentiment', 'span'
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_aspect_labels: int = 5,
        num_sentiment_labels: int = 3,
        num_span_labels: int = 11,
        dropout: float = 0.1,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.num_aspect_labels = num_aspect_labels
        self.num_sentiment_labels = num_sentiment_labels
        self.num_span_labels = num_span_labels

        # Loss weights: default to equal if not specified
        self.loss_weights = loss_weights or {
            "aspect": 0.2, "sentiment": 0.4, "span": 0.4,
        }

        # --- Shared BERT backbone ---
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768

        # --- Dropout ---
        self.dropout = nn.Dropout(dropout)

        # --- Classification heads ---
        self.aspect_head = nn.Linear(hidden_size, num_aspect_labels)
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment_labels)

        # V2: Token-level span extraction head
        # Applied to every token's hidden state, not just [CLS].
        # Same math as NER: z_t = W·h_t + b for each token t.
        self.span_head = nn.Linear(hidden_size, num_span_labels)

        # V2 iter3: Quantum-inspired uncertainty projection
        # Projects [CLS] → complex state vector → density matrix
        # Provides interference magnitude and von Neumann entropy
        # alongside standard softmax confidence.
        self.quantum_projection = QuantumProjection(
            input_dim=hidden_size,
            num_classes=num_sentiment_labels,
        )

        # Xavier init for all heads
        self._init_head(self.aspect_head)
        self._init_head(self.sentiment_head)
        self._init_head(self.span_head)

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
        span_params = sum(p.numel() for p in self.span_head.parameters())
        quantum_params = sum(p.numel() for p in self.quantum_projection.parameters())
        total = backbone_params + aspect_params + sentiment_params + span_params + quantum_params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(
            f"Model parameters:\n"
            f"  BERT backbone:   {backbone_params:>12,} ({backbone_params / total:.1%})\n"
            f"  Aspect head:     {aspect_params:>12,} ({aspect_params / total:.1%})\n"
            f"  Sentiment head:  {sentiment_params:>12,} ({sentiment_params / total:.1%})\n"
            f"  Span head (V2):  {span_params:>12,} ({span_params / total:.1%})\n"
            f"  Quantum proj:    {quantum_params:>12,} ({quantum_params / total:.1%})\n"
            f"  Total:           {total:>12,}\n"
            f"  Trainable:       {trainable:>12,}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_labels: Optional[torch.Tensor] = None,
        sentiment_labels: Optional[torch.Tensor] = None,
        span_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BERT backbone → triple heads.

        Args:
            input_ids:        (B, T) token IDs
            attention_mask:    (B, T) 1 for real tokens, 0 for padding
            aspect_labels:     (B,) review-level aspect labels
            sentiment_labels:  (B,) review-level sentiment labels
            span_labels:       (B, T) token-level BIO labels (-100 = ignore)

        Returns:
            Dict with logits for all three heads + losses if labels provided.

        Shape walkthrough (B=16, T=256):
            input_ids:          (16, 256)
                ↓ BERT
            hidden_states:      (16, 256, 768)
                ↓ [CLS] = index 0
            cls_output:         (16, 768)
                ↓ heads
            aspect_logits:      (16, 5)
            sentiment_logits:   (16, 3)
            span_logits:        (16, 256, 11)   ← V2: per-token BIO scores
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state  # (B, T, 768)

        # [CLS] for sequence-level heads
        cls_output = self.dropout(hidden_states[:, 0, :])  # (B, 768)

        # Token-level for span head
        token_output = self.dropout(hidden_states)  # (B, T, 768)

        # --- Three heads ---
        aspect_logits = self.aspect_head(cls_output)       # (B, 5)
        sentiment_logits = self.sentiment_head(cls_output)  # (B, 3)
        span_logits = self.span_head(token_output)          # (B, T, 11)

        result = {
            "aspect_logits": aspect_logits,
            "sentiment_logits": sentiment_logits,
            "span_logits": span_logits,
        }

        # --- Three-way joint loss ---
        # L = α·L_aspect + β·L_sentiment + γ·L_span
        if aspect_labels is not None and sentiment_labels is not None:
            aspect_loss = F.cross_entropy(aspect_logits, aspect_labels)
            sentiment_loss = F.cross_entropy(sentiment_logits, sentiment_labels)

            w = self.loss_weights
            joint_loss = (
                w["aspect"] * aspect_loss
                + w["sentiment"] * sentiment_loss
            )

            result["aspect_loss"] = aspect_loss.detach()
            result["sentiment_loss"] = sentiment_loss.detach()

            # Span loss: per-token CE with -100 masking
            # F.cross_entropy ignores positions where label == -100,
            # so [CLS], [SEP], [PAD] tokens don't contribute gradient.
            if span_labels is not None and span_labels.max() >= 0:
                # Reshape: (B*T, 11) vs (B*T,) for cross_entropy
                span_loss = F.cross_entropy(
                    span_logits.view(-1, self.num_span_labels),
                    span_labels.view(-1),
                    ignore_index=-100,
                )
                joint_loss = joint_loss + w["span"] * span_loss
                result["span_loss"] = span_loss.detach()
            else:
                result["span_loss"] = torch.tensor(0.0, device=aspect_logits.device)

            result["loss"] = joint_loss

        return result

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference-only forward pass (no loss computation).

        Returns predicted labels, confidence scores, span extractions,
        and quantum-inspired uncertainty measures.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)

        aspect_probs = F.softmax(output["aspect_logits"], dim=-1)
        sentiment_probs = F.softmax(output["sentiment_logits"], dim=-1)

        # V2: Token-level span predictions
        span_logits = output["span_logits"]                   # (B, T, 11)
        span_probs = F.softmax(span_logits, dim=-1)           # (B, T, 11)
        span_preds = span_logits.argmax(dim=-1)               # (B, T)
        span_confidence = span_probs.max(dim=-1).values       # (B, T)

        # V2 iter3: Quantum-inspired uncertainty via density matrix
        # Run [CLS] through quantum projection for uncertainty metadata.
        # This does NOT replace the softmax sentiment prediction — it
        # augments it with interference and entropy measures.
        outputs_bert = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs_bert.last_hidden_state[:, 0, :]
        quantum_out = self.quantum_projection(cls_output)

        return {
            "aspect_preds": aspect_probs.argmax(dim=-1),
            "aspect_probs": aspect_probs,
            "aspect_confidence": aspect_probs.max(dim=-1).values,
            "sentiment_preds": sentiment_probs.argmax(dim=-1),
            "sentiment_probs": sentiment_probs,
            "sentiment_confidence": sentiment_probs.max(dim=-1).values,
            # V2 span extraction
            "span_preds": span_preds,
            "span_probs": span_probs,
            "span_confidence": span_confidence,
            # V2 iter3: quantum uncertainty
            "quantum_probs": quantum_out["probs"],              # (B, k) density matrix probs
            "quantum_entropy": quantum_out["entropy"],          # (B,) von Neumann entropy
            "quantum_interference": quantum_out["interference"],# (B,) off-diagonal magnitude
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
    loss_weights = config.get("loss_weights", {
        "aspect": 0.2, "sentiment": 0.4, "span": 0.4,
    })

    model = AspectSentimentModel(
        model_name=model_cfg["name"],
        num_aspect_labels=model_cfg["num_aspect_labels"],
        num_sentiment_labels=model_cfg["num_sentiment_labels"],
        num_span_labels=model_cfg.get("num_span_labels", 11),
        dropout=model_cfg["dropout"],
        loss_weights=loss_weights,
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
    print(f"\nInference forward pass:")
    print(f"  aspect_preds:      {preds['aspect_preds']}")             # (4,)
    print(f"  aspect_confidence: {preds['aspect_confidence']}")        # (4,)
    print(f"  sentiment_preds:   {preds['sentiment_preds']}")          # (4,)
    print(f"  sentiment_conf:    {preds['sentiment_confidence']}")     # (4,)

    # Test freeze
    model.freeze_backbone(num_layers_to_freeze=10)
    print(f"\nAfter freezing 10 layers:")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print(f"{'='*60}")
