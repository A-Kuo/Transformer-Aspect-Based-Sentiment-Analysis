"""
V3 hybrid backbone: BERT attention sandwich with SSM middle blocks.

  Layers 0–3:   pretrained BertLayer (local self-attention)
  Layers 4–9:   bidirectional SSM (Mamba2 when CUDA + mamba_ssm; else conv fallback)
  Layers 10–11: pretrained BertLayer

Classification heads match V2: [CLS] aspect/sentiment, token-level BIO span,
quantum projection on [CLS].
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers.modeling_outputs import BaseModelOutput

from .quantum_uncertainty import QuantumProjection

logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba2

    _MAMBA_CLASS = Mamba2
except ImportError:
    _MAMBA_CLASS = None


def _to_2d_attention_mask(
    attention_mask: Optional[torch.Tensor],
    seq_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Convert HF extended mask (B,1,1,L) or (B,L) to (B,L) float in {0,1}."""
    if attention_mask is None:
        return None
    if attention_mask.dim() == 2:
        return attention_mask.to(dtype=torch.float32, device=device)
    # extended mask: large negative = masked position
    m = attention_mask.squeeze(1).squeeze(1)
    return (m > -500).to(dtype=torch.float32, device=device)


def _call_bert_layer(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    head_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Invoke BertLayer across minor transformers API differences."""
    try:
        if head_mask is not None:
            out = layer(
                hidden_states,
                attention_mask,
                head_mask=head_mask,
            )
        else:
            out = layer(hidden_states, attention_mask)
    except TypeError:
        out = layer(hidden_states, attention_mask=attention_mask)
    if isinstance(out, tuple):
        return out[0]
    return out


class BidirectionalSSMBlock(nn.Module):
    """
    Bidirectional selective SSM when Mamba2 is available on CUDA; otherwise
    a depthwise separable conv + FFN (O(L)), suitable for CPU / CI.
    """

    def __init__(self, d_model: int, ssm_cfg: Dict):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        d_state = int(ssm_cfg.get("d_state", 64))
        d_conv = int(ssm_cfg.get("d_conv", 4))
        expand = int(ssm_cfg.get("expand", 2))

        self._use_mamba = _MAMBA_CLASS is not None
        if self._use_mamba:
            try:
                self.fwd = _MAMBA_CLASS(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                self.bwd = _MAMBA_CLASS(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            except Exception as e:  # pragma: no cover
                logger.warning("Mamba2 init failed (%s); using fallback SSM block.", e)
                self._use_mamba = False

        if not self._use_mamba:
            self.conv = nn.Conv1d(
                d_model, d_model, kernel_size=5, padding=2, groups=d_model, bias=False
            )
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, D)
            attention_mask: HF-style 4D or (B, T)
        """
        residual = hidden_states
        x = self.norm(hidden_states)
        mask_2d = _to_2d_attention_mask(attention_mask, x.size(1), x.device)

        use_mamba_here = self._use_mamba and x.is_cuda
        if use_mamba_here:
            hf = self.fwd(x)
            hb = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
            mixed = hf + hb
        else:
            # CPU-friendly O(T) local mixing
            xt = x.transpose(1, 2)
            mixed = self.conv(xt).transpose(1, 2)
            mixed = mixed + self.ffn(x)

        if mask_2d is not None:
            mixed = mixed * mask_2d.unsqueeze(-1)

        return residual + mixed


class HybridBertEncoder(nn.Module):
    """
    Replaces `BertEncoder`: 4 + 6 + 2 layers by default, compatible with
    `BertModel.forward` calling `self.encoder(...)`.
    """

    def __init__(
        self,
        config,
        layer_early: List[nn.Module],
        layer_ssm: List[BidirectionalSSMBlock],
        layer_late: List[nn.Module],
    ):
        super().__init__()
        self.config = config
        self.layer_early = nn.ModuleList(layer_early)
        self.layer_ssm = nn.ModuleList(layer_ssm)
        self.layer_late = nn.ModuleList(layer_late)

    @classmethod
    def from_bert_model(cls, bert: BertModel, ssm_cfg: Dict) -> "HybridBertEncoder":
        layers = list(bert.encoder.layer)
        ne = int(ssm_cfg.get("num_early_attention", 4))
        ns = int(ssm_cfg.get("num_ssm", 6))
        nl = int(ssm_cfg.get("num_late_attention", 2))
        if ne + ns + nl != len(layers):
            raise ValueError(
                f"SSM sandwich layers {ne}+{ns}+{nl} must equal BERT depth {len(layers)}"
            )
        hidden = bert.config.hidden_size
        early = layers[:ne]
        late = layers[ne + ns :]
        ssm_blocks = [BidirectionalSSMBlock(hidden, ssm_cfg) for _ in range(ns)]
        return cls(bert.config, early, ssm_blocks, late)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states: tuple = ()
        all_self_attentions: tuple = ()

        layer_idx = 0
        for layer_module in self.layer_early:
            h = (
                head_mask[layer_idx]
                if head_mask is not None and layer_idx < len(head_mask)
                else None
            )
            hidden_states = _call_bert_layer(layer_module, hidden_states, attention_mask, h)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_idx += 1

        for ssm in self.layer_ssm:
            hidden_states = ssm(hidden_states, attention_mask=attention_mask)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        for layer_module in self.layer_late:
            h = (
                head_mask[layer_idx]
                if head_mask is not None and layer_idx < len(head_mask)
                else None
            )
            hidden_states = _call_bert_layer(layer_module, hidden_states, attention_mask, h)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_idx += 1

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=all_self_attentions if output_attentions else None,
        )


def build_hybrid_bert(model_name: str, ssm_cfg: Dict) -> BertModel:
    """Load BERT and swap the encoder for `HybridBertEncoder`."""
    bert = BertModel.from_pretrained(model_name)
    bert.encoder = HybridBertEncoder.from_bert_model(bert, ssm_cfg)
    return bert


class AspectSentimentHybridModel(nn.Module):
    """V3 dual / triple head model with hybrid backbone."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        ssm_cfg: Optional[Dict] = None,
        num_aspect_labels: int = 5,
        num_sentiment_labels: int = 3,
        num_span_labels: int = 11,
        dropout: float = 0.1,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        ssm_cfg = ssm_cfg or {}
        self.ssm_cfg = ssm_cfg
        self.num_aspect_labels = num_aspect_labels
        self.num_sentiment_labels = num_sentiment_labels
        self.num_span_labels = num_span_labels
        self.loss_weights = loss_weights or {
            "aspect": 0.2,
            "sentiment": 0.4,
            "span": 0.4,
        }

        self.backbone = build_hybrid_bert(model_name, ssm_cfg)
        hidden_size = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.aspect_head = nn.Linear(hidden_size, num_aspect_labels)
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment_labels)
        self.span_head = nn.Linear(hidden_size, num_span_labels)
        self.quantum_projection = QuantumProjection(
            input_dim=hidden_size,
            num_classes=num_sentiment_labels,
        )

        nn.init.xavier_uniform_(self.aspect_head.weight)
        nn.init.zeros_(self.aspect_head.bias)
        nn.init.xavier_uniform_(self.sentiment_head.weight)
        nn.init.zeros_(self.sentiment_head.bias)
        nn.init.xavier_uniform_(self.span_head.weight)
        nn.init.zeros_(self.span_head.bias)

        self._log_params()

    def _log_params(self) -> None:
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = (
            sum(p.numel() for p in self.aspect_head.parameters())
            + sum(p.numel() for p in self.sentiment_head.parameters())
            + sum(p.numel() for p in self.span_head.parameters())
            + sum(p.numel() for p in self.quantum_projection.parameters())
        )
        total = backbone_params + head_params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "Hybrid model parameters:\n"
            f"  Backbone:        {backbone_params:>12,} ({backbone_params / total:.1%})\n"
            f"  Heads + quantum: {head_params:>12,} ({head_params / total:.1%})\n"
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
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state

        cls_output = self.dropout(hidden_states[:, 0, :])
        token_output = self.dropout(hidden_states)

        aspect_logits = self.aspect_head(cls_output)
        sentiment_logits = self.sentiment_head(cls_output)
        span_logits = self.span_head(token_output)

        result: Dict[str, torch.Tensor] = {
            "aspect_logits": aspect_logits,
            "sentiment_logits": sentiment_logits,
            "span_logits": span_logits,
        }

        if aspect_labels is not None and sentiment_labels is not None:
            aspect_loss = F.cross_entropy(aspect_logits, aspect_labels)
            sentiment_loss = F.cross_entropy(sentiment_logits, sentiment_labels)
            w = self.loss_weights
            joint_loss = w["aspect"] * aspect_loss + w["sentiment"] * sentiment_loss
            result["aspect_loss"] = aspect_loss.detach()
            result["sentiment_loss"] = sentiment_loss.detach()

            if span_labels is not None and span_labels.max() >= 0:
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
        self.eval()
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = outputs.last_hidden_state
            cls_output = hidden_states[:, 0, :]
            token_output = hidden_states

            aspect_logits = self.aspect_head(cls_output)
            sentiment_logits = self.sentiment_head(cls_output)
            span_logits = self.span_head(token_output)

            aspect_probs = F.softmax(aspect_logits, dim=-1)
            sentiment_probs = F.softmax(sentiment_logits, dim=-1)
            span_probs = F.softmax(span_logits, dim=-1)
            span_preds = span_logits.argmax(dim=-1)

            quantum_out = self.quantum_projection(cls_output)

        return {
            "aspect_preds": aspect_probs.argmax(dim=-1),
            "aspect_probs": aspect_probs,
            "aspect_confidence": aspect_probs.max(dim=-1).values,
            "sentiment_preds": sentiment_probs.argmax(dim=-1),
            "sentiment_probs": sentiment_probs,
            "sentiment_confidence": sentiment_probs.max(dim=-1).values,
            "span_preds": span_preds,
            "span_probs": span_probs,
            "span_confidence": span_probs.max(dim=-1).values,
            "quantum_probs": quantum_out["probs"],
            "quantum_entropy": quantum_out["entropy"],
            "quantum_interference": quantum_out["interference"],
        }

    def freeze_backbone(self, num_layers_to_freeze: int = 10) -> None:
        """Freeze embeddings then first N sandwich blocks (12 total)."""
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False

        enc = self.backbone.encoder
        blocks: List[nn.Module] = (
            list(enc.layer_early) + list(enc.layer_ssm) + list(enc.layer_late)
        )
        for i, block in enumerate(blocks):
            if i < num_layers_to_freeze:
                for p in block.parameters():
                    p.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "Froze first %s/12 sandwich blocks (+ embeddings). Trainable: %s / %s (%.1f%%)",
            num_layers_to_freeze,
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / max(total, 1),
        )

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Unfroze all parameters. Trainable: %s", f"{trainable:,}")


def build_model(config: dict) -> AspectSentimentHybridModel:
    """Build V3 hybrid model from a loaded config dict."""
    model_cfg = config["model"]
    ssm_cfg = config.get("ssm", {})
    loss_weights = config.get(
        "loss_weights",
        {"aspect": 0.2, "sentiment": 0.4, "span": 0.4},
    )
    model = AspectSentimentHybridModel(
        model_name=model_cfg["name"],
        ssm_cfg=ssm_cfg,
        num_aspect_labels=model_cfg["num_aspect_labels"],
        num_sentiment_labels=model_cfg["num_sentiment_labels"],
        num_span_labels=model_cfg.get("num_span_labels", 11),
        dropout=model_cfg["dropout"],
        loss_weights=loss_weights,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    logger.info("Hybrid model on device: %s", device)
    return model
