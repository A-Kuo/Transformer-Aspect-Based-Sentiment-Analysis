"""
Token-level BIO span utilities for V3 (aligned with V2 `V2/data.py` labeling).

Hidden states for the span head come from the **hybrid** backbone (SSM middle
+ attention sandwich); decoding logic is architecture-agnostic.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

# --- BIO layout (must match V2/data.py) ---
BIO_O = 0


def bio_b(aspect_id: int) -> int:
    return 2 * aspect_id + 1


def bio_i(aspect_id: int) -> int:
    return 2 * aspect_id + 2


def label_id_to_aspect_id(label_id: int) -> Optional[int]:
    """Map BIO class id to aspect id, or None for O."""
    if label_id == BIO_O:
        return None
    if label_id % 2 == 1:
        return (label_id - 1) // 2
    return (label_id - 2) // 2


def build_bio_label_map(aspect_names: Dict[int, str]) -> Dict[int, str]:
    """Human-readable BIO names from config `aspects` mapping."""
    m = {0: "O"}
    for aid, name in aspect_names.items():
        m[bio_b(int(aid))] = f"B-{name}"
        m[bio_i(int(aid))] = f"I-{name}"
    return m


def decode_bio_spans(
    pred_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    aspect_names: Optional[Dict[int, str]] = None,
) -> List[Dict]:
    """
    Decode a single sequence of BIO predictions to contiguous spans.

    Args:
        pred_ids: (seq_len,) int tensor of predicted BIO class per token.
        attention_mask: (seq_len,) 1 for real tokens, 0 for padding.
        aspect_names: optional id -> name from config.

    Returns:
        List of dicts: aspect_id, label (B-name), start_token, end_token (exclusive).
    """
    ids = pred_ids.detach().cpu().tolist()
    mask = attention_mask.detach().cpu().tolist()
    spans: List[Dict] = []
    current: Optional[Tuple[int, int, int]] = None  # (aspect_id, start, last)

    for t, (pid, m) in enumerate(zip(ids, mask)):
        if m == 0:
            break
        aid = label_id_to_aspect_id(pid)
        if pid == BIO_O or aid is None:
            if current is not None:
                a0, s0, last = current
                spans.append(_span_dict(a0, s0, last + 1, aspect_names))
                current = None
            continue
        if pid == bio_b(aid):
            if current is not None:
                a0, s0, last = current
                spans.append(_span_dict(a0, s0, last + 1, aspect_names))
            current = (aid, t, t)
        elif pid == bio_i(aid) and current is not None and current[0] == aid:
            current = (aid, current[1], t)
        else:
            if current is not None:
                a0, s0, last = current
                spans.append(_span_dict(a0, s0, last + 1, aspect_names))
            current = None

    if current is not None:
        a0, s0, last = current
        spans.append(_span_dict(a0, s0, last + 1, aspect_names))

    return spans


def _span_dict(
    aspect_id: int,
    start: int,
    end: int,
    aspect_names: Optional[Dict[int, str]],
) -> Dict:
    name = (aspect_names or {}).get(aspect_id, str(aspect_id))
    return {
        "aspect_id": aspect_id,
        "aspect_name": name,
        "label": f"B-{name}",
        "start_token": start,
        "end_token": end,
    }


def spans_from_bio_predictions(
    span_preds: torch.Tensor,
    attention_mask: torch.Tensor,
    aspect_names: Optional[Dict[int, str]] = None,
) -> List[List[Dict]]:
    """
    Batch wrapper for `decode_bio_spans`.

    Args:
        span_preds: (B, T)
        attention_mask: (B, T)
    """
    out: List[List[Dict]] = []
    for b in range(span_preds.size(0)):
        out.append(
            decode_bio_spans(span_preds[b], attention_mask[b], aspect_names=aspect_names)
        )
    return out
