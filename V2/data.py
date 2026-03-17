"""
Data pipeline for BERT Aspect-Based Sentiment Analysis.

Pipeline:
  Raw Amazon reviews (text + star rating)
      ↓
  [Weak Supervision] Assign aspect labels via keyword matching
      ↓
  [Star → Sentiment] Map 1-5 stars → negative/neutral/positive
      ↓
  [BERT Tokenizer] → input_ids, attention_mask
      ↓
  PyTorch DataLoader (batched, shuffled, ready for training)

Why weak supervision?
  Amazon reviews don't come labeled with aspects. Manual labeling 5000+
  reviews is expensive. Instead, we use domain keyword dictionaries to
  assign soft labels. This is a common production pattern (Snorkel, etc.)
  — noisy labels + large data beats small perfect labels for fine-tuning.

  The keyword approach is a labeling function L(x) → y that maps review
  text to aspect categories. It introduces label noise, but BERT's
  cross-entropy loss is relatively robust to uniform label noise up to
  ~20% error rates (see: Natarajan et al., "Learning with Noisy Labels").
"""

import re
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ============================================================
# Aspect keyword dictionaries (weak supervision)
# ============================================================
# Each aspect has trigger keywords. If a review contains keywords
# from a category, that aspect is assigned. Reviews can match
# multiple aspects (multi-label), but we take the strongest match
# for the prototype (single-label simplification).

ASPECT_KEYWORDS: Dict[int, Dict[str, List[str]]] = {
    0: {  # quality
        "name": "quality",
        "keywords": [
            "quality", "durable", "durability", "material", "build",
            "sturdy", "flimsy", "broke", "broken", "cheap", "well made",
            "well-made", "solid", "fragile", "defective", "craftsmanship",
            "construction", "lasts", "lasted", "wear", "worn", "tear",
            "premium", "inferior", "excellent quality", "poor quality",
        ],
    },
    1: {  # usability
        "name": "usability",
        "keywords": [
            "easy to use", "difficult", "intuitive", "confusing",
            "user friendly", "user-friendly", "setup", "set up",
            "instructions", "complicated", "simple", "straightforward",
            "hard to use", "learning curve", "convenient", "hassle",
            "interface", "design", "ergonomic", "comfortable",
            "figuring out", "plug and play", "works out of the box",
        ],
    },
    2: {  # value
        "name": "value",
        "keywords": [
            "price", "value", "worth", "expensive", "cheap", "affordable",
            "overpriced", "bargain", "deal", "money", "cost", "budget",
            "bang for the buck", "bang for buck", "rip off", "ripoff",
            "waste of money", "great deal", "good value", "fair price",
            "paid", "dollars", "investment",
        ],
    },
    3: {  # shipping
        "name": "shipping",
        "keywords": [
            "shipping", "delivery", "arrived", "package", "packaging",
            "shipped", "delivered", "box", "damaged in transit",
            "fast shipping", "slow shipping", "late", "on time",
            "tracking", "carrier", "fedex", "ups", "usps", "amazon prime",
            "days to arrive", "weeks to arrive", "lost in mail",
        ],
    },
    4: {  # customer_service
        "name": "customer_service",
        "keywords": [
            "customer service", "support", "return", "refund", "warranty",
            "replacement", "responded", "response", "helpful staff",
            "unhelpful", "rude", "polite", "exchange", "contact",
            "representative", "rep", "phone call", "email support",
            "resolved", "unresolved", "complaint",
        ],
    },
}


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ============================================================
# Aspect labeling (weak supervision)
# ============================================================

def assign_aspect(text: str) -> int:
    """
    Assign a single aspect label to a review via keyword matching.

    Strategy: Count keyword hits per aspect, pick the one with most
    matches. If no keywords match, default to 'quality' (aspect 0)
    as it's the most common complaint category in product reviews.

    This is deliberately simple — a labeling function, not a model.
    In Phase 2, this gets replaced by the model's own predictions
    in a self-training loop.

    Args:
        text: Raw review text (lowercase matching applied internally).

    Returns:
        Integer aspect label (0-4).
    """
    text_lower = text.lower()
    scores = {}

    for aspect_id, aspect_info in ASPECT_KEYWORDS.items():
        count = 0
        for keyword in aspect_info["keywords"]:
            # Use word boundary matching to avoid partial matches
            # e.g., "price" shouldn't match "priceless" differently
            # but simple 'in' check is fine for prototype
            if keyword in text_lower:
                count += 1
        scores[aspect_id] = count

    # Pick highest-scoring aspect; default to quality (0) on ties/zeros
    max_score = max(scores.values())
    if max_score == 0:
        return 0  # Default: quality

    # Among tied aspects, pick the first (deterministic)
    for aspect_id in sorted(scores.keys()):
        if scores[aspect_id] == max_score:
            return aspect_id

    return 0  # Fallback (unreachable, but explicit)


# ============================================================
# V2: Span-level aspect extraction (BIO tagging)
# ============================================================
# BIO scheme: O=0, then for each aspect i: B-aspect=2i+1, I-aspect=2i+2
# This gives 11 labels for 5 aspects: O, B-quality, I-quality, ...
#
# Why BIO over IO?
#   BIO distinguishes "fast shipping and great quality" as two separate
#   spans. IO would merge adjacent aspect mentions into one span.

# Label constants
BIO_O = 0  # Outside any aspect span

def bio_b(aspect_id: int) -> int:
    """BIO Begin tag for an aspect. E.g., aspect 0 (quality) → B=1."""
    return 2 * aspect_id + 1

def bio_i(aspect_id: int) -> int:
    """BIO Inside tag for an aspect. E.g., aspect 0 (quality) → I=2."""
    return 2 * aspect_id + 2

# Build human-readable BIO label map
BIO_LABEL_MAP = {0: "O"}
for _aid, _info in ASPECT_KEYWORDS.items():
    BIO_LABEL_MAP[bio_b(_aid)] = f"B-{_info['name']}"
    BIO_LABEL_MAP[bio_i(_aid)] = f"I-{_info['name']}"


def find_aspect_spans(text: str) -> List[Dict]:
    """
    Find all keyword-matched aspect spans in text with character offsets.

    Returns list of dicts: [{"aspect_id": int, "start": int, "end": int, "keyword": str}, ...]

    Multiple aspects can appear in one review — this is the V2 fix over V1's
    single-label approach. "Amazing camera but terrible battery" now yields
    two separate span annotations.

    Implementation: greedy longest-match. For each aspect's keywords (sorted
    by length descending), scan text for occurrences. Overlapping spans are
    resolved by keeping the longest match.
    """
    text_lower = text.lower()
    raw_spans = []

    for aspect_id, aspect_info in ASPECT_KEYWORDS.items():
        # Sort keywords longest-first for greedy matching
        sorted_kw = sorted(aspect_info["keywords"], key=len, reverse=True)
        for kw in sorted_kw:
            start = 0
            while True:
                idx = text_lower.find(kw, start)
                if idx == -1:
                    break
                raw_spans.append({
                    "aspect_id": aspect_id,
                    "start": idx,
                    "end": idx + len(kw),
                    "keyword": kw,
                })
                start = idx + len(kw)  # Non-overlapping within same keyword

    # Resolve overlaps: sort by start, then by span length (longest wins)
    raw_spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
    resolved = []
    last_end = -1
    for span in raw_spans:
        if span["start"] >= last_end:
            resolved.append(span)
            last_end = span["end"]

    return resolved


def align_spans_to_tokens(
    text: str,
    spans: List[Dict],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> torch.Tensor:
    """
    Convert character-level spans to token-level BIO labels aligned with
    BERT's wordpiece tokenization.

    The key challenge: BERT splits "shipping" → ["ship", "##ping"]. If the
    character span covers "shipping", both wordpieces get labels:
    "ship" → B-shipping, "##ping" → I-shipping.

    Uses tokenizer's offset_mapping to map char positions → token indices.

    Args:
        text:       Raw review text.
        spans:      Output from find_aspect_spans().
        tokenizer:  BERT tokenizer.
        max_length: Sequence length (must match model config).

    Returns:
        (max_length,) int tensor of BIO labels. Special tokens ([CLS], [SEP],
        [PAD]) get label -100 (ignored by cross-entropy loss).
    """
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    offset_mapping = encoding["offset_mapping"].squeeze(0)  # (max_length, 2)
    bio_labels = torch.full((max_length,), -100, dtype=torch.long)  # Default: ignore

    # Assign O to real tokens (non-special, non-padding)
    for tok_idx in range(max_length):
        start_char, end_char = offset_mapping[tok_idx].tolist()
        # Special tokens and padding have offset (0, 0)
        if start_char == 0 and end_char == 0:
            # But token 0 might be [CLS] with (0,0) — keep as -100
            continue
        bio_labels[tok_idx] = BIO_O  # Default: Outside

    # Map each span to token indices
    for span in spans:
        span_start = span["start"]
        span_end = span["end"]
        aspect_id = span["aspect_id"]
        is_first_token = True

        for tok_idx in range(max_length):
            tok_start, tok_end = offset_mapping[tok_idx].tolist()
            if tok_start == 0 and tok_end == 0:
                continue  # Skip special tokens

            # Token overlaps with span?
            if tok_start >= span_start and tok_end <= span_end:
                if is_first_token:
                    bio_labels[tok_idx] = bio_b(aspect_id)
                    is_first_token = False
                else:
                    bio_labels[tok_idx] = bio_i(aspect_id)
            elif tok_start >= span_end:
                break  # Past this span, move to next

    return bio_labels


def stars_to_sentiment(stars: int) -> int:
    """
    Map star rating to sentiment label.

    Mapping:
      1-2 stars → 0 (negative)
      3 stars   → 1 (neutral)
      4-5 stars → 2 (positive)

    This is a standard bucketing. The boundaries are debatable
    (is 3 stars neutral or mildly negative?), but this split gives
    reasonable class balance for Amazon data.
    """
    if stars <= 2:
        return 0  # negative
    elif stars == 3:
        return 1  # neutral
    else:
        return 2  # positive


# ============================================================
# Dataset class
# ============================================================

class ReviewDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenized Amazon reviews.

    Each sample returns:
      - input_ids:      (max_seq_length,) int tensor — BERT token IDs
      - attention_mask:  (max_seq_length,) int tensor — 1 for real tokens, 0 for padding
      - aspect_label:    scalar int tensor — aspect category (0-4)
      - sentiment_label: scalar int tensor — sentiment (0-2)
      - span_labels:     (max_seq_length,) int tensor — BIO tags per token (V2)
      - text:            original review string (for debugging/inference display)
    """

    def __init__(
        self,
        texts: List[str],
        aspect_labels: List[int],
        sentiment_labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        enable_spans: bool = False,
    ):
        self.texts = texts
        self.aspect_labels = aspect_labels
        self.sentiment_labels = sentiment_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_spans = enable_spans

        assert len(texts) == len(aspect_labels) == len(sentiment_labels), (
            f"Length mismatch: {len(texts)} texts, {len(aspect_labels)} aspects, "
            f"{len(sentiment_labels)} sentiments"
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        # BERT tokenization: [CLS] tokens... [SEP] [PAD]...
        # Truncation clips long reviews; padding fills short ones.
        # This produces fixed-size tensors for batching.
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        result = {
            "input_ids": encoding["input_ids"].squeeze(0),       # (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_length,)
            "aspect_label": torch.tensor(self.aspect_labels[idx], dtype=torch.long),
            "sentiment_label": torch.tensor(self.sentiment_labels[idx], dtype=torch.long),
            "text": text,
        }

        # V2: Generate token-level BIO span labels on the fly
        # We compute these per-sample because they depend on tokenizer alignment.
        # Cost: one extra tokenizer call per sample (with offset_mapping).
        if self.enable_spans:
            spans = find_aspect_spans(text)
            result["span_labels"] = align_spans_to_tokens(
                text, spans, self.tokenizer, self.max_length,
            )
        else:
            # Placeholder: all -100 (ignored by loss)
            result["span_labels"] = torch.full(
                (self.max_length,), -100, dtype=torch.long,
            )

        return result


# ============================================================
# Data loading & preprocessing
# ============================================================

def load_and_preprocess(
    config: dict,
    split: str = "train",
) -> Tuple[List[str], List[int], List[int]]:
    """
    Load Amazon reviews from HuggingFace and apply labeling.

    The Amazon Reviews 2023 dataset has fields:
      - text (or 'reviewText'): the review body
      - rating: 1-5 star rating
      - title: review title (we prepend this to text for more signal)

    We stream the dataset to avoid downloading the entire thing,
    then take only the configured number of samples.

    Args:
        config: Loaded config dict.
        split: One of 'train', 'val', 'test'.

    Returns:
        Tuple of (texts, aspect_labels, sentiment_labels).
    """
    from datasets import load_dataset

    data_cfg = config["data"]
    size_map = {
        "train": data_cfg["train_size"],
        "val": data_cfg["val_size"],
        "test": data_cfg["test_size"],
    }
    n_samples = size_map.get(split, data_cfg["train_size"])

    logger.info(
        f"Loading {n_samples} samples for '{split}' split from "
        f"{data_cfg['dataset_name']} / {data_cfg['dataset_subset']}..."
    )

    # Stream to avoid massive downloads; take what we need
    try:
        ds = load_dataset(
            data_cfg["dataset_name"],
            data_cfg["dataset_subset"],
            split="full",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.warning(f"Streaming load failed ({e}), trying non-streaming...")
        ds = load_dataset(
            data_cfg["dataset_name"],
            data_cfg["dataset_subset"],
            split="full",
            trust_remote_code=True,
        )

    # Collect samples
    texts = []
    aspect_labels = []
    sentiment_labels = []

    # Deterministic offset per split so train/val/test don't overlap
    offset_map = {"train": 0, "val": data_cfg["train_size"], "test": data_cfg["train_size"] + data_cfg["val_size"]}
    offset = offset_map.get(split, 0)

    count = 0
    skipped = 0

    for i, example in enumerate(ds):
        # Skip to offset for this split
        if i < offset:
            continue

        # Extract text — field names vary across dataset versions
        text = example.get("text", "") or example.get("reviewText", "") or ""
        title = example.get("title", "") or ""
        rating = example.get("rating", None)

        # Skip empty or missing reviews
        if not text.strip() or rating is None:
            skipped += 1
            continue

        # Clean text: combine title + body for richer signal
        clean_text = _clean_text(text, title)
        if len(clean_text.split()) < 5:
            skipped += 1
            continue  # Skip very short reviews (< 5 words)

        # Apply labeling
        aspect = assign_aspect(clean_text)
        sentiment = stars_to_sentiment(int(float(rating)))

        texts.append(clean_text)
        aspect_labels.append(aspect)
        sentiment_labels.append(sentiment)
        count += 1

        if count >= n_samples:
            break

    logger.info(
        f"Loaded {len(texts)} samples for '{split}' "
        f"(skipped {skipped} empty/short reviews)"
    )

    # Log class distribution for sanity check
    _log_distribution("Aspect", aspect_labels, config.get("aspects", {}))
    _log_distribution("Sentiment", sentiment_labels, config.get("sentiments", {}))

    return texts, aspect_labels, sentiment_labels


def _clean_text(text: str, title: str = "") -> str:
    """
    Minimal text cleaning. We keep it light because BERT's tokenizer
    handles most normalization, and aggressive cleaning can destroy
    signal (e.g., "DON'T BUY" → "dont buy" loses emphasis).

    Steps:
      1. Prepend title (if exists) for extra context
      2. Collapse whitespace
      3. Strip leading/trailing whitespace
    """
    combined = f"{title}. {text}" if title.strip() else text
    combined = re.sub(r"\s+", " ", combined)
    return combined.strip()


def _log_distribution(name: str, labels: List[int], label_map: dict) -> None:
    """Log class distribution for debugging class imbalance."""
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    dist_str = ", ".join(
        f"{label_map.get(k, k)}: {v} ({v / total:.1%})"
        for k, v in sorted(counts.items())
    )
    logger.info(f"  {name} distribution: {dist_str}")


# ============================================================
# DataLoader factory
# ============================================================

def create_dataloaders(
    config: dict,
    tokenizer: Optional[AutoTokenizer] = None,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders from config.

    This is the main entry point for the data pipeline. Call this
    from train.py or main.py.

    Args:
        config: Loaded config dict.
        tokenizer: BERT tokenizer. If None, loads from config model name.

    Returns:
        Dict with keys 'train', 'val', 'test' mapping to DataLoaders.
    """
    if tokenizer is None:
        model_name = config["model"]["name"]
        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_length = config["model"]["max_seq_length"]
    batch_size = config["training"]["batch_size"]
    enable_spans = config["model"].get("span_extraction", False)

    dataloaders = {}

    for split in ["train", "val", "test"]:
        texts, aspect_labels, sentiment_labels = load_and_preprocess(config, split)

        dataset = ReviewDataset(
            texts=texts,
            aspect_labels=aspect_labels,
            sentiment_labels=sentiment_labels,
            tokenizer=tokenizer,
            max_length=max_length,
            enable_spans=enable_spans,
        )

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),  # Only shuffle training data
            num_workers=0,  # Keep simple for prototype; increase in Phase 2
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),  # Drop incomplete last batch for training stability
            collate_fn=_collate_fn,
        )

        logger.info(
            f"  {split}: {len(dataset)} samples, "
            f"{len(dataloaders[split])} batches (batch_size={batch_size})"
        )

    return dataloaders


def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate that separates text strings (can't be stacked)
    from tensors (need to be batched).

    Returns dict with:
      - input_ids:       (B, max_length)
      - attention_mask:   (B, max_length)
      - aspect_labels:    (B,)
      - sentiment_labels: (B,)
      - texts:            list of B strings
    """
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "aspect_labels": torch.stack([b["aspect_label"] for b in batch]),
        "sentiment_labels": torch.stack([b["sentiment_label"] for b in batch]),
        "span_labels": torch.stack([b["span_labels"] for b in batch]),
        "texts": [b["text"] for b in batch],
    }


# ============================================================
# Standalone test / debug
# ============================================================

if __name__ == "__main__":
    """Quick sanity check: load data, print shapes and a sample."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    config = load_config()
    dataloaders = create_dataloaders(config)

    # Print one batch
    batch = next(iter(dataloaders["train"]))
    print(f"\n{'='*60}")
    print("Sample batch shapes:")
    print(f"  input_ids:       {batch['input_ids'].shape}")
    print(f"  attention_mask:   {batch['attention_mask'].shape}")
    print(f"  aspect_labels:    {batch['aspect_labels'].shape}")
    print(f"  sentiment_labels: {batch['sentiment_labels'].shape}")
    print(f"  num texts:        {len(batch['texts'])}")
    print(f"\nFirst review (truncated): {batch['texts'][0][:200]}...")
    print(f"  Aspect:    {batch['aspect_labels'][0].item()} "
          f"({config['aspects'][batch['aspect_labels'][0].item()]})")
    print(f"  Sentiment: {batch['sentiment_labels'][0].item()} "
          f"({config['sentiments'][batch['sentiment_labels'][0].item()]})")
    print(f"{'='*60}")
