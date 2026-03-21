"""
Evaluation suite for BERT Aspect-Based Sentiment Analysis.

Metrics computed:

  1. Per-task accuracy (aspect, sentiment)
  2. Per-class precision, recall, F1 (macro and weighted averages)
  3. Confusion matrices (aspect 5×5, sentiment 3×3)
  4. Per-aspect sentiment accuracy (e.g., how well do we classify
     sentiment specifically for "shipping" reviews?)
  5. Latency percentiles (p50, p95, p99) from inference pipeline

Why these metrics?

  - Accuracy alone is misleading with class imbalance. If 60% of
    reviews are "quality" aspect, a model predicting "quality" always
    gets 60% accuracy. F1 penalizes this by requiring both precision
    and recall.

  - Macro F1 treats all classes equally (good for rare aspects).
    Weighted F1 weights by class frequency (good for overall quality).
    We report both.

  - Confusion matrix shows WHERE the model fails. If "value" is
    systematically confused with "quality" (both mention "cheap"),
    we know our weak supervision keywords need refinement.

  - Latency percentiles matter for SLA: p50 is typical experience,
    p99 is worst-case. Our target: p99 < 500ms.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.data import load_config, create_dataloaders
from src.model import AspectSentimentModel, build_model
from src.inference import Predictor
from src.train import load_checkpoint

logger = logging.getLogger(__name__)


def evaluate_model(
    predictor: Predictor,
    dataloader,
    config: dict,
) -> Dict:
    """
    Run full evaluation on a dataloader.

    Args:
        predictor: Initialized Predictor with loaded model.
        dataloader: PyTorch DataLoader (typically test set).
        config: Config dict for label maps.

    Returns:
        Dict with all metrics (suitable for JSON serialization).
    """
    all_aspect_preds = []
    all_aspect_labels = []
    all_sentiment_preds = []
    all_sentiment_labels = []
    latencies_ms = []

    logger.info(f"Evaluating on {len(dataloader)} batches...")

    for batch in dataloader:
        texts = batch["texts"]
        aspect_labels = batch["aspect_labels"].numpy()
        sentiment_labels = batch["sentiment_labels"].numpy()

        # Time the prediction
        t0 = time.perf_counter()
        results = predictor.predict_batch(texts)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms / len(texts))  # Per-review latency

        # Collect predictions
        for r in results:
            all_aspect_preds.append(r["aspect_id"])
            all_sentiment_preds.append(r["sentiment_id"])

        all_aspect_labels.extend(aspect_labels.tolist())
        all_sentiment_labels.extend(sentiment_labels.tolist())

    # Convert to arrays
    aspect_preds = np.array(all_aspect_preds)
    aspect_labels = np.array(all_aspect_labels)
    sentiment_preds = np.array(all_sentiment_preds)
    sentiment_labels = np.array(all_sentiment_labels)
    latencies = np.array(latencies_ms)

    # --- Compute metrics ---
    metrics = {
        "num_samples": len(aspect_labels),
        "aspect": _compute_classification_metrics(
            y_true=aspect_labels,
            y_pred=aspect_preds,
            label_map=config.get("aspects", {}),
            task_name="aspect",
        ),
        "sentiment": _compute_classification_metrics(
            y_true=sentiment_labels,
            y_pred=sentiment_preds,
            label_map=config.get("sentiments", {}),
            task_name="sentiment",
        ),
        "latency": _compute_latency_metrics(latencies),
        "cross_task": _compute_cross_task_metrics(
            aspect_labels=aspect_labels,
            aspect_preds=aspect_preds,
            sentiment_labels=sentiment_labels,
            sentiment_preds=sentiment_preds,
            aspect_map=config.get("aspects", {}),
        ),
    }

    return metrics


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: dict,
    task_name: str,
) -> Dict:
    """
    Compute classification metrics for one task.

    Returns accuracy, per-class precision/recall/F1, macro & weighted
    averages, and confusion matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0,
    )

    # Averages
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Build per-class breakdown
    per_class = {}
    unique_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    for i, label_id in enumerate(unique_labels):
        label_name = label_map.get(label_id, f"{task_name}_{label_id}")
        if i < len(precision):
            per_class[label_name] = {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }

    result = {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

    # Log summary
    logger.info(
        f"  {task_name}: accuracy={accuracy:.3f}, "
        f"macro_f1={macro_f1:.3f}, weighted_f1={weighted_f1:.3f}"
    )

    return result


def _compute_latency_metrics(latencies_ms: np.ndarray) -> Dict:
    """
    Compute latency percentiles for SLA monitoring.

    Production SLA: p99 < 500ms.
    """
    if len(latencies_ms) == 0:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0}

    result = {
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 2),
        "mean_ms": round(float(np.mean(latencies_ms)), 2),
        "std_ms": round(float(np.std(latencies_ms)), 2),
        "meets_sla": bool(np.percentile(latencies_ms, 99) < 500),
    }

    logger.info(
        f"  Latency: p50={result['p50_ms']:.1f}ms, "
        f"p95={result['p95_ms']:.1f}ms, "
        f"p99={result['p99_ms']:.1f}ms "
        f"({'✓ SLA met' if result['meets_sla'] else '✗ SLA BREACH'})"
    )

    return result


def _compute_cross_task_metrics(
    aspect_labels: np.ndarray,
    aspect_preds: np.ndarray,
    sentiment_labels: np.ndarray,
    sentiment_preds: np.ndarray,
    aspect_map: dict,
) -> Dict:
    """
    Per-aspect sentiment accuracy.

    This answers: "For reviews about shipping, how accurately do we
    classify sentiment?" This is critical because some aspects may
    have noisier weak supervision labels than others.
    """
    cross = {}

    for aspect_id, aspect_name in aspect_map.items():
        # Mask: only reviews where TRUE aspect matches this category
        mask = aspect_labels == aspect_id
        if mask.sum() == 0:
            continue

        # Sentiment accuracy for this aspect subset
        sent_acc = accuracy_score(
            sentiment_labels[mask], sentiment_preds[mask]
        )

        # Also check how often we correctly predict this aspect
        aspect_acc = (aspect_preds[mask] == aspect_id).mean()

        cross[aspect_name] = {
            "num_samples": int(mask.sum()),
            "aspect_recall": round(float(aspect_acc), 4),
            "sentiment_accuracy": round(float(sent_acc), 4),
        }

    return cross


def save_metrics(metrics: Dict, output_path: str) -> None:
    """Save metrics dict as formatted JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved → {path}")


def print_report(metrics: Dict) -> None:
    """Print a human-readable evaluation report."""
    print(f"\n{'='*70}")
    print("EVALUATION REPORT")
    print(f"{'='*70}")
    print(f"Samples evaluated: {metrics['num_samples']}")

    for task in ["aspect", "sentiment"]:
        m = metrics[task]
        print(f"\n--- {task.upper()} ---")
        print(f"  Accuracy:    {m['accuracy']:.3f}")
        print(f"  Macro F1:    {m['macro_f1']:.3f}")
        print(f"  Weighted F1: {m['weighted_f1']:.3f}")
        print(f"  Per-class:")
        for cls_name, cls_metrics in m["per_class"].items():
            print(
                f"    {cls_name:20s}  "
                f"P={cls_metrics['precision']:.3f}  "
                f"R={cls_metrics['recall']:.3f}  "
                f"F1={cls_metrics['f1']:.3f}  "
                f"(n={cls_metrics['support']})"
            )

    lat = metrics["latency"]
    print(f"\n--- LATENCY ---")
    print(f"  p50:  {lat['p50_ms']:.1f}ms")
    print(f"  p95:  {lat['p95_ms']:.1f}ms")
    print(f"  p99:  {lat['p99_ms']:.1f}ms")
    print(f"  SLA:  {'✓ MET (<500ms)' if lat['meets_sla'] else '✗ BREACH'}")

    if "cross_task" in metrics:
        print(f"\n--- PER-ASPECT SENTIMENT ---")
        for name, vals in metrics["cross_task"].items():
            print(
                f"  {name:20s}  "
                f"sent_acc={vals['sentiment_accuracy']:.3f}  "
                f"asp_recall={vals['aspect_recall']:.3f}  "
                f"(n={vals['num_samples']})"
            )

    print(f"{'='*70}\n")


def run_evaluation(
    checkpoint_path: str = "models/checkpoint_best.pt",
    config_path: str = "config.yaml",
    output_path: str = "results/metrics.json",
) -> Dict:
    """
    End-to-end evaluation entry point.

    Loads model → creates test dataloader → evaluates → saves metrics.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        config_path: Path to config.yaml.
        output_path: Where to save metrics JSON.

    Returns:
        Metrics dict.
    """
    config = load_config(config_path)

    # Build predictor from checkpoint
    predictor = Predictor.from_checkpoint(checkpoint_path, config_path)

    # Create test dataloader only
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    dataloaders = create_dataloaders(config, tokenizer=tokenizer)

    # Evaluate
    metrics = evaluate_model(predictor, dataloaders["test"], config)

    # Save and display
    save_metrics(metrics, output_path)
    print_report(metrics)

    return metrics


# ============================================================
# Standalone entry point
# ============================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Default: evaluate best checkpoint on test set
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "models/checkpoint_best.pt"

    if not Path(checkpoint).exists():
        logger.warning(
            f"Checkpoint not found: {checkpoint}. "
            f"Running evaluation with untrained model (baseline)."
        )
        config = load_config()
        predictor = Predictor.from_pretrained()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        dataloaders = create_dataloaders(config, tokenizer=tokenizer)
        metrics = evaluate_model(predictor, dataloaders["test"], config)
        save_metrics(metrics, "results/metrics_baseline.json")
        print_report(metrics)
    else:
        run_evaluation(checkpoint_path=checkpoint)
