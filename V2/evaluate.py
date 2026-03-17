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
    all_span_labels = []
    all_span_preds = []
    sarcasm_data = []
    quantum_data = []
    latencies_ms = []

    # Check if span extraction is enabled
    enable_spans = config["model"].get("span_extraction", False)
    device = next(predictor.model.parameters()).device

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

        # V2: Collect sarcasm routing data
        if results and "sarcasm_score" in results[0]:
            for idx_r, r in enumerate(results):
                sarcasm_data.append({
                    "sarcasm_score": r["sarcasm_score"],
                    "sarcasm_route": r["sarcasm_route"],
                    "sentiment_true": sentiment_labels[idx_r],
                    "sentiment_pred": r["sentiment_id"],
                    "was_inverted": "sentiment_id_original" in r,
                })

        # V2 iter3: Collect quantum uncertainty data
        if results and "quantum_entropy" in results[0]:
            for idx_r, r in enumerate(results):
                quantum_data.append({
                    "entropy": r["quantum_entropy"],
                    "interference": r["quantum_interference"],
                    "sentiment_true": sentiment_labels[idx_r],
                    "sentiment_pred": r["sentiment_id"],
                    "correct": sentiment_labels[idx_r] == r["sentiment_id"],
                })

        # V2: Collect span predictions using batch tensors directly
        # (not via predict_batch, because we need tokenization-aligned labels)
        if enable_spans and "span_labels" in batch:
            with torch.no_grad():
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                preds = predictor.model.predict(input_ids, attention_mask)
                span_preds_batch = preds["span_preds"].cpu().numpy()
                span_labels_batch = batch["span_labels"].numpy()
                for i in range(len(texts)):
                    all_span_labels.append(span_labels_batch[i])
                    all_span_preds.append(span_preds_batch[i])

    # Convert to arrays
    aspect_preds = np.array(all_aspect_preds)
    aspect_labels = np.array(all_aspect_labels)
    sentiment_preds = np.array(all_sentiment_preds)
    sentiment_labels = np.array(all_sentiment_labels)
    latencies = np.array(latencies_ms)

    # --- Compute metrics ---
    from src.data import BIO_LABEL_MAP

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

    # V2: Span extraction metrics
    if all_span_labels:
        metrics["span"] = _compute_span_metrics(
            all_span_labels, all_span_preds, BIO_LABEL_MAP,
        )

    # V2: Sarcasm routing metrics
    if sarcasm_data:
        metrics["sarcasm_routing"] = _compute_sarcasm_routing_metrics(sarcasm_data)

    # V2 iter3: Quantum uncertainty metrics
    if quantum_data:
        metrics["quantum_uncertainty"] = _compute_quantum_uncertainty_metrics(quantum_data)

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


def _compute_span_metrics(
    all_span_labels: List[np.ndarray],
    all_span_preds: List[np.ndarray],
    bio_label_map: dict,
) -> Dict:
    """
    Compute span-level extraction metrics (V2).

    Two metric types:
      1. Token-level F1: treats each token as an independent classification.
         Simple but inflated — getting 90% of O tokens right is easy.

      2. Span-level F1: extracts contiguous B-I spans, checks if predicted
         spans match ground truth spans (exact aspect + overlap).
         This is the metric that actually matters for ABSA.

    A span is defined as a contiguous B-X I-X I-X sequence.
    """
    # 1. Token-level (filter out -100 ignore tokens)
    flat_labels = []
    flat_preds = []
    for labels, preds in zip(all_span_labels, all_span_preds):
        mask = labels != -100
        flat_labels.extend(labels[mask].tolist())
        flat_preds.extend(preds[mask].tolist())

    flat_labels = np.array(flat_labels)
    flat_preds = np.array(flat_preds)

    token_acc = accuracy_score(flat_labels, flat_preds) if len(flat_labels) > 0 else 0.0
    token_f1 = f1_score(flat_labels, flat_preds, average="macro", zero_division=0) if len(flat_labels) > 0 else 0.0

    # Per-BIO-class breakdown (skip O for clarity)
    non_o_labels = sorted(set(flat_labels.tolist()) | set(flat_preds.tolist()))
    per_class = {}
    if len(flat_labels) > 0:
        prec, rec, f1, sup = precision_recall_fscore_support(
            flat_labels, flat_preds, labels=non_o_labels, average=None, zero_division=0,
        )
        for i, label_id in enumerate(non_o_labels):
            name = bio_label_map.get(label_id, f"bio_{label_id}")
            per_class[name] = {
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(sup[i]),
            }

    # 2. Span-level F1
    def extract_spans(bio_seq):
        """Extract (aspect_id, start, end) tuples from BIO sequence."""
        spans = []
        current_aspect = None
        start = None
        for i, label in enumerate(bio_seq):
            if label <= 0:  # O or -100
                if current_aspect is not None:
                    spans.append((current_aspect, start, i))
                    current_aspect = None
            elif label % 2 == 1:  # B-tag (odd numbers)
                if current_aspect is not None:
                    spans.append((current_aspect, start, i))
                current_aspect = (label - 1) // 2
                start = i
            else:  # I-tag (even numbers)
                expected_aspect = (label - 2) // 2
                if current_aspect != expected_aspect:
                    if current_aspect is not None:
                        spans.append((current_aspect, start, i))
                    current_aspect = expected_aspect
                    start = i
        if current_aspect is not None:
            spans.append((current_aspect, start, len(bio_seq)))
        return set(spans)

    total_true_spans = 0
    total_pred_spans = 0
    total_correct = 0

    for labels, preds in zip(all_span_labels, all_span_preds):
        mask = labels != -100
        true_spans = extract_spans(labels[mask].tolist())
        pred_spans = extract_spans(preds[mask].tolist())
        total_true_spans += len(true_spans)
        total_pred_spans += len(pred_spans)
        total_correct += len(true_spans & pred_spans)

    span_precision = total_correct / max(total_pred_spans, 1)
    span_recall = total_correct / max(total_true_spans, 1)
    span_f1 = (
        2 * span_precision * span_recall / max(span_precision + span_recall, 1e-8)
    )

    result = {
        "token_accuracy": round(float(token_acc), 4),
        "token_macro_f1": round(float(token_f1), 4),
        "span_precision": round(float(span_precision), 4),
        "span_recall": round(float(span_recall), 4),
        "span_f1": round(float(span_f1), 4),
        "total_true_spans": total_true_spans,
        "total_pred_spans": total_pred_spans,
        "total_correct_spans": total_correct,
        "per_bio_class": per_class,
    }

    logger.info(
        f"  spans: token_acc={token_acc:.3f}, "
        f"span_f1={span_f1:.3f} "
        f"(P={span_precision:.3f}, R={span_recall:.3f}, "
        f"{total_correct}/{total_true_spans} spans matched)"
    )

    return result


def _compute_sarcasm_routing_metrics(sarcasm_data: List[Dict]) -> Dict:
    """
    Compute sarcasm routing effectiveness metrics (V2).

    Key question: "Of the reviews the model misclassified,
    what % had high sarcasm scores?" If this % is significant,
    the routing system has measurable value.

    Metrics:
      - Route distribution: how many reviews went trust/flag/invert
      - Per-route accuracy: sentiment accuracy within each route
      - Sarcasm-misclass correlation: do misclassified reviews have
        higher sarcasm scores than correct ones?
    """
    from collections import Counter

    n = len(sarcasm_data)
    if n == 0:
        return {}

    # Route distribution
    routes = [d["sarcasm_route"] for d in sarcasm_data]
    route_counts = Counter(routes)
    route_dist = {
        route: {"count": count, "pct": round(count / n, 4)}
        for route, count in route_counts.items()
    }

    # Per-route sentiment accuracy
    route_accuracy = {}
    for route in ["trust", "flag", "invert"]:
        subset = [d for d in sarcasm_data if d["sarcasm_route"] == route]
        if subset:
            correct = sum(1 for d in subset if d["sentiment_true"] == d["sentiment_pred"])
            route_accuracy[route] = {
                "accuracy": round(correct / len(subset), 4),
                "count": len(subset),
            }

    # Sarcasm score distribution for correct vs incorrect predictions
    scores_correct = [
        d["sarcasm_score"] for d in sarcasm_data
        if d["sentiment_true"] == d["sentiment_pred"]
    ]
    scores_incorrect = [
        d["sarcasm_score"] for d in sarcasm_data
        if d["sentiment_true"] != d["sentiment_pred"]
    ]

    score_analysis = {
        "mean_score_correct": round(float(np.mean(scores_correct)), 4) if scores_correct else 0.0,
        "mean_score_incorrect": round(float(np.mean(scores_incorrect)), 4) if scores_incorrect else 0.0,
        "num_correct": len(scores_correct),
        "num_incorrect": len(scores_incorrect),
    }

    # Core metric: do misclassified reviews have higher sarcasm scores?
    if scores_correct and scores_incorrect:
        score_gap = score_analysis["mean_score_incorrect"] - score_analysis["mean_score_correct"]
        score_analysis["score_gap"] = round(score_gap, 4)
        score_analysis["sarcasm_correlates_with_errors"] = score_gap > 0.05
    else:
        score_analysis["score_gap"] = 0.0
        score_analysis["sarcasm_correlates_with_errors"] = False

    # Inversion effectiveness: for inverted predictions,
    # were they originally wrong and now correct?
    inverted = [d for d in sarcasm_data if d["was_inverted"]]
    if inverted:
        # After inversion, how many are now correct?
        inversion_correct = sum(
            1 for d in inverted if d["sentiment_true"] == d["sentiment_pred"]
        )
        inversion_stats = {
            "num_inverted": len(inverted),
            "accuracy_after_inversion": round(inversion_correct / len(inverted), 4),
        }
    else:
        inversion_stats = {"num_inverted": 0, "accuracy_after_inversion": 0.0}

    result = {
        "route_distribution": route_dist,
        "per_route_accuracy": route_accuracy,
        "score_analysis": score_analysis,
        "inversion_effectiveness": inversion_stats,
    }

    logger.info(
        f"  sarcasm routing: {route_counts} | "
        f"score_gap={score_analysis['score_gap']:.3f} "
        f"(correlates={score_analysis['sarcasm_correlates_with_errors']})"
    )

    return result


def _compute_quantum_uncertainty_metrics(quantum_data: List[Dict]) -> Dict:
    """
    Compute quantum-inspired uncertainty metrics (V2 iter3).

    Core question: Does high von Neumann entropy / high interference
    correlate with misclassification? If yes, the density matrix
    representation captures something softmax confidence doesn't.

    Metrics:
      1. Mean entropy for correct vs incorrect predictions
         (gap = how well entropy predicts errors)
      2. Mean interference for correct vs incorrect
      3. Entropy-accuracy curve: accuracy within entropy buckets
      4. Interference-accuracy curve: same for interference
    """
    n = len(quantum_data)
    if n == 0:
        return {}

    correct = [d for d in quantum_data if d["correct"]]
    incorrect = [d for d in quantum_data if not d["correct"]]

    # 1. Entropy analysis
    ent_correct = [d["entropy"] for d in correct]
    ent_incorrect = [d["entropy"] for d in incorrect]
    entropy_analysis = {
        "mean_entropy_correct": round(float(np.mean(ent_correct)), 4) if ent_correct else 0.0,
        "mean_entropy_incorrect": round(float(np.mean(ent_incorrect)), 4) if ent_incorrect else 0.0,
        "num_correct": len(correct),
        "num_incorrect": len(incorrect),
    }
    if ent_correct and ent_incorrect:
        gap = entropy_analysis["mean_entropy_incorrect"] - entropy_analysis["mean_entropy_correct"]
        entropy_analysis["entropy_gap"] = round(gap, 4)
        entropy_analysis["entropy_predicts_errors"] = gap > 0.01
    else:
        entropy_analysis["entropy_gap"] = 0.0
        entropy_analysis["entropy_predicts_errors"] = False

    # 2. Interference analysis
    interf_correct = [d["interference"] for d in correct]
    interf_incorrect = [d["interference"] for d in incorrect]
    interference_analysis = {
        "mean_interference_correct": round(float(np.mean(interf_correct)), 4) if interf_correct else 0.0,
        "mean_interference_incorrect": round(float(np.mean(interf_incorrect)), 4) if interf_incorrect else 0.0,
    }
    if interf_correct and interf_incorrect:
        gap = interference_analysis["mean_interference_incorrect"] - interference_analysis["mean_interference_correct"]
        interference_analysis["interference_gap"] = round(gap, 4)
        interference_analysis["interference_predicts_errors"] = gap > 0.01
    else:
        interference_analysis["interference_gap"] = 0.0
        interference_analysis["interference_predicts_errors"] = False

    # 3. Bucketed accuracy: bin samples by entropy, check accuracy per bucket
    # This shows WHERE in the entropy range the model breaks down
    import math
    max_entropy = math.log(3)  # 3 sentiment classes
    entropy_buckets = {}
    bucket_boundaries = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Normalized entropy [0,1]
    for i in range(len(bucket_boundaries) - 1):
        lo, hi = bucket_boundaries[i], bucket_boundaries[i + 1]
        bucket_name = f"{lo:.1f}-{hi:.1f}"
        in_bucket = [
            d for d in quantum_data
            if lo <= d["entropy"] / max_entropy < hi
        ]
        if in_bucket:
            acc = sum(1 for d in in_bucket if d["correct"]) / len(in_bucket)
            entropy_buckets[bucket_name] = {
                "accuracy": round(acc, 4),
                "count": len(in_bucket),
            }

    # 4. Overall summary stat: Spearman correlation between entropy and error
    # (if scipy available, otherwise skip)
    correlation = None
    try:
        from scipy.stats import spearmanr
        entropies = [d["entropy"] for d in quantum_data]
        errors = [0 if d["correct"] else 1 for d in quantum_data]
        if len(set(entropies)) > 1 and len(set(errors)) > 1:
            rho, pval = spearmanr(entropies, errors)
            correlation = {
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(pval), 6),
                "significant": pval < 0.05,
            }
    except ImportError:
        pass

    result = {
        "entropy_analysis": entropy_analysis,
        "interference_analysis": interference_analysis,
        "entropy_accuracy_buckets": entropy_buckets,
    }
    if correlation:
        result["entropy_error_correlation"] = correlation

    logger.info(
        f"  quantum: entropy_gap={entropy_analysis['entropy_gap']:.4f}, "
        f"interference_gap={interference_analysis['interference_gap']:.4f}, "
        f"predicts_errors={entropy_analysis['entropy_predicts_errors']}"
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
