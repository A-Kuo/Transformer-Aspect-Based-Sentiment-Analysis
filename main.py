#!/usr/bin/env python3
"""
BERT Aspect-Based Sentiment Analysis — CLI Entry Point

Usage:
    python main.py train                         # Train model on Amazon reviews
    python main.py train --freeze 10             # Freeze first 10 BERT layers
    python main.py evaluate                      # Evaluate best checkpoint on test set
    python main.py evaluate --baseline           # Evaluate untrained BERT (random heads)
    python main.py predict "Great product!"      # Single review prediction
    python main.py predict --file reviews.txt    # Batch prediction from file
    python main.py test                          # Run smoke tests
    python main.py info                          # Print config + architecture summary

Pipeline overview:

    ┌─────────┐     ┌───────────┐     ┌───────────┐     ┌──────────┐
    │  train   │ ──→ │ checkpoint│ ──→ │ evaluate  │ ──→ │ metrics  │
    │          │     │  .pt file │     │           │     │  .json   │
    └─────────┘     └───────────┘     └───────────┘     └──────────┘
                          │
                          ▼
                    ┌───────────┐
                    │  predict  │ ──→  aspect + sentiment + confidence
                    └───────────┘

Every subcommand is self-contained: loads config, builds what it
needs, runs, saves outputs. No hidden state between commands.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from src.data import load_config

# ============================================================
# Logging setup
# ============================================================

def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ============================================================
# Subcommands
# ============================================================

def cmd_train(args: argparse.Namespace) -> None:
    """
    Train the dual-head BERT model.

    Pipeline: load config -> build dataloaders -> build model ->
              (optionally freeze backbone) -> train -> save checkpoint.

    The training loop uses:
      - AdamW with decoupled weight decay
      - Linear warmup + cosine annealing (OneCycleLR)
      - Mixed-precision fp16 on CUDA
      - Gradient accumulation (effective batch = batch_size * accum_steps)
      - Early stopping on validation loss
    """
    from src.data import compute_class_weights, create_dataloaders
    from src.model import build_model
    from src.train import Trainer

    config = load_config(args.config)

    # Override config from CLI flags
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    logging.info("=" * 60)
    logging.info("PHASE 1 — TRAINING")
    logging.info("=" * 60)

    dataloaders = create_dataloaders(config)

    aspect_class_weights = None
    sentiment_class_weights = None
    if config["training"].get("class_weighted_loss", False):
        train_dataset = dataloaders["train"].dataset
        aspect_class_weights = compute_class_weights(
            train_dataset.aspect_labels, config["model"]["num_aspect_labels"]
        )
        sentiment_class_weights = compute_class_weights(
            train_dataset.sentiment_labels, config["model"]["num_sentiment_labels"]
        )
        logging.info(f"Class-weighted loss enabled. Aspect weights: {aspect_class_weights.tolist()}")
        logging.info(f"Class-weighted loss enabled. Sentiment weights: {sentiment_class_weights.tolist()}")

    model = build_model(
        config,
        aspect_class_weights=aspect_class_weights,
        sentiment_class_weights=sentiment_class_weights,
    )

    # Optional: freeze early layers for small dataset
    if args.freeze is not None and args.freeze > 0:
        model.freeze_backbone(num_layers_to_freeze=args.freeze)
        logging.info(
            f"Froze first {args.freeze}/12 encoder layers "
            f"(~{args.freeze/12*100:.0f}% of backbone frozen)"
        )

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
    )

    t0 = time.time()
    history = trainer.train()
    elapsed = time.time() - t0

    logging.info(f"Training complete in {elapsed:.1f}s")
    logging.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    logging.info(f"  Final val loss:   {history['val_loss'][-1]:.4f}")
    logging.info(f"  Best val loss:    {trainer.best_val_loss:.4f}")
    logging.info(f"  Checkpoint:       {config['paths']['model_dir']}checkpoint_best.pt")
    logging.info(f"  History:          {config['paths']['results_dir']}training_history.json")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """
    Evaluate model on the held-out test set.

    Computes:
      - Per-task accuracy, macro/weighted F1
      - Per-class precision, recall, F1
      - Confusion matrices (5x5 aspect, 3x3 sentiment)
      - Per-aspect sentiment accuracy (cross-task analysis)
      - Latency percentiles (p50/p95/p99) with SLA check

    Saves results to results/metrics.json.
    """
    from src.data import create_dataloaders
    from src.evaluate import evaluate_model, print_report, save_metrics
    from src.inference import Predictor

    config = load_config(args.config)
    checkpoint_path = args.checkpoint

    logging.info("=" * 60)
    logging.info("PHASE 1 — EVALUATION")
    logging.info("=" * 60)

    if args.baseline:
        logging.info("Evaluating UNTRAINED baseline (random heads)")
        predictor = Predictor.from_pretrained(args.config)
        output_path = "results/metrics_baseline.json"
    else:
        if not Path(checkpoint_path).exists():
            logging.error(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"  Run 'python main.py train' first, or use --baseline for untrained model."
            )
            sys.exit(1)
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        predictor = Predictor.from_checkpoint(checkpoint_path, args.config)
        output_path = "results/metrics.json"

    dataloaders = create_dataloaders(config)
    metrics = evaluate_model(predictor, dataloaders["test"], config)

    save_metrics(metrics, output_path)
    print_report(metrics)

    # Warn if below accuracy floor
    acc_floor = config.get("monitoring", {}).get("accuracy_floor", 0.0)
    sent_acc = metrics["sentiment"]["accuracy"]
    if sent_acc < acc_floor:
        logging.warning(
            f"Sentiment accuracy {sent_acc:.3f} below floor {acc_floor}. "
            f"Consider more data or hyperparameter tuning."
        )


def cmd_predict(args: argparse.Namespace) -> None:
    """
    Run inference on text input.

    Supports:
      - Single text from CLI argument
      - Batch from file (one review per line)
      - Interactive mode (type reviews, Ctrl+D to exit)
    """
    from src.inference import Predictor, format_prediction

    load_config(args.config)
    checkpoint_path = args.checkpoint

    if Path(checkpoint_path).exists():
        predictor = Predictor.from_checkpoint(checkpoint_path, args.config)
        logging.info(f"Loaded trained model from {checkpoint_path}")
    else:
        logging.warning(
            f"No checkpoint at {checkpoint_path} — using untrained model. "
            f"Run 'python main.py train' first for meaningful predictions."
        )
        predictor = Predictor.from_pretrained(args.config)

    # --- Single text ---
    if args.text:
        text = " ".join(args.text)
        result = predictor.predict_one(text)
        print(f"\n{format_prediction(result, verbose=args.verbose)}\n")
        return

    # --- File input ---
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logging.error(f"File not found: {file_path}")
            sys.exit(1)

        texts = [
            line.strip()
            for line in file_path.read_text().splitlines()
            if line.strip()
        ]
        logging.info(f"Processing {len(texts)} reviews from {file_path}")

        results = predictor.predict_batch(texts)

        print(f"\n{'='*70}")
        for i, r in enumerate(results, 1):
            print(f"[{i}/{len(results)}]")
            print(format_prediction(r, verbose=args.verbose))
            print("-" * 70)

        # Summary stats
        aspects = [r["aspect"] for r in results]
        sentiments = [r["sentiment"] for r in results]
        print(f"\nSummary ({len(results)} reviews):")
        print(f"  Aspects:    {dict(sorted(((a, aspects.count(a)) for a in set(aspects)), key=lambda x: -x[1]))}")
        print(f"  Sentiments: {dict(sorted(((s, sentiments.count(s)) for s in set(sentiments)), key=lambda x: -x[1]))}")
        print(f"  Avg latency: {sum(r['latency_ms'] for r in results)/len(results):.1f}ms/review")
        print()
        return

    # --- Interactive mode ---
    print("\nInteractive prediction mode (type a review, Enter to predict, Ctrl+D to exit)\n")
    try:
        while True:
            text = input("Review > ").strip()
            if not text:
                continue
            result = predictor.predict_one(text)
            print(f"  -> {result['aspect']} ({result['aspect_confidence']:.0%}) | "
                  f"{result['sentiment']} ({result['sentiment_confidence']:.0%}) | "
                  f"{result['latency_ms']:.0f}ms\n")
    except (EOFError, KeyboardInterrupt):
        print("\nDone.")


def cmd_test(args: argparse.Namespace) -> None:
    """Run the smoke test suite via pytest."""
    import subprocess

    logging.info("Running smoke tests...")
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_core.py",
        "-v", "--tb=short",
    ]
    if args.quick:
        cmd.extend([
            "-k", "TestConfig or TestEvaluationMetrics or "
                  "test_aspect_keywords or test_assign_aspect or test_stars_to_sentiment"
        ])
        logging.info("Quick mode: skipping tests that require BERT download")

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    sys.exit(result.returncode)


def cmd_info(args: argparse.Namespace) -> None:
    """Print project configuration and architecture summary."""
    config = load_config(args.config)

    print(f"\n{'='*60}")
    print("BERT ASPECT-BASED SENTIMENT ANALYSIS — Phase 1 Prototype")
    print(f"{'='*60}")

    print("\nArchitecture:")
    print(f"  Backbone:       {config['model']['name']} (~110M params)")
    print(f"  Aspect head:    Linear(768 -> {config['model']['num_aspect_labels']})")
    print(f"  Sentiment head: Linear(768 -> {config['model']['num_sentiment_labels']})")
    print(f"  Max seq length: {config['model']['max_seq_length']} tokens")
    print(f"  Dropout:        {config['model']['dropout']}")

    print("\nData:")
    print(f"  Dataset:        {config['data']['dataset_name']}")
    print(f"  Subset:         {config['data']['dataset_subset']}")
    print(f"  Train/Val/Test: {config['data']['train_size']}/{config['data']['val_size']}/{config['data']['test_size']}")

    print("\nTraining:")
    t = config["training"]
    print(f"  Epochs:         {t['epochs']}")
    print(f"  Batch size:     {t['batch_size']} (effective: {t['batch_size'] * t['gradient_accumulation_steps']})")
    print(f"  Learning rate:  {t['learning_rate']}")
    print(f"  Warmup:         {t['warmup_ratio']*100:.0f}% of steps")
    print(f"  FP16:           {t['fp16']}")
    print(f"  Early stopping: patience={t['early_stopping_patience']}")

    print("\nLabels:")
    print(f"  Aspects:    {list(config['aspects'].values())}")
    print(f"  Sentiments: {list(config['sentiments'].values())}")

    print("\nPaths:")
    for k, v in config["paths"].items():
        print(f"  {k:15s} {v}")

    # Check for existing artifacts
    model_dir = Path(config["paths"]["model_dir"])
    results_dir = Path(config["paths"]["results_dir"])

    if (model_dir / "checkpoint_best.pt").exists():
        import torch
        ckpt = torch.load(model_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        print(f"\n[trained] Checkpoint found (epoch {ckpt['epoch']}, val_loss={ckpt['best_val_loss']:.4f})")
    else:
        print("\n[untrained] No checkpoint yet. Run: python main.py train")

    if (results_dir / "metrics.json").exists():
        import json
        with open(results_dir / "metrics.json") as f:
            m = json.load(f)
        print(f"[evaluated] Aspect acc: {m['aspect']['accuracy']:.3f} | Sentiment acc: {m['sentiment']['accuracy']:.3f}")
    else:
        print("[unevaluated] Run: python main.py evaluate")

    print(f"\n{'='*60}\n")


# ============================================================
# Argument parser
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bert-sentiment",
        description="BERT Aspect-Based Sentiment Analysis — Phase 1 Prototype",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py info                                    # Show config
  python main.py train                                   # Train with defaults
  python main.py train --freeze 10 --epochs 5            # Freeze + more epochs
  python main.py evaluate                                # Eval best checkpoint
  python main.py evaluate --baseline                     # Eval untrained model
  python main.py predict "Amazing quality, fast shipping" # Single prediction
  python main.py predict --file reviews.txt              # Batch from file
  python main.py predict                                 # Interactive mode
  python main.py test                                    # Full test suite
  python main.py test --quick                            # Quick tests (no BERT download)
        """,
    )

    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, help="Override number of epochs")
    train_parser.add_argument("--lr", type=float, help="Override learning rate")
    train_parser.add_argument("--batch-size", type=int, help="Override batch size")
    train_parser.add_argument(
        "--freeze", type=int, default=None,
        help="Freeze first N BERT encoder layers (0-12). "
             "Recommended: 10 for small datasets (<10K samples)",
    )

    # --- evaluate ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on test set")
    eval_parser.add_argument(
        "--checkpoint", default="models/checkpoint_best.pt",
        help="Path to model checkpoint (default: models/checkpoint_best.pt)",
    )
    eval_parser.add_argument(
        "--baseline", action="store_true",
        help="Evaluate untrained BERT (random classification heads)",
    )

    # --- predict ---
    pred_parser = subparsers.add_parser("predict", help="Run predictions")
    pred_parser.add_argument(
        "text", nargs="*", default=None,
        help="Review text to classify (omit for interactive mode)",
    )
    pred_parser.add_argument(
        "--file", "-f", type=str, default=None,
        help="Path to file with one review per line",
    )
    pred_parser.add_argument(
        "--checkpoint", default="models/checkpoint_best.pt",
        help="Path to model checkpoint",
    )
    pred_parser.add_argument(
        "--verbose", action="store_true",
        help="Show full probability distributions",
    )

    # --- test ---
    test_parser = subparsers.add_parser("test", help="Run smoke tests")
    test_parser.add_argument(
        "--quick", action="store_true",
        help="Skip tests requiring BERT download (~440MB)",
    )

    # --- info ---
    subparsers.add_parser("info", help="Print config and architecture summary")

    return parser


# ============================================================
# Entry point
# ============================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
        "test": cmd_test,
        "info": cmd_info,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn is None:
        parser.print_help()
        sys.exit(1)

    try:
        cmd_fn(args)
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
