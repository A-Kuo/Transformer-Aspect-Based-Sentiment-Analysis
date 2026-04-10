#!/usr/bin/env python3
"""
V3 CLI — hybrid SSM–attention ABSA (train / predict / info).

Run from repository root:
  python -m V3.main train --config V3/config_hybrid.yaml
  python -m V3.main predict "Great quality, slow shipping" --config V3/config_hybrid.yaml
  python -m V3.main info --config V3/config_hybrid.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure repo root is on path when running `python V3/main.py` (not only `-m V3.main`)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data import load_config


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_train(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from V2.data import create_dataloaders
    from V3.model_hybrid import build_model
    from V3.train_hybrid import HybridTrainer

    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    logging.info("=" * 60)
    logging.info("V3 — HYBRID TRAINING")
    logging.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    dataloaders = create_dataloaders(config, tokenizer=tokenizer)
    model = build_model(config)

    if args.freeze is not None and args.freeze > 0:
        model.freeze_backbone(num_layers_to_freeze=args.freeze)

    trainer = HybridTrainer(
        model=model,
        config=config,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
    )

    t0 = time.time()
    history = trainer.train()
    elapsed = time.time() - t0

    logging.info("Training complete in %.1fs", elapsed)
    logging.info("  Final train loss: %.4f", history["train_loss"][-1])
    logging.info("  Final val loss:   %.4f", history["val_loss"][-1])
    logging.info("  Best val loss:    %.4f", trainer.best_val_loss)
    logging.info("  Checkpoint:       %scheckpoint_best.pt", config["paths"]["model_dir"])


def cmd_predict(args: argparse.Namespace) -> None:
    from V3.inference import Predictor, format_prediction

    if args.checkpoint:
        predictor = Predictor.from_checkpoint(args.checkpoint, config_path=args.config)
    else:
        predictor = Predictor.from_pretrained(config_path=args.config)

    text = " ".join(args.text) if args.text else ""
    if not text.strip():
        logging.error("No text provided.")
        sys.exit(1)

    pred = predictor.predict_one(text)
    print(format_prediction(pred, verbose=args.verbose))


def cmd_info(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ssm = config.get("ssm", {})
    print("=" * 60)
    print("V3 — Hybrid SSM–Attention ABSA")
    print("=" * 60)
    print(f"Backbone:      {config['model']['name']} encoder with SSM sandwich")
    print(f"  early attn:  {ssm.get('num_early_attention', 4)} layers")
    print(f"  SSM blocks:  {ssm.get('num_ssm', 6)} (Mamba2 on CUDA if installed)")
    print(f"  late attn:   {ssm.get('num_late_attention', 2)} layers")
    print(f"Max seq len:   {config['model']['max_seq_length']}")
    print(f"Span BIO:      {config['model'].get('span_extraction', False)}")
    print(f"Sarcasm route: {config.get('sarcasm', {}).get('enabled', False)}")
    print(f"Model dir:     {config['paths']['model_dir']}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 hybrid ABSA CLI")
    parser.add_argument("--config", default="V3/config_hybrid.yaml", help="Path to YAML config")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Fine-tune hybrid model")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--lr", type=float, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--freeze", type=int, default=None, help="Freeze first N sandwich blocks")
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Run inference on one review")
    p_pred.add_argument("text", nargs="+", help="Review text")
    p_pred.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    p_pred.add_argument("--verbose", action="store_true")
    p_pred.set_defaults(func=cmd_predict)

    p_info = sub.add_parser("info", help="Print V3 config summary")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
