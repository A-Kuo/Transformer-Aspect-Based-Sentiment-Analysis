"""
Training loop for BERT Aspect-Based Sentiment Analysis.

Training pipeline:
    DataLoader → Model.forward() → Joint Loss → Backward → Optimizer Step
                                                    ↑
                                            LR Scheduler (warmup + decay)

Key engineering decisions:

  1. Linear warmup + cosine decay: BERT's pretrained weights sit in a
     sharp loss basin. Large initial LR steps can kick parameters out
     of this basin, destroying pretrained knowledge. Warmup gradually
     increases LR over the first 10% of steps, letting the new heads
     (randomly initialized) catch up to the backbone's scale before
     full-strength updates begin.

  2. Mixed precision (fp16): Forward pass runs in float16 on GPU,
     halving memory for activations and enabling larger batches.
     Gradients are computed in fp16 but accumulated in fp32 (loss
     scaling prevents underflow). ~2x speedup on modern GPUs.

  3. Gradient accumulation: Effective batch = batch_size × accum_steps.
     With batch_size=16 and accum_steps=2, effective batch is 32.
     This simulates larger batches without the memory cost, giving
     more stable gradient estimates for BERT fine-tuning.

  4. Early stopping: If validation loss doesn't improve for
     `patience` epochs, stop training. Prevents overfitting on our
     small 5000-sample dataset.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast

from src.data import load_config, create_dataloaders
from src.model import AspectSentimentModel, build_model

logger = logging.getLogger(__name__)


class Trainer:
    """
    Manages the full training lifecycle.

    Responsibilities:
      - Optimizer & scheduler setup
      - Training loop with mixed precision
      - Validation after each epoch
      - Checkpointing (save best model)
      - Early stopping
      - Logging metrics to JSON
    """

    def __init__(
        self,
        model: AspectSentimentModel,
        config: dict,
        train_loader,
        val_loader,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = next(model.parameters()).device

        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.grad_accum_steps = train_cfg["gradient_accumulation_steps"]
        self.patience = train_cfg["early_stopping_patience"]
        self.use_fp16 = train_cfg["fp16"] and self.device.type == "cuda"

        # Paths
        self.model_dir = Path(config["paths"]["model_dir"])
        self.results_dir = Path(config["paths"]["results_dir"])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # --- Optimizer: AdamW ---
        # AdamW decouples weight decay from the gradient update.
        # Standard Adam applies decay inside the momentum update,
        # which interacts poorly with adaptive learning rates.
        # AdamW applies decay directly to weights: w = w - λ·w
        # This gives cleaner regularization for transformer fine-tuning.
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            betas=(0.9, 0.999),  # Standard Adam betas
            eps=1e-8,
        )

        # --- LR Scheduler: OneCycleLR ---
        # Combines warmup (linear ramp to max LR) with cosine annealing
        # back to ~0. The warmup phase is critical: new classification
        # heads start with random weights, so early large LR updates
        # to the backbone would be driven by random head gradients.
        total_steps = (
            len(train_loader) // self.grad_accum_steps * self.epochs
        )
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=train_cfg["learning_rate"],
            total_steps=total_steps,
            pct_start=train_cfg["warmup_ratio"],  # 10% warmup
            anneal_strategy="cos",
        )

        # --- Mixed precision ---
        # GradScaler prevents float16 gradient underflow by scaling
        # the loss before backward, then unscaling gradients before
        # the optimizer step. If gradients contain inf/nan (overflow),
        # the scaler skips that step entirely.
        self.scaler = GradScaler(enabled=self.use_fp16)

        # Tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_aspect_loss": [],
            "train_sentiment_loss": [],
            "val_aspect_loss": [],
            "val_sentiment_loss": [],
            "learning_rate": [],
            "epoch_time_s": [],
        }
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train(self) -> Dict:
        """
        Full training run across all epochs.

        Returns:
            Training history dict.
        """
        logger.info(
            f"Starting training: {self.epochs} epochs, "
            f"device={self.device}, fp16={self.use_fp16}, "
            f"grad_accum={self.grad_accum_steps}"
        )

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Train one epoch
            train_metrics = self._train_epoch(epoch)

            # Validate
            val_metrics = self._validate(epoch)

            epoch_time = time.time() - epoch_start

            # Record metrics
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_aspect_loss"].append(train_metrics["aspect_loss"])
            self.history["train_sentiment_loss"].append(train_metrics["sentiment_loss"])
            self.history["val_aspect_loss"].append(val_metrics["aspect_loss"])
            self.history["val_sentiment_loss"].append(val_metrics["sentiment_loss"])
            self.history["learning_rate"].append(
                self.optimizer.param_groups[0]["lr"]
            )
            self.history["epoch_time_s"].append(round(epoch_time, 1))

            # Log summary
            logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"(asp: {train_metrics['aspect_loss']:.4f}, "
                f"sent: {train_metrics['sentiment_loss']:.4f}) | "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"(asp: {val_metrics['aspect_loss']:.4f}, "
                f"sent: {val_metrics['sentiment_loss']:.4f}) | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Checkpointing: save if best val loss
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
                logger.info(f"  ✓ New best val loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
                logger.info(
                    f"  No improvement. Patience: "
                    f"{self.patience_counter}/{self.patience}"
                )

            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {self.patience} epochs)"
                )
                break

        # Save final training history
        self._save_history()
        logger.info(f"Training complete. Best val loss: {self.best_val_loss:.4f}")

        return self.history

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Implements gradient accumulation:
          - Forward + backward on each mini-batch
          - Only step optimizer every `grad_accum_steps` batches
          - This simulates larger effective batch size

        Returns:
            Dict with average loss values for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_aspect_loss = 0.0
        total_sentiment_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            aspect_labels = batch["aspect_labels"].to(self.device)
            sentiment_labels = batch["sentiment_labels"].to(self.device)

            # Forward pass with mixed precision
            with autocast(device_type=self.device.type, enabled=self.use_fp16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    aspect_labels=aspect_labels,
                    sentiment_labels=sentiment_labels,
                )
                # Scale loss by accumulation steps so gradients average correctly
                loss = outputs["loss"] / self.grad_accum_steps

            # Backward pass (gradients computed in fp16, accumulated in fp32)
            self.scaler.scale(loss).backward()

            # Optimizer step every grad_accum_steps
            if (step + 1) % self.grad_accum_steps == 0:
                # Gradient clipping: prevents exploding gradients that
                # can destabilize BERT fine-tuning. Max norm of 1.0 is
                # standard — if ||grad|| > 1.0, scale down proportionally.
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Track losses (unscaled)
            total_loss += outputs["loss"].item()
            total_aspect_loss += outputs["aspect_loss"].item()
            total_sentiment_loss += outputs["sentiment_loss"].item()
            num_batches += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "aspect_loss": total_aspect_loss / max(num_batches, 1),
            "sentiment_loss": total_sentiment_loss / max(num_batches, 1),
        }

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """
        Run validation pass (no gradients).

        Returns:
            Dict with average loss values.
        """
        self.model.eval()
        total_loss = 0.0
        total_aspect_loss = 0.0
        total_sentiment_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            aspect_labels = batch["aspect_labels"].to(self.device)
            sentiment_labels = batch["sentiment_labels"].to(self.device)

            with autocast(device_type=self.device.type, enabled=self.use_fp16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    aspect_labels=aspect_labels,
                    sentiment_labels=sentiment_labels,
                )

            total_loss += outputs["loss"].item()
            total_aspect_loss += outputs["aspect_loss"].item()
            total_sentiment_loss += outputs["sentiment_loss"].item()
            num_batches += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "aspect_loss": total_aspect_loss / max(num_batches, 1),
            "sentiment_loss": total_sentiment_loss / max(num_batches, 1),
        }

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.

        Saves:
          - Model state dict (weights)
          - Optimizer state (for resuming training)
          - Scheduler state
          - Epoch number and best val loss

        We save the full state so training can be resumed from any
        checkpoint. For deployment, only model weights are needed.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        # Save latest
        path = self.model_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, path)

        # Save best separately
        if is_best:
            best_path = self.model_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"  Saved best checkpoint → {best_path}")

    def _save_history(self) -> None:
        """Save training history as JSON for plotting and analysis."""
        history_path = self.results_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved → {history_path}")


def load_checkpoint(
    model: AspectSentimentModel,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Load model weights from checkpoint.

    Args:
        model: Initialized model (architecture must match checkpoint).
        checkpoint_path: Path to .pt checkpoint file.
        device: Target device. If None, uses model's current device.

    Returns:
        Checkpoint dict (contains epoch, config, etc.).
    """
    if device is None:
        device = next(model.parameters()).device

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        f"Loaded checkpoint from epoch {checkpoint['epoch']} "
        f"(val_loss={checkpoint['best_val_loss']:.4f})"
    )
    return checkpoint


def run_training(config: Optional[dict] = None) -> Dict:
    """
    End-to-end training entry point.

    Loads config → builds dataloaders → builds model → trains → saves.

    Args:
        config: Config dict. If None, loads from config.yaml.

    Returns:
        Training history dict.
    """
    if config is None:
        config = load_config()

    logger.info("Building dataloaders...")
    dataloaders = create_dataloaders(config)

    logger.info("Building model...")
    model = build_model(config)

    # Optional: freeze backbone for small datasets
    # model.freeze_backbone(num_layers_to_freeze=10)

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
    )

    history = trainer.train()
    return history


# ============================================================
# Standalone entry point
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    history = run_training()
    print(f"\nFinal train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss:   {history['val_loss'][-1]:.4f}")
