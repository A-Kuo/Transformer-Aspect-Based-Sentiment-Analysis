"""
Training loop for V3 hybrid model (forwards optional span_labels like V2 data).
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from src.train import Trainer

logger = logging.getLogger(__name__)


class HybridTrainer(Trainer):
    """
    Same as `src.train.Trainer` but passes `span_labels` when batches include them
    (V2 `ReviewDataset` + collate with `span_extraction: true`).
    """

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_aspect_loss = 0.0
        total_sentiment_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            aspect_labels = batch["aspect_labels"].to(self.device)
            sentiment_labels = batch["sentiment_labels"].to(self.device)

            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "aspect_labels": aspect_labels,
                "sentiment_labels": sentiment_labels,
            }
            if "span_labels" in batch:
                kwargs["span_labels"] = batch["span_labels"].to(self.device)

            with autocast(device_type=self.device.type, enabled=self.use_fp16):
                outputs = self.model(**kwargs)
                loss = outputs["loss"] / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

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

            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "aspect_labels": aspect_labels,
                "sentiment_labels": sentiment_labels,
            }
            if "span_labels" in batch:
                kwargs["span_labels"] = batch["span_labels"].to(self.device)

            with autocast(device_type=self.device.type, enabled=self.use_fp16):
                outputs = self.model(**kwargs)

            total_loss += outputs["loss"].item()
            total_aspect_loss += outputs["aspect_loss"].item()
            total_sentiment_loss += outputs["sentiment_loss"].item()
            num_batches += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "aspect_loss": total_aspect_loss / max(num_batches, 1),
            "sentiment_loss": total_sentiment_loss / max(num_batches, 1),
        }
