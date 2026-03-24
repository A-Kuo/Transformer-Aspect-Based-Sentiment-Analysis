# V3 vs V2 benchmark checklist

Use this after training both stacks on the **same** data split and evaluation code.

## Environment

- **V2 baseline:** `V2/model.py` + `V2/config.yaml`, train via your V2 entrypoint (or adapt `src/train.py` with V2 model + `V2/data.py` batches including `span_labels`).
- **V3 hybrid:** `python -m V3.main train --config V3/config_hybrid.yaml` (uses `V2/data.create_dataloaders` + `HybridTrainer`).
- **Optional Mamba:** `pip install -e ".[v3]"` on **Linux + CUDA** for real Mamba2 middle blocks. Otherwise V3 uses the **CPU/GPU-safe conv fallback** (still linear-time in `T` per block).

## Metrics to record

| Metric | V2 | V3 | Notes |
|--------|----|----|-------|
| Val loss (best) | | | Early stopping comparable patience |
| Aspect accuracy / F1 | | | Same test set |
| Sentiment accuracy / F1 | | | Apply same sarcasm post-processing for fair compare |
| Span F1 (BIO) | | | Token-level, mask `-100` |
| Train wall-clock (3 epochs) | | | Same hardware, batch size |
| Inference ms / review | | | Batch size from config |
| Peak GPU memory | | | Longer sequences stress V2 attention more |

## Sequence-length scaling

1. Set `model.max_seq_length` to **256** in both configs; record throughput and (if applicable) memory.
2. Repeat with **512** (if GPU memory allows). Expect V3’s middle stack to show **larger relative gains** when `T` grows (attention remains quadratic only in the first/last BERT slices).

## Accuracy gate (from V3 plan)

- Target: V3 within **~2%** of V2 on primary sentiment/aspect metrics before merging V3 to `main`.
- If V3 lags: lower LR, train SSM middle longer, or freeze early BERT layers first (`--freeze 4`).

## Sarcasm + quantum

- **Sarcasm:** Routing is **identical** to V2 (rule-based `SarcasmDetector`). Compare `sarcasm_route` distribution on a fixed review list.
- **Quantum:** Compare `quantum_entropy` vs misclassification rate; high entropy should correlate with hard/sarcastic samples (hypothesis; log correlation).
