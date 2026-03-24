# NLP Transformer Analysis

Aspect-based sentiment analysis on product reviews using a fine-tuned BERT backbone.

The model predicts both **what** a review discusses (quality, shipping, value, usability, customer service) and **how** the reviewer feels about it (positive, neutral, negative) while being mindful of sarcasm.

## Installation

```bash
git clone https://github.com/A-Kuo/NLP-Transformer-Sentiments.git
cd NLP-Transformer-Sentiments

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev]"

# Optional: Mamba2 middle blocks for V3 (Linux/CUDA typical; see V3/V3_ITERATION_PLAN.md)
# pip install -e ".[v3]"
```

## Quick Start

```bash
python main.py train
python main.py evaluate
python main.py predict "Great quality but shipping took 3 weeks"
python main.py info
```

Or with `make`:

```bash
make train
make evaluate
make test
```

## Project Structure

```text
.
├── main.py                  # CLI entry point
├── config.yaml              # All hyperparameters
├── pyproject.toml           # Python packaging and tool config
├── Makefile                 # Common workflow commands
├── requirements.txt         # Pinned pip dependencies
├── src/
│   ├── __init__.py
│   ├── data.py              # Data loading, tokenization, weak supervision
│   ├── model.py             # BERT + dual classification heads
│   ├── train.py             # Training loop, checkpointing, early stopping
│   ├── inference.py         # Single and batch prediction
│   └── evaluate.py          # Metrics, confusion matrices, latency reporting
├── tests/
│   ├── __init__.py
│   └── test_core.py         # Smoke tests for the full pipeline
├── notebooks/               # Exploration and analysis notebooks
├── V2/                      # Experimental iteration 2 track
│   └── V2_ITERATION_PLAN.md
├── V3/                      # Iteration 3: hybrid SSM–attention backbone (see V3/V3_ITERATION_PLAN.md)
│   ├── config_hybrid.yaml
│   ├── model_hybrid.py
│   └── main.py              # python -m V3.main train|predict|info
├── data/                    # Downloaded datasets (gitignored)
├── models/                  # Saved checkpoints (gitignored)
└── results/                 # Metrics JSON output
```

## Architecture

```text
Amazon Review Text
      ↓
[BERT Tokenizer] → input_ids, attention_mask
      ↓
[BERT Encoder (bert-base-uncased)]
      ↓ [CLS] pooled output (768-dim)
      ↓
  ┌───┴───┐
  ↓       ↓
[Aspect  [Sentiment
 Head]    Head]
  ↓       ↓
5-class  3-class
softmax  softmax
```

Joint loss: `L = α · L_aspect + (1 − α) · L_sentiment`

Multi-task learning provides implicit regularization — the shared backbone must learn representations useful for both tasks, reducing overfitting on small datasets.

## Development

```bash
make install-dev    # Install with dev dependencies and pre-commit hooks
make test           # Run full test suite
make test-quick     # Run tests that skip BERT download (~440 MB)
make lint           # Lint with ruff
make lint-fix       # Auto-fix lint issues
make clean          # Remove generated artifacts
```

## Status

- [x] Project scaffold and config
- [x] Data pipeline with weak supervision
- [x] Model architecture (dual-head BERT)
- [x] Training loop with mixed precision
- [x] Inference pipeline
- [x] Evaluation suite
- [x] CLI entry point
- [x] Smoke tests
- [x] CI pipeline (GitHub Actions)
- [ ] Exploration notebooks
- [ ] V2: multi-aspect span extraction, sarcasm-aware routing, quantum-inspired uncertainty

## Security Note

This repository loads PyTorch checkpoints with `weights_only=False` (required for model state dicts containing custom objects). Only load checkpoints from trusted sources. The HuggingFace dataset loader uses `trust_remote_code=True` for the Amazon Reviews dataset.

## Citation

```bibtex
@misc{kuo2026nlptransformer,
  title   = {Aspect-Based Sentiment Analysis with BERT},
  author  = {Austin Kuo},
  year    = {2026},
  url     = {https://github.com/A-Kuo/NLP-Transformer-Sentiments}
}
```

## Acknowledgments

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — Devlin et al., 2019
- [Amazon Reviews 2023 Dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) — McAuley et al.
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## License

MIT License — see [LICENSE](LICENSE) for details.
