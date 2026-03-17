# BERT Aspect-Based Sentiment Analysis at Scale

**Problem:** Manually reviewing millions of product reviews is impossible. Generic sentiment models miss *what* customers are complaining about — a 3-star review might praise quality but trash shipping.

**Solution:** Fine-tuned BERT with dual classification heads: one for **aspect detection** (what is being discussed?) and one for **sentiment scoring** (positive / neutral / negative per aspect). This gives structured, actionable signal from unstructured text.

**Target metrics (Phase 1 prototype):**
- Aspect identification: >90% accuracy
- Sentiment classification: >88% accuracy  
- Inference latency: <500ms p99
- Throughput: 10k reviews/min (batch)

---

## Quick Start

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download data + train
python main.py download
python main.py train

# 3. Evaluate
python main.py evaluate --checkpoint models/best/

# 4. Predict
python main.py predict --text "Great quality but shipping took 3 weeks"
```

## Project Structure

```
bert-sentiment/
├── config.yaml              # All hyperparameters (single source of truth)
├── requirements.txt
├── main.py                  # CLI entry point
├── src/
│   ├── data.py              # Data loading, tokenization, aspect labeling
│   ├── model.py             # BERT + dual heads (aspect + sentiment)
│   ├── train.py             # Training loop, checkpointing, logging
│   ├── inference.py         # Single-doc and batch prediction
│   └── evaluate.py          # Metrics: accuracy, F1, per-aspect breakdown
├── tests/
│   └── test_core.py         # Smoke tests for pipeline
├── data/                    # Downloaded datasets (gitignored)
├── models/                  # Saved checkpoints (gitignored)
├── results/                 # Metrics JSON, plots
└── monitoring/              # Phase 2: drift detection, dashboards
```

## Architecture

```
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

## ML Mathematics

The dual-head architecture shares a BERT backbone and branches at the [CLS] token:

- **Shared representation:** BERT's self-attention layers learn contextual embeddings where each token attends to every other token via scaled dot-product attention: `Attention(Q,K,V) = softmax(QK^T / √d_k) V`
- **Aspect head:** Linear projection from 768-dim [CLS] → 5 aspect classes. Cross-entropy loss.
- **Sentiment head:** Linear projection from 768-dim [CLS] → 3 sentiment classes. Cross-entropy loss.
- **Joint loss:** `L = α * L_aspect + (1-α) * L_sentiment` where α balances the two tasks. Multi-task learning provides implicit regularization — the shared backbone must learn representations useful for *both* tasks, reducing overfitting.

## Status

- [x] Project scaffold + config
- [ ] Data pipeline
- [ ] Model architecture
- [ ] Training loop
- [ ] Inference pipeline
- [ ] Evaluation suite
- [ ] CLI entry point
- [ ] Tests

---

*Phase 2 roadmap: Docker containerization, Airflow orchestration, Redis caching, Streamlit monitoring dashboard, automated drift detection & retraining.*
