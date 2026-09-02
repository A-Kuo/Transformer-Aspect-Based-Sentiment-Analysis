# Transformer Aspect-Based Sentiment Analysis

**BERT-based aspect and sentiment modeling for product reviews**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/transformers)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

> *Standard sentiment asks “positive or negative?” This project asks “*which theme* (quality, shipping, …) and *what polarity*?”
>
> — uses a shared transformer encoder and task-specific heads.*

---

## The problem

| Input | Plain sentiment | This project (ABSA-style) |
|-------|-----------------|---------------------------|
| “Great food but terrible service” | Often confused | Aspect + sentiment heads trained on weak labels (Amazon reviews) |
| “Fast shipping, broken on arrival” | Negative overall | Separates shipping vs. quality signals where data supports it |

**Scope of this repository:** fine-tuning **BERT** (`bert-base-uncased`) on **McAuley-Lab/Amazon-Reviews-2023** with keyword weak supervision — not a drop-in PyPI package named `transformer-absa`. Install **from source** (below).

There is **no** `pip install transformer-absa` or `from absa import …` in this codebase; use **`src.inference.Predictor`** and the CLI in `main.py`.

---

## Architecture (this repo)

This implementation uses a **multi-task transformer** setup:

- **Shared encoder:** BERT producing token representations; **`[CLS]`** used for sequence-level heads.
- **Aspect head:** `Linear(768 → K)` for a small set of aspect categories (config-driven).
- **Sentiment head:** `Linear(768 → 3)` for negative / neutral / positive.
- **Joint loss:** `L = α · L_aspect + (1 − α) · L_sentiment`.

Experimental tracks:

- **`V2/`** — span BIO tagging, sarcasm routing, quantum-inspired uncertainty (see `V2/V2_ITERATION_PLAN.md`).
- **`V3/`** — hybrid attention–SSM middle stack (`python -m V3.main`, see `V3/V3_ITERATION_PLAN.md`).

Conceptual ABSA diagrams in older docs (biaffine triples, restaurant “AspectTriple” APIs) are **not** implemented as a separate `absa` Python package here.

---

## Success criteria & target metrics

Training is a weak-supervision setup — aspects are keyword-matched, sentiment comes from star ratings — on a modest Amazon-reviews subset (5,000 train / 1,000 val / 500 test by default), fine-tuning `bert-base-uncased` for 3 epochs. `config.yaml`'s `monitoring` section defines two-tier engineering thresholds for that specific setup:

- **floor** — the minimum bar; below this, something is likely broken (a labeling bug, undertraining, class weights not applying), not just "needs more tuning."
- **target** — the aspirational bar for a healthy first-pass run given this setup's known constraints (label noise, small dataset, few epochs).

| Metric | Floor | Target |
|---|---|---|
| Aspect accuracy | 0.55 | 0.70 |
| Aspect macro F1 | 0.45 | 0.60 |
| Sentiment accuracy | 0.70 | 0.82 |
| Sentiment macro F1 | 0.60 | 0.75 |
| Sentiment recall — negative / neutral / positive | 0.60 / 0.45 / 0.70 | — |

**These are project-specific engineering targets, not industry ABSA benchmarks.** Published numbers for domain-adapted models (e.g. FinBERT on financial text, ClinicalBERT on clinical notes) use much larger domain-specific corpora and often cleaner labels, so they aren't a fair comparison for a weak-supervision prototype on general product reviews — don't cite them as this project's expected performance.

Amazon reviews skew heavily positive (most ratings are 4–5 stars), which without correction collapses training into mostly predicting "positive" regardless of input — this is why the per-class sentiment recall floors exist separately from overall accuracy, and why `training.class_weighted_loss` (`src/data.py::compute_class_weights`, sklearn's inverse-frequency "balanced" heuristic) is on by default.

`python main.py evaluate` and `notebooks/kaggle_train_and_push.ipynb` both print a PASS/FAIL report against these thresholds (`src.evaluate.check_targets` / `print_targets_report`) right after evaluation. The Streamlit dashboard's **Model Info** page (`make app`, or `streamlit run app.py`) renders the same comparison live against whichever model is currently loaded — from the Neon-backed model registry once a Kaggle run has been pushed, or from a local `python main.py evaluate` run otherwise. Recalibrate the thresholds once real results land; they're a first-pass hypothesis, not a guarantee.

---

## Installation

```bash
git clone https://github.com/A-Kuo/Transformer-Aspect-Based-Sentiment-Analysis.git
cd Transformer-Aspect-Based-Sentiment-Analysis

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev]"

# Optional: Mamba2 middle blocks for V3 (Linux/CUDA typical; see V3/V3_ITERATION_PLAN.md)
# pip install -e ".[v3]"

# Optional: Streamlit dashboard (app.py)
# pip install -e ".[app]"

# Optional: Neon Postgres model registry + Hugging Face Hub weight delivery
# (scripts/load_absa_results.py needs this; app.py degrades gracefully without it)
# pip install -e ".[db]"
```

---

## Quick start (CLI)

```bash
python main.py train
python main.py evaluate
python main.py predict "Great quality but shipping took 3 weeks"
python main.py info
streamlit run app.py     # or: make app
```

Or with `make`: `make train`, `make evaluate`, `make test`.

---

## Python API (local checkpoint)

After `python main.py train`, load weights from disk (default: `models/checkpoint_best.pt`; paths in `config.yaml`).

```python
from src.inference import Predictor

predictor = Predictor.from_checkpoint(
    "models/checkpoint_best.pt",
    config_path="config.yaml",
)
result = predictor.predict_one(
    "The food was excellent but the waiter was rude and slow."
)
# result["aspect"], result["sentiment"], confidences, etc.
```

Untrained heads (baseline): `Predictor.from_pretrained(config_path="config.yaml")`.

---

## Project layout

```text
.
├── main.py                          # CLI (train/evaluate/predict/info)
├── app.py                           # Streamlit triage dashboard (make app)
├── config.yaml                      # Hyperparameters + success-criteria targets
├── src/                             # data, model, train, inference, evaluate, db
├── tests/                           # Smoke tests
├── V2/                              # Experimental iteration 2
├── V3/                              # Hybrid SSM–attention iteration 3
├── notebooks/                       # Inference/data/feature walkthroughs +
│                                     #   kaggle_train_and_push.ipynb
├── db/schema.sql                    # Neon Postgres model-registry schema
├── scripts/load_absa_results.py     # Push a Kaggle run's results into Neon
├── models/                          # Checkpoints (gitignored)
└── results/                         # Metrics (gitignored)
```

A trained model reaches the dashboard one of two ways: push results from `notebooks/kaggle_train_and_push.ipynb` via `scripts/load_absa_results.py` (weights on Hugging Face Hub, provenance/metrics in Neon), or train locally with `python main.py train` (reads `models/checkpoint_best.pt` directly). Either way, `app.py` falls back gracefully — Neon-backed run → local checkpoint → untrained baseline — never crashing on a missing/unreachable data source.

---

## ABSA landscape (brief)

1. **Feature-based & lexicon** (pre-neural): rules and polarity lexicons.
2. **LSTM / CNN pipelines** (~2015–2019): staged target and opinion modeling.
3. **End-to-end transformers** (2020–present): this repo uses a **single encoder + heads** on `[CLS]`; token-level extensions live under `V2/` / `V3/`.

**Papers this work relates to:**

- **Li et al. (2019)** — Xin Li, Lidong Bing, Piji Li, and Wai Lam. “A Unified Model for Opinion Target Extraction and Target Sentiment Prediction.” *AAAI 2019*. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/4383).
- **Chen et al. (2020)** — Shaowei Chen, Yu Wang, Jie Liu, and Yubo Wang. “Inducing Target-Specific Latent Structures for Aspect Sentiment Classification.” *EMNLP 2020*.
- **Yan et al. (2021)** — Hang Yan et al. “A Unified Generative Framework for Aspect-Based Sentiment Analysis.” *ACL 2021*.

---

## Related repositories

- [Fine-Tuned-SEC-Filing-Extraction-Pipeline](https://github.com/A-Kuo/Fine-Tuned-SEC-Filing-Extraction-Pipeline) — optional downstream integration (not required for this repo).
- [NLPTransformerAnalysis-archive](https://github.com/A-Kuo/NLPTransformerAnalysis-archive) — archived history.

---

## Security note

Checkpoints are loaded with `weights_only=False` where PyTorch requires it for full state dicts. Only load checkpoints from trusted sources. The Hugging Face dataset loader may use `trust_remote_code=True` for the Amazon Reviews dataset.

---

## Citation

```bibtex
@misc{kuo2026transformer_absa,
  title   = {Transformer Aspect-Based Sentiment Analysis with BERT},
  author  = {Austin Kuo},
  year    = {2026},
  url     = {https://github.com/A-Kuo/Transformer-Aspect-Based-Sentiment-Analysis}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

> *Model sentiment understanding is one step closer to pretending to bypass the Turing Test*
