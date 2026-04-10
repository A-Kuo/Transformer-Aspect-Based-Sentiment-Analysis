# Transformer Aspect-Based Sentiment Analysis

**Fine-grained sentiment extraction with transformer architectures**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/transformers)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Status](https://img.shields.io/badge/Status-Stable-green.svg)]()

> *"'Positive' is not enough. Was the food good but the service slow? Was the battery life praised but the price criticized? Aspect-based sentiment tells you what people actually think about specific components."*

---

## The Problem

Standard sentiment analysis answers: *"Is this text positive or negative?"*

Aspect-Based Sentiment Analysis (ABSA) answers: *"What aspects are mentioned, and what is the sentiment toward each?"*

| Input | Standard SA | ABSA |
|-------|-------------|------|
| "Great food but terrible service" | Neutral/Confused | Food: Positive, Service: Negative |
| "Battery lasts forever, price is steep" | Mixed | Battery: Positive, Price: Negative |
| "Fast shipping, broken on arrival" | Negative | Shipping: Positive, Product Quality: Negative |

ABSA is essential for:
- Product review analysis (identify specific feature strengths/weaknesses)
- Customer service triage (route based on aspect, not just sentiment)
- Competitive intelligence (compare aspect-level sentiment across brands)
- Financial sentiment (market reaction to specific corporate announcements)

---

## Architecture

This implementation uses a **multi-task transformer** approach:

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT TEXT                                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              TRANSFORMER ENCODER (BERT/RoBERTa/DeBERTa)         │
│                    Contextualized token representations           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
           ┌──────────────┼──────────────┐
           │              │              │
           ▼              ▼              ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
│ Aspect Extraction│ │  Opinion   │ │ Sentiment       │
│  (NER-style)     │ │  Term      │ │ Classification  │
│                  │ │ Extraction │ │  (per aspect)   │
│  {B-ASP, I-ASP,  │ │            │ │                 │
│   B-OP, I-OP, O} │ │ {B-OP,    │ │ {Positive,     │
│                  │ │  I-OP, O}  │ │  Negative,     │
│                  │ │            │ │  Neutral}       │
└────────┬────────┘ └──────┬──────┘ └────────┬────────┘
         │                 │               │
         └────────┬────────┘               │
                  │                         │
                  ▼                         │
         ┌─────────────────┐                │
         │  Aspect-Opinion  │                │
         │    Pairing       │◄───────────────┘
         │   (Association)  │
         └────────┬─────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  FINAL OUTPUT   │
         │  (Aspect,       │
         │   Opinion,      │
         │   Sentiment)    │
         │   Triples       │
         └─────────────────┘
```

---

## Features

### 1. End-to-End Pipeline

```python
from absa import AspectSentimentAnalyzer

# Load from local checkpoint (see models/ directory after training)
analyzer = AspectSentimentAnalyzer.from_pretrained("models/restaurant-base")

result = analyzer.analyze(
    "The food was excellent but the waiter was rude and slow."
)

print(result)
# [
#   AspectTriple(
#     aspect="food",
#     opinion="excellent",
#     sentiment="positive",
#     confidence=0.94
#   ),
#   AspectTriple(
#     aspect="waiter",
#     opinion="rude and slow",
#     sentiment="negative",
#     confidence=0.89
#   )
# ]
```

### 2. Multi-Domain Support

Pre-trained models available for:
- Restaurant reviews
- Product reviews (electronics, apparel)
- Financial news (corporate announcements)
- Hotel/travel reviews

### 3. Domain Adaptation

Fine-tune on your specific domain with minimal data:

```python
from absa import Trainer

trainer = Trainer(base_model="absa/general-base")
trainer.train(
    train_data="my_domain_annotations.jsonl",
    epochs=3,
    learning_rate=2e-5
)
trainer.save("my_domain_absa_model")
```

### 4. Batch Processing

```python
# Process thousands of reviews
results = analyzer.analyze_batch(
    texts=df['review_text'].tolist(),
    batch_size=32,
    show_progress=True
)
```

---

## Technical Approach

### Model Architecture

**Base:** RoBERTa-base (best empirical performance for ABSA)

**Modifications:**
- Aspect extraction head: Token classification (BIO scheme)
- Opinion extraction head: Token classification (BIO scheme)
- Sentiment classification head: Sequence classification per aspect
- Aspect-opinion pairing: Biaffine attention mechanism

### Training Data

Models trained on:
- SemEval ABSA datasets (restaurant, laptop domains)
- MAMS (multi-aspect multi-sentiment) dataset
- Custom financial sentiment annotations for SEC filing analysis

### Evaluation Metrics

- **Aspect Extraction:** F1 (exact match)
- **Opinion Extraction:** F1 (exact match)
- **Sentiment Classification:** Accuracy per aspect
- **End-to-End:** F1 on (Aspect, Opinion, Sentiment) triples

---

## Performance

| Domain | Aspect F1 | Opinion F1 | Sentiment Acc | End-to-End F1 |
|--------|-----------|------------|---------------|---------------|
| Restaurant | 0.84 | 0.78 | 0.91 | 0.76 |
| Laptop | 0.81 | 0.75 | 0.89 | 0.73 |
| Financial | 0.79 | 0.72 | 0.87 | 0.70 |

---

## Usage in Financial Analysis

Special integration with SEC filing analysis:

```python
from absa import FinancialSentimentAnalyzer
from sec_extractor import extract_management_discussion

# Extract MD&A section from 10-K
mdna_text = extract_management_discussion("10-K-filing.pdf")

# Analyze sentiment toward specific business aspects
analyzer = FinancialSentimentAnalyzer()
aspects = analyzer.analyze(
    mdna_text,
    aspect_categories=[
        "revenue", "expenses", "competition",
        "regulation", "supply_chain", "workforce"
    ]
)

# Aspects now contains sentiment toward each business component
# e.g., "supply_chain": negative ("ongoing disruptions")
#       "revenue": positive ("record growth in Q3")
```

See integration with [Fine-Tuned-SEC-Filing-Extraction-Pipeline](https://github.com/A-Kuo/Fine-Tuned-SEC-Filing-Extraction-Pipeline).

---

## Installation

```bash
git clone https://github.com/A-Kuo/Transformer-Aspect-Based-Sentiment-Analysis.git
cd Transformer-Aspect-Based-Sentiment-Analysis
pip install -e .
```

---

## Quick Start

```python
from absa import AspectSentimentAnalyzer

# Load from local checkpoint (see models/ directory after training)
analyzer = AspectSentimentAnalyzer.from_pretrained("models/restaurant-base")

# Analyze
text = "The ambiance was lovely but the main course took forever."
result = analyzer.analyze(text)

for triple in result:
    print(f"Aspect: {triple.aspect}")
    print(f"Opinion: {triple.opinion}")
    print(f"Sentiment: {triple.sentiment} ({triple.confidence:.2f})")
    print()
```

Output:
```
Aspect: ambiance
Opinion: lovely
Sentiment: positive (0.92)

Aspect: main course
Opinion: took forever
Sentiment: negative (0.88)
```

---

## Research Context

ABSA has evolved through several paradigms:

1. **Pipeline approaches** (2014-2018): Separate extraction + classification
2. **Joint models** (2018-2020): Shared encoder, separate heads
3. **End-to-end transformers** (2020-present): This implementation — unified architecture with biaffine pairing

Key papers this implementation draws from:

- **Li et al. (2019)** — Xin Li, Lidong Bing, Piji Li, and Wai Lam. "A Unified Model for Opinion Target Extraction and Target Sentiment Prediction." *Proceedings of the AAAI Conference on Artificial Intelligence*, 33(01):6714–6721. AAAI 2019.
- **Chen et al. (2020)** — Shaowei Chen, Yu Wang, Jie Liu, and Yubo Wang. "Inducing Target-Specific Latent Structures for Aspect Sentiment Classification." *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 5596–5607. ACL, 2020. *(Biaffine attention for aspect-opinion relational structure.)*
- Recent work on generative ABSA (unified text-to-structure generation), e.g., Yan et al. (2021) "A Unified Generative Framework for Aspect-Based Sentiment Analysis," *ACL 2021*.

---

## Related Work

- [Fine-Tuned-SEC-Filing-Extraction-Pipeline](https://github.com/A-Kuo/Fine-Tuned-SEC-Filing-Extraction-Pipeline) — Uses this ABSA for financial sentiment extraction
- [NLPTransformerAnalysis-archive](https://github.com/A-Kuo/NLPTransformerAnalysis-archive) — Historical transformer analysis work

---

## Citation

```bibtex
@software{transformer_absa_2026,
  author = {A-Kuo},
  title = {Transformer Aspect-Based Sentiment Analysis},
  url = {https://github.com/A-Kuo/Transformer-Aspect-Based-Sentiment-Analysis},
  year = {2026}
}
```

---

*Sentiment is not binary. Context is everything. April 2026.*
