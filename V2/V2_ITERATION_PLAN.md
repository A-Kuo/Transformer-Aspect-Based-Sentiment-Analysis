# BERT Sentiment V2 — Iteration Plan

## Motivation

V1 was a clean BERT fine-tuning exercise. V2 is the harder question:
*What happens when the model is confidently wrong, and how do we build systems that catch it?*

---

## Scope

V2 does not attempt to solve all challenges from the research literature.
It picks **three** that form a coherent story about uncertainty-aware NLP.

**In scope:**

1. **Sarcasm-Aware Routing** — Detect when the model will fail, route to a fallback
2. **Multi-Aspect Span Extraction** — Fix V1's single-label architectural flaw
3. **Quantum-Inspired Uncertainty Representation** — Replace naive softmax confidence with density matrix encoding

**Out of scope (acknowledged, not attempted):**

- Multilingual / cross-cultural bias — future work
- Domain adaptation — V2 stays on Amazon reviews
- Full Airflow/Kafka production infra — described in docs, not implemented

---

## Iteration 1: Multi-Aspect Span Extraction

**What changes:** V1 assigns one aspect per review. Real reviews mention multiple aspects.
"Amazing camera but terrible battery life" → two aspect-sentiment pairs, not one.

**Architecture delta:**

```
V1:  [CLS] → Linear(768→5)  → single aspect label
V2:  Each token → Linear(768→7) → BIO tags per aspect
     B-quality, I-quality, B-shipping, I-shipping, ... O
```

**Files touched:**
- `src/model.py` — Add token-level classification head alongside [CLS] heads
- `src/data.py` — Extend weak supervision to produce token-level BIO labels from keyword spans
- `src/evaluate.py` — Add span-level F1 (exact match + partial overlap)
- `config.yaml` — Add `extraction_mode: span` flag, BIO label count

**Math note:**
Token classification uses the same cross-entropy as V1, but applied per-token:
`L_span = -(1/T) Σ_t Σ_c y_tc · log(softmax(z_t)_c)`
where T = sequence length, c = BIO class. The joint loss becomes:
`L = α·L_span + β·L_sentiment + γ·L_overall`

**Output format changes:**
```json
{
  "aspects": [
    {"span": "Amazing camera", "aspect": "quality", "sentiment": "positive", "confidence": 0.92},
    {"span": "terrible battery life", "aspect": "quality", "sentiment": "negative", "confidence": 0.88}
  ]
}
```

**Estimated scope:** ~200 lines new/modified across model + data + evaluate

---

## Iteration 2: Sarcasm-Aware Routing

**What changes:** Add a lightweight sarcasm detection gate before sentiment classification.
When sarcasm probability is high, the system either inverts confidence, flags for human review,
or routes to an LLM-based reanalysis path.

**Motivation:**
Sarcasm is the primary failure mode for sentiment classifiers. A routing gate before
classification allows the system to detect when it will fail and handle those cases
differently — the same uncertainty-aware pattern used in hallucination detection systems
where high entropy triggers fallback behavior.

**Architecture delta:**

```
Review text
    ↓
[Sarcasm Detector] → sarcasm_score (0-1)
    ↓
    ├─ score < 0.3 → [BERT Sentiment] → normal prediction
    ├─ 0.3 ≤ score < 0.7 → [BERT Sentiment] → prediction + uncertainty flag
    └─ score ≥ 0.7 → [Fallback: rule-based / LLM] → flagged prediction
```

**Implementation approach:**
- Small fine-tuned classifier (DistilBERT or even a frozen BERT + linear probe)
  trained on existing sarcasm datasets (iSarcasm, SemEval-2022 Task 6)
- NOT a second full BERT — the sarcasm head can share the backbone via multi-task,
  or use a distilled model for latency budget
- The routing logic lives in `src/inference.py` as a pre-classification gate

**Files touched:**
- `src/sarcasm.py` — New file: sarcasm detector + routing logic
- `src/inference.py` — Predictor gains a sarcasm gate before prediction
- `src/evaluate.py` — Track sarcasm detection recall + routing accuracy
- `src/data.py` — Optional: loader for sarcasm training data

**Key metric:**
"Of the reviews our model misclassified in V1, what % had high sarcasm scores?"
If the answer is significant, the routing system has measurable value.

**Estimated scope:** ~300 lines (new sarcasm.py + inference modifications)

---

## Iteration 3: Quantum-Inspired Density Matrix Uncertainty

**What changes:** Replace softmax confidence with density matrix representation of sentiment state.

**Why density matrices over softmax:**
Every other sentiment project reports `P(positive) = 0.73`. That number is uncalibrated
and hides the structure of the uncertainty. A density matrix encodes:
- Diagonal elements: probability of each sentiment class (same as softmax)
- Off-diagonal elements: **interference** between classes (quantum coherence)

The word "cheap" in isolation exists in a superposition:
`|ψ⟩ = α|positive⟩ + β|negative⟩`

The density matrix `ρ = |ψ⟩⟨ψ|` captures not just the probabilities |α|² and |β|²,
but the interference term αβ* which measures HOW MUCH the two readings
conflict with each other. High off-diagonal magnitude = high ambiguity.

This is NOT quantum computing. It's quantum probability theory as a mathematical
framework running on classical hardware via numpy. The 2024 ACM Computing Surveys
paper and the 2025 QI-CNN paper both take this approach.

**Architecture delta:**

```
BERT [CLS] embedding (768-dim)
    ↓
[Projection] → sentiment_state (k-dim complex vector, k = num_sentiments)
    ↓
[Density Matrix] → ρ = |ψ⟩⟨ψ| (k×k Hermitian matrix)
    ↓
├─ diag(ρ) → class probabilities (replaces softmax)
├─ off_diag(ρ) → interference / ambiguity score
└─ von Neumann entropy S(ρ) = -Tr(ρ log ρ) → calibrated uncertainty
```

**Implementation:**
- `src/quantum_uncertainty.py` — New file: density matrix construction,
  von Neumann entropy, interference magnitude
- Projection layer: `Linear(768 → 2k)` outputs real + imaginary parts,
  reshaped into complex vector, normalized to unit length
- Loss: still cross-entropy on `diag(ρ)`, but add entropy regularization
  term that penalizes overconfident predictions on ambiguous inputs

**What this gives us that softmax doesn't:**
- `interference_score` per prediction — quantifies genuine semantic ambiguity
  (not just model uncertainty). "Not bad" has high interference between
  positive and negative; "Excellent product" has low interference.
- `von_neumann_entropy` — single scalar summarizing total uncertainty,
  mathematically grounded (unlike softmax temperature hacks)
- Direct connection to the quantum NLP literature

**Files touched:**
- `src/quantum_uncertainty.py` — New (~150 lines)
- `src/model.py` — Add density matrix head option alongside softmax
- `src/inference.py` — Report interference + entropy alongside confidence
- `src/evaluate.py` — Correlation: high entropy ↔ misclassification rate

**Estimated scope:** ~250 lines (new file + modifications)

---

## Implementation Order & Dependencies

```
Iteration 1 (Multi-Aspect Spans)
    │
    │  V1 model still works, this extends it
    │  Can be tested independently
    │
    ▼
Iteration 2 (Sarcasm Routing)
    │
    │  Depends on having predictions to route
    │  Benefits from span extraction (sarcasm is often aspect-specific)
    │
    ▼
Iteration 3 (Quantum-Inspired Uncertainty)
    │
    │  Depends on having a working pipeline to add uncertainty layer to
    │  Benefits from sarcasm routing (can validate: high entropy ↔ sarcasm?)
    │
    ▼
Documentation update to reflect V2 architecture
```

Each iteration is independently testable and demoable.

---

## Total Estimated Code Delta

| Iteration | New Files | Modified Files | ~Lines |
|-----------|-----------|----------------|--------|
| 1. Multi-Aspect | — | model, data, evaluate, config | ~200 |
| 2. Sarcasm | sarcasm.py | inference, evaluate, data | ~300 |
| 3. Quantum | quantum_uncertainty.py | model, inference, evaluate | ~250 |
| **Total** | **2 new files** | **5 modified** | **~750** |

V1 was ~3,100 lines. V2 adds ~750, bringing total to ~3,850.
The ratio is right: V2 is a targeted extension, not a rewrite.
