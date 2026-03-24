# BERT / Hybrid SSM V3 — Iteration Plan

## Positioning

V3 builds **after** V2 is merged to `main`. It keeps V2 capabilities:

- Multi-aspect **span extraction** (BIO tagging on token-level hidden states)
- **Sarcasm-aware routing** (rule-based detector + polarity / flag / invert paths)
- **Quantum-inspired uncertainty** (density matrix projection on `[CLS]`)

and replaces the monolithic BERT encoder with a **hybrid attention–SSM stack** (attention sandwich).

## Motivation

- **Middle layers as SSM:** Selective state-space blocks scale **O(L)** in sequence length; self-attention is **O(L²)**. Gains matter most at **long contexts** (e.g. 512+ tokens); at 256 tokens the win is modest but the stack is still a valid research baseline.
- **Attention sandwich:** Early attention captures **local syntax** (helpful for span boundaries); late attention **re-aggregates** for classification; SSM middle propagates context along the sequence.
- **No change to routing semantics:** Sarcasm is still a **pre-model gate**; the hybrid backbone only changes **how** hidden states are computed after tokenization.

## Architecture

```text
input_ids, attention_mask
        ↓
[BERT embeddings + positional]
        ↓
Layers 0–3:  BertLayer (self-attention + FFN)  [from pretrained BERT]
        ↓
Layers 4–9:  Bidirectional SSM blocks
             (Mamba2 forward + flipped Mamba2 when installed;
              else depthwise-conv + FFN fallback for CPU/CI)
        ↓
Layers 10–11: BertLayer
        ↓
last_hidden_state (B, T, 768)
        ↓
┌───────────────┬────────────────┬─────────────────┐
│ [CLS] heads   │ token span head │ quantum on [CLS] │
│ aspect/sent.  │ BIO (11 labels) │ (V2 parity)      │
└───────────────┴────────────────┴─────────────────┘
```

## Dependencies

- **Core:** `torch`, `transformers` (same as V1/V2).
- **Optional (GPU / Linux typical):** `mamba-ssm`, `causal-conv1d` — install with `pip install -e ".[v3]"`.
- If Mamba is missing, training and tests still run using the **CPU-friendly SSM fallback** (documented; not byte-identical to Mamba).

## Files (this branch)

| Path | Role |
|------|------|
| [V3/model_hybrid.py](model_hybrid.py) | `HybridBertModel`, `BidirectionalSSMBlock`, `AspectSentimentHybridModel`, `build_model` |
| [V3/config_hybrid.yaml](config_hybrid.yaml) | Model, SSM, data, training, paths, sarcasm, loss weights |
| [V3/span_extraction.py](span_extraction.py) | BIO label IDs + `spans_from_bio_predictions` (decode for eval / demos) |
| [V3/sarcasm.py](sarcasm.py) | Same routing API as V2 (rule-based detector) |
| [V3/quantum_uncertainty.py](quantum_uncertainty.py) | Density matrix projection (ported from V2) |
| [V3/train_hybrid.py](train_hybrid.py) | `HybridTrainer` — passes `span_labels` when present |
| [V3/inference.py](inference.py) | `Predictor` for hybrid checkpoint + sarcasm routing |
| [V3/main.py](main.py) | CLI: `train`, `predict`, `info` |
| [V3/test_hybrid.py](test_hybrid.py) | Smoke tests (fallback SSM if Mamba unavailable) |
| [V3/BENCHMARK.md](BENCHMARK.md) | How to compare V2 vs V3 (speed, length, accuracy gates) |

## Implementation order (completed in repo)

1. Config + optional deps (`pyproject.toml` `[project.optional-dependencies] v3`).
2. Hybrid backbone + full ABSA model in `model_hybrid.py`.
3. Span utilities + ported sarcasm + quantum modules.
4. `HybridTrainer` + `main.py` CLI.
5. Tests + benchmark checklist in `BENCHMARK.md`.

## Merge criteria (before V3 → main)

1. `pytest V3/test_hybrid.py` passes (CPU with fallback or GPU with Mamba).
2. Sarcasm routing behavior matches V2 (same thresholds / paths).
3. Optional: run `BENCHMARK.md` checklist on Colab — record wall-clock and metrics vs V2 baseline.

## Branch

Development branch: **`iteration-3`**. Create from `main` after V2 is integrated, or use `iteration-3` as a long-lived experimental branch until merge criteria are met.
