"""
Quantum-Inspired Uncertainty Representation (V2 Iteration 3).

Why density matrices instead of softmax?

  Softmax gives: P(positive)=0.7, P(negative)=0.3
  This hides the STRUCTURE of the uncertainty. Is the model confused
  because "cheap" could mean "affordable" (positive) or "low-quality"
  (negative)? Or because the review is genuinely neutral?

  A density matrix encodes both:
    - Diagonal elements: class probabilities (same info as softmax)
    - Off-diagonal elements: INTERFERENCE between classes

  The word "cheap" in isolation exists in a quantum-like superposition:
      |ψ⟩ = α|positive⟩ + β|negative⟩

  The density matrix ρ = |ψ⟩⟨ψ| captures not just |α|² and |β|²
  (the probabilities), but also αβ* (the interference term), which
  measures HOW MUCH the two readings conflict with each other.

  High |αβ*| = genuine semantic ambiguity (the text supports both readings).
  Low |αβ*|  = the model is confident in one reading.

Mathematical foundation:

  In quantum mechanics, a pure state |ψ⟩ ∈ ℂ^k (unit vector) has
  density matrix ρ = |ψ⟩⟨ψ| (rank-1 positive semidefinite, trace 1).

  For sentiment with k=3 classes:
    |ψ⟩ = [α₀, α₁, α₂]ᵀ ∈ ℂ³,  ||ψ|| = 1
    ρ = |ψ⟩⟨ψ| = 3×3 Hermitian matrix

  Properties:
    ρᵢᵢ = |αᵢ|² = probability of class i  (diagonal)
    ρᵢⱼ = αᵢαⱼ* = interference between i and j  (off-diagonal)
    Tr(ρ) = 1  (probabilities sum to 1)
    ρ ≥ 0  (positive semidefinite)

  Von Neumann entropy:
    S(ρ) = -Tr(ρ log ρ) = -Σᵢ λᵢ log(λᵢ)
    where λᵢ are eigenvalues of ρ.

    For pure states (rank 1): S = 0 (maximum certainty)
    For maximally mixed ρ = I/k: S = log(k) (maximum uncertainty)

    Unlike softmax entropy, von Neumann entropy on the density matrix
    is a proper quantum information measure that accounts for coherence.

  Interference magnitude:
    I(ρ) = Σᵢ≠ⱼ |ρᵢⱼ|² / Σᵢ ρᵢᵢ²
    This ratio captures how much "off-diagonal energy" exists
    relative to the diagonal. High I = high ambiguity.

Implementation note:
  This runs on classical hardware via torch/numpy. The density matrix
  is 3×3 (or k×k) — trivial to compute. The quantum formalism is
  the mathematical contribution, not quantum hardware.

References:
  - Kayal et al. (2025), "Quantum-Inspired ABSA Using NLP", Springer
  - ACM Computing Surveys (2023), "Quantum-Cognitively Inspired SA Models"
  - QI-CNN (IEEE 2021), density matrices in convolutional sentiment models
"""

import logging
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================
# Core density matrix operations
# ============================================================

def construct_density_matrix(psi: torch.Tensor) -> torch.Tensor:
    """
    Construct density matrix from complex state vector.

    ρ = |ψ⟩⟨ψ| = outer product of ψ with its conjugate transpose.

    Args:
        psi: (B, k) complex tensor, unit-normalized state vectors.

    Returns:
        (B, k, k) complex tensor, density matrices.
        Each ρ[b] is Hermitian, PSD, trace 1.
    """
    # |ψ⟩⟨ψ| = ψ ⊗ ψ* (outer product)
    # psi: (B, k) → psi_bra: (B, 1, k).conj()
    # result: (B, k, 1) @ (B, 1, k) = (B, k, k)
    return torch.bmm(
        psi.unsqueeze(-1),           # (B, k, 1)
        psi.unsqueeze(-2).conj(),    # (B, 1, k)
    )


def von_neumann_entropy(rho: torch.Tensor) -> torch.Tensor:
    """
    Compute von Neumann entropy: S(ρ) = -Tr(ρ log ρ).

    Equivalent to: S = -Σᵢ λᵢ log(λᵢ) where λᵢ are eigenvalues.

    For pure states: S = 0 (one eigenvalue = 1, rest = 0).
    For maximally mixed: S = log(k) (all eigenvalues = 1/k).

    We compute via eigendecomposition since ρ is Hermitian (guaranteed
    real non-negative eigenvalues).

    Args:
        rho: (B, k, k) complex density matrices.

    Returns:
        (B,) real tensor, entropy values in [0, log(k)].
    """
    # Eigenvalues of Hermitian matrix (real, non-negative)
    # torch.linalg.eigvalsh handles complex Hermitian → real eigenvalues
    eigenvalues = torch.linalg.eigvalsh(rho)  # (B, k) real

    # Clamp to avoid log(0): replace tiny/negative values with epsilon
    eigenvalues = eigenvalues.clamp(min=1e-12)

    # S = -Σ λᵢ log(λᵢ)
    entropy = -torch.sum(eigenvalues * torch.log(eigenvalues), dim=-1)  # (B,)

    return entropy


def interference_magnitude(rho: torch.Tensor) -> torch.Tensor:
    """
    Compute interference magnitude from off-diagonal elements.

    I(ρ) = Σᵢ≠ⱼ |ρᵢⱼ|² / (Σᵢ ρᵢᵢ² + ε)

    This measures how much "coherence" exists between sentiment classes.
    High interference = the text genuinely supports multiple readings
    (e.g., "cheap" as positive or negative).
    Low interference = the model has a clear winner.

    For a pure state |ψ⟩ = [1, 0, 0]ᵀ: I = 0 (no interference).
    For |ψ⟩ = [1/√2, 1/√2, 0]ᵀ: I = 1 (maximum 2-class interference).

    Args:
        rho: (B, k, k) complex density matrices.

    Returns:
        (B,) real tensor, interference magnitude ≥ 0.
    """
    rho_abs_sq = (rho.abs()) ** 2  # |ρᵢⱼ|²

    # Diagonal sum: Σᵢ |ρᵢᵢ|²
    diag_sum = torch.diagonal(rho_abs_sq, dim1=-2, dim2=-1).sum(dim=-1)  # (B,)

    # Total sum: Σᵢⱼ |ρᵢⱼ|²
    total_sum = rho_abs_sq.sum(dim=(-2, -1))  # (B,)

    # Off-diagonal sum
    offdiag_sum = total_sum - diag_sum  # (B,)

    # Normalize by diagonal energy
    interference = offdiag_sum / (diag_sum + 1e-12)

    return interference.real  # Should already be real, but ensure dtype


def density_matrix_to_probs(rho: torch.Tensor) -> torch.Tensor:
    """
    Extract class probabilities from density matrix diagonal.

    P(class i) = ρᵢᵢ = |αᵢ|²

    This is equivalent to softmax probabilities but derived from the
    quantum state vector. The diagonal of ρ = |ψ⟩⟨ψ| gives the Born
    rule probabilities.

    Args:
        rho: (B, k, k) complex density matrices.

    Returns:
        (B, k) real tensor, probability distribution per sample.
    """
    probs = torch.diagonal(rho, dim1=-2, dim2=-1).real  # (B, k)
    # Renormalize in case of numerical drift
    probs = probs.clamp(min=0)
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
    return probs


# ============================================================
# Neural network projection layer
# ============================================================

class QuantumProjection(nn.Module):
    """
    Project BERT's real-valued [CLS] embedding into a complex-valued
    quantum state vector.

    Architecture:
        h_CLS (768-dim real)
            ↓
        Linear(768 → 2k)  →  [real₀, imag₀, real₁, imag₁, ..., real_{k-1}, imag_{k-1}]
            ↓
        Reshape to complex: αᵢ = realᵢ + j·imagᵢ
            ↓
        L2 normalize: |ψ⟩ = α / ||α||  (unit vector in ℂ^k)
            ↓
        Density matrix: ρ = |ψ⟩⟨ψ|  (k×k Hermitian, trace 1)

    The projection learns to map BERT's 768-dim semantic space into
    a k-dim complex space where the geometry of the state vector
    encodes both class membership (diagonal of ρ) and class
    interference (off-diagonal of ρ).

    Parameters:
        input_dim:  BERT hidden size (768)
        num_classes: Number of sentiment classes (3)
    """

    def __init__(self, input_dim: int = 768, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes

        # Project to 2k dimensions: k real parts + k imaginary parts
        self.projection = nn.Linear(input_dim, 2 * num_classes)

        # Xavier init for stable training
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, h_cls: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Project [CLS] embedding to quantum state and density matrix.

        Args:
            h_cls: (B, 768) real tensor from BERT [CLS].

        Returns:
            Dict with:
              - psi: (B, k) complex state vector (unit normalized)
              - rho: (B, k, k) complex density matrix
              - probs: (B, k) class probabilities from diag(ρ)
              - entropy: (B,) von Neumann entropy
              - interference: (B,) interference magnitude
        """
        k = self.num_classes

        # Project: (B, 768) → (B, 2k)
        projected = self.projection(h_cls)  # (B, 2k)

        # Split into real and imaginary parts
        real_parts = projected[:, :k]      # (B, k)
        imag_parts = projected[:, k:]      # (B, k)

        # Construct complex state vector
        psi = torch.complex(real_parts, imag_parts)  # (B, k) complex

        # L2 normalize to unit vector: |ψ⟩ = α / ||α||
        # This ensures Tr(ρ) = ⟨ψ|ψ⟩ = 1 (valid probability distribution)
        psi_norm = psi / (psi.abs().norm(dim=-1, keepdim=True) + 1e-12)

        # Density matrix: ρ = |ψ⟩⟨ψ|
        rho = construct_density_matrix(psi_norm)

        # Extract observables
        probs = density_matrix_to_probs(rho)
        entropy = von_neumann_entropy(rho)
        interf = interference_magnitude(rho)

        return {
            "psi": psi_norm,          # (B, k) complex unit vector
            "rho": rho,               # (B, k, k) complex density matrix
            "probs": probs,           # (B, k) real probabilities
            "entropy": entropy,       # (B,) real, in [0, log(k)]
            "interference": interf,   # (B,) real, ≥ 0
        }


# ============================================================
# Uncertainty analysis utilities
# ============================================================

def classify_uncertainty(
    entropy: torch.Tensor,
    interference: torch.Tensor,
    num_classes: int = 3,
) -> Dict[str, torch.Tensor]:
    """
    Classify uncertainty into interpretable categories.

    Uses entropy thresholds relative to maximum possible entropy
    (log(k) for k classes):

      - "confident":  entropy < 0.3 * log(k), low interference
      - "uncertain":  entropy 0.3-0.7 * log(k)
      - "ambiguous":  entropy > 0.7 * log(k), OR high interference

    The distinction between "uncertain" and "ambiguous" matters:
      - Uncertain: model doesn't have enough signal (need more context)
      - Ambiguous: text genuinely supports multiple readings (inherent)

    Args:
        entropy: (B,) von Neumann entropy values.
        interference: (B,) interference magnitude values.
        num_classes: k (for computing max entropy).

    Returns:
        Dict with category labels and normalized scores.
    """
    import math
    max_entropy = math.log(num_classes)

    # Normalize entropy to [0, 1]
    normalized_entropy = (entropy / max_entropy).clamp(0, 1)

    # Categories
    # High interference + any entropy = ambiguous (genuine semantic conflict)
    # High entropy + low interference = uncertain (model confused)
    # Low entropy + low interference = confident
    is_high_interference = interference > 0.5
    is_high_entropy = normalized_entropy > 0.7
    is_medium_entropy = normalized_entropy > 0.3

    categories = torch.where(
        is_high_interference,
        torch.tensor(2),  # ambiguous
        torch.where(
            is_high_entropy,
            torch.tensor(1),  # uncertain
            torch.where(
                is_medium_entropy,
                torch.tensor(1),  # uncertain
                torch.tensor(0),  # confident
            )
        )
    )

    category_names = {0: "confident", 1: "uncertain", 2: "ambiguous"}

    return {
        "categories": categories,           # (B,) int: 0/1/2
        "category_names": category_names,
        "normalized_entropy": normalized_entropy,  # (B,) float [0,1]
    }


def format_quantum_state(
    probs: torch.Tensor,
    entropy: float,
    interference: float,
    sentiment_map: Dict,
) -> str:
    """
    Format quantum uncertainty info for human display.

    Example output:
        Quantum state: |ψ⟩ ≈ 0.84|positive⟩ + 0.54|negative⟩
        ρ diagonal:    [0.71, 0.00, 0.29]
        Entropy:       0.612 / 1.099 (56% of max)
        Interference:  0.41 (genuine ambiguity detected)
    """
    import math
    max_ent = math.log(len(sentiment_map))
    ent_pct = (entropy / max_ent) * 100 if max_ent > 0 else 0

    prob_str = ", ".join(
        f"{sentiment_map.get(i, f's{i}')}={p:.2f}"
        for i, p in enumerate(probs.tolist())
    )

    lines = [
        f"  Density matrix probs: [{prob_str}]",
        f"  Von Neumann entropy:  {entropy:.3f} / {max_ent:.3f} ({ent_pct:.0f}% of max)",
        f"  Interference:         {interference:.3f}"
        + (" (ambiguous)" if interference > 0.5 else " (coherent)" if interference < 0.1 else ""),
    ]
    return "\n".join(lines)


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print(f"\n{'='*60}")
    print("QUANTUM UNCERTAINTY MODULE TEST")
    print(f"{'='*60}")

    k = 3  # sentiment classes
    B = 4  # batch

    # Test 1: Pure state (no ambiguity)
    # |ψ⟩ = [1, 0, 0] → all probability on class 0
    psi_pure = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * B)
    proj = QuantumProjection(input_dim=6, num_classes=3)
    # Override projection to identity-like for testing
    with torch.no_grad():
        proj.projection.weight.zero_()
        proj.projection.bias.zero_()
        proj.projection.weight[:3, :3] = torch.eye(3)

    # Manual test with known state
    psi_known = torch.complex(
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0]]),
    )
    psi_known = psi_known / psi_known.abs().norm(dim=-1, keepdim=True)
    rho = construct_density_matrix(psi_known)
    probs = density_matrix_to_probs(rho)
    ent = von_neumann_entropy(rho)
    interf = interference_magnitude(rho)

    print("\nTest 1: Pure state |1,0,0⟩")
    print(f"  probs:        {probs[0].tolist()}")
    print(f"  entropy:      {ent[0].item():.6f} (expected: ~0)")
    print(f"  interference: {interf[0].item():.6f} (expected: ~0)")
    assert ent[0].item() < 0.01, "Pure state should have ~0 entropy"
    assert interf[0].item() < 0.01, "Pure state should have ~0 interference"
    print("  ✅ PASSED")

    # Test 2: Maximum superposition (equal weight on 2 classes)
    # |ψ⟩ = [1/√2, 1/√2, 0] → 50/50 between class 0 and 1
    psi_super = torch.complex(
        torch.tensor([[0.7071, 0.7071, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0]]),
    )
    psi_super = psi_super / psi_super.abs().norm(dim=-1, keepdim=True)
    rho2 = construct_density_matrix(psi_super)
    probs2 = density_matrix_to_probs(rho2)
    ent2 = von_neumann_entropy(rho2)
    interf2 = interference_magnitude(rho2)

    print("\nTest 2: Superposition |1/√2, 1/√2, 0⟩")
    print(f"  probs:        {[round(p, 3) for p in probs2[0].tolist()]}")
    print(f"  entropy:      {ent2[0].item():.6f} (expected: ~0, pure state)")
    print(f"  interference: {interf2[0].item():.6f} (expected: >0)")
    # Note: pure state always has 0 entropy even with superposition!
    # Von Neumann entropy = 0 for all pure states. The INTERFERENCE
    # measure is what distinguishes superposition from concentration.
    assert interf2[0].item() > 0.5, "Superposition should have high interference"
    print("  ✅ PASSED (interference captures ambiguity, entropy stays 0 for pure states)")

    # Test 3: QuantumProjection layer
    print("\nTest 3: QuantumProjection(768→3)")
    proj = QuantumProjection(input_dim=768, num_classes=3)
    fake_cls = torch.randn(B, 768)
    output = proj(fake_cls)

    print(f"  psi shape:         {output['psi'].shape}")          # (4, 3) complex
    print(f"  rho shape:         {output['rho'].shape}")          # (4, 3, 3) complex
    print(f"  probs shape:       {output['probs'].shape}")        # (4, 3) real
    print(f"  entropy shape:     {output['entropy'].shape}")      # (4,)
    print(f"  interference shape:{output['interference'].shape}")  # (4,)
    print(f"  probs sum:         {output['probs'].sum(dim=-1).tolist()}")  # should be ~1.0
    assert output['psi'].shape == (B, 3)
    assert output['rho'].shape == (B, 3, 3)
    assert output['probs'].shape == (B, 3)
    assert all(abs(s - 1.0) < 0.01 for s in output['probs'].sum(dim=-1).tolist())
    print("  ✅ PASSED")

    # Test 4: Uncertainty classification
    print("\nTest 4: classify_uncertainty()")
    ent_test = torch.tensor([0.01, 0.5, 1.0])
    interf_test = torch.tensor([0.0, 0.2, 0.8])
    cats = classify_uncertainty(ent_test, interf_test, num_classes=3)
    print(f"  categories: {cats['categories'].tolist()}")
    # Low entropy, low interference → confident (0)
    # Medium entropy, low interference → uncertain (1)
    # High entropy, high interference → ambiguous (2)
    assert cats['categories'][0].item() == 0  # confident
    assert cats['categories'][2].item() == 2  # ambiguous (high interference)
    print("  ✅ PASSED")

    print(f"\n{'='*60}")
    print("ALL QUANTUM UNCERTAINTY TESTS PASSED")
    print(f"{'='*60}")
