"""
Centered Kernel Alignment (CKA) for layer similarity measurement.

CKA measures representational similarity between neural network layers,
invariant to orthogonal transformations and isotropic scaling. Used in
Stage 1 to monitor information loss during attention -> SSM conversion.

References:
    Kornblith et al., "Similarity of Neural Network Representations
    Revisited" (ICML 2019)
    Nguyen et al., "Do Wide Neural Networks Really Need to be Wide?"
    (AAAI 2021) — mini-batch CKA

Wayy Research, 2024-2026.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def _center_gram(K: torch.Tensor) -> torch.Tensor:
    """Center a Gram matrix: H @ K @ H where H = I - 1/n."""
    n = K.shape[0]
    unit = torch.ones(n, n, device=K.device, dtype=K.dtype) / n
    return K - unit @ K - K @ unit + unit @ K @ unit


def _hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Hilbert-Schmidt Independence Criterion (biased estimator)."""
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    # HSIC = (1/n^2) * tr(Kc @ Lc)
    n = K.shape[0]
    return torch.trace(Kc @ Lc) / (n * n)


# ---------------------------------------------------------------------------
# Linear CKA
# ---------------------------------------------------------------------------

def linear_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Linear CKA between activation matrices X and Y.

    CKA(K, L) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    where K = X X^T, L = Y Y^T are linear kernels.

    Args:
        X: (n_samples, d_x) activation matrix from layer X.
        Y: (n_samples, d_y) activation matrix from layer Y.
        eps: numerical stability constant.

    Returns:
        Scalar CKA similarity in [0, 1].
    """
    assert X.shape[0] == Y.shape[0], (
        f"Sample count mismatch: {X.shape[0]} vs {Y.shape[0]}"
    )

    # Center columns
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Cross-covariance
    YtX = Y.T @ X  # (d_y, d_x)
    numerator = torch.norm(YtX, p="fro") ** 2

    # Self-covariance norms
    XtX_norm = torch.norm(X.T @ X, p="fro")
    YtY_norm = torch.norm(Y.T @ Y, p="fro")

    denominator = XtX_norm * YtY_norm + eps
    return numerator / denominator


# ---------------------------------------------------------------------------
# RBF CKA
# ---------------------------------------------------------------------------

def _rbf_kernel(
    X: torch.Tensor,
    sigma: Optional[float] = None,
) -> torch.Tensor:
    """Compute RBF (Gaussian) kernel matrix.

    If sigma is None, uses the median heuristic.
    """
    # Squared pairwise distances
    sq_dists = torch.cdist(X, X, p=2.0) ** 2

    if sigma is None:
        # Median heuristic
        median_dist = torch.median(sq_dists[sq_dists > 0])
        sigma = torch.sqrt(median_dist / 2.0).item()
        if sigma < 1e-10:
            sigma = 1.0

    return torch.exp(-sq_dists / (2.0 * sigma * sigma))


def rbf_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    RBF kernel CKA between activation matrices.

    More expressive than linear CKA — captures nonlinear relationships.
    Slower due to O(n^2) kernel computation.

    Args:
        X: (n_samples, d_x) activations.
        Y: (n_samples, d_y) activations.
        sigma_x: RBF bandwidth for X (None = median heuristic).
        sigma_y: RBF bandwidth for Y (None = median heuristic).
        eps: numerical stability.

    Returns:
        Scalar CKA similarity in [0, 1].
    """
    assert X.shape[0] == Y.shape[0]

    K = _rbf_kernel(X, sigma_x)
    L = _rbf_kernel(Y, sigma_y)

    hsic_kl = _hsic(K, L)
    hsic_kk = _hsic(K, K)
    hsic_ll = _hsic(L, L)

    denominator = torch.sqrt(hsic_kk * hsic_ll) + eps
    return hsic_kl / denominator


# ---------------------------------------------------------------------------
# Mini-batch CKA (memory efficient)
# ---------------------------------------------------------------------------

@dataclass
class MinibatchCKAAccumulator:
    """
    Accumulates cross-covariance statistics over mini-batches for
    memory-efficient linear CKA computation.

    Instead of materializing full (n, d) activation matrices, we
    accumulate X^T X, Y^T Y, and Y^T X across batches, then compute
    CKA from the accumulated statistics.

    Usage:
        acc = MinibatchCKAAccumulator(d_x=768, d_y=1024)
        for X_batch, Y_batch in loader:
            acc.update(X_batch, Y_batch)
        score = acc.compute()
    """

    d_x: int
    d_y: int
    device: str = "cpu"
    _XtX: Optional[torch.Tensor] = field(default=None, repr=False)
    _YtY: Optional[torch.Tensor] = field(default=None, repr=False)
    _YtX: Optional[torch.Tensor] = field(default=None, repr=False)
    _sum_x: Optional[torch.Tensor] = field(default=None, repr=False)
    _sum_y: Optional[torch.Tensor] = field(default=None, repr=False)
    _n: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear accumulated statistics."""
        dev = self.device
        self._XtX = torch.zeros(self.d_x, self.d_x, device=dev)
        self._YtY = torch.zeros(self.d_y, self.d_y, device=dev)
        self._YtX = torch.zeros(self.d_y, self.d_x, device=dev)
        self._sum_x = torch.zeros(self.d_x, device=dev)
        self._sum_y = torch.zeros(self.d_y, device=dev)
        self._n = 0

    @torch.no_grad()
    def update(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Accumulate a mini-batch of activations.

        Args:
            X: (batch, d_x) activations from one model.
            Y: (batch, d_y) activations from the other model.
        """
        assert X.shape[0] == Y.shape[0]
        X = X.to(self.device, dtype=torch.float32)
        Y = Y.to(self.device, dtype=torch.float32)

        self._XtX += X.T @ X
        self._YtY += Y.T @ Y
        self._YtX += Y.T @ X
        self._sum_x += X.sum(dim=0)
        self._sum_y += Y.sum(dim=0)
        self._n += X.shape[0]

    def compute(self, eps: float = 1e-10) -> float:
        """
        Compute linear CKA from accumulated statistics.

        We need centered versions:
            X_c = X - mean(X)
            X_c^T X_c = X^T X - n * mean_x * mean_x^T

        Returns:
            CKA score as a Python float.
        """
        if self._n == 0:
            raise ValueError("No samples accumulated. Call update() first.")

        n = self._n
        mean_x = self._sum_x / n
        mean_y = self._sum_y / n

        # Centered cross/self covariances
        XtX_c = self._XtX - n * mean_x.outer(mean_x)
        YtY_c = self._YtY - n * mean_y.outer(mean_y)
        YtX_c = self._YtX - n * mean_y.outer(mean_x)

        numerator = torch.norm(YtX_c, p="fro") ** 2
        denom = torch.norm(XtX_c, p="fro") * torch.norm(YtY_c, p="fro") + eps

        return (numerator / denom).item()


def minibatch_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int = 256,
    eps: float = 1e-10,
) -> float:
    """
    Convenience function: compute linear CKA in mini-batches.

    Avoids O(n * max(d_x, d_y)) peak memory from holding full matrices.

    Args:
        X: (n_samples, d_x) full activation matrix.
        Y: (n_samples, d_y) full activation matrix.
        batch_size: number of samples per mini-batch.
        eps: numerical stability.

    Returns:
        CKA score as a Python float.
    """
    n = X.shape[0]
    acc = MinibatchCKAAccumulator(
        d_x=X.shape[1],
        d_y=Y.shape[1],
        device=str(X.device),
    )

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        acc.update(X[start:end], Y[start:end])

    return acc.compute(eps=eps)


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def cka_permutation_test(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_permutations: int = 1000,
    kernel: str = "linear",
    seed: int = 42,
) -> dict[str, float]:
    """
    Permutation test for CKA significance.

    Null hypothesis: no representational similarity (shuffled samples).
    If p < 0.05, the observed CKA is significantly above chance.

    Args:
        X: (n_samples, d_x) activations.
        Y: (n_samples, d_y) activations.
        n_permutations: number of permutation replicates.
        kernel: "linear" or "rbf".
        seed: random seed for reproducibility.

    Returns:
        dict with keys: observed_cka, p_value, null_mean, null_std.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    cka_fn = linear_cka if kernel == "linear" else rbf_cka
    observed = cka_fn(X, Y).item()

    null_scores: list[float] = []
    n = X.shape[0]

    for _ in range(n_permutations):
        perm = torch.randperm(n)
        Y_perm = Y[perm]
        null_scores.append(cka_fn(X, Y_perm).item())

    null_arr = np.array(null_scores)
    p_value = float(np.mean(null_arr >= observed))

    return {
        "observed_cka": observed,
        "p_value": p_value,
        "null_mean": float(null_arr.mean()),
        "null_std": float(null_arr.std()),
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

@dataclass
class CKAHeatmapData:
    """Data container for a layer-wise CKA heatmap."""

    scores: NDArray[np.floating]  # (n_teacher_layers, n_student_layers)
    teacher_layer_names: list[str]
    student_layer_names: list[str]

    def to_dict(self) -> dict:
        """Serialize for JSON/YAML logging."""
        return {
            "scores": self.scores.tolist(),
            "teacher_layers": self.teacher_layer_names,
            "student_layers": self.student_layer_names,
        }


def compute_layerwise_cka(
    teacher_activations: dict[str, torch.Tensor],
    student_activations: dict[str, torch.Tensor],
    kernel: str = "linear",
    batch_size: Optional[int] = None,
) -> CKAHeatmapData:
    """
    Compute all-pairs CKA between teacher and student layers.

    Args:
        teacher_activations: {layer_name: (n, d)} from hook collection.
        student_activations: {layer_name: (n, d)} from hook collection.
        kernel: "linear" or "rbf".
        batch_size: if set, use mini-batch CKA for memory efficiency.

    Returns:
        CKAHeatmapData with the full similarity matrix.
    """
    t_names = sorted(teacher_activations.keys())
    s_names = sorted(student_activations.keys())

    scores = np.zeros((len(t_names), len(s_names)), dtype=np.float64)

    for i, tn in enumerate(t_names):
        t_act = teacher_activations[tn]
        for j, sn in enumerate(s_names):
            s_act = student_activations[sn]

            if batch_size is not None:
                scores[i, j] = minibatch_cka(t_act, s_act, batch_size)
            elif kernel == "linear":
                scores[i, j] = linear_cka(t_act, s_act).item()
            else:
                scores[i, j] = rbf_cka(t_act, s_act).item()

    return CKAHeatmapData(
        scores=scores,
        teacher_layer_names=t_names,
        student_layer_names=s_names,
    )
