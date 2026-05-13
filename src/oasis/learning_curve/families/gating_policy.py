from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class GatingPolicy(Protocol):
    def apply(self, logits: np.ndarray) -> np.ndarray: ...
    def regularization_loss(self, logits: np.ndarray) -> float: ...


@dataclass(frozen=True, slots=True)
class DenseGatingPolicy:
    """Softmax over all experts."""

    def apply(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=-1, keepdims=True)

    def regularization_loss(self, logits: np.ndarray) -> float:
        return 0.0


@dataclass(frozen=True, slots=True)
class TopKGatingPolicy:
    """Activate only the top-k experts; zero out the rest before softmax.

    regularization_loss uses the Switch Transformer auxiliary load-balance
    term: n_experts * sum_i(f_i * P_i), where f_i is the fraction of samples
    routed to expert i and P_i is the mean router probability for expert i.
    The minimum value (perfectly balanced load) equals k.
    """

    k: int

    def apply(self, logits: np.ndarray) -> np.ndarray:
        # Threshold = k-th largest value along the expert axis (keepdims via list index).
        threshold = np.partition(logits, -self.k, axis=-1)[..., [-self.k]]
        masked = np.where(logits >= threshold, logits, -np.inf)
        # Numerically stable softmax over the surviving logits; -inf slots → 0.
        shifted = masked - masked.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=-1, keepdims=True)

    def regularization_loss(self, logits: np.ndarray) -> float:
        if logits.ndim < 2:
            return 0.0
        weights = self.apply(logits)  # (n_samples, n_experts)
        n_experts = logits.shape[-1]
        f = (weights > 0).astype(float).mean(axis=0)  # fraction routed to each expert
        P = weights.mean(axis=0)  # mean router probability per expert
        return float(n_experts * (f * P).sum())
