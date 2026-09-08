"""Shared numerical utilities."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def softmax(x: Array, axis: int = -1) -> Array:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _normalise_distribution(x: Array, axis: int, eps: float) -> Array:
    x = np.asarray(x, dtype=np.float64)
    if np.any(x < 0):
        raise ValueError("Probability inputs must be non-negative.")
    denominator = np.sum(x, axis=axis, keepdims=True)
    if np.any(denominator <= eps):
        raise ValueError("Probability inputs must have positive mass.")
    return x / denominator


def js_divergence(p: Array, q: Array, axis: int = -1, eps: float = 1e-12) -> Array:
    """Jensen-Shannon divergence in nats, robust to zero probabilities."""
    p = _normalise_distribution(p, axis, eps)
    q = _normalise_distribution(q, axis, eps)
    m = 0.5 * (p + q)
    p_safe = np.clip(p, eps, None)
    q_safe = np.clip(q, eps, None)
    m_safe = np.clip(m, eps, None)
    kl_pm = np.sum(p * (np.log(p_safe) - np.log(m_safe)), axis=axis)
    kl_qm = np.sum(q * (np.log(q_safe) - np.log(m_safe)), axis=axis)
    return 0.5 * (kl_pm + kl_qm)


def contrastive_logits(original: Array, negative: Array, alpha: float) -> Array:
    """Generic contrastive decoding: (1 + alpha) original - alpha negative."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    original = np.asarray(original)
    negative = np.asarray(negative)
    if original.shape != negative.shape:
        raise ValueError("original and negative logits must have identical shapes.")
    return (1.0 + alpha) * original - alpha * negative

