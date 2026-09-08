"""SEASON: Self-Diagnostic Contrastive Decoding (Eqs. 1-6).

These functions implement the paper's model-independent core. Integration with a
specific VideoLLM requires hooks that expose vision-layer activations, decoder
attentions, and three next-token logit streams.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .common import js_divergence, softmax


Array = np.ndarray
LayerFn = Callable[[Array], Array]


def temporal_homogenize_layer(current_pre_features: Array, global_context: Array, beta: float) -> Array:
    """Apply SEASON Eq. 2 to one vision-encoder layer.

    Inputs use shape ``[frames, ...feature dimensions...]``. ``global_context``
    must be the layer-wise frame mean from the *standard* forward pass.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1].")
    x = np.asarray(current_pre_features)
    d = np.asarray(global_context)
    expected = x.shape[1:]
    if d.shape not in (expected, (1,) + expected):
        raise ValueError(f"global_context shape must be {expected} or {(1,) + expected}.")
    return (1.0 - beta) * x + beta * d


def temporal_homogenize(initial_frame_features: Array, layers: Sequence[LayerFn], beta: float) -> Array:
    """Progressively homogenize all vision layers exactly as SEASON Eq. 2.

    A first, unmodified pass records each layer's pre-homogenization global
    context ``d_l``. A second pass recurrently injects those contexts.
    """
    if not layers:
        return np.asarray(initial_frame_features).copy()

    standard = np.asarray(initial_frame_features)
    contexts: list[Array] = []
    for layer in layers:
        standard = np.asarray(layer(standard))
        contexts.append(np.mean(standard, axis=0, keepdims=True))

    homogenized = np.asarray(initial_frame_features)
    for layer, context in zip(layers, contexts):
        pre = np.asarray(layer(homogenized))
        homogenized = temporal_homogenize_layer(pre, context, beta)
    return homogenized


def spatial_negative(features: Array, noise_std: float, seed: int | None = None) -> Array:
    """Construct the Gaussian spatial negative used by SEASON/VCD.

    The paper does not report a single noise standard deviation, so it remains
    explicit rather than silently choosing a benchmark-specific value.
    """
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")
    rng = np.random.default_rng(seed)
    features = np.asarray(features)
    return features + rng.normal(0.0, noise_std, size=features.shape)


def frame_attention_distribution(
    attentions: Array,
    query_index: int,
    frame_token_slices: Sequence[slice],
    layer_indices: Sequence[int] | None = None,
) -> Array:
    """Compute SEASON Eq. 4 from decoder attention matrices.

    ``attentions`` has shape ``[layers, heads, query_tokens, key_tokens]``.
    The function sums heads, selected layers, and patch tokens per frame, then
    applies softmax over frames.
    """
    a = np.asarray(attentions, dtype=np.float64)
    if a.ndim != 4:
        raise ValueError("attentions must have shape [layers, heads, query, key].")
    selected = a if layer_indices is None else a[np.asarray(layer_indices, dtype=int)]
    if not -a.shape[2] <= query_index < a.shape[2]:
        raise IndexError("query_index is outside the attention query dimension.")
    key_scores = selected[:, :, query_index, :].sum(axis=(0, 1))
    frame_scores = np.asarray([key_scores[s].sum() for s in frame_token_slices])
    return softmax(frame_scores)


def diagnostic_weights(
    original_frame_attention: Array,
    spatial_frame_attention: Array,
    temporal_frame_attention: Array,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """Return (w_spatial, w_temporal) from SEASON Eq. 5."""
    ds = float(js_divergence(original_frame_attention, spatial_frame_attention))
    dt = float(js_divergence(original_frame_attention, temporal_frame_attention))
    total = ds + dt
    if total <= eps:
        return 0.5, 0.5
    return ds / total, dt / total


def season_logits(
    original_logits: Array,
    spatial_logits: Array,
    temporal_logits: Array,
    alpha: float,
    w_spatial: float,
    w_temporal: float,
) -> Array:
    """Return pre-softmax SEASON logits from Eq. 6."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    if min(w_spatial, w_temporal) < 0:
        raise ValueError("diagnostic weights must be non-negative.")
    total = w_spatial + w_temporal
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError("diagnostic weights must sum to one.")
    o = np.asarray(original_logits)
    s = np.asarray(spatial_logits)
    t = np.asarray(temporal_logits)
    if o.shape != s.shape or o.shape != t.shape:
        raise ValueError("all logit tensors must have identical shapes.")
    negative = w_spatial * s + w_temporal * t
    return (1.0 + alpha) * o - alpha * negative

