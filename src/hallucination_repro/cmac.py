"""CMAC/IMD/CMPC model-independent kernels (paper Eqs. 8-13)."""

from __future__ import annotations

import numpy as np

from .common import contrastive_logits, softmax


Array = np.ndarray


def refined_position_ids(system_tokens: int, image_tokens: int, query_tokens: int, gamma: float = 2.0) -> Array:
    """Build zero-based CMPC positions corresponding to paper Eq. 13.

    Image positions and their total span are compressed by ``gamma`` while text
    order and unit spacing are preserved. Float positions are intentional: RoPE
    accepts continuous angles even though stock HF APIs may require a small patch.
    """
    if min(system_tokens, image_tokens, query_tokens) < 0:
        raise ValueError("token counts must be non-negative.")
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    prefix = np.arange(system_tokens, dtype=np.float64)
    # Eq. 13 is one-based: mb + i/gamma for i=1..n. Subtracting one
    # produces the exact zero-based form below.
    image_anchor = max(system_tokens - 1, 0)
    image = image_anchor + (np.arange(image_tokens, dtype=np.float64) + 1.0) / gamma
    suffix_start = system_tokens + image_tokens / gamma
    suffix = suffix_start + np.arange(query_tokens, dtype=np.float64)
    return np.concatenate([prefix, image, suffix])


def cross_modal_mask(attention_logits: Array, text_queries: slice, image_keys: slice) -> Array:
    """Create the mean-thresholded cross-modal binary mask from CMAC Eq. 8.

    Leading dimensions (typically batch and heads) are preserved; each matrix is
    thresholded by its own cross-modal mean to avoid mixing heads or samples.
    """
    logits = np.asarray(attention_logits)
    if logits.ndim < 2:
        raise ValueError("attention_logits needs at least query and key dimensions.")
    cross = logits[..., text_queries, image_keys]
    if cross.shape[-2] == 0 or cross.shape[-1] == 0:
        raise ValueError("cross-modal slice cannot be empty.")
    threshold = np.mean(cross, axis=(-2, -1), keepdims=True)
    local = cross > threshold
    mask = np.zeros_like(logits, dtype=bool)
    mask[..., text_queries, image_keys] = local
    return mask


def distort_attention_output(
    attention_logits: Array,
    values: Array,
    mask: Array,
    image_keys: slice,
) -> Array:
    """Apply IMD value distortion and weighted aggregation (Eq. 9).

    Shapes are ``[..., query, key]`` and ``[..., key, value_dim]``. Following
    the released IMCCD implementation, the replacement is one scalar mean per
    leading batch/head unit, computed over image-token and value dimensions.
    """
    logits = np.asarray(attention_logits, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if logits.shape != mask.shape:
        raise ValueError("mask and attention_logits must have identical shapes.")
    if logits.shape[:-2] != values.shape[:-2] or logits.shape[-1] != values.shape[-2]:
        raise ValueError("attention and value tensor shapes are incompatible.")
    weights = softmax(logits, axis=-1)
    image_values = values[..., image_keys, :]
    if image_values.shape[-2] == 0:
        raise ValueError("image_keys slice cannot be empty.")
    replacement = image_values.mean(axis=(-2, -1), keepdims=True)
    replacement = np.broadcast_to(replacement, values.shape)
    effective_values = np.where(mask[..., None], replacement[..., None, :, :], values[..., None, :, :])
    return np.sum(weights[..., None] * effective_values, axis=-2)


def cmac_logits(original_logits: Array, distorted_logits: Array, alpha: float = 3.0) -> Array:
    """Return CMAC contrastive logits from Eq. 12."""
    return contrastive_logits(original_logits, distorted_logits, alpha)
