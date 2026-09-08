"""R²-TAR functional-head identification and class-conditioned rescaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class HeadSelection:
    perception: Array
    reasoning: Array


def visual_attention_ratio(attention: Array, query_tokens: slice, visual_tokens: slice) -> Array:
    """Compute S_v for every head (paper Eq. 7).

    ``attention`` has shape ``[..., heads, query, key]`` and is expected to be
    post-softmax. The ratio is averaged across the selected query positions.
    """
    a = np.asarray(attention, dtype=np.float64)
    if a.ndim < 3:
        raise ValueError("attention must have at least [head, query, key] dimensions.")
    selected_queries = a[..., query_tokens, :]
    visual_mass = selected_queries[..., visual_tokens].sum(axis=-1)
    if visual_mass.shape[-1] == 0:
        raise ValueError("query_tokens slice cannot be empty.")
    return visual_mass.mean(axis=-1)


def assign_head_gains(
    visual_ratios: Array,
    layer_index: int,
    perception_last_layer: int,
    reasoning_first_layer: int,
    tau_perception: float,
    tau_reasoning: float,
    gain_perception: float,
    gain_reasoning: float,
) -> tuple[Array, HeadSelection]:
    """Identify heads (Eq. 8) and assign gains (Eq. 9)."""
    if tau_reasoning >= tau_perception:
        raise ValueError("tau_reasoning must be less than tau_perception.")
    if min(gain_perception, gain_reasoning) < 1.0:
        raise ValueError("R2-TAR uses amplification gains >= 1.")
    ratios = np.asarray(visual_ratios, dtype=np.float64)
    perception = (layer_index <= perception_last_layer) & (ratios >= tau_perception)
    reasoning = (layer_index >= reasoning_first_layer) & (ratios <= tau_reasoning)
    gains = np.ones_like(ratios)
    gains = np.where(perception, gain_perception, gains)
    gains = np.where(reasoning, gain_reasoning, gains)
    return gains, HeadSelection(perception=perception, reasoning=reasoning)


def rescale_head_outputs(head_outputs: Array, gains: Array, head_axis: int = -2) -> Array:
    """Scale per-head outputs before concatenation/output projection (Eq. 10)."""
    outputs = np.asarray(head_outputs)
    gains = np.asarray(gains)
    axis = head_axis if head_axis >= 0 else outputs.ndim + head_axis
    if not 0 <= axis < outputs.ndim:
        raise ValueError("head_axis is invalid.")
    if outputs.shape[axis] != gains.shape[-1]:
        raise ValueError("last gain dimension must match the head dimension.")
    gain_prefix = gains.shape[:-1]
    if gain_prefix and gain_prefix != outputs.shape[: len(gain_prefix)]:
        raise ValueError("leading gain dimensions must match leading output dimensions.")
    if len(gain_prefix) > axis:
        raise ValueError("gains have too many leading dimensions for head_axis.")
    shape = list(gain_prefix)
    shape += [1] * (axis - len(gain_prefix))
    shape += [gains.shape[-1]]
    shape += [1] * (outputs.ndim - axis - 1)
    return outputs * gains.reshape(shape)
