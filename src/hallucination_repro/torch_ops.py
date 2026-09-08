"""Optional PyTorch kernels for patching real model attention implementations.

PyTorch is intentionally not a base dependency. Import this module only inside a
GPU environment that already contains the backbone's compatible torch version.
The functions operate on the tensor layouts used by common Hugging Face eager
attention implementations.
"""

from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the deployment env
    raise ImportError("torch_ops requires PyTorch; install the version required by your backbone.") from exc


def season_diagnostic_weights(original, spatial, temporal, eps: float = 1e-12):
    """Batched SEASON Eq. 5 for frame distributions on the final axis."""
    def jsd(p, q):
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
        m = 0.5 * (p + q)
        return 0.5 * (
            (p * (p.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(-1)
            + (q * (q.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(-1)
        )

    ds, dt = jsd(original, spatial), jsd(original, temporal)
    total = ds + dt
    ws = torch.where(total > eps, ds / total.clamp_min(eps), torch.full_like(total, 0.5))
    wt = torch.where(total > eps, dt / total.clamp_min(eps), torch.full_like(total, 0.5))
    return ws, wt


def season_contrastive_logits(original, spatial, temporal, alpha: float, w_spatial, w_temporal):
    """SEASON Eq. 6; weights broadcast across vocabulary."""
    while w_spatial.ndim < original.ndim:
        w_spatial = w_spatial.unsqueeze(-1)
        w_temporal = w_temporal.unsqueeze(-1)
    return (1 + alpha) * original - alpha * (w_spatial * spatial + w_temporal * temporal)


def cmac_cross_modal_mask(attn_logits, text_queries: slice, image_keys: slice):
    """CMAC Eq. 8 on tensors shaped [batch, heads, query, key]."""
    cross = attn_logits[..., text_queries, image_keys]
    threshold = cross.mean(dim=(-2, -1), keepdim=True)
    local = cross > threshold
    mask = torch.zeros_like(attn_logits, dtype=torch.bool)
    mask[..., text_queries, image_keys] = local
    return mask


def cmac_distorted_attention(attn_logits, value_states, mask, image_keys: slice):
    """CMAC Eq. 9 using the released IMCCD scalar-mean replacement."""
    weights = torch.softmax(attn_logits, dim=-1, dtype=torch.float32).to(value_states.dtype)
    replacement = value_states[..., image_keys, :].mean(dim=(-2, -1), keepdim=True)
    replacement = replacement.expand_as(value_states)
    effective = torch.where(mask.unsqueeze(-1), replacement.unsqueeze(-3), value_states.unsqueeze(-3))
    return (weights.unsqueeze(-1) * effective).sum(dim=-2)


def r2tar_gains(
    attn_weights,
    layer_index: int,
    query_tokens: slice,
    visual_tokens: slice,
    perception_last_layer: int,
    reasoning_first_layer: int,
    tau_perception: float,
    tau_reasoning: float,
    gain_perception: float,
    gain_reasoning: float,
):
    """R²-TAR Eqs. 7-9 for [batch, heads, query, key] attention."""
    visual_ratio = attn_weights[..., query_tokens, visual_tokens].sum(-1).mean(-1)
    gains = torch.ones_like(visual_ratio)
    if layer_index <= perception_last_layer:
        gains = torch.where(visual_ratio >= tau_perception, gain_perception, gains)
    if layer_index >= reasoning_first_layer:
        gains = torch.where(visual_ratio <= tau_reasoning, gain_reasoning, gains)
    return gains


def r2tar_rescale(head_outputs, gains):
    """R²-TAR Eq. 10 for outputs [batch, query, heads, head_dim]."""
    return head_outputs * gains[:, None, :, None]

