"""Model-agnostic numerical kernels for three hallucination-mitigation papers."""

from .common import contrastive_logits, js_divergence, softmax
from .season import (
    diagnostic_weights,
    frame_attention_distribution,
    season_logits,
    spatial_negative,
    temporal_homogenize,
    temporal_homogenize_layer,
)
from .cmac import (
    cmac_logits,
    cross_modal_mask,
    distort_attention_output,
    refined_position_ids,
)
from .r2tar import (
    HeadSelection,
    assign_head_gains,
    rescale_head_outputs,
    visual_attention_ratio,
)

__all__ = [
    "HeadSelection",
    "assign_head_gains",
    "cmac_logits",
    "contrastive_logits",
    "cross_modal_mask",
    "diagnostic_weights",
    "distort_attention_output",
    "frame_attention_distribution",
    "js_divergence",
    "refined_position_ids",
    "rescale_head_outputs",
    "season_logits",
    "softmax",
    "spatial_negative",
    "temporal_homogenize",
    "temporal_homogenize_layer",
    "visual_attention_ratio",
]

