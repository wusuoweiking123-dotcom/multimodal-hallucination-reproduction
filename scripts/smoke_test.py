#!/usr/bin/env python3
"""Run a deterministic end-to-end numerical smoke test without model weights."""

from __future__ import annotations

import json

import numpy as np

from hallucination_repro import (
    assign_head_gains,
    cmac_logits,
    diagnostic_weights,
    season_logits,
    softmax,
    visual_attention_ratio,
)


def main() -> None:
    original = np.array([3.0, 1.0, -1.0])
    spatial = np.array([1.0, 2.0, 0.0])
    temporal = np.array([2.0, 0.0, 1.0])
    ws, wt = diagnostic_weights(
        np.array([0.8, 0.2]), np.array([0.55, 0.45]), np.array([0.2, 0.8])
    )
    season = season_logits(original, spatial, temporal, 1.0, ws, wt)
    cmac = cmac_logits(original, spatial, 3.0)

    attention = np.array([
        [[0.4, 0.4, 0.1, 0.1]],
        [[0.005, 0.005, 0.495, 0.495]],
    ])
    ratio = visual_attention_ratio(attention, slice(0, 1), slice(0, 2))
    gains, selected = assign_head_gains(
        ratio, 5, 7, 3, 0.22, 0.01, 1.16, 1.30
    )
    report = {
        "season": {"w_spatial": ws, "w_temporal": wt, "probabilities": softmax(season).tolist()},
        "cmac": {"probabilities": softmax(cmac).tolist()},
        "r2tar": {
            "visual_ratios": ratio.tolist(),
            "gains": gains.tolist(),
            "perception_heads": np.flatnonzero(selected.perception).tolist(),
            "reasoning_heads": np.flatnonzero(selected.reasoning).tolist(),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

