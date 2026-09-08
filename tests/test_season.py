import unittest

import numpy as np

from hallucination_repro.season import (
    diagnostic_weights,
    frame_attention_distribution,
    season_logits,
    temporal_homogenize,
    temporal_homogenize_layer,
)


class SeasonTests(unittest.TestCase):
    def test_beta_extremes(self):
        x = np.array([[1.0, 2.0], [5.0, 8.0]])
        d = x.mean(axis=0, keepdims=True)
        np.testing.assert_allclose(temporal_homogenize_layer(x, d, 0.0), x)
        np.testing.assert_allclose(temporal_homogenize_layer(x, d, 1.0), np.repeat(d, 2, axis=0))

    def test_progressive_temporal_homogenization(self):
        frames = np.array([[1.0], [3.0]])
        layers = [lambda x: x + 1.0, lambda x: x * 2.0]
        out = temporal_homogenize(frames, layers, beta=1.0)
        np.testing.assert_allclose(out, np.array([[6.0], [6.0]]))

    def test_frame_distribution_and_weights(self):
        a = np.zeros((2, 3, 2, 4))
        a[:, :, 1, :2] = 2.0
        a[:, :, 1, 2:] = 0.1
        p = frame_attention_distribution(a, 1, [slice(0, 2), slice(2, 4)])
        self.assertGreater(p[0], p[1])
        ws, wt = diagnostic_weights(p, p, p[::-1])
        self.assertAlmostEqual(ws, 0.0)
        self.assertAlmostEqual(wt, 1.0)

    def test_eq6(self):
        o = np.array([2.0, 1.0])
        s = np.array([0.0, 4.0])
        t = np.array([1.0, 3.0])
        got = season_logits(o, s, t, alpha=1.0, w_spatial=0.25, w_temporal=0.75)
        expected = 2 * o - (0.25 * s + 0.75 * t)
        np.testing.assert_allclose(got, expected)


if __name__ == "__main__":
    unittest.main()

