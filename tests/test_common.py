import unittest

import numpy as np

from hallucination_repro.common import contrastive_logits, js_divergence, softmax


class CommonTests(unittest.TestCase):
    def test_softmax_is_normalized_and_shift_invariant(self):
        x = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(softmax(x).sum(-1), 1.0)
        np.testing.assert_allclose(softmax(x), softmax(x + 1000.0))

    def test_jsd_properties(self):
        p = np.array([0.8, 0.2])
        q = np.array([0.1, 0.9])
        self.assertAlmostEqual(float(js_divergence(p, p)), 0.0)
        self.assertAlmostEqual(float(js_divergence(p, q)), float(js_divergence(q, p)))
        self.assertGreater(float(js_divergence(p, q)), 0.0)

    def test_zero_alpha_is_baseline(self):
        original = np.array([1.0, -2.0])
        negative = np.array([-4.0, 5.0])
        np.testing.assert_allclose(contrastive_logits(original, negative, 0.0), original)


if __name__ == "__main__":
    unittest.main()

