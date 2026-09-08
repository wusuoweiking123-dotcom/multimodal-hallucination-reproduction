import unittest

import numpy as np

from hallucination_repro.r2tar import assign_head_gains, rescale_head_outputs, visual_attention_ratio


class R2TarTests(unittest.TestCase):
    def test_visual_ratio(self):
        # Two heads, two queries, four keys. Rows sum to one.
        a = np.array([
            [[0.4, 0.4, 0.1, 0.1], [0.3, 0.3, 0.2, 0.2]],
            [[0.01, 0.01, 0.49, 0.49], [0.02, 0.02, 0.48, 0.48]],
        ])
        ratios = visual_attention_ratio(a, slice(0, 2), slice(0, 2))
        np.testing.assert_allclose(ratios, [0.7, 0.03])

    def test_overlapping_layer_bands_select_by_ratio(self):
        gains, selected = assign_head_gains(
            np.array([0.30, 0.005, 0.10]),
            layer_index=5,
            perception_last_layer=7,
            reasoning_first_layer=3,
            tau_perception=0.22,
            tau_reasoning=0.01,
            gain_perception=1.16,
            gain_reasoning=1.30,
        )
        np.testing.assert_allclose(gains, [1.16, 1.30, 1.0])
        np.testing.assert_array_equal(selected.perception, [True, False, False])
        np.testing.assert_array_equal(selected.reasoning, [False, True, False])

    def test_rescaling(self):
        outputs = np.ones((2, 3, 4))  # tokens, heads, head_dim
        got = rescale_head_outputs(outputs, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(got[:, 1], 2.0)
        np.testing.assert_allclose(got[:, 2], 3.0)

    def test_batched_rescaling(self):
        outputs = np.ones((2, 4, 3, 5))  # batch, tokens, heads, head_dim
        gains = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        got = rescale_head_outputs(outputs, gains)
        np.testing.assert_allclose(got[0, :, 2], 3.0)
        np.testing.assert_allclose(got[1, :, 0], 4.0)


if __name__ == "__main__":
    unittest.main()
