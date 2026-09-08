import unittest

import numpy as np

from hallucination_repro.cmac import cross_modal_mask, distort_attention_output, refined_position_ids


class CmacTests(unittest.TestCase):
    def test_positions_compress_only_image_span(self):
        positions = refined_position_ids(2, 4, 3, gamma=2.0)
        np.testing.assert_allclose(positions, [0, 1, 1.5, 2, 2.5, 3, 4, 5, 6])

    def test_mask_is_confined_to_cross_modal_block(self):
        logits = np.zeros((1, 1, 5, 5))
        logits[..., 3:, 1:3] = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = cross_modal_mask(logits, slice(3, 5), slice(1, 3))
        self.assertEqual(int(mask.sum()), 2)
        self.assertFalse(mask[..., :3, :].any())
        self.assertFalse(mask[..., :, :1].any())

    def test_distortion_changes_only_selected_value_contributions(self):
        logits = np.zeros((1, 1, 1, 2))
        values = np.array([[[[0.0, 2.0], [10.0, 20.0]]]])
        no_mask = np.zeros_like(logits, dtype=bool)
        baseline = distort_attention_output(logits, values, no_mask, slice(0, 2))
        np.testing.assert_allclose(baseline, [[[[5.0, 11.0]]]])
        mask = np.array([[[[True, False]]]])
        distorted = distort_attention_output(logits, values, mask, slice(0, 2))
        # Image/value scalar mean is 8; first key becomes [8, 8].
        np.testing.assert_allclose(distorted, [[[[9.0, 14.0]]]])


if __name__ == "__main__":
    unittest.main()
