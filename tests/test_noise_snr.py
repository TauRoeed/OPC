"""Unit tests for embedding SNR metrics and noise axis/component resolution."""

from __future__ import annotations

import unittest

import numpy as np

from utils.noise_levels import (
    VALID_NOISE_LEVELS,
    noise_eps,
    noise_target_sides,
    resolve_noise_spec,
)
from utils.noise_snr import (
    cosine_retention,
    embedding_noise_metrics,
    isolate_component_metrics,
    signal_frac,
    snr_db,
)


class TestNoiseSNR(unittest.TestCase):
    def test_identical_embeddings_perfect_cosine_and_high_snr(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(64, 8)).astype(np.float32)
        m = embedding_noise_metrics(x, x.copy(), epsilons=[0.0, 0.0])
        self.assertAlmostEqual(m["cosine_retention"], 1.0, places=6)
        self.assertAlmostEqual(m["rmse"], 0.0, places=6)
        self.assertGreater(m["snr_db"], 100.0)
        self.assertAlmostEqual(m["signal_frac"], 1.0)

    def test_snr_drops_with_stronger_perturbation(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(128, 16)).astype(np.float64)
        noise = rng.normal(size=x.shape)
        mild = x + 0.05 * noise
        strong = x + 0.5 * noise
        self.assertGreater(snr_db(x, mild), snr_db(x, strong))
        self.assertGreater(cosine_retention(x, mild), cosine_retention(x, strong))

    def test_signal_frac(self):
        self.assertAlmostEqual(signal_frac([0.1, 0.2, 0.05]), 0.65)

    def test_noise_eps_combined_and_components(self):
        self.assertEqual(noise_eps("low", "combined"), (0.05, 0.05, 0.0))
        # Axis alone no longer remaps to a single component.
        self.assertEqual(noise_eps("high", "context"), (0.20, 0.25, 0.10))
        self.assertEqual(noise_eps("high", "action"), (0.20, 0.25, 0.10))
        # Component isolation uses the matching column from the level table.
        self.assertEqual(noise_eps("high", "combined", "linear"), (0.20, 0.0, 0.0))
        self.assertEqual(noise_eps("high", "combined", "general"), (0.20, 0.0, 0.0))
        self.assertEqual(noise_eps("high", "combined", "cluster"), (0.0, 0.25, 0.0))
        self.assertEqual(noise_eps("high", "combined", "metadata"), (0.0, 0.0, 0.10))
        # Legacy axis=metadata still means metadata component.
        self.assertEqual(noise_eps("high", "metadata"), (0.0, 0.0, 0.10))
        self.assertTrue(
            {"low", "medium", "high", "extreme", "brutal", "catastrophic"}
            <= set(VALID_NOISE_LEVELS)
        )

    def test_noise_target_sides(self):
        self.assertEqual(noise_target_sides("context"), (True, False))
        self.assertEqual(noise_target_sides("action"), (False, True))
        self.assertEqual(noise_target_sides("combined"), (True, True))
        self.assertEqual(noise_target_sides("metadata"), (True, True))

    def test_resolve_axis_and_component(self):
        spec = resolve_noise_spec("high", axis="context", component="cluster")
        self.assertEqual((spec["eps1"], spec["eps2"], spec["eps_meta"]), (0.0, 0.25, 0.0))
        self.assertTrue(spec["apply_user"])
        self.assertFalse(spec["apply_item"])
        self.assertEqual(spec["component"], "cluster")

    def test_isolate_component_metrics(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(32, 4)).astype(np.float32)
        n1 = rng.normal(size=x.shape).astype(np.float32)
        n2 = rng.normal(size=x.shape).astype(np.float32)
        out = isolate_component_metrics(
            x, [n1, n2], [0.2, 0.0], component_names=["linear", "cluster"]
        )
        self.assertAlmostEqual(out["cluster"]["rmse"], 0.0)
        self.assertGreater(out["linear"]["rmse"], 0.0)
        self.assertLess(out["linear"]["snr_db"], out["cluster"]["snr_db"])


if __name__ == "__main__":
    unittest.main()
