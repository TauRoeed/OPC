"""Vector-level distance metrics between clean and biased embeddings."""

from __future__ import annotations

import unittest

import numpy as np

from utils.noise_snr import cosine_retention, dataset_snr_report, embedding_noise_metrics, snr_db


class TestNoiseSNR(unittest.TestCase):
    def test_identical_embeddings_perfect_cosine_and_high_snr(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(64, 8)).astype(np.float32)
        m = embedding_noise_metrics(x, x.copy())
        self.assertAlmostEqual(m["cosine_retention"], 1.0, places=6)
        self.assertAlmostEqual(m["rmse"], 0.0, places=6)
        self.assertGreater(m["snr_db"], 100.0)

    def test_snr_drops_with_stronger_perturbation(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(128, 16)).astype(np.float64)
        noise = rng.normal(size=x.shape)
        mild = x + 0.05 * noise
        strong = x + 0.5 * noise
        self.assertGreater(snr_db(x, mild), snr_db(x, strong))
        self.assertGreater(cosine_retention(x, mild), cosine_retention(x, strong))

    def test_dataset_report_sides(self):
        rng = np.random.default_rng(2)
        ds = {k: rng.normal(size=(20, 4)) for k in ("emb_x", "our_x", "emb_a", "our_a")}
        ds["our_a"] = ds["emb_a"].copy()
        rep = dataset_snr_report(ds)
        self.assertAlmostEqual(rep["action"]["cosine_retention"], 1.0)
        self.assertLess(rep["context"]["cosine_retention"], 0.9)
        self.assertAlmostEqual(
            rep["cosine_mean"], 0.5 * (rep["action"]["cosine_retention"] + rep["context"]["cosine_retention"])
        )


if __name__ == "__main__":
    unittest.main()
