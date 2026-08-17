"""Tests for bounded q and rand_ctr utilities."""

from __future__ import annotations

import unittest

import numpy as np

from utils.bounded_q_model import BoundedErrorRewardModel, ConstantRewardModel
from utils.rand_ctr import calibrate_ctr_from_rand, estimate_rand_ctr
from utils.simulation_utils import SyntheticBanditEnv, generate_dataset


class TestBoundedQ(unittest.TestCase):
    def test_eps_zero_is_oracle(self):
        rng = np.random.default_rng(0)
        emb_x = rng.normal(size=(32, 8)).astype(np.float32)
        emb_a = rng.normal(size=(40, 8)).astype(np.float32)
        from training.trainer_trials import AnalyticRewardModel

        oracle = AnalyticRewardModel(
            action_context=emb_a, ctr=0.1, kind="oracle"
        )
        bad = ConstantRewardModel(0.5, n_actions=40)
        bounded = BoundedErrorRewardModel(oracle, bad, eps=0.0, n_actions=40)
        ctx = emb_x[:5]
        np.testing.assert_allclose(
            bounded.predict_user_action_block(ctx, 0, 40),
            oracle.predict_user_action_block(ctx, 0, 40),
            rtol=1e-5,
        )

    def test_eps_one_is_bad_constant(self):
        rng = np.random.default_rng(1)
        emb_x = rng.normal(size=(16, 4)).astype(np.float32)
        emb_a = rng.normal(size=(20, 4)).astype(np.float32)
        from training.trainer_trials import AnalyticRewardModel

        oracle = AnalyticRewardModel(
            action_context=emb_a, ctr=0.2, kind="oracle"
        )
        bad_val = 0.33
        bad = ConstantRewardModel(bad_val, n_actions=20)
        bounded = BoundedErrorRewardModel(oracle, bad, eps=1.0, n_actions=20)
        q = bounded.predict_user_action_block(emb_x[:3], 0, 20)
        self.assertTrue(np.allclose(q, bad_val, atol=1e-5))

    def test_linf_bound(self):
        rng = np.random.default_rng(2)
        emb_x = rng.normal(size=(24, 6)).astype(np.float32)
        emb_a = rng.normal(size=(30, 6)).astype(np.float32)
        from training.trainer_trials import AnalyticRewardModel

        oracle = AnalyticRewardModel(
            action_context=emb_a, ctr=0.15, kind="oracle"
        )
        bad = ConstantRewardModel(0.0, n_actions=30)
        eps = 0.4
        bounded = BoundedErrorRewardModel(oracle, bad, eps=eps, n_actions=30)
        o = oracle.predict_user_action_block(emb_x, 0, 30)
        b = bounded.predict_user_action_block(emb_x, 0, 30)
        self.assertLessEqual(float(np.max(np.abs(o - b))), eps + 1e-5)


class TestRandCTR(unittest.TestCase):
    def test_estimate_and_calibrate(self):
        rng = np.random.default_rng(3)
        emb_x = rng.normal(size=(50, 8)).astype(np.float32)
        emb_a = rng.normal(size=(60, 8)).astype(np.float32)
        params = {
            "n_users": 50,
            "n_actions": 60,
            "emb_dim": 8,
            "n_clusters": 8,
            "eps1": 0.05,
            "eps2": 0.05,
            "eps_meta": 0.0,
            "ctr": 0.1,
            "noise_mode": "kmeans_templates",
        }
        ds = generate_dataset(
            params=params, seed=0, emb_a=emb_a, emb_x=emb_x, store_original=True
        )
        est = estimate_rand_ctr(ds, n_samples=20_000, seed=1)
        self.assertIn("rand_ctr", est)
        self.assertGreaterEqual(est["rand_ctr"], 0.0)
        self.assertLessEqual(est["rand_ctr"], 1.0)

        cal = calibrate_ctr_from_rand(
            emb_x, emb_a, target_rand_ctr=0.08, n_samples=10_000, seed=2
        )
        self.assertIn("ctr_calibrated", cal)
        self.assertGreater(cal["ctr_calibrated"], 0.0)


if __name__ == "__main__":
    unittest.main()
