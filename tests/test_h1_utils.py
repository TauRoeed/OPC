"""Tests for bounded q and rand_ctr utilities."""

from __future__ import annotations

import unittest

import numpy as np

from utils.bounded_q_model import BoundedErrorRewardModel, ConstantRewardModel
from utils.rand_ctr import estimate_rand_ctr
from utils.simulation_utils import calc_uniform_reward, generate_dataset


class TestBoundedQ(unittest.TestCase):
    def test_eps_zero_is_oracle(self):
        rng = np.random.default_rng(0)
        emb_x = rng.normal(size=(32, 8)).astype(np.float32)
        emb_a = rng.normal(size=(40, 8)).astype(np.float32)
        from training.trainer_trials import AnalyticRewardModel

        oracle = AnalyticRewardModel(
            action_context=emb_a, scale=1.0, offset=-2.0, kind="oracle"
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
            action_context=emb_a, scale=0.8, offset=-1.5, kind="oracle"
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
            action_context=emb_a, scale=1.2, offset=-1.0, kind="oracle"
        )
        bad = ConstantRewardModel(0.0, n_actions=30)
        eps = 0.4
        bounded = BoundedErrorRewardModel(oracle, bad, eps=eps, n_actions=30)
        o = oracle.predict_user_action_block(emb_x, 0, 30)
        b = bounded.predict_user_action_block(emb_x, 0, 30)
        self.assertLessEqual(float(np.max(np.abs(o - b))), eps + 1e-5)


class TestRandCTR(unittest.TestCase):
    def test_world_calibrated_to_random_policy_ctr(self):
        # H1 worlds: the uniform random policy's CTR is the calibration target.
        rng = np.random.default_rng(3)
        emb_x = rng.normal(size=(300, 8)).astype(np.float32)
        emb_a = rng.normal(size=(400, 8)).astype(np.float32)
        params = {"bias": "medium", "ctr": 0.08, "ctr_reference": "uniform"}
        ds = generate_dataset(params=params, seed=0, emb_a=emb_a, emb_x=emb_x)
        exact = calc_uniform_reward(ds)
        self.assertAlmostEqual(ds["world"]["uniform_ctr"], 0.08, places=6)  # calibration sample
        self.assertLess(abs(exact - 0.08), 0.004)  # full population
        est = estimate_rand_ctr(ds, n_samples=200_000, seed=1)
        self.assertLess(abs(est["rand_q_mean"] - exact), 0.004)
        self.assertEqual(est["density_regime"], "ModerateReward")


if __name__ == "__main__":
    unittest.main()
