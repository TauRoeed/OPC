"""Fixture-based tests for KuaiRec / KuaiRand loaders (no network)."""

from __future__ import annotations

import unittest
from pathlib import Path

from BPR.bpr_config import load_bpr_dataset_config
from BPR.dataload import build_csr_from_interactions, load_kuairand, load_kuairec
from BPR.generate_artifacts import _dataset_bundle

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TestKuaiLoaders(unittest.TestCase):
    def test_load_kuairec_fixture_positives_and_csr(self):
        root = FIXTURES / "kuairec"
        ratings, users, items = load_kuairec(
            str(root), download=False, watch_ratio_min=2.0
        )
        self.assertEqual(len(ratings), 4)
        self.assertTrue({"user_id", "item_id"} <= set(ratings.columns))
        self.assertIn("feat", items.columns)
        feat = str(items.loc[items["item_id"] == 10, "feat"].iloc[0])
        self.assertIn("|", feat)
        data = build_csr_from_interactions(
            ratings, user_col="user_id", item_col="item_id", item_info=items
        )
        self.assertEqual(data.X.shape[0], ratings["user_id"].nunique())
        self.assertEqual(data.X.nnz, 4)

    def test_load_kuairand_fixture_clicks(self):
        root = FIXTURES / "kuairand-pure"
        ratings, users, items = load_kuairand(str(root), download=False)
        self.assertGreaterEqual(len(ratings), 4)
        self.assertEqual(ratings[["user_id", "item_id"]].duplicated().sum(), 0)
        self.assertIn("video_type", items.columns)
        data = build_csr_from_interactions(
            ratings, user_col="user_id", item_col="item_id", item_info=items
        )
        self.assertEqual(data.X.nnz, len(ratings))

    def test_kuairec_dataset_bundle_config(self):
        cfg = load_bpr_dataset_config("kuairec")
        ratings, users, data = _dataset_bundle(
            "kuairec",
            str(FIXTURES / "kuairec"),
            data_cfg=cfg["data"],
            download=False,
        )
        self.assertGreater(data.X.nnz, 0)
        self.assertGreater(len(users), 0)

    def test_kuairand_dataset_bundle_config(self):
        cfg = load_bpr_dataset_config("kuairand")
        ratings, users, data = _dataset_bundle(
            "kuairand",
            str(FIXTURES / "kuairand-pure"),
            data_cfg=cfg["data"],
            download=False,
        )
        self.assertGreater(data.X.nnz, 0)


if __name__ == "__main__":
    unittest.main()
