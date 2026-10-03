"""CausE's protocol port (training/cause_protocol.py): row-for-row equality with the audit's reconstructed files,
and the released bootstrap evaluation."""
import os
import random
from pathlib import Path

import numpy as np
import pytest

from training import cause_protocol as P

AUDIT = Path(os.path.expanduser("~/code/CausE/repro"))
RAW = AUDIT / ".local" / "raw"
needs_audit = pytest.mark.skipif(not (RAW / "ml-100k" / "u.data").exists() or not (AUDIT / "data" / "ml100k_skew_s0").exists(),
                                 reason="the standalone CausE audit (raw MovieLens and its reconstructed files) is not on this machine")


def _csv(path):
    return np.loadtxt(path, delimiter=",", dtype=np.int64, ndmin=2)


def _same(ours, path):
    ref = _csv(path)
    got = np.stack(ours, axis=1)
    assert got.shape == ref.shape and np.array_equal(got, ref), path.name


@needs_audit
def test_ml100k_split_matches_the_audit_files_row_for_row():
    split = P.build_skew_split("ml100k", RAW, seed=0)
    d = AUDIT / "data" / "ml100k_skew_s0"
    for tag in P.TAGS:
        _same(P.tag_rows(split, tag), d / f"user_prod_dict.skew.train.{tag}.csv")
    for part in ("test", "valid_test", "valid_train"):
        _same(P.eval_rows(split, part), d / f"user_prod_dict.skew.{part}.shared.csv")
    assert (split["num_users"], split["num_products"]) == (944, 1683)  # the released code's ML-100K defaults


@needs_audit
def test_fig1_injection_levels_match_the_audit_files():
    levels = (0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15)
    split = P.build_skew_split("ml100k", RAW, seed=0, st_levels=levels, st_train_frac=0.15)
    d = AUDIT / "data" / "ml100k_fig1_s0"
    for lv in levels:
        pct = "%03d" % int(round(lv * 1000))
        sub = split["st_levels"][lv]
        _same(P.tag_rows(split, "adapt_2i", st_rows=sub), d / f"user_prod_dict.skew.train.adapt_2i_st{pct}.csv")
        _same(P.tag_rows(split, "adapt_0", st_rows=sub), d / f"user_prod_dict.skew.train.adapt_0_st{pct}.csv")
        _same(P.tag_rows(split, "adapt_blend", st_rows=sub), d / f"user_prod_dict.skew.train.adapt_blend_st{pct}.csv")


@needs_audit
def test_split_proportions_follow_the_paper():
    split = P.build_skew_split("ml100k", RAW, seed=0)
    n = len(split["y"])
    frac = {k: len(split[k]) / n for k in ("sc", "st", "test", "valid_train", "valid_test")}
    assert frac["test"] == pytest.approx(0.20, abs=0.01) and frac["st"] == pytest.approx(0.10, abs=0.01)
    assert frac["sc"] == pytest.approx(0.60, abs=0.01)
    assert split["y"].mean() == pytest.approx(0.2125, abs=0.001)  # five-star share of ML-100K
    assert split["accept_item"].max() <= 0.9 + 1e-12


def test_bootstrap_metrics_follow_the_released_code():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(float)
    z = rng.normal(size=500) + 1.5 * (y - 0.3)
    m = P.upstream_bootstrap_metrics(z, y)
    random.seed(0)  # first resample, as utils.generate_bootstrap_batch(0, n)
    ids = [random.randint(0, 499) for _ in range(400)]
    yb, zb = y[ids], z[ids]
    pb = 1 / (1 + np.exp(-zb))
    odds = y.sum() / (y == 0).sum()  # compute_empircal_cr: clicks / non-clicks (the odds), as released
    ap_mse = np.mean((yb - odds) ** 2)
    assert P.upstream_bootstrap_metrics(z, y, n_boot=1)["lift_mse"] == pytest.approx(100 * (ap_mse - np.mean((yb - pb) ** 2)) / ap_mse)
    assert m["lift_mse"] > m["lift_mse_rate"]  # the odds baseline is worse than the rate, so it inflates the lift
    assert 0.5 < m["auc"] < 1.0
