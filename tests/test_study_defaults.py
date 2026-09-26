"""Study CLI defaults (2026-09-26): the reward model shares the policy's budget (--reward-data train,
cross-fitted by user in 5 folds) and the validation split is fixed at 20,000 rows. The older setting
stays one flag away: --reward-data external (cross-fitting then defaults to off), --val-size 0."""

import argparse
import sys

import numpy as np
import pandas as pd
import pytest

from training.run_full_study import (
    DEFAULT_CROSSFIT_FOLDS,
    DEFAULT_REWARD_DATA,
    DEFAULT_VAL_SIZE,
    _resolve_val_size_configs,
    _study_budget_from_args,
)


def test_constants():
    assert (DEFAULT_REWARD_DATA, DEFAULT_CROSSFIT_FOLDS, DEFAULT_VAL_SIZE) == ("train", 5, 20_000)


def test_budget_resolution():
    ns = lambda **kw: argparse.Namespace(**kw)
    assert _study_budget_from_args(ns(reward_data="train", crossfit_folds=None)) == ("train", 5)
    assert _study_budget_from_args(ns(reward_data="external", crossfit_folds=None)) == ("external", 0)
    assert _study_budget_from_args(ns(reward_data="train", crossfit_folds=3)) == ("train", 3)
    assert _study_budget_from_args(ns(reward_data="train", crossfit_folds=0)) == ("train", 0)


def test_validation_resolution():
    ns = lambda **kw: argparse.Namespace(**{"val_sizes": None, **kw})
    assert _resolve_val_size_configs(ns(val_size=20_000)) == [(20_000, "20000")]
    assert _resolve_val_size_configs(ns(val_size=0)) == [(None, "frac")]  # the older fraction rule
    assert _resolve_val_size_configs(ns(val_size=None)) == [(None, "frac")]
    assert _resolve_val_size_configs(ns(val_size=20_000, val_sizes=[1000, 3000])) == [(1000, "1000"), (3000, "3000")]


def test_sequential_runner_passes_the_budget_fair_defaults(tmp_path, monkeypatch):
    import training.run_full_study as rfs

    seen = []

    def stub(**kw):
        seen.append(kw)
        empty = pd.DataFrame()
        return empty, empty, empty, empty, {"dataset": kw["dataset_name"]}, {}

    monkeypatch.setattr(rfs, "_run_condition", stub)
    (tmp_path / "emb").mkdir()
    for side, n in (("user", 30), ("item", 20)):
        np.save(tmp_path / "emb" / f"toy_{side}_factors.npy", np.zeros((n, 4), dtype=np.float32))
    base = ["run_full_study", "--datasets", "toy", "--bias-configs", "low", "--seeds", "1", "--train-sizes", "1000",
            "--emb-dir", str(tmp_path / "emb"), "--out-dir", str(tmp_path / "out"), "--no-skip-completed"]
    for extra, want in (([], ("train", 5, 20_000, "__qhat=train__cf=5__val=20000")),
                        (["--reward-data", "external", "--val-size", "0"], ("external", 0, None, "__seed=1"))):
        seen.clear()
        monkeypatch.setattr(sys, "argv", base + ["--run-tag", f"t{len(extra)}"] + extra)
        rfs.main()
        (kw,) = seen
        assert (kw["reward_data"], kw["crossfit_folds"], kw["val_size"]) == want[:3]
        assert str(kw["run_dir"]).endswith(want[3])


def test_parallel_runner_configs_carry_the_defaults(tmp_path, monkeypatch, capsys):
    import training.run_full_study_parallel as par

    monkeypatch.setattr(sys, "argv", ["run_full_study_parallel", "--help"])
    with pytest.raises(SystemExit):
        par.main()
    text = " ".join(capsys.readouterr().out.split())
    assert "train (default" in text and "default 5 with train" in text and "default 20000" in text
