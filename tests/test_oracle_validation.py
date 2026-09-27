"""Validation of the Stage 1 oracle bounds (training/oracle_validation.py): the same class, objective and seeds,
a wider learning-rate range and a longer budget."""

import pandas as pd
import pytest

from training.oracle_repair import oracle_repair
from training.oracle_validation import _next_down, _next_up, main, validate_world
from training.run_full_study import build_condition_world
from utils.seeding import seed_everything

FAST = dict(steps=60, fit_users=600, batch_users=128)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from test_reproducibility import _toy_embeddings

    tmp = tmp_path_factory.mktemp("oracle_validation")
    _toy_embeddings(tmp)
    seed_everything(0)
    return tmp, build_condition_world("toy", tmp, "high/none/none", 0.05, 0)[0]


def test_half_decade_grid():
    assert [_next_up(x) for x in (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)] == [3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0]
    assert [_next_down(x) for x in (1e-2, 3e-3, 1e-3)] == [3e-3, 1e-3, 3e-4]


def test_candidates_reproduce_stage1_and_extend_past_the_edge(world):
    _, d = world
    stage1 = {"oracle_linear_lr": 1e-2, "oracle_linear+scale_lr": 1e-3}  # an upper and a lower edge won
    rows = pd.DataFrame(validate_world(d, stage1=stage1, seed=0, **FAST))
    lin, scl = rows[rows.cls == "linear"], rows[rows.cls == "linear+scale"]
    base = lin[lin.steps == 60]
    assert {1e-2, 3e-2, 0.1, 0.3} <= set(base.lr)  # the grid beyond the old top
    if base.loc[base.value.idxmax(), "lr"] == base.lr.max():
        assert base.lr.max() > 0.3  # the top won: extended further
    assert set(scl[scl.steps == 60].lr) == {1e-3, 3e-4}  # one point below the lower edge that won
    for cls, g in (("linear", lin), ("linear+scale", scl)):
        best = g[g.steps == 60].value.idxmax()
        assert (g.steps == 180).sum() == 1 and g.loc[g.steps == 180, "lr"].item() == g.loc[best, "lr"]  # budget x3 at the best
        assert (g.steps == 540).sum() <= 1  # x9 only when the value still moved
    # the Stage 1 winner is reproduced exactly: same class, objective, users and seed
    ref = oracle_repair(d, classes=("linear",), lrs=(1e-2,), seed=0, **FAST)
    got = lin[(lin.lr == 1e-2) & (lin.steps == 60)].iloc[0]
    assert got["value"] == ref["oracle_linear_value"] and got["greedy"] == ref["oracle_linear_greedy"]


def test_deterministic(world):
    _, d = world
    stage1 = {"oracle_linear_lr": 0.3, "oracle_linear+scale_lr": 3e-3}
    kw = dict(stage1=stage1, seed=0, linear_lrs=(0.3,), max_extensions=0, **FAST)
    a = pd.DataFrame(validate_world(d, **kw)).drop(columns="seconds")
    b = pd.DataFrame(validate_world(d, **kw)).drop(columns="seconds")
    pd.testing.assert_frame_equal(a, b)
    assert set(a[a.cls == "linear+scale"].lr) == {3e-3}  # an interior Stage 1 winner: nothing added


def test_cli_writes_candidates_and_skips_done(world, tmp_path):
    from training.oracle_repair import main as oracle_main

    tmp, _ = world
    common = ["--datasets", "toy", "--bias-configs", "high/none/none", "--seeds", "0", "--steps", "40",
              "--fit-users", "400", "--batch-users", "128", "--emb-dir", str(tmp)]
    oracle_main(common + ["--classes", "linear", "linear+scale", "--lrs", "0.003", "0.01", "--out", str(tmp_path / "s1")])
    argv = common + ["--stage1", str(tmp_path / "s1"), "--out", str(tmp_path / "v")]
    main(argv)
    rows = pd.read_csv(tmp_path / "v" / "oracle_candidates.csv")
    assert set(rows.cls) == {"linear", "linear+scale"} and (rows.dataset == "toy").all()
    assert {"lr", "steps", "value", "greedy", "logit_scale", "stage1_lr", "trace_last"} <= set(rows.columns)
    main(argv)  # already done: nothing appended
    assert len(pd.read_csv(tmp_path / "v" / "oracle_candidates.csv")) == len(rows)
    assert (tmp_path / "v" / "validation_settings.json").exists()
