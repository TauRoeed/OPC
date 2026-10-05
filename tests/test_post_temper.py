"""--post-temper: after training, each trained policy's logits are scaled by the factor in
POST_TEMPER_GRID with the best selection score on validation (the arm's own score); the trial is the
tempered policy from then on. The ranking never changes. Off by default."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    from test_reproducibility import _toy_embeddings

    from training.run_full_study import _dataset_paths
    from utils.simulation_utils import generate_dataset

    root = tmp_path_factory.mktemp("toy_pt")
    _toy_embeddings(root)
    up, ip, _, _ = _dataset_paths(root, "toy")
    return root, generate_dataset({"bias": "medium", "ctr": 0.05}, seed=0, emb_x=np.load(up), emb_a=np.load(ip))


def test_post_temper_picks_the_best_scale_on_the_grid(toy):
    import training.trainer_trials as tt
    from scipy.stats import t as student_t

    _, ds = toy
    split = tt._build_regression_logged_split(ds, ds["our_x"], ds["our_a"], 2000, 2000, 0, split_seed=1, regression_size=2000)
    lk = tt._scores_lookup_from_bundle(tt.fit_shared_regression_bundle(ds, split["reg_data"], reward_model="regression"), "cpu")
    x, a = ds["our_x"], ds["our_a"]
    for mode, weights in (("logged", ("clip", 10.0)), ("logged", ("dm", 0.0)), ("uniform", ("none", np.inf))):
        got_x, s = tt._post_temper(split["val_data"], x, a, lk, ds, propensity_mode=mode, weights=weights)
        scores = {}
        for g in tt.POST_TEMPER_GRID:
            v, _, _ = tt._split_dr_vec_and_ess(split["val_data"], x * np.float32(g), a, lk, ds, propensity_mode=mode, weights=weights)
            scores[g] = v.mean() - student_t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        assert s == max(scores, key=scores.get) and s in tt.POST_TEMPER_GRID
        np.testing.assert_allclose(got_x, x * np.float32(s), rtol=1e-7)
    assert 1.0 in tt.POST_TEMPER_GRID  # the trained policy itself is always a candidate


def test_study_arms_use_the_tempered_policies(toy, tmp_path):
    from training.run_full_study import ALL_STUDY_METHODS, _finalize_summary_df, _run_condition
    from training.trainer_trials import POST_TEMPER_GRID

    root, _ = toy
    kw = dict(dataset_name="toy", emb_dir=root, bias="medium", ctr=0.05, seed=0, train_sizes=[1000], n_trials=3,
              batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact",
              policy_reward_mc_sim=8, slim=True, shared_regression_size=2000, return_extra=True,
              # every arm but BLOB, which needs --sampler random (tests/test_blob.py runs it end to end)
              methods=tuple(m for m in ALL_STUDY_METHODS if m != "blob"))
    runs = {}
    for on in (False, True):
        run_dir = tmp_path / f"pt{int(on)}"
        run_dir.mkdir()
        opc, nop, _, _, meta, extra = _run_condition(**kw, run_dir=run_dir, post_temper=on)
        assert meta["post_temper"] is on
        summary = _finalize_summary_df(opc, nop, meta, extra=extra)
        assert (summary["post_temper"] == on).all()
        runs[on] = pd.read_csv(run_dir / "trials_long.csv")
    off, on = runs[False], runs[True]
    assert "post_scale" not in off.columns
    trained = on[on.method != "tempered_logger"]
    assert trained["post_scale"].isin(POST_TEMPER_GRID).all()
    assert on[on.method == "tempered_logger"]["post_scale"].isna().all()  # no training, nothing to temper
    # the same training (same seeds): tempering only rescales, so the rankings (greedy values) agree
    k = ["method", "train_size", "trial_number"]
    m = off[k + ["actual_reward_greedy"]].merge(on[k + ["actual_reward_greedy", "post_scale"]], on=k)
    np.testing.assert_allclose(m["actual_reward_greedy_x"], m["actual_reward_greedy_y"], rtol=1e-12)
    same = on.set_index(k).loc[off.set_index(k).index]
    moved = same["post_scale"] != 1.0
    assert (np.isclose(same.loc[~moved, "actual_reward"], off.set_index(k).loc[~moved.values, "actual_reward"])).all()


@pytest.mark.parametrize("module", ["run_full_study", "run_full_study_parallel"])
def test_cli_has_post_temper(module, monkeypatch, capsys):
    import sys

    main = __import__(f"training.{module}", fromlist=["main"]).main
    monkeypatch.setattr(sys, "argv", [module, "--help"])
    with pytest.raises(SystemExit):
        main()
    assert "--post-temper" in capsys.readouterr().out
