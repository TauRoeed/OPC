"""The CausE arm end to end on the toy world (training/cause_trials.py through _run_condition)."""
import numpy as np
import pandas as pd
import pytest

from test_reproducibility import _toy_embeddings
from training.cause_trials import cause_method_label
from training.run_full_study import _run_condition, _summary_has_methods
from training.trainer_trials import LOGGED_RUN_IDX, LazyRegressionSplitCache
from utils.budget_split import budget_counts

N = 1000
RHOS = (0.0, 0.25)
OPTIONS = {"rhos": list(RHOS), "variants": ["prod", "avg"], "dim": 8, "optimizer": "momentum_decay",
           "tie": "one_way", "batch_size": 128, "n_trials": 3}


def _run(tmp_path, tag, methods, cause_options=OPTIONS, seed=0):
    run_dir = tmp_path / tag
    run_dir.mkdir()
    return _run_condition(dataset_name="toy", emb_dir=tmp_path, bias="medium", ctr=0.05, seed=seed, train_sizes=[N],
                          n_trials=2, batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None,
                          policy_reward_mode="exact", policy_reward_mc_sim=8, run_dir=run_dir, slim=True,
                          shared_regression_size=2000, methods=methods, sampler="random", return_extra=True,
                          cause_options=cause_options)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("cause")
    _toy_embeddings(root)
    with_cause = _run(root, "with", ("opc", "cause"))
    without = _run(root, "without", ("opc",))
    again = _run(root, "again", ("opc", "cause"))
    return root, with_cause, without, again


def test_every_prediction_and_rho_is_reported(runs):
    _root, (_opc, _np, _ot, _nt, meta, extra), *_ = runs
    expected = {cause_method_label(p, r) for p in ("prod_c", "prod_t", "avg") for r in RHOS}
    assert set(extra) == expected
    assert meta["cause"]["variants"] == ["prod", "avg"] and meta["cause"]["dim"] == 8 and meta["cause"]["n_trials"] == 3
    for label in expected:
        summary, trials = extra[label]
        row = summary.loc[N]
        n_c, n_t = budget_counts(N, row["cause_rho"])
        assert (row["n_control"], row["n_treatment"], row["n_total"]) == (n_c, n_t, N)
        assert row["control_policy"] == "warm_logger" and row["treatment_policy"] == "uniform"
        assert row["cause_dim"] == 8 and row["cause_tie"] == "one_way" and row["stage"] == "development"
        assert row["n_trials"] == 3 and len(trials) == 3
        assert np.isfinite(row["policy_rewards"]) and np.isfinite(row["policy_rewards_greedy"])
        assert row["oracle_selected_value_greedy"] >= row["policy_rewards_greedy"] - 1e-12
        assert row["selected_trial"] in set(trials["trial"])


def test_cause_sees_a_prefix_of_the_training_rows_opc_trains_on(runs):
    """Same world, same split: the warm rows CausE gets are the first rows of OPC's training split."""
    root, (_opc, _np, _ot, _nt, meta, extra), *_ = runs
    from training.run_full_study import build_condition_world

    dataset, *_ = build_condition_world("toy", root, "medium", 0.05, 0, world_options={})
    cache = LazyRegressionSplitCache(dataset, [N], val_size=1000, val_min=1000, condition_seed=0, regression_size=2000)
    warm = cache[(N, LOGGED_RUN_IDX)]["train_data"]
    for r in RHOS:
        row = extra[cause_method_label("prod_c", r)][0].loc[N]
        assert row["control_reward_sum"] == pytest.approx(float(np.sum(warm["r"][: int(row["n_control"])])))
        assert row["opc_collection_reward_sum"] == pytest.approx(float(np.sum(warm["r"])))
        assert row["replaced_warm_reward_sum"] + row["control_reward_sum"] == pytest.approx(float(np.sum(warm["r"])))
    zero = extra[cause_method_label("avg", 0.0)][0].loc[N]
    assert zero["n_treatment"] == 0 and zero["exploration_cost_expected"] == 0.0


def test_prod_c_and_prod_t_share_their_trained_models(runs):
    _root, (_opc, _np, _ot, _nt, _meta, extra), *_ = runs
    c = extra[cause_method_label("prod_c", 0.25)][1]
    t = extra[cause_method_label("prod_t", 0.25)][1]
    cols = ["trial", "lr", "epochs", "l2_pen", "cf_pen", "alpha", "prod_c_val_nll", "prod_t_val_nll"]
    pd.testing.assert_frame_equal(c[cols].reset_index(drop=True), t[cols].reset_index(drop=True))


def test_configurations_do_not_depend_on_rho(runs):
    _root, (_opc, _np, _ot, _nt, _meta, extra), *_ = runs
    a = extra[cause_method_label("avg", 0.0)][1][["lr", "epochs", "l2_pen", "cf_pen"]]
    b = extra[cause_method_label("avg", 0.25)][1][["lr", "epochs", "l2_pen", "cf_pen"]]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))  # random search: paired


def test_opc_outputs_do_not_change_when_cause_runs(runs):
    _root, with_cause, without, _again = runs
    pd.testing.assert_frame_equal(with_cause[0].drop(columns=["_learned_user_emb", "_learned_item_emb"], errors="ignore"),
                                  without[0].drop(columns=["_learned_user_emb", "_learned_item_emb"], errors="ignore"))
    drop = [c for c in with_cause[2].columns if "time" in c.lower() or c.startswith("datetime")]
    pd.testing.assert_frame_equal(with_cause[2].drop(columns=drop), without[2].drop(columns=drop))
    assert without[4]["cause"] is None


def test_cause_is_deterministic(runs):
    _root, with_cause, _without, again = runs
    for label, (summary, trials) in with_cause[5].items():
        pd.testing.assert_frame_equal(summary, again[5][label][0])
        drop = ["seconds"]
        pd.testing.assert_frame_equal(trials.drop(columns=drop), again[5][label][1].drop(columns=drop))


def test_skip_completed_recognises_cause_labels(tmp_path):
    path = tmp_path / "summary_metrics.csv"
    pd.DataFrame({"method": ["opc", "cause_prod_c_r000"]}).to_csv(path, index=False)
    assert _summary_has_methods(path, ("opc", "cause"))
    pd.DataFrame({"method": ["opc"]}).to_csv(path, index=False)
    assert not _summary_has_methods(path, ("opc", "cause"))


def test_a_cause_only_run_gives_the_same_cause_results(runs, tmp_path):
    """The split M5 layout: CausE alone (no q_hat fits) reproduces the CausE rows of the combined run."""
    root, with_cause, *_ = runs
    alone = _run(root, "alone", ("cause",))
    assert set(alone[5]) == set(with_cause[5])
    for label, (summary, trials) in with_cause[5].items():
        pd.testing.assert_frame_equal(summary, alone[5][label][0])
        pd.testing.assert_frame_equal(trials.drop(columns=["seconds"]), alone[5][label][1].drop(columns=["seconds"]))


def test_cause_trains_on_the_cpu_when_asked(runs):
    root, *_ = runs
    out = _run(root, "cpu", ("cause",), cause_options={**OPTIONS, "rhos": [0.25], "variants": ["avg"], "device": "cpu"})
    (summary, _trials), = out[5].values()
    assert (summary["cause_train_device"] == "cpu").all() and out[4]["cause"]["device"] == "cpu"
