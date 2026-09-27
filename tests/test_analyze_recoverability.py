"""Recoverability tables (training/analyze_recoverability.py) on hand-made oracle rows."""

import numpy as np
import pandas as pd
import pytest

from training.analyze_recoverability import MIN_LOSS, derive, learned_recovery, summary


def _rows():
    base = dict(dataset="ml", seed=0, logger_ceiling=0.30)
    rows = []
    for bias, lv, lg, ov, og in (("none", 0.24, 0.30, 0.29, 0.30), ("high/none/none", 0.18, 0.22, 0.30, 0.29),
                                 ("none/none/high", 0.17, 0.21, 0.20, 0.24)):
        r = dict(base, bias=bias, logger_value=lv, logger_greedy=lg)
        for cls in ("linear", "linear+scale", "scale"):
            r[f"oracle_{cls}_value"], r[f"oracle_{cls}_greedy"] = ov, (lg if cls == "scale" else og)
        rows.append(r)
    return pd.DataFrame(rows)


def test_derived_quantities():
    df = derive(_rows()).set_index("bias")
    warp = df.loc["w-high.g-none.v-none"]
    assert warp["V_clean"] == pytest.approx(0.24) and warp["V_clean_greedy"] == pytest.approx(0.30)
    assert warp["representation_loss"] == pytest.approx(0.06) and warp["representation_loss_greedy"] == pytest.approx(0.08)
    assert warp["gain_linear+scale_greedy"] == pytest.approx(0.07)
    assert warp["recoverability_linear+scale_greedy"] == pytest.approx(0.875)
    assert warp["recoverability_linear+scale"] == pytest.approx(0.12 / 0.06)  # above 1: not clipped
    none = df.loc["none"]
    assert none["representation_loss_greedy"] == 0 and np.isnan(none["recoverability_linear+scale_greedy"])  # no ratio
    assert df.loc["w-none.g-none.v-high", "recoverability_linear+scale_greedy"] == pytest.approx(0.03 / 0.09)
    assert df.loc["w-high.g-none.v-none", "recoverability_scale_greedy"] == pytest.approx(0.0)  # sharpening keeps the ranking
    # the learner's class takes the better of the two fits (scale fixed, scale learned)
    assert warp["oracle_repair_value"] == pytest.approx(0.30) and warp["recoverability_repair_greedy"] == pytest.approx(0.875)
    assert MIN_LOSS > 0


def test_summary_and_learned_join():
    df = derive(_rows())
    s = summary(df)
    assert list(s.index) == ["none", "w-high.g-none.v-none", "w-none.g-none.v-high"]
    assert s.loc["w-high.g-none.v-none", "gain greedy %"] == pytest.approx(7.0)
    assert s.loc["w-high.g-none.v-none", "bias type"] == "warp only (high)"
    learned = pd.DataFrame([dict(dataset="ml", bias="high/none/none", seed=0, method="opc", train_size=5000,
                                 V_method=0.24, V_method_greedy=0.255)])
    m = learned_recovery(learned, _rows()).iloc[0]
    assert m["learned_gain"] == pytest.approx(0.06) and m["fraction_of_oracle_repair"] == pytest.approx(0.06 / 0.12)
    assert m["fraction_of_oracle_repair_greedy"] == pytest.approx(0.035 / 0.07)


def test_load_learned_reads_a_study_condition(tmp_path):
    """A toy study condition with all four arms (paired random sampler) and a toy oracle row join into
    per-arm recovery rows."""
    from test_reproducibility import _toy_embeddings

    from training.analyze_recoverability import load_learned
    from training.run_full_study import _condition_run_key, _finalize_summary_df, _run_condition

    _toy_embeddings(tmp_path)
    run = tmp_path / "run_toy"
    from utils.representation_bias import resolve_bias_configs

    (bias,) = resolve_bias_configs(["high/none/none"])  # canonical, as the runners name folders
    cond = run / _condition_run_key("toy", bias, 0.05, 0, {}, "1000", reward_data="train", crossfit_folds=5)
    cond.mkdir(parents=True)
    opc, nop, _, _, meta, extra = _run_condition(
        dataset_name="toy", emb_dir=tmp_path, bias=bias, ctr=0.05, seed=0, train_sizes=[1000], n_trials=2,
        batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact",
        policy_reward_mc_sim=8, slim=True, shared_regression_size=2000, run_dir=cond, return_extra=True,
        methods=("opc", "no_propensity", "dm", "tempered_logger"), sampler="random", reward_data="train",
        crossfit_folds=5, learn_logit_scale=True)
    _finalize_summary_df(opc, nop, meta, extra=extra).to_csv(cond / "summary_metrics.csv", index=False)
    rows = load_learned(run)
    assert set(rows["method"]) == {"opc", "no_propensity", "dm", "tempered_logger"} and (rows["n_trials"] == 2).all()
    assert (rows["bias"] == "w-high.g-none.v-none").all() and (rows["regret"] >= 0).all()
    assert rows["V_tempered"].nunique() == 1 and rows[["V_method", "V_method_greedy", "ess_raw"]].notna().all().all()
    oracle = pd.DataFrame([dict(dataset="toy", bias="w-high.g-none.v-none", seed=0, logger_value=0.1, logger_greedy=0.12,
                                logger_ceiling=0.3, **{f"oracle_{c}_{k}": v for c in ("linear", "linear+scale", "scale")
                                                       for k, v in (("value", 0.25), ("greedy", 0.27))}),
                           dict(dataset="toy", bias="none", seed=0, logger_value=0.24, logger_greedy=0.3, logger_ceiling=0.3,
                                **{f"oracle_{c}_{k}": 0.3 for c in ("linear", "linear+scale", "scale") for k in ("value", "greedy")})])
    m = learned_recovery(rows, oracle)
    assert m["fraction_of_oracle_repair"].notna().all()
    np.testing.assert_allclose(m["learned_gain"], m["V_method"] - 0.1)
