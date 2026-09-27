"""run_meta.json and the summaries record the condition's bias label, also when the opt-in baseline arms run.

Before the fix, the baseline-arm loop in _run_condition reused the name `label`, so runs with --methods dm or
tempered_logger recorded bias_label and noise_level as the last arm's name ("tempered_logger")."""

import pandas as pd

from training.run_full_study import _finalize_summary_df, _run_condition
from utils.representation_bias import bias_label, parse_bias


def test_bias_label_survives_the_baseline_arms(tmp_path):
    from test_reproducibility import _toy_embeddings

    _toy_embeddings(tmp_path)
    kw = dict(dataset_name="toy", emb_dir=tmp_path, bias="high/none/none", ctr=0.05, seed=0, train_sizes=[1000],
              n_trials=1, batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None,
              policy_reward_mode="exact", policy_reward_mc_sim=8, slim=True, shared_regression_size=2000,
              methods=("opc", "no_propensity", "dm", "tempered_logger"))
    opc, nop, _, _, meta, extra = _run_condition(**kw, run_dir=tmp_path, return_extra=True)
    want = bias_label(parse_bias("high/none/none"))  # 'w-high.g-none.v-none'
    assert meta["bias_label"] == want and meta["noise_level"] == want
    summary = _finalize_summary_df(opc, nop, meta, extra=extra)
    assert set(summary["method"]) == {"opc", "no_propensity", "dm", "tempered_logger"}
    assert (summary["noise_level"] == want).all()
    assert set(pd.read_csv(tmp_path / "trials_long.csv")["method"]) == {"opc", "no_propensity", "dm", "tempered_logger"}
