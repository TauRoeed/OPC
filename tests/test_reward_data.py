"""--reward-data: the regression reward model from the separate reg slice (external, default) or
from each train size's own training rows (train), shared by every arm; the splits stay the same."""

import numpy as np
import pandas as pd
import pytest


def _kw(tmp_path):
    from test_reproducibility import _toy_embeddings

    _toy_embeddings(tmp_path)
    return dict(dataset_name="toy", emb_dir=tmp_path, bias="medium", ctr=0.05, seed=0, train_sizes=[1000, 3000],
                n_trials=2, batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None,
                policy_reward_mode="exact", policy_reward_mc_sim=8, slim=True, shared_regression_size=2000)


def test_run_key_marks_the_train_mode():
    from training.run_full_study import REWARD_DATA_MODES, _condition_run_key

    assert REWARD_DATA_MODES == ("external", "train")
    ext = _condition_run_key("ml", "medium", 0.05, 1, {})
    assert ext == _condition_run_key("ml", "medium", 0.05, 1, {}, reward_data="external")
    assert _condition_run_key("ml", "medium", 0.05, 1, {}, reward_data="train") == ext + "__qhat=train"
    assert _condition_run_key("ml", "medium", 0.05, 1, {}, "5000", reward_data="train").endswith("__qhat=train__val=5000")


def test_train_mode_fits_one_reward_model_per_size_on_its_training_rows(tmp_path, monkeypatch):
    import training.run_full_study as rfs

    calls = []
    real = rfs.fit_shared_regression_bundle

    def spy(dataset, reg_data, **kw):
        calls.append(len(reg_data["r"]))
        return real(dataset, reg_data, **kw)

    monkeypatch.setattr(rfs, "fit_shared_regression_bundle", spy)
    kw = _kw(tmp_path)
    runs = {}
    for mode in ("external", "train"):
        calls.clear()
        run_dir = tmp_path / mode
        run_dir.mkdir()
        opc_df, noprop_df, _, _, meta, extra = rfs._run_condition(**kw, run_dir=run_dir, reward_data=mode,
                                                                  methods=("opc", "dm"), return_extra=True)
        assert meta["reward_data"] == mode
        assert (rfs._finalize_summary_df(opc_df, noprop_df, meta, extra=extra)["reward_data"] == mode).all()
        # the external slice is always fit; train mode adds one model per train size, on its own rows
        assert calls == ([2000] if mode == "external" else [2000, 1000, 3000]), calls
        runs[mode] = pd.read_csv(run_dir / "trials_long.csv")
    # same splits, same logger: the logger row agrees; the reward model (hence DM's scores) does not
    ext, tr = runs["external"], runs["train"]
    assert ext["initial_reward"].iloc[0] == tr["initial_reward"].iloc[0]
    dm_e, dm_t = ext[ext.method == "dm"], tr[tr.method == "dm"]
    assert not np.allclose(dm_e["r_hat"].to_numpy(), dm_t["r_hat"].to_numpy())


def test_trainer_scores_with_the_size_bundle(tmp_path):
    """With q_hat = 0.07 everywhere for one size, DM-only's selection score is 0.07 for every trial."""
    from test_baselines import _trainer

    from training.run_full_study import _dataset_paths
    from training.trainer_trials import fit_shared_regression_bundle
    from utils.simulation_utils import generate_dataset

    from test_reproducibility import _toy_embeddings

    _toy_embeddings(tmp_path)
    up, ip, _, _ = _dataset_paths(tmp_path, "toy")
    ds = generate_dataset({"bias": "medium", "ctr": 0.05}, seed=0, emb_x=np.load(up), emb_a=np.load(ip))
    const = fit_shared_regression_bundle(ds, {}, reward_model="oracle", q_error=1.0, q_bad_value=0.07, materialize_qhat="never")
    _, trials = _trainer(ds, tmp_path, method_label="dm", policy_loss_types=("dm",), select_estimator="dm",
                         size_regression_bundles={1000: const})
    np.testing.assert_allclose(trials["r_hat"], 0.07, rtol=1e-5)


def test_train_mode_rejects_reward_models_without_data(tmp_path):
    from training.run_full_study import _run_condition

    kw = _kw(tmp_path)
    with pytest.raises(ValueError, match="reward_data"):
        _run_condition(**kw, run_dir=tmp_path, reward_data="sideways")
    with pytest.raises(ValueError, match="regression"):
        _run_condition(**kw, run_dir=tmp_path, reward_data="train", reward_model="oracle")


@pytest.mark.parametrize("module", ["run_full_study", "run_full_study_parallel"])
def test_cli_has_reward_data(module, monkeypatch, capsys):
    import sys

    main = __import__(f"training.{module}", fromlist=["main"]).main
    monkeypatch.setattr(sys, "argv", [module, "--help"])
    with pytest.raises(SystemExit):
        main()
    assert "--reward-data" in capsys.readouterr().out
