"""One seed reproduces a study condition exactly; a different seed does not."""

import numpy as np
import pandas as pd

from training.run_full_study import _run_condition
from utils.seeding import derive_seed

N_USERS, N_ITEMS, DIM = 400, 600, 8  # enough items for a 5% logger next to a 30% best item


def test_derive_seed_is_stable():
    # Fixed values: a change here silently changes every derived RNG stream.
    assert derive_seed(0, "opc", 5000, "trial", 0) == derive_seed(0, "opc", 5000, "trial", 0)
    assert derive_seed(0, "opc", 5000, "trial", 0) != derive_seed(1, "opc", 5000, "trial", 0)
    assert derive_seed(0, "opc", 5000, "trial", 0) != derive_seed(0, "opc", 5000, "trial", 1)
    assert derive_seed(0, "optuna", "opc", 5000) == 1962295420
    assert derive_seed(7, "mlp_reward", "trial", 3) == 3432253218


def _toy_embeddings(tmp_path):
    rng = np.random.default_rng(0)
    np.save(tmp_path / "toy_user_factors.npy", rng.standard_normal((N_USERS, DIM)).astype(np.float32))
    np.save(tmp_path / "toy_item_factors.npy", rng.standard_normal((N_ITEMS, DIM)).astype(np.float32))


def _run(tmp_path, seed, tag):
    run_dir = tmp_path / tag
    run_dir.mkdir()
    opc_df, noprop_df, opc_trials, noprop_trials, _ = _run_condition(
        dataset_name="toy",
        emb_dir=tmp_path,
        bias="low",
        ctr=0.05,
        seed=seed,
        train_sizes=[1000],
        n_trials=2,
        batch_size=None,
        val_size=1000,
        val_frac=0.15,
        val_min=1000,
        val_max=None,
        policy_reward_mode="exact",
        policy_reward_mc_sim=8,
        run_dir=run_dir,
        slim=True,
        shared_regression_size=2000,
    )
    drop = [c for c in opc_trials.columns if "time" in c.lower() or c.startswith("datetime")]
    return (
        pd.concat([opc_df, noprop_df], ignore_index=True).drop(columns=["_learned_user_emb", "_learned_item_emb"], errors="ignore"),
        pd.concat([opc_trials, noprop_trials], ignore_index=True).drop(columns=drop, errors="ignore"),
    )


def test_same_seed_reproduces_condition(tmp_path):
    _toy_embeddings(tmp_path)
    summary_a, trials_a = _run(tmp_path, seed=0, tag="a")
    summary_b, trials_b = _run(tmp_path, seed=0, tag="b")
    summary_c, trials_c = _run(tmp_path, seed=1, tag="c")
    pd.testing.assert_frame_equal(summary_a, summary_b)
    pd.testing.assert_frame_equal(trials_a, trials_b)
    assert not summary_a["policy_rewards"].equals(summary_c["policy_rewards"])
