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


def _run(tmp_path, seed, tag, world_options=None, with_meta=False):
    run_dir = tmp_path / tag
    run_dir.mkdir()
    opc_df, noprop_df, opc_trials, noprop_trials, meta = _run_condition(
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
        world_options=world_options,
    )
    assert meta["bpr"]["status"] == "missing"  # toy embeddings have no BPR meta file
    drop = [c for c in opc_trials.columns if "time" in c.lower() or c.startswith("datetime")]
    frames = (
        pd.concat([opc_df.reset_index(), noprop_df.reset_index()], ignore_index=True)
        .drop(columns=["_learned_user_emb", "_learned_item_emb"], errors="ignore"),
        pd.concat([opc_trials, noprop_trials], ignore_index=True).drop(columns=drop, errors="ignore"),
    )
    return (*frames, meta) if with_meta else frames


def test_same_seed_reproduces_condition(tmp_path):
    _toy_embeddings(tmp_path)
    summary_a, trials_a = _run(tmp_path, seed=0, tag="a")
    summary_b, trials_b = _run(tmp_path, seed=0, tag="b")
    summary_c, trials_c = _run(tmp_path, seed=1, tag="c")
    pd.testing.assert_frame_equal(summary_a, summary_b)
    pd.testing.assert_frame_equal(trials_a, trials_b)
    assert not summary_a["policy_rewards"].equals(summary_c["policy_rewards"])


def test_popularity_world_reproduces_and_learns_its_weight(tmp_path):
    _toy_embeddings(tmp_path)
    b = 1.5 * np.random.default_rng(1).standard_normal(N_ITEMS)
    np.save(tmp_path / "toy_item_bias.npy", b.astype(np.float32))
    world = {"pop_strength": 1.0, "logger_pop_strength": 2.0}  # the logger over-weighs popularity
    summary_a, trials_a, meta = _run(tmp_path, seed=0, tag="a", world_options=world, with_meta=True)
    summary_b, trials_b = _run(tmp_path, seed=0, tag="b", world_options=world)
    pd.testing.assert_frame_equal(summary_a, summary_b)
    pd.testing.assert_frame_equal(trials_a, trials_b)
    assert (meta["world"]["pop_strength"], meta["world"]["logger_pop_strength"]) == (1.0, 2.0)
    assert meta["world"]["popularity"]["item_bias"]
    # baseline rows carry the logger's weight; trained policies moved away from it
    base = summary_a[summary_a["index"] == 0]["pop_weight"]
    np.testing.assert_allclose(base, 2.0, rtol=1e-6)
    learned = summary_a[summary_a["index"] > 0]["pop_weight"]
    assert len(learned) == 2 and np.all(np.abs(learned - 2.0) > 1e-4), learned
    assert trials_a["pop_weight"].notna().all() and np.all(np.abs(trials_a["pop_weight"] - 2.0) > 1e-4)
    # the default world ignores the item bias file and has no popularity columns
    summary_c, trials_c = _run(tmp_path, seed=0, tag="c")
    assert "pop_weight" not in summary_c.columns and "pop_weight" not in trials_c.columns


def test_reward_features_reach_the_reward_model(tmp_path, monkeypatch):
    import training.run_full_study as rfs

    _toy_embeddings(tmp_path)
    seen = []
    fit = rfs.fit_shared_regression_bundle

    def spy(*args, **kwargs):
        bundle = fit(*args, **kwargs)
        seen.append(bundle["regression_model"].features)
        return bundle

    monkeypatch.setattr(rfs, "fit_shared_regression_bundle", spy)
    runs = {}
    for features in ("concat", "interaction"):
        run_dir = tmp_path / features
        run_dir.mkdir()
        *_, meta = _run_condition(dataset_name="toy", emb_dir=tmp_path, bias="low", ctr=0.05, seed=0, train_sizes=[1000],
                                  n_trials=2, batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None,
                                  policy_reward_mode="exact", policy_reward_mc_sim=8, run_dir=run_dir, slim=True,
                                  shared_regression_size=2000, reward_features=features)
        assert meta["reward_features"] == features
        runs[features] = pd.read_csv(run_dir / "trials_long.csv")
    assert seen == ["concat", "interaction"]
    assert not runs["concat"]["r_hat"].equals(runs["interaction"]["r_hat"])
    assert rfs._finalize_summary_df(pd.DataFrame({"x": [1.0]}, index=[0]), None, {"ctr": 0.05, "reward_features": "concat"})["reward_features"].iloc[0] == "concat"
