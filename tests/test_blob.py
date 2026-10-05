"""models/blob.py: the batched BLOB layer equals separate single-model runs, the point prediction, and the released
initialization."""
import numpy as np
import pytest
import torch

from models.blob import (BlobBandit, BlobBanditBatch, BlobPriors, TFAdam, TFAdamBatch, fit_blob_batch, prepare_psi,
                         tf_param_order)

P, K, N = 15, 5, 300


def _data(seed=0):
    rng = np.random.default_rng(seed)
    psi = rng.random((P, K)).astype(np.float32)
    x = rng.standard_normal((N, K)).astype(np.float32)
    a = rng.integers(0, P, N)
    y = (rng.random(N) < 1 / (1 + np.exp(-(x * psi[a]).sum(1) + 1))).astype(np.float32)
    return psi, x, a, y


PRIORS = [BlobPriors(), BlobPriors(wb_m=0.0, kappa_s=0.5), BlobPriors(wa_m=2.0, wc_m=-1.0, kappa_s=0.1)]
LRS = [1e-3, 3e-3, 1e-2]


@pytest.mark.parametrize("family", ["mnq", "nq"])
def test_batched_trials_equal_separate_runs(family):
    psi, x, a, y = _data()
    batch = BlobBanditBatch(psi, PRIORS, family=family)
    singles = [BlobBandit(psi, family=family, priors=p) for p in PRIORS]
    opt_b = TFAdamBatch(batch.params_in_order(), LRS)
    opts = [TFAdam(tf_param_order(m), lr=lr) for m, lr in zip(singles, LRS)]
    xt, at, yt = torch.as_tensor(x), torch.as_tensor(a), torch.as_tensor(y)
    gen = torch.Generator().manual_seed(3)
    for _ in range(3):
        order = torch.randperm(N, generator=gen)
        for s in range(0, N, 64):
            idx = order[s:s + 64]
            noise = torch.randn(len(PRIORS), len(idx), 4, generator=gen)
            loss = batch.neg_elbo(xt[idx], at[idx], yt[idx], N, noise)
            opt_b.zero_grad()
            loss.sum().backward()
            opt_b.step()
            for t, (m, o) in enumerate(zip(singles, opts)):
                nz = noise[t]
                lt, _, _ = m.neg_elbo(xt[idx], at[idx], yt[idx], N, {"wa": nz[:, 0:1], "wb": nz[:, 1:2],
                                                                     "band": nz[:, 2:3], "bias": nz[:, 3:4]})
                torch.testing.assert_close(loss[t], lt, rtol=1e-5, atol=1e-6)
                o.zero_grad()
                lt.backward()
                o.step()
    for t, m in enumerate(singles):
        beta_b, kappa_b, wc_b = batch.point(t)
        beta_s, kappa_s = m.point_beta()
        torch.testing.assert_close(beta_b, beta_s, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(kappa_b, kappa_s, rtol=1e-4, atol=1e-6)
        assert wc_b == pytest.approx(float(m.bias_means.detach()), rel=1e-5)


def test_released_initialization_is_the_prior():
    psi, *_ = _data()
    m = BlobBandit(psi, family="mnq", priors=BlobPriors(kappa_s=0.2))
    assert float(m.wa_means) == -1.0 and float(m.wb_means) == -6.0 and float(m.bias_means) == -4.5
    assert torch.allclose(torch.exp(m.kappa_logstd), torch.full((P, 1), 0.2))
    assert float(torch.exp(m.wb_logstd)) == pytest.approx(1.0)  # wb's prior std is wa_s in the release
    assert not m.zeta_means.any() and float(m.kl()) == pytest.approx(0.0, abs=1e-4)  # posterior = prior


def test_point_prediction_is_omega_beta_plus_kappa():
    psi, x, *_ = _data()
    m = BlobBandit(psi, family="nq")
    with torch.no_grad():
        m.zeta_means.normal_()
        m.kappa_means.normal_()
        m.wb_means.fill_(0.5)
    beta, kappa = m.point_beta()
    loc, cov, chol = (torch.as_tensor(v) for v in prepare_psi(psi))
    zeta = m.zeta_means.detach().reshape(K, K)
    expect = torch.nn.functional.softplus(m.wa_means.detach()) * loc + \
        torch.nn.functional.softplus(m.wb_means.detach()) * cov @ zeta @ chol.T
    torch.testing.assert_close(beta, expect)
    xt = torch.as_tensor(x[:7])
    torch.testing.assert_close(m.predict_logits(xt), xt @ expect.T + m.kappa_means.detach()[:, 0])


def test_fit_blob_batch_learns_and_freezes_a_diverging_trial():
    psi, x, a, y = _data(1)
    model = BlobBanditBatch(psi, [BlobPriors(), BlobPriors(wb_m=0.0)], family="nq")
    before = model.wb_means.detach().clone()
    info = fit_blob_batch(model, x, a, y, epochs=4, lrs=[1e-2, 1e3], batch_size=64, order_seed=0)
    assert info["steps"] == 4 * int(np.ceil(N / 64))
    assert info["finite"][0] and not torch.equal(model.wb_means.detach()[0], before[0])
    assert np.isfinite(model.wa_means.detach()[0].numpy()).all()


@pytest.fixture(scope="module")
def blob_runs(tmp_path_factory):
    from test_reproducibility import _toy_embeddings
    from training.run_full_study import _run_condition

    root = tmp_path_factory.mktemp("blob_arm")
    _toy_embeddings(root)
    out = []
    for rep in range(2):  # twice: the arm is deterministic
        run_dir = root / f"run{rep}"
        run_dir.mkdir()
        options = {"families": ["nq", "mnq"], "n_trials": 4, "epochs": [2, 5], "wb_m": [-6.0, 0.0],
                   "kappa_s": [0.01, 0.5], "batch_size": 128, "temper": True}
        out.append(_run_condition(dataset_name="toy", emb_dir=root, bias="medium", ctr=0.05, seed=0, train_sizes=[1000],
                                  n_trials=2, batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None,
                                  policy_reward_mode="exact", policy_reward_mc_sim=8, run_dir=run_dir, slim=True,
                                  shared_regression_size=2000, methods=("blob",), sampler="random", return_extra=True,
                                  blob_options=options)[5])
    return out


def test_blob_arm_end_to_end(blob_runs):
    extra = blob_runs[0]
    assert set(extra) == {"blob_nq", "blob_mnq"}
    for label, (summary, trials) in extra.items():
        row = summary.loc[1000]
        assert row["n_trials"] == 4 == len(trials) and row["n_total"] == 1000
        assert set(trials["kappa_s"]) <= {0.01, 0.5} and set(trials["wb_m"]) <= {-6.0, 0.0}
        finite = trials[trials["val_nll"] < 1e9]
        assert int(row["selected_trial"]) == int(finite.loc[finite["val_nll"].idxmin(), "trial"])
        for col in ("policy_rewards", "policy_rewards_greedy", "policy_rewards_tempered", "val_nll", "val_dr_greedy",
                    "sp_wa", "temper_scale"):
            assert np.isfinite(row[col]), col
        assert row["oracle_selected_value_greedy"] >= row["policy_rewards_greedy"] - 1e-12
        assert row["val_dr_tempered_low"] <= row["val_dr_tempered"]


def test_blob_arm_is_deterministic(blob_runs):
    import pandas as pd

    for label in blob_runs[0]:
        pd.testing.assert_frame_equal(blob_runs[0][label][0], blob_runs[1][label][0])
        drop = ["seconds"]
        pd.testing.assert_frame_equal(blob_runs[0][label][1].drop(columns=drop), blob_runs[1][label][1].drop(columns=drop))


def test_supplied_source_has_unit_rms_users():
    from training.blob_trials import supplied_source

    rng = np.random.default_rng(0)
    ds = {"our_x": 3.0 * rng.standard_normal((50, 6)), "our_a": rng.standard_normal((20, 6))}
    omega, psi = supplied_source(ds)
    assert np.sqrt(np.mean(omega.astype(np.float64) ** 2)) == pytest.approx(1.0, rel=1e-5)
    np.testing.assert_allclose(psi, ds["our_a"].astype(np.float32))
    np.testing.assert_allclose(omega * np.sqrt(np.mean(ds["our_x"] ** 2)), ds["our_x"], rtol=1e-5)
