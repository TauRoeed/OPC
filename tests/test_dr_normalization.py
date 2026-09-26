"""The OPC training objective and the batch size (Optuna searches it).

With self-normalization per minibatch ('sndr', scope 'batch', the form used through 63a3cc1) the
objective itself changes with the batch size. The 'dr' loss (no self-normalization) and 'sndr' with
scope 'global' (the full-data mean weight, refreshed every epoch) are the same objective at every
batch size: averaged over any equal-size partition of the rows, the minibatch losses and gradients
equal the full-data ones."""

import numpy as np
import pytest
import torch

from models.custom_losses import DRPolicyLoss, SNDRPolicyLoss, dr_correction


def _data(seed=0, n=96, n_actions=25):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, n_actions, generator=g, dtype=torch.float64)
    scores = torch.rand(n, n_actions, generator=g, dtype=torch.float64) * 0.4
    pi0 = torch.softmax(torch.randn(n, n_actions, generator=g, dtype=torch.float64), dim=1)
    actions = torch.multinomial(pi0, 1, generator=g).squeeze(1)
    pscore = pi0[torch.arange(n), actions]
    rewards = torch.bernoulli(scores[torch.arange(n), actions].clamp(0.05, 0.95), generator=g)
    return logits, scores, actions, rewards, pscore


def _loss_and_grad(loss, logits, scores, actions, rewards, pscore, rows):
    lg = logits[rows].clone().requires_grad_(True)
    val = loss(pscore[rows], scores[rows], torch.softmax(lg, dim=1), rewards[rows], actions[rows])
    (grad,) = torch.autograd.grad(val, lg)
    return float(val), grad


def _partition_average(loss, data, batch):
    n = data[0].shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(1))
    vals, grads = [], torch.zeros_like(data[0])
    for s in range(0, n, batch):
        rows = perm[s : s + batch]
        v, g = _loss_and_grad(loss, *data, rows)
        vals.append(v)
        grads[rows] = g * (len(rows) / n)  # d(mean over batches)/d(logits) for equal batches
    return float(np.mean(vals)), grads * (n / batch) / (n / batch)


@pytest.mark.parametrize("weights", ["none", "shrink:100", "clip:10"])
@pytest.mark.parametrize("log_trick", [True, False])
def test_dr_loss_is_the_same_objective_at_every_batch_size(weights, log_trick):
    data = _data()
    loss = DRPolicyLoss(use_log_trick=log_trick, weights=weights)
    full_val, full_grad = _loss_and_grad(loss, *data, torch.arange(data[0].shape[0]))
    for batch in (8, 16, 32, 48):
        val, grad = _partition_average(loss, data, batch)
        assert val == pytest.approx(full_val, rel=1e-12, abs=1e-14)
        torch.testing.assert_close(grad * (data[0].shape[0] / batch) / (data[0].shape[0] / batch),
                                   full_grad * 1.0, rtol=1e-9, atol=1e-12)


def test_batch_normalized_sndr_is_not():
    data = _data()
    loss = SNDRPolicyLoss(use_log_trick=True, weights="none", normalization="batch")
    full_val, _ = _loss_and_grad(loss, *data, torch.arange(data[0].shape[0]))
    vals = {b: _partition_average(loss, data, b)[0] for b in (8, 16, 48)}
    assert all(abs(v - full_val) > 1e-6 for v in vals.values()), (full_val, vals)
    assert len({round(v, 9) for v in vals.values()}) == 3  # a different objective per batch size


@pytest.mark.parametrize("log_trick", [True, False])
def test_global_normalizer_is_full_batch_sndr_at_every_batch_size(log_trick):
    logits, scores, actions, rewards, pscore = data = _data(2)
    n = logits.shape[0]
    full = SNDRPolicyLoss(use_log_trick=log_trick, weights="shrink:100", normalization="batch")
    full_val, _ = _loss_and_grad(full, *data, torch.arange(n))  # batch = all rows: full-data SNDR
    glob = SNDRPolicyLoss(use_log_trick=log_trick, weights="shrink:100", normalization="global")
    with pytest.raises(RuntimeError, match="set_global_normalizer"):
        _loss_and_grad(glob, *data, torch.arange(8))
    pi_a = torch.softmax(logits, dim=1)[torch.arange(n), actions]
    iw, _ = glob._prepare_iw(pi_a, pscore)
    glob.set_global_normalizer(float(iw.mean()))
    for batch in (8, 24, 96):
        assert _partition_average(glob, data, batch)[0] == pytest.approx(full_val, rel=1e-12)
    with pytest.raises(ValueError):
        glob.set_global_normalizer(0.0)
    with pytest.raises(ValueError):
        SNDRPolicyLoss(normalization="per-row")


def test_dr_correction_modes():
    iw, r, q = torch.tensor([0.5, 2.0, 4.0]), torch.tensor([1.0, 0.0, 1.0]), torch.tensor([0.2, 0.3, 0.4])
    torch.testing.assert_close(dr_correction(iw, r, q, None), iw * (r - q))
    torch.testing.assert_close(dr_correction(iw, r, q, "none"), iw * (r - q))
    torch.testing.assert_close(dr_correction(iw, r, q), iw * (r - q) / iw.mean())  # the older default
    torch.testing.assert_close(dr_correction(iw, r, q, 2.0), iw * (r - q) / 2.0)


def test_training_refreshes_the_full_data_normalizer_each_epoch(monkeypatch):
    from torch.utils.data import DataLoader

    from models.models import CFModel, make_policy_transform
    from training import training_utils
    from utils.simulation_utils import CustomCFDatasetPS, collate_prebatched

    rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"  # train() expects the model on the GPU when there is one
    n_users, n_items, n = 40, 30, 300
    model = CFModel(n_users, n_items, 6, initial_user_embeddings=torch.tensor(rng.normal(size=(n_users, 6)), dtype=torch.float32),
                    initial_actions_embeddings=torch.tensor(rng.normal(size=(n_items, 6)), dtype=torch.float32),
                    user_transform=make_policy_transform("linear", 6), action_transform=make_policy_transform("linear", 6)).to(device)
    users, actions = rng.integers(0, n_users, n), rng.integers(0, n_items, n)
    ds = CustomCFDatasetPS(users, actions, rng.binomial(1, 0.2, n), rng.uniform(0.01, 0.2, n))
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=collate_prebatched)
    scores = torch.rand(n_users, n_items, device=device)
    loss = SNDRPolicyLoss(use_log_trick=True, weights="shrink:100", normalization="global")
    # the normalizer is the mean transformed weight of every training row under the current policy
    with torch.no_grad():
        pi = model(torch.as_tensor(users, device=device))[:, :, 0][torch.arange(n, device=device), torch.as_tensor(actions, device=device)]
        want = float(loss._prepare_iw(pi, ds.pscore.to(device))[0].mean())
    state = torch.get_rng_state()
    got = training_utils.full_data_mean_weight(model, ds, loss, device, cells=30 * 7)  # several chunks
    assert got == pytest.approx(want, rel=1e-6)
    assert torch.equal(torch.get_rng_state(), state)  # the training shuffle's stream is untouched
    seen = []
    real = loss.set_global_normalizer
    monkeypatch.setattr(loss, "set_global_normalizer", lambda v: seen.append(v) or real(v))
    training_utils.train(model, loader, scores, criterion=loss, num_epochs=3, lr=1e-2, device=device)
    assert len(seen) == 3 and seen[0] == pytest.approx(want, rel=1e-6) and seen[2] != seen[0]


def test_trainer_wiring_and_defaults(monkeypatch, capsys):
    import sys

    import training.trainer_trials as tt

    assert "dr" in tt.VALID_POLICY_LOSSES and tt.SN_SCOPES == ("batch", "global")
    dr = tt._policy_loss_from_name("dr", train_weights="shrink:100")
    assert isinstance(dr, DRPolicyLoss) and dr.normalization == "none"
    assert tt._policy_loss_from_name("sndr").normalization == "batch"  # the older loss, unchanged
    assert tt._policy_loss_from_name("sndr", sn_scope="global").normalization == "global"
    assert tt._policy_loss_from_name("kl", sn_scope="global").normalization == "global"
    with pytest.raises(ValueError):
        tt._policy_loss_from_name("sndr", sn_scope="epoch")
    for module in ("run_full_study", "run_full_study_parallel"):
        main = __import__(f"training.{module}", fromlist=["main"]).main
        monkeypatch.setattr(sys, "argv", [module, "--help"])
        with pytest.raises(SystemExit):
            main()
        text = " ".join(capsys.readouterr().out.split())
        assert "--sn-scope" in text and "default dr" in text


def test_study_runs_the_dr_and_global_losses(tmp_path):
    import pandas as pd
    from test_reproducibility import _toy_embeddings

    from training.run_full_study import _finalize_summary_df, _run_condition

    _toy_embeddings(tmp_path)
    kw = dict(dataset_name="toy", emb_dir=tmp_path, bias="medium", ctr=0.05, seed=0, train_sizes=[1000], n_trials=2,
              batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact",
              policy_reward_mc_sim=8, slim=True, shared_regression_size=2000, methods=("opc",))
    values = {}
    for name, losses, scope in (("dr", ("dr",), "batch"), ("global", ("sndr",), "global"), ("batch", ("sndr",), "batch")):
        run_dir = tmp_path / name
        run_dir.mkdir()
        opc, nop, _, _, meta = _run_condition(**kw, run_dir=run_dir, policy_loss_types=losses, sn_scope=scope)
        summary = _finalize_summary_df(opc, nop, meta)
        assert (summary["sn_scope"] == scope).all() and meta["policy_loss_types"] == list(losses)
        values[name] = pd.read_csv(run_dir / "trials_long.csv")["actual_reward"].to_numpy()
    # three different objectives (on this toy, global and batch SNDR differ only slightly)
    assert not np.array_equal(values["dr"], values["batch"]) and not np.array_equal(values["global"], values["batch"])
