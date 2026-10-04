"""The OPC training objective and the batch size (Optuna searches it).

With self-normalization per minibatch (legacy 'sndr', scope 'batch') the objective itself changes
with the batch size. 'dr' (no self-normalization) is a mean of per-row terms, and so is 'sndr' with
scope 'global' while its normalizer is held fixed: with the trainer's minibatch weighting
(``training_utils.minibatch_loss``), their minibatch losses and gradients summed over any partition
of the rows, the short final batch included, equal the full-data ones. 'global' is not exact SNDR:
its normalizer is refreshed once per epoch and carries no gradient (tests/test_objective_gradients.py
compares every variant with the literal SNDR ratio)."""

import numpy as np
import pytest
import torch

from models.custom_losses import DRPolicyLoss, SNDRPolicyLoss, dr_correction
from training.training_utils import minibatch_loss


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
    return float(val.detach()), grad


def _epoch(loss, data, batch):
    """One pass over a random partition of the rows into minibatches of ``batch`` (the last one short
    when ``batch`` does not divide n), each weighted as the trainer does (``minibatch_loss``): the summed
    losses and per-row logit gradients, times b / n. For a per-row loss this is the full-data mean."""
    logits, scores, actions, rewards, pscore = data
    n = logits.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(1))
    total, grads = 0.0, torch.zeros_like(logits)
    for s in range(0, n, batch):
        rows = perm[s : s + batch]
        lg = logits[rows].clone().requires_grad_(True)
        val = minibatch_loss(loss, pscore[rows], scores[rows], torch.softmax(lg, dim=1), rewards[rows], actions[rows], batch)
        (grads[rows],) = torch.autograd.grad(val, lg)
        total += float(val.detach())
    return total * batch / n, grads * batch / n


BATCHES = (8, 16, 32, 48, 7, 25, 50, 95)  # n = 96: the last four leave a short final batch


@pytest.mark.parametrize("weights", ["none", "shrink:100", "clip:10"])
@pytest.mark.parametrize("log_trick", [True, False])
def test_dr_loss_is_the_same_objective_at_every_batch_size(weights, log_trick):
    data = _data()
    loss = DRPolicyLoss(use_log_trick=log_trick, weights=weights)
    full_val, full_grad = _loss_and_grad(loss, *data, torch.arange(data[0].shape[0]))
    for batch in BATCHES:
        val, grad = _epoch(loss, data, batch)
        assert val == pytest.approx(full_val, rel=1e-12, abs=1e-14)
        torch.testing.assert_close(grad, full_grad, rtol=1e-9, atol=1e-12)


def test_batch_normalized_sndr_is_not():
    data = _data()
    loss = SNDRPolicyLoss(use_log_trick=True, weights="none", normalization="batch")
    full_val, _ = _loss_and_grad(loss, *data, torch.arange(data[0].shape[0]))
    vals = {b: _epoch(loss, data, b)[0] for b in (8, 16, 48, 25)}
    assert all(abs(v - full_val) > 1e-6 for v in vals.values()), (full_val, vals)
    assert len({round(v, 9) for v in vals.values()}) == 4  # a different objective per batch size


@pytest.mark.parametrize("log_trick", [True, False])
def test_global_scope_at_its_refresh_point_is_the_one_batch_surrogate(log_trick):
    """With the normalizer set to the full-data mean weight at the current parameters (the trainer does
    this at the start of each epoch), the global-scope loss summed over any partition has the value and
    gradient of the batch-scope loss on all rows at once. Under the log trick that one-batch loss is
    itself a stop-gradient surrogate, not the SNDR ratio (tests/test_objective_gradients.py); this test
    says nothing about later steps, where the global normalizer is stale."""
    logits, scores, actions, rewards, pscore = data = _data(2)
    n = logits.shape[0]
    one = SNDRPolicyLoss(use_log_trick=log_trick, weights="shrink:100", normalization="batch")
    one_val, one_grad = _loss_and_grad(one, *data, torch.arange(n))
    glob = SNDRPolicyLoss(use_log_trick=log_trick, weights="shrink:100", normalization="global")
    with pytest.raises(RuntimeError, match="set_global_normalizer"):
        _loss_and_grad(glob, *data, torch.arange(8))
    pi_a = torch.softmax(logits, dim=1)[torch.arange(n), actions]
    iw, _ = glob._prepare_iw(pi_a, pscore)
    glob.set_global_normalizer(float(iw.mean()))
    for batch in BATCHES + (96,):
        val, grad = _epoch(glob, data, batch)
        assert val == pytest.approx(one_val, rel=1e-12)
        if log_trick:  # without it, the one batch's denominator carries a gradient and the two differ
            torch.testing.assert_close(grad, one_grad, rtol=1e-9, atol=1e-12)
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
    import argparse
    import sys

    import training.trainer_trials as tt

    assert "dr" in tt.VALID_POLICY_LOSSES and tt.SN_SCOPES == ("batch", "global", "exact")
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
        assert "--sn-scope" in text and "working development default" in text
    # the working development default since f5cade9 (2026-09-27): dr (legacy SNDR stays available)
    real_parse, seen = argparse.ArgumentParser.parse_args, {}

    class Parsed(Exception):
        pass

    def parse_defaults(self, args=None, namespace=None):
        seen["args"] = real_parse(self, [], namespace)
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_defaults)
    for module in ("run_full_study", "run_full_study_parallel"):
        with pytest.raises(Parsed):
            __import__(f"training.{module}", fromlist=["main"]).main()
        assert list(seen["args"].policy_losses) == ["dr"] and seen["args"].sn_scope == "batch"


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
