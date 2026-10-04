"""Exact SNDR (``--sn-scope exact``): the trainer's minibatch directions add up to the gradient of the full-data
self-normalized DR ratio V = mean DM + sum g(w)(r - q) / sum g(w), at constants refreshed every epoch.

Legacy ('batch') and 'global' SNDR are stop-gradient surrogates (tests/test_objective_gradients.py). The exact
form follows from grad(N/S) = (grad N - (N/S) grad S) / S with S = mean g(w), N = mean g(w)(r - q): per row,
g(w) (r - q - N/S) / S, i.e. DR with the self-normalized mean residual as a baseline, divided by S."""
import pytest
import torch

from models.custom_losses import DRPolicyLoss, SNDRPolicyLoss, dr_correction
from test_objective_gradients import SPECS, _Softmax, _close, _epoch_direction, _far, _g, _grad, _literal, _loader
from test_objective_gradients import world  # noqa: F401  (the module fixture)
from training import training_utils
from training.training_utils import full_data_sn_constants


def _constants(world, loss, theta=None):
    model = _Softmax(world["X"], world["theta"] if theta is None else theta)
    return full_data_sn_constants(model, _loader(world, 384).dataset, world["q_hat"], loss, "cpu", cells=12 * 50)


@pytest.mark.parametrize("spec", SPECS)
def test_exact_sndr_follows_the_gradient_of_the_sndr_ratio(world, spec):
    loss = SNDRPolicyLoss(use_log_trick=False, weights=spec, normalization="exact")
    with pytest.raises(RuntimeError, match="set_sn_constants"):
        _epoch_direction(world, loss, 384)
    loss.set_sn_constants(*_constants(world, loss))
    dm, w, res = _literal(world, world["theta"])
    assert loss.sn_constants == pytest.approx((float(_g(w, spec).mean()), float((_g(w, spec) * res).mean())), rel=1e-10)
    got, sizes = _epoch_direction(world, loss, 384)
    assert sizes == [384, 384, 232]  # the real DataLoader with its short final batch
    exact = _grad(lambda dm, w, res: dm.mean() + (_g(w, spec) * res).sum() / _g(w, spec).sum(), world)
    _close(got, exact)
    _far(got, _grad(lambda dm, w, res: dm.mean() + (_g(w, spec) * res).mean(), world))  # not DR
    for batch in (7, 100, 1000):  # any partition: a mean of per-row terms
        _close(_epoch_direction(world, loss, batch)[0], exact)


def test_exact_sndr_is_dr_with_the_self_normalized_baseline(world):
    """At S = 1 and N = 0 the row term is DR's; in general it is DR's correction with r - q - N/S, over S."""
    iw, r, q = torch.tensor([0.5, 2.0, 4.0]), torch.tensor([1.0, 0.0, 1.0]), torch.tensor([0.2, 0.3, 0.4])
    torch.testing.assert_close(dr_correction(iw, r, q, ("exact", 1.0, 0.0)), dr_correction(iw, r, q, "none"))
    torch.testing.assert_close(dr_correction(iw, r, q, ("exact", 2.0, 0.3)), iw * (r - q - 0.15) / 2.0)
    with pytest.raises(ValueError):
        dr_correction(iw, r, q, ("other", 1.0, 0.0))


def test_exact_sndr_refuses_the_log_trick_and_bad_constants():
    with pytest.raises(ValueError, match="direct gradient"):
        SNDRPolicyLoss(use_log_trick=True, normalization="exact")
    loss = SNDRPolicyLoss(use_log_trick=False, normalization="exact")
    for bad in ((0.0, 0.1), (-1.0, 0.0), (float("nan"), 0.0), (1.0, float("inf"))):
        with pytest.raises(ValueError):
            loss.set_sn_constants(*bad)
    assert loss.per_example_additive and loss.needs_sn_constants and not DRPolicyLoss().needs_sn_constants


def test_training_refreshes_the_constants_every_epoch(world, monkeypatch):
    loss = SNDRPolicyLoss(use_log_trick=False, weights="shrink:4", normalization="exact")
    model = _Softmax(world["X"], world["theta"])
    want = _constants(world, loss)
    seen, real = [], loss.set_sn_constants
    monkeypatch.setattr(loss, "set_sn_constants", lambda s, n: seen.append((s, n)) or real(s, n))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # train() asserts a GPU model when one exists
    state = torch.get_rng_state()
    got = full_data_sn_constants(model, _loader(world, 384).dataset, world["q_hat"], loss, "cpu")
    assert got == pytest.approx(want, rel=1e-12) and torch.equal(torch.get_rng_state(), state)
    training_utils.train(model, _loader(world, 128), world["q_hat"], criterion=loss, num_epochs=3, lr=0.05, device="cpu")
    assert len(seen) == 3 and seen[0] == pytest.approx(want, rel=1e-12) and seen[2] != seen[0]


def test_study_runs_exact_sndr(tmp_path):
    from test_reproducibility import _toy_embeddings

    from training.run_full_study import _finalize_summary_df, _run_condition

    _toy_embeddings(tmp_path)
    kw = dict(dataset_name="toy", emb_dir=tmp_path, bias="medium", ctr=0.05, seed=0, train_sizes=[1000], n_trials=2,
              batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact",
              policy_reward_mc_sim=8, slim=True, shared_regression_size=2000, methods=("opc",),
              policy_loss_types=("sndr",), sn_scope="exact")
    with pytest.raises(ValueError, match="direct"):
        _run_condition(**kw, run_dir=tmp_path, opc_gradient="log-trick")
    run_dir = tmp_path / "exact"
    run_dir.mkdir()
    opc, nop, _, _, meta = _run_condition(**kw, run_dir=run_dir, opc_gradient="direct")
    summary = _finalize_summary_df(opc, nop, meta)
    assert (summary["sn_scope"] == "exact").all() and summary["policy_rewards"].notna().all()
