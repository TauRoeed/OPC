"""The reward models of docs/opc_adaptive_estimator_study.md §5.4 (models/reward_models.py): recovery of a known logit,
the lookup interface, cross-fitting without a fold's own rows."""
import numpy as np
import pytest
import torch

from models.reward_models import BilinearRewardLookup, crossfit_bilinear, fit_bilinear_reward


def _synthetic(kind, n_users=300, n_items=200, k=6, n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_users, k))
    A = rng.normal(size=(n_items, k))
    P, Q = rng.normal(scale=0.5, size=(k, 2)), rng.normal(scale=0.5, size=(k, 2))
    inter = X @ A.T + (X @ P) @ (A @ Q).T
    if kind == "lowrank":
        logit = -2.0 + 0.6 * inter + (X @ rng.normal(scale=0.2, size=k))[:, None]
    else:
        va, vb = rng.normal(scale=0.3, size=k), rng.normal(scale=0.3, size=k)
        logit = np.exp(X @ va)[:, None] * (0.6 * inter + 0.2) + (X @ vb)[:, None] - 2.0
    q = 1 / (1 + np.exp(-logit))
    u = rng.integers(0, n_users, n)
    a = rng.integers(0, n_items, n)
    r = (rng.random(n) < q[u, a]).astype(float)
    return X, A, q, u, a, r


@pytest.mark.parametrize("kind", ["lowrank", "hetero"])
def test_the_fit_recovers_a_known_click_model(kind):
    X, A, q, u, a, r = _synthetic(kind)
    m = fit_bilinear_reward(X[u], A[a], r, kind=kind, seed=1, device="cpu", rank=2)
    lk = BilinearRewardLookup(m, X, A, "cpu")
    qh = lk[np.arange(len(X))].double().numpy()
    assert np.sqrt(np.mean((qh - q) ** 2)) < 0.02
    assert np.corrcoef(qh.ravel(), q.ravel())[0, 1] > 0.98


def test_the_lookup_interface_is_consistent():
    X, A, q, u, a, r = _synthetic("lowrank", n=5000)
    m = fit_bilinear_reward(X[u], A[a], r, kind="lowrank", seed=1, device="cpu", rank=2, max_iter=50)
    lk = BilinearRewardLookup(m, X, A, "cpu")
    users = np.array([3, 7, 7, 120])
    rows = lk[users].numpy()
    assert np.allclose(lk.qhat_block_numpy(users, 10, 40), rows[:, 10:40])
    assert np.allclose(lk.qhat_block_fn("cpu")(users, 0, A.shape[0]).numpy(), rows)
    acts = np.array([0, 5, 9, 199])
    pairs = lk.regression_model.predict_pairs(lk.user_context[users], acts)
    assert np.allclose(pairs, rows[np.arange(4), acts], atol=1e-6)


def test_crossfitting_never_fits_a_fold_on_its_own_rows():
    X, A, q, u, a, r = _synthetic("lowrank", n=6000)
    ds = {"our_x": X, "our_a": A}
    rows = {"x_idx": u, "a": a, "r": r}
    fold = np.random.default_rng(3).integers(0, 5, len(X))
    folds, full = crossfit_bilinear(ds, rows, fold, kind="lowrank", seed=2, device="cpu")
    k = 2
    keep = fold[u] != k
    ref = fit_bilinear_reward(X[u[keep]], A[a[keep]], r[keep], kind="lowrank", seed=2 * 31 + k, device="cpu")
    users = np.arange(20)
    assert torch.allclose(folds[k][users], BilinearRewardLookup(ref, X, A, "cpu")[users])
    # changing fold k's own clicks leaves fold k's model unchanged
    r2 = r.copy()
    r2[fold[u] == k] = 1 - r2[fold[u] == k]
    folds2, _ = crossfit_bilinear(ds, {"x_idx": u, "a": a, "r": r2}, fold, kind="lowrank", seed=2, device="cpu")
    assert torch.allclose(folds2[k][users], folds[k][users])
    assert not torch.allclose(folds2[0][users], folds[0][users])
