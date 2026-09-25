"""BPR test-set metrics (BPR/evaluate.py): the test split, ranks against brute force on every
backend (numpy, torch cpu, torch cuda), the metric formulas, and a trained toy model."""

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from BPR.bpr_minibatch import BPRConfig, MiniBatchBPR, validation_split
from BPR.evaluate import held_out_ranks, popularity_scores, ranking_metrics, split_test_items

BACKENDS = ["numpy", "cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _toy_likes(n_users=400, n_items=160, groups=4, seed=0):
    """Users in taste groups; each likes mostly items of its own group (plus a few others)."""
    rng = np.random.default_rng(seed)
    g_user = rng.integers(0, groups, n_users)
    g_item = np.arange(n_items) % groups
    p = np.where(g_user[:, None] == g_item[None, :], 0.25, 0.02)
    return csr_matrix((rng.random((n_users, n_items)) < p).astype(np.float32))


def _brute_ranks(U, V, b, X_train, users, items):
    out, cand = [], []
    for u, i in zip(users, items):
        s = (U[u].astype(np.float64) @ V.T.astype(np.float64)) + (0.0 if b is None else b.astype(np.float64))
        liked = X_train[u].indices
        mask = np.ones(len(s), dtype=bool)
        mask[liked] = False
        c = s[mask]
        out.append((c > s[i]).sum() + 0.5 * ((c == s[i]).sum() - 1))
        cand.append(mask.sum())
    return np.array(out), np.array(cand)


def test_split_is_disjoint_complete_and_seeded():
    X = _toy_likes()
    cfg = BPRConfig(random_state=0)
    X_rest, users, items = split_test_items(X, cfg, min_positives=4)
    deg = np.diff(X.indptr)
    assert set(users) == set(np.flatnonzero(deg >= 4)) and len(np.unique(users)) == len(users)
    assert X_rest.nnz + len(users) == X.nnz
    assert all(X[u, i] == 1 and X_rest[u, i] == 0 for u, i in zip(users[:50], items[:50]))
    assert (np.diff(X_rest.indptr)[users] >= 3).all()  # the recipe's validation split still applies
    _, vu, _ = validation_split(X_rest, cfg)
    assert set(users) <= set(vu) or len(vu) == cfg.val_users
    again = split_test_items(X, BPRConfig(random_state=0), min_positives=4)
    np.testing.assert_array_equal(again[2], items)
    assert not np.array_equal(split_test_items(X, BPRConfig(random_state=1), min_positives=4)[2], items)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ranks_match_brute_force_with_ties_and_masking(backend):
    rng = np.random.default_rng(3)
    X = _toy_likes(seed=3)
    X_train, users, items = split_test_items(X, BPRConfig(random_state=0))
    U = rng.normal(size=(X.shape[0], 6)).astype(np.float32)
    V = rng.normal(size=(X.shape[1], 6)).astype(np.float32)
    b = rng.normal(size=X.shape[1]).astype(np.float32)
    V[1], b[1] = V[0], b[0]  # items 0 and 1 always tie
    brute, cand = _brute_ranks(U, V, b, X_train, users, items)
    for cells in (None, 3 * X.shape[1]):  # one block, and 3 users per block
        ranks, c = held_out_ranks(U, V, b, X_train, users, items, backend=backend, block_cells=cells)
        np.testing.assert_array_equal(c, cand)
        np.testing.assert_array_equal(ranks, brute)
    tied = [k for k, i in enumerate(items) if i in (0, 1) and X_train[users[k], 1 - i] == 0]
    assert tied and all(ranks[k] % 1 == 0.5 for k in tied)  # the twin item counts half


def test_metrics_formulas():
    ranks = np.array([0.0, 4.0, 19.0, 50.0])
    m = ranking_metrics(ranks, np.array([100, 100, 100, 100]))
    assert m["users"] == 4
    assert m["recall@5"] == pytest.approx(0.5) and m["recall@20"] == pytest.approx(0.75)
    assert m["ndcg@5"] == pytest.approx((1.0 + 1.0 / np.log2(6.0)) / 4)
    assert m["ndcg@20"] == pytest.approx((1.0 + 1.0 / np.log2(6.0) + 1.0 / np.log2(21.0)) / 4)
    assert m["mpr"] == pytest.approx(np.mean(ranks / 99.0))


def test_popularity_scores_and_ranks():
    X = _toy_likes()
    X_train, users, items = split_test_items(X, BPRConfig(random_state=0))
    pop = popularity_scores(X_train)
    np.testing.assert_array_equal(pop, np.asarray(X_train.sum(axis=0)).ravel())
    zu, zi = np.zeros((X.shape[0], 1), np.float32), np.zeros((X.shape[1], 1), np.float32)
    ranks, cand = held_out_ranks(zu, zi, pop, X_train, users, items)
    brute, bc = _brute_ranks(zu, zi, pop, X_train, users, items)
    np.testing.assert_array_equal(ranks, brute)


def test_trained_model_beats_random_and_popularity():
    X = _toy_likes(n_users=600, n_items=200)
    cfg = BPRConfig(factors=8, max_epochs=60, patience=5, batch_size=1024, random_state=0)
    X_rest, users, items = split_test_items(X, cfg)
    model = MiniBatchBPR(cfg).fit(X_rest, log=None)
    trained = ranking_metrics(*held_out_ranks(model.U, model.V, model.b, X_rest, users, items))
    rng = np.random.default_rng(0)
    rand = ranking_metrics(*held_out_ranks(rng.normal(size=model.U.shape).astype(np.float32),
                                           rng.normal(size=model.V.shape).astype(np.float32), None, X_rest, users, items))
    zu, zi = np.zeros((X.shape[0], 1), np.float32), np.zeros((X.shape[1], 1), np.float32)
    pop = ranking_metrics(*held_out_ranks(zu, zi, popularity_scores(X_rest), X_rest, users, items))
    assert rand["mpr"] == pytest.approx(0.5, abs=0.05)
    assert trained["mpr"] < 0.3 and trained["mpr"] < pop["mpr"]
    assert trained["recall@20"] > pop["recall@20"] > 0
