"""BPR v2 (BPR/bpr_minibatch.py): gradients, sampling, the held-out split, early stopping,
refit, determinism, and that it learns planted structure with the bias tracking popularity."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.stats import chisquare, spearmanr

import BPR.bpr_minibatch as bm
from BPR.bpr_minibatch import (
    BPRConfig,
    MiniBatchBPR,
    TripleSampler,
    _segment_sum,
    data_fingerprint,
    evaluate_ranking,
    holdout_split,
)


def _planted(n_users=600, n_items=300, d=4, per_user=25, seed=0):
    """Interactions drawn from latent tastes plus item popularity."""
    rng = np.random.default_rng(seed)
    P, Q = rng.normal(size=(n_users, d)), rng.normal(size=(n_items, d))
    pop = rng.normal(scale=1.5, size=n_items)
    logits = P @ Q.T + pop[None, :]
    rows, cols = [], []
    for u in range(n_users):
        p = np.exp(logits[u] - logits[u].max())
        items = rng.choice(n_items, size=per_user, replace=False, p=p / p.sum())
        rows += [u] * per_user
        cols += list(items)
    return csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n_users, n_items)), pop


# ------------------------------------------------------------------ gradients
@pytest.mark.parametrize("item_bias", [True, False])
def test_gradients_match_finite_differences(item_bias):
    rng = np.random.default_rng(1)
    m = MiniBatchBPR(BPRConfig(factors=4, item_bias=item_bias, regularization=0.03, bias_regularization=0.02))
    m._init_params(7, 9)
    m.U = rng.normal(size=m.U.shape)  # float64 for an exact check
    m.V = rng.normal(size=m.V.shape)
    if item_bias:
        m.b = rng.normal(size=m.b.shape)
    u = np.array([0, 3, 3, 6, 1, 0])
    i = np.array([1, 2, 5, 2, 8, 1])
    j = np.array([4, 0, 2, 7, 5, 3])  # item 2 is liked in one triple and a negative in another
    grads, _ = m.batch_gradients(u, i, j)
    analytic = {"U": dict(zip(*_segment_sum(u, grads["u"])))}
    rows, g = _segment_sum(np.concatenate([i, j]), np.concatenate([grads["i"], grads["j"]]))
    analytic["V"] = dict(zip(rows, g))
    if item_bias:
        rows, g = _segment_sum(np.concatenate([i, j]), np.concatenate([grads["bi"], grads["bj"]]))
        analytic["b"] = dict(zip(rows, g))
    h = 1e-6
    for name, arr in (("U", m.U), ("V", m.V)) + ((("b", m.b),) if item_bias else ()):
        for row, g_row in analytic[name].items():
            for col in (range(arr.shape[1]) if arr.ndim == 2 else [None]):
                idx = (row, col) if col is not None else (row,)
                old = arr[idx]
                arr[idx] = old + h; up = m.batch_loss(u, i, j)
                arr[idx] = old - h; down = m.batch_loss(u, i, j)
                arr[idx] = old
                numeric = (up - down) / (2 * h)
                got = g_row[col] if col is not None else g_row
                assert got == pytest.approx(numeric, rel=1e-6, abs=1e-8), (name, idx)


def test_segment_sum():
    rows, sums = _segment_sum(np.array([3, 1, 3, 0, 1]), np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]))
    assert rows.tolist() == [0, 1, 3] and sums[:, 0].tolist() == [4.0, 7.0, 4.0]


# ------------------------------------------------------------------ sampling
def test_uniform_negatives_are_uniform_and_never_liked():
    X = csr_matrix((np.ones(5), ([0] * 5, [0, 1, 2, 3, 4])), shape=(2, 20))
    s = TripleSampler(X, BPRConfig(), np.random.default_rng(0))
    j, keep = s.negatives(np.zeros(60_000, dtype=np.int64))
    assert keep.all() and not np.isin(j, [0, 1, 2, 3, 4]).any()
    counts = np.bincount(j, minlength=20)[5:]
    assert chisquare(counts).pvalue > 1e-3


def test_popularity_negatives_follow_counts():
    rng = np.random.default_rng(2)
    n_items = 12
    per_item = rng.integers(1, 40, size=n_items)
    rows = np.concatenate([np.arange(1, 1 + c) for c in per_item])     # users 1..c like item k
    cols = np.concatenate([np.full(c, k) for k, c in enumerate(per_item)])
    X = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(50, n_items))  # user 0 likes nothing
    s = TripleSampler(X, BPRConfig(negatives="popularity", negative_gamma=0.75), np.random.default_rng(0))
    j, keep = s.negatives(np.zeros(200_000, dtype=np.int64))
    assert keep.all()
    expected = per_item ** 0.75
    counts = np.bincount(j, minlength=n_items)
    assert chisquare(counts, expected / expected.sum() * counts.sum()).pvalue > 1e-3


def test_heavy_users_get_exact_negatives_and_rounds_only_recheck_redraws(monkeypatch):
    # user 0 likes all but items 3 and 7 of 40; user 1 likes everything; user 2 likes nothing
    n = 40
    liked0 = [k for k in range(n) if k not in (3, 7)]
    rows = [0] * len(liked0) + [1] * n
    cols = liked0 + list(range(n))
    X = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(3, n))
    s = TripleSampler(X, BPRConfig(), np.random.default_rng(0))
    sizes, liked_counts = [], []
    real = TripleSampler.is_liked

    def recording(self, u, j):
        out = real(self, u, j)
        sizes.append(len(u))
        liked_counts.append(int(out.sum()))
        return out

    monkeypatch.setattr(TripleSampler, "is_liked", recording)
    u = np.array([0] * 4000 + [1] * 5 + [2] * 1000)
    j, keep = s.negatives(u)
    # each round re-checks exactly the entries the previous round found liked
    assert sizes[0] == len(u) and sizes[1:] == liked_counts[:-1]
    assert set(j[:4000].tolist()) == {3, 7} and keep[:4000].all()
    assert abs(np.mean(j[:4000] == 3) - 0.5) < 0.05
    assert not keep[4000:4005].any() and s.dropped == 5  # a user who likes everything has no negative
    assert keep[4005:].all()


def test_heavy_user_fallback_respects_popularity_weights():
    n = 30
    counts = np.arange(1, n + 1)  # item k liked by k+1 users (users 1..)
    rows = np.concatenate([np.arange(1, 1 + c) for c in counts])
    cols = np.concatenate([np.full(c, k) for k, c in enumerate(counts)])
    liked0 = [k for k in range(n) if k not in (2, 25)]  # user 0 leaves only items 2 and 25
    rows = np.concatenate([rows, np.zeros(len(liked0), dtype=int)])
    cols = np.concatenate([cols, liked0])
    X = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(40, n))
    s = TripleSampler(X, BPRConfig(negatives="popularity", negative_gamma=1.0), np.random.default_rng(1))
    assert np.array_equal(s.item_counts, np.bincount(cols, minlength=n))
    j = np.array([s._draw_unliked(0)[0] for _ in range(4000)])
    w2, w25 = s.item_counts[2], s.item_counts[25]
    assert set(j.tolist()) == {2, 25}
    assert np.mean(j == 25) == pytest.approx(w25 / (w2 + w25), abs=0.03)


def test_interaction_vs_user_sampling_of_positives():
    # user 0 likes 1 item, user 1 likes 9 items: interaction sampling picks user 1 90% of the time
    X = csr_matrix((np.ones(10), ([0] + [1] * 9, [0] + list(range(1, 10)))), shape=(2, 30))
    for sampling, share in (("interaction", 0.9), ("user", 0.5)):
        s = TripleSampler(X, BPRConfig(sampling=sampling), np.random.default_rng(0))
        u, i = s.positives(100_000)
        assert np.mean(u == 1) == pytest.approx(share, abs=0.01)
        assert X[u, i].min() == 1  # always a liked item


# ------------------------------------------------------------------ split and metric
def test_holdout_split():
    X, _ = _planted(n_users=200, n_items=80, per_user=6)
    X = X.tolil(); X[0, :] = 0; X[0, 5] = 1; X[0, 6] = 1  # user 0 has 2 liked items: not held out
    X = X.tocsr()
    X_train, users, items = holdout_split(X, min_positives=3, seed=0)
    assert 0 not in users and len(users) == len(set(users.tolist())) == 199
    assert np.all(np.asarray(X[users, items]).ravel() == 1)
    assert np.all(np.asarray(X_train[users, items]).ravel() == 0)
    assert X_train.nnz == X.nnz - len(users)
    assert (X_train + csr_matrix((np.ones(len(users)), (users, items)), shape=X.shape) != X).nnz == 0
    again = holdout_split(X, min_positives=3, seed=0)[2]
    other = holdout_split(X, min_positives=3, seed=1)[2]
    assert np.array_equal(items, again) and not np.array_equal(items, other)


def test_evaluate_ranking_hand_example():
    U = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[3.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.0, 5.0]])
    X_train = csr_matrix(([1.0], ([0], [0])), shape=(2, 4))  # user 0 already liked item 0
    # user 0: item 0 masked, held-out item 1 ranks first; user 1: held-out item 2 ties at 0 with items 0, 1
    out = evaluate_ranking(U, V, None, X_train, [0, 1], [1, 2], k=1)
    assert out["recall"] == 0.5 and out["ndcg"] == pytest.approx(0.5)
    out = evaluate_ranking(U, V, np.array([0.0, 0.0, 0.0, -10.0]), X_train, [1], [3], k=1)
    assert out["recall"] == 0.0  # the bias pushes item 3 below the others


# ------------------------------------------------------------------ training behaviour
def test_early_stopping_keeps_the_best_epoch(monkeypatch):
    curve = iter([0.10, 0.30, 0.20, 0.25, 0.29, 0.10, 0.10])
    snaps = []

    def fake_eval(U, V, b, *args, **kwargs):
        snaps.append(U.copy())
        return {"recall": next(curve), "ndcg": 0.0}

    monkeypatch.setattr(bm, "evaluate_ranking", fake_eval)
    X, _ = _planted(n_users=100, n_items=60, per_user=8)
    m = MiniBatchBPR(BPRConfig(factors=4, batch_size=256, patience=3, max_epochs=20, refit=False)).fit(X, log=None)
    assert m.best_epoch == 2 and len(m.history) == 5  # epochs 3-5 bring no gain, then it stops
    np.testing.assert_array_equal(m.user_factors, snaps[1])


def test_refit_trains_on_all_interactions_for_the_best_epochs(monkeypatch):
    seen = []
    real = bm.TripleSampler

    class Recorder(real):
        def __init__(self, X, *a, **k):
            seen.append(X.nnz)
            super().__init__(X, *a, **k)

    monkeypatch.setattr(bm, "TripleSampler", Recorder)
    X, _ = _planted(n_users=150, n_items=60, per_user=8)
    m = MiniBatchBPR(BPRConfig(factors=4, batch_size=256, max_epochs=6, patience=2)).fit(X, log=None)
    assert seen == [X.nnz - m.n_val_users, X.nnz]  # search without the held-out items, refit on all
    assert m.epochs_trained == m.best_epoch


def test_same_seed_same_vectors_other_seed_differs():
    X, _ = _planted(n_users=120, n_items=50, per_user=8)
    cfg = dict(factors=4, batch_size=256, early_stopping=False, epochs=3)
    a = MiniBatchBPR(BPRConfig(**cfg, random_state=5)).fit(X, log=None)
    b = MiniBatchBPR(BPRConfig(**cfg, random_state=5)).fit(X, log=None)
    c = MiniBatchBPR(BPRConfig(**cfg, random_state=6)).fit(X, log=None)
    for k in ("user_factors", "item_factors", "item_bias"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
        assert not np.array_equal(getattr(a, k), getattr(c, k))


def test_learns_planted_structure_and_bias_tracks_popularity():
    X, pop = _planted()
    m = MiniBatchBPR(BPRConfig(factors=8, batch_size=1024, max_epochs=60, patience=4)).fit(X, log=None)
    counts = np.asarray(X.sum(axis=0)).ravel()
    val = m.summary()["validation"]
    random_recall = 20 / X.shape[1]
    assert val["recall"] > 4 * random_recall
    assert spearmanr(m.item_bias, counts).correlation > 0.8
    assert spearmanr(m.item_bias, pop).correlation > 0.7
    no_bias = MiniBatchBPR(BPRConfig(factors=8, batch_size=1024, item_bias=False, early_stopping=False, epochs=3)).fit(X, log=None)
    assert no_bias.item_bias is None


# ------------------------------------------------------------------ config and fingerprint
def test_config_validation():
    with pytest.raises(ValueError, match="unknown BPR settings"):
        BPRConfig.from_dict({"factors": 8, "samples_per_epoch": 10})
    for bad in ({"negatives": "hard"}, {"sampling": "items"}, {"batch_size": 0}, {"learning_rate": 0}):
        with pytest.raises(ValueError):
            BPRConfig(**bad)


def test_data_fingerprint_changes_with_data():
    X, _ = _planted(n_users=50, n_items=30, per_user=5)
    Y = X.tolil(); Y[0, 0] = 1 - Y[0, 0]
    assert data_fingerprint(X) == data_fingerprint(X.copy())
    assert data_fingerprint(X) != data_fingerprint(Y.tocsr())
