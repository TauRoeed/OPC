"""CausE objective (models/cause.py) against hand calculations on tiny inputs, and its structural properties."""
import math

import numpy as np
import pytest
import torch

import models.cause as C


def _model(variant="prod", n_users=2, n_items=2, dim=2, seed=0):
    layout = C.CausELayout(variant, n_items)
    g = torch.Generator().manual_seed(seed)
    return C.CausEModel.for_layout(layout, n_users, dim, generator=g), layout


def _set(model, **values):
    with torch.no_grad():
        for k, v in values.items():
            getattr(model, k).copy_(torch.as_tensor(v, dtype=torch.float32).reshape(getattr(model, k).shape))


def _ce(z, y):
    return max(z, 0.0) - z * y + math.log1p(math.exp(-abs(z)))


def test_prod_loss_equals_hand_calculation():
    m, _ = _model("prod")  # rows: 0, 1 control; 2, 3 treatment
    _set(m, user_emb=[[1.0, 2.0], [0.5, -1.0]], item_emb=[[0.1, 0.2], [0.3, -0.4], [0.5, 0.0], [-0.2, 0.1]],
         user_bias=[0.1, -0.2], item_bias=[0.05, 0.0, -0.1, 0.2], global_bias=[0.3], alpha=0.5)
    users = torch.tensor([0, 1, 0]); rows = torch.tensor([0, 3, 1]); y = torch.tensor([1.0, 0.0, 0.0])
    # logits by hand: alpha*<u,p> + b_u + b_k + b
    z = [0.5 * (1 * 0.1 + 2 * 0.2) + 0.1 + 0.05 + 0.3,
         0.5 * (0.5 * -0.2 + -1 * 0.1) - 0.2 + 0.2 + 0.3,
         0.5 * (1 * 0.3 + 2 * -0.4) + 0.1 + 0.0 + 0.3]
    ce = np.mean([_ce(zi, yi) for zi, yi in zip(z, [1, 0, 0])])
    # tie: rows 0 -> 2, 3 -> 3 (treatment, 0), 1 -> 3 ; L1 then batch mean
    tie = (abs(0.1 - 0.5) + abs(0.2 - 0.0) + 0.0 + abs(0.3 + 0.2) + abs(-0.4 - 0.1)) / 3
    l2 = 0.5 * (1 + 4 + 0.25 + 1) + 0.5 * (0.01 + 0.04 + 0.09 + 0.16 + 0.25 + 0.04 + 0.01) \
        + 0.5 * (0.05 ** 2 + 0.1 ** 2 + 0.2 ** 2) + 0.5 * (0.1 ** 2 + 0.2 ** 2)
    total, ce_t = m.loss(users, rows, y, l2_pen=0.01, cf_pen=2.0)
    assert ce_t.item() == pytest.approx(ce, rel=1e-6)
    assert float(m.tie(rows)) == pytest.approx(tie, rel=1e-6)
    assert float(m.l2()) == pytest.approx(l2, rel=1e-6)
    assert total.item() == pytest.approx(ce + 0.01 * l2 + 2.0 * tie, rel=1e-6)


def test_avg_tie_equals_hand_calculation():
    m, layout = _model("avg")  # rows 0, 1 items; row 2 pooled
    _set(m, item_emb=[[3.0, 4.0], [0.0, -2.0], [1.0, 0.0]])
    rows = torch.tensor([0, 1, 2, 0])
    # normalised rows (0.6, 0.8), (0, -1), (1, 0); pooled (1, 0)
    expected = (abs(0.6 - 1) + 0.8 + (1 + 1) + 0.0 + abs(0.6 - 1) + 0.8) / 4
    assert float(m.tie(rows)) == pytest.approx(expected, rel=1e-6)
    assert layout.pooled_row == 2


def test_one_way_tie_gradient_is_sign_times_frequency_and_lazy():
    m, _ = _model("prod", n_items=3)  # control 0..2, treatment 3..5
    _set(m, item_emb=[[0.5, -0.5], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    rows = torch.tensor([0, 0, 4])  # item 0 twice (control), treatment row of item 1 once; item 2 absent
    m.zero_grad()
    m.tie(rows).backward()
    g = m.item_emb.grad
    assert torch.allclose(g[0], torch.tensor([1.0, -1.0]) * 2 / 3)  # sign(θc - θt) * count / |B|
    assert torch.all(g[3:] == 0)  # one-way: treatment rows get no tie gradient
    assert torch.all(g[1] == 0) and torch.all(g[2] == 0)  # item 1's control row not in the batch; item 2 absent


def test_symmetric_tie_moves_the_treatment_rows_oppositely():
    m, _ = _model("prod", n_items=2)
    _set(m, item_emb=[[0.5, -0.5], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    m.zero_grad()
    m.tie(torch.tensor([0]), symmetric=True).backward()
    assert torch.allclose(m.item_emb.grad[2], -m.item_emb.grad[0])
    assert torch.allclose(m.item_emb.grad[0], torch.tensor([1.0, -1.0]))


def test_avg_tie_gradient_is_direction_only():
    m, _ = _model("avg", n_items=3, dim=4, seed=3)
    rows = torch.tensor([0, 1, 1, 2])
    m.zero_grad()
    m.tie(rows).backward()
    g, p = m.item_emb.grad, m.item_emb.detach()
    for k in (0, 1, 2):
        assert abs(float(g[k] @ p[k])) < 1e-6  # no radial component: the normalised tie only rotates
    assert torch.all(g[3] == 0)  # pooled row (one-way)


def test_layout_rows():
    prod = C.CausELayout("prod", 4)
    avg = C.CausELayout("avg", 4)
    a = np.array([0, 3, 1]); t = np.array([False, True, True])
    assert prod.train_rows(a, t).tolist() == [0, 7, 5] and prod.n_rows == 8
    assert avg.train_rows(a, t).tolist() == [0, 4, 4] and avg.n_rows == 5
    assert prod.prediction_rows("treatment").tolist() == [4, 5, 6, 7]
    assert avg.prediction_rows("control").tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        avg.prediction_rows("treatment")


def test_policy_vectors_reproduce_the_ranking_logits():
    m, layout = _model("prod", n_users=3, n_items=4, dim=3, seed=5)
    _set(m, alpha=0.7, user_bias=[1.0, -2.0, 0.5], global_bias=[0.3])
    with torch.no_grad():
        m.item_bias.copy_(torch.arange(8, dtype=torch.float32) / 10)
    for side in ("control", "treatment"):
        rows = layout.prediction_rows(side)
        ux, ia = m.policy_vectors(rows)
        scores = ux @ ia.T
        users = torch.arange(3).repeat_interleave(4)
        full = m.logits(users, torch.as_tensor(rows).repeat(3)).detach().numpy().reshape(3, 4)
        # equal up to per-user constants (b_u + b): same ranking and same softmax
        assert np.allclose(scores - full, (scores - full)[:, :1], atol=1e-6)


def test_xavier_init_matches_the_tf_limit():
    t = C.xavier_uniform((2000, 50), torch.Generator().manual_seed(1))
    limit = math.sqrt(6 / 2050)
    assert float(t.abs().max()) <= limit and float(t.abs().max()) > 0.99 * limit
    assert abs(float(t.std()) - limit / math.sqrt(3)) < 0.01 * limit


def test_alpha_starts_at_the_released_value():
    m, _ = _model()
    assert float(m.alpha) == pytest.approx(1e-8)
    assert float(m.user_bias.abs().sum() + m.item_bias.abs().sum() + m.global_bias.abs().sum()) == 0.0


def test_momentum_decay_learning_rate_schedule():
    m, _ = _model()
    opt = C.CausEOptimizer(m, kind="momentum_decay", lr=0.4, total_steps=4)
    lrs = []
    for _ in range(5):
        lrs.append(opt.current_lr())
        for p in m.parameters():
            p.grad = torch.zeros_like(p)
        opt.step()
    assert lrs == pytest.approx([0.4, 0.3, 0.2, 0.1, 0.0])
    assert C.CausEOptimizer(m, kind="sgd", lr=0.4).current_lr() == 0.4


def test_epoch_batches_replay_one_order_and_cover_every_row():
    batches = list(C.epoch_batches(10, 4, 3, seed=9))
    assert [len(b) for b in batches] == [4, 4, 2] * 3  # partial last batch, as tf.data batch()
    assert np.array_equal(np.concatenate(batches[:3]), np.concatenate(batches[3:6]))  # cached order
    assert sorted(np.concatenate(batches[:3]).tolist()) == list(range(10))
    fresh = list(C.epoch_batches(10, 4, 2, seed=9, reshuffle=True))
    assert not np.array_equal(np.concatenate(fresh[:3]), np.concatenate(fresh[3:]))
    assert C.n_steps(10, 4, 3) == 9


def test_fit_is_deterministic_and_learns_a_planted_signal():
    rng = np.random.default_rng(0)
    n_users, n_items, n = 30, 12, 3000
    users = rng.integers(0, n_users, n); items = rng.integers(0, n_items, n)
    treat = rng.random(n) < 0.2
    p = 1 / (1 + np.exp(-(items / n_items * 4 - 2)))  # item effect only
    labels = (rng.random(n) < p).astype(np.float32)

    def fit(seed):
        layout = C.CausELayout("prod", n_items)
        model = C.CausEModel.for_layout(layout, n_users, 4, generator=torch.Generator().manual_seed(seed))
        info = C.fit_cause(model, users, layout.train_rows(items, treat), labels, epochs=20, batch_size=64,
                           optimizer="momentum_decay", lr=0.5, cf_pen=0.1, seed=seed)
        return model, layout, info

    m1, layout, info = fit(3)
    m2, _, _ = fit(3)
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(a, b)
    assert info["finite"] and info["steps"] == 20 * math.ceil(n / 64)
    b_c = m1.item_bias.detach().numpy()[layout.prediction_rows("control")]
    assert np.corrcoef(b_c, np.arange(n_items))[0, 1] > 0.9  # item biases recover the planted item effect


def test_prediction_metrics():
    z = np.array([2.0, -1.0, 0.5, -3.0]); y = np.array([1, 0, 0, 1])
    m = C.prediction_metrics(z, y)
    p = 1 / (1 + np.exp(-z))
    assert m["nll"] == pytest.approx(float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))))
    assert m["mse"] == pytest.approx(float(np.mean((p - y) ** 2)))
    assert m["auc"] == pytest.approx(0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("optimizer,l2_pen,variant", [("momentum_decay", 0.0, "prod"), ("sgd", 1e-3, "prod"),
                                                      ("momentum_decay", 1e-4, "avg")])
def test_cuda_graph_training_is_bit_identical_to_eager(optimizer, l2_pen, variant):
    rng = np.random.default_rng(1)
    n_users, n_items, n = 50, 20, 1300  # two full batches of 512 and a partial one per epoch
    users = rng.integers(0, n_users, n); items = rng.integers(0, n_items, n)
    treat = rng.random(n) < 0.2
    labels = (rng.random(n) < 0.3).astype(np.float32)
    layout = C.CausELayout(variant, n_items)
    models = []
    for graph in (False, True):
        m = C.CausEModel.for_layout(layout, n_users, 8, generator=torch.Generator().manual_seed(4))
        with torch.no_grad():
            m.alpha.fill_(0.5)
        C.fit_cause(m, users, layout.train_rows(items, treat), labels, epochs=4, batch_size=512, optimizer=optimizer,
                    lr=0.3, l2_pen=l2_pen, cf_pen=1.0, seed=2, device="cuda", cuda_graph=graph)
        models.append(m)
    for a, b in zip(models[0].parameters(), models[1].parameters()):
        assert torch.equal(a, b)
