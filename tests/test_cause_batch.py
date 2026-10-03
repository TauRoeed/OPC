"""The batched trainer (several CausE trials in one model) against separate single-trial runs and TensorFlow."""
from pathlib import Path

import numpy as np
import pytest
import torch

import models.cause as C

FIXTURE = Path(__file__).parent / "fixtures" / "cause_tf_reference.npz"
TF_VARS = ["user_embeddings", "product_embeddings", "user_b", "prod_b", "global_bias", "alpha"]
ATTRS = ["user_emb", "item_emb", "user_bias", "item_bias", "global_bias", "alpha"]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _data(n_users=40, n_items=15, n=1300, seed=3):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, n_users, n), rng.integers(0, n_items, n), rng.random(n) < 0.25,
            (rng.random(n) < 0.3).astype(np.float32), n_users, n_items)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("variant,optimizer", [("prod", "momentum_decay"), ("prod", "sgd"), ("avg", "momentum_decay")])
def test_batch_training_equals_separate_trainings(device, variant, optimizer):
    users, items, treat, labels, n_users, n_items = _data()
    layout = C.CausELayout(variant, n_items)
    rows = layout.train_rows(items, treat)
    cfg = [(0.3, 0.0, 1.0), (0.05, 1e-3, 0.0), (0.8, 1e-4, 10.0), (0.1, 0.0, 0.1)]  # (lr, l2, tie) per trial

    def fresh(i):
        m = C.CausEModel.for_layout(layout, n_users, 6, generator=torch.Generator().manual_seed(10 + i))
        with torch.no_grad():
            m.alpha.fill_(0.4 + 0.1 * i)  # active embeddings, so every term matters
        return m

    singles = []
    for i, (lr, l2, cf) in enumerate(cfg):
        m = fresh(i)
        C.fit_cause(m, users, rows, labels, epochs=3, batch_size=512, optimizer=optimizer, lr=lr, l2_pen=l2, cf_pen=cf,
                    seed=99, device=device)
        singles.append(m)
    batch = [fresh(i) for i in range(len(cfg))]
    info = C.fit_cause_batch(batch, users, rows, labels, epochs=3, batch_size=512, optimizer=optimizer,
                             lrs=[c[0] for c in cfg], l2_pens=[c[1] for c in cfg], cf_pens=[c[2] for c in cfg],
                             order_seed=99, device=device)
    assert info["finite"].all() and info["steps"] == 3 * 3
    for s, b in zip(singles, batch):
        for a in ATTRS:
            x, y = getattr(s, a), getattr(b, a)
            if device == "cpu":
                assert torch.equal(x, y), a  # the same operations in the same order: bit for bit
            else:  # CUDA sums duplicate indices of the flattened tables in another order: float32 rounding only
                assert float((x - y).abs().max()) <= 2e-6 * float(x.abs().max()), a


def test_a_batch_of_one_matches_tensorflow():
    ref = np.load(FIXTURE)
    for case in ("prod_sgd_l2", "prod_mom_sym", "prod_mom_l2", "avg_mom_sym"):
        g = lambda k: ref[f"{case}/{k}"]
        variant, opt, lr, l2, cf, _a0, sym = (str(x) for x in g("meta"))
        init = {v: g("init_" + v) for v in TF_VARS}
        m = C.CausEModel(init["user_embeddings"].shape[0], init["product_embeddings"].shape[0],
                         init["user_embeddings"].shape[1], variant=variant,
                         tie_offset=6 if variant == "prod" else None, pooled_row=0 if variant == "avg" else None)
        with torch.no_grad():
            for v, a in zip(TF_VARS, ATTRS):
                getattr(m, a).copy_(torch.as_tensor(init[v]).reshape(getattr(m, a).shape))
        bm = C.CausEBatchModel([m])
        bopt = C.CausEBatchOptimizer(bm, kind=opt, lrs=torch.tensor([float(lr)]), total_steps=10)
        for i in range(10):
            u = torch.as_tensor(g(f"batch{i}_users"), dtype=torch.long)
            r = torch.as_tensor(g(f"batch{i}_products"), dtype=torch.long)
            y = torch.as_tensor(g(f"batch{i}_labels"), dtype=torch.float32)
            bm.zero_grad()
            loss, _ = bm.loss(u, r, y, l2_pen=torch.tensor([float(l2)]), cf_pen=torch.tensor([float(cf)]),
                              symmetric=sym == "True")
            loss.backward()
            bopt.apply(torch.tensor([np.float32(float(lr) * bopt.factor_at(i))]))
        bm.trial_model(0, m)
        err = max(float(np.max(np.abs(getattr(m, a).detach().numpy().reshape(-1) - g("final_" + v).reshape(-1))))
                  for v, a in zip(TF_VARS, ATTRS))
        assert err < 5e-6, (case, err)


def test_batch_keeps_a_diverging_trial_to_itself():
    users, items, treat, labels, n_users, n_items = _data()
    layout = C.CausELayout("prod", n_items)
    rows = layout.train_rows(items, treat)
    models = [C.CausEModel.for_layout(layout, n_users, 6, generator=torch.Generator().manual_seed(i)) for i in range(2)]
    with torch.no_grad():
        for m in models:
            m.alpha.fill_(1.0)
    info = C.fit_cause_batch(models, users, rows, labels, epochs=5, lrs=[1e6, 0.1], l2_pens=[0.0, 0.0],
                             cf_pens=[1.0, 1.0], optimizer="sgd", order_seed=1)
    assert not info["finite"][0] and info["finite"][1]
    assert all(torch.isfinite(p).all() for p in models[1].parameters())
