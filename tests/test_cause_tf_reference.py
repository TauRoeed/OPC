"""models/cause.py against the authors' unmodified TensorFlow graph (criteo-research/CausE @ 957e556).

The fixture comes from scripts/cause_reference/make_tf_fixture.py (Python 3.6 / TF 1.9, outside OPC). It holds the
initial parameters, a fixed sequence of tiny batches, TF's per-step losses, the tie difference TF computed in
its graph, and TF's final parameters, for 11 cases: CausE-prod and CausE-avg; the released plain SGD and the
paper's momentum with linear decay (the audit's paper variant); the one-way and symmetric tie; with and without L2.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import models.cause as C

FIXTURE = Path(__file__).parent / "fixtures" / "cause_tf_reference.npz"
TF_VARS = ["user_embeddings", "product_embeddings", "user_b", "prod_b", "global_bias", "alpha"]
ATTRS = ["user_emb", "item_emb", "user_bias", "item_bias", "global_bias", "alpha"]
N_PRODUCTS = 6  # the fixture's --num_products (id 0 = CausE-avg's pooled product; prod doubles the table)
ATOL = 5e-6


@pytest.fixture(scope="module")
def ref():
    return np.load(FIXTURE)


def _cases(ref):
    return sorted({k.split("/")[0] for k in ref.files if "/" in k})


def _setup(ref, case):
    g = lambda k: ref[f"{case}/{k}"]
    variant, opt, lr, l2, cf, _alpha0, sym = (str(x) for x in g("meta"))
    init = {v: g("init_" + v) for v in TF_VARS}
    model = C.CausEModel(init["user_embeddings"].shape[0], init["product_embeddings"].shape[0],
                         init["user_embeddings"].shape[1], variant=variant,
                         tie_offset=N_PRODUCTS if variant == "prod" else None,
                         pooled_row=0 if variant == "avg" else None)
    with torch.no_grad():
        for v, a in zip(TF_VARS, ATTRS):
            getattr(model, a).copy_(torch.as_tensor(init[v]).reshape(getattr(model, a).shape))
    batches, signs = [], []
    i = 0
    while f"{case}/batch{i}_users" in ref.files:
        batches.append((torch.as_tensor(g(f"batch{i}_users"), dtype=torch.long),
                        torch.as_tensor(g(f"batch{i}_products"), dtype=torch.long),
                        torch.as_tensor(g(f"batch{i}_labels"), dtype=torch.float32)))
        signs.append(torch.as_tensor(np.sign(g(f"batch{i}_tie_diff")), dtype=torch.float32))
        i += 1
    cfg = dict(variant=variant, optimizer=opt, lr=float(lr), l2_pen=float(l2), cf_pen=float(cf), symmetric=sym == "True")
    return model, batches, signs, cfg, g


def _replay(model, batches, cfg, signs=None):
    """Replay the batches; ``signs`` injects TF's tie subgradient signs (CausE-avg's pooled-row rounding)."""
    opt = C.CausEOptimizer(model, kind=cfg["optimizer"], lr=cfg["lr"], total_steps=len(batches))
    hist = []
    for k, (u, r, y) in enumerate(batches):
        model.zero_grad(set_to_none=False)
        if signs is None:
            total, ce = model.loss(u, r, y, l2_pen=cfg["l2_pen"], cf_pen=cfg["cf_pen"], symmetric=cfg["symmetric"])
        else:
            ce = F.binary_cross_entropy_with_logits(model.logits(u, r), y)
            total = ce + cfg["l2_pen"] * model.l2()
            c = model.item_emb[model.pooled_row]
            c = c if cfg["symmetric"] else c.detach()
            diff = C._tf_l2_normalize(model.item_emb[r], dim=1) - C._tf_l2_normalize(c, dim=0)
            total = total + cfg["cf_pen"] * (diff * signs[k]).sum(dim=-1).mean()
        total.backward()
        opt.step()
        hist.append((float(total.detach()), float(ce.detach())))
    return np.array(hist)


def _max_param_err(model, g):
    return max(float(np.max(np.abs(getattr(model, a).detach().numpy().reshape(-1) - g("final_" + v).reshape(-1))))
               for v, a in zip(TF_VARS, ATTRS))


def test_fixture_provenance_is_the_unmodified_release(ref):
    prov = " ".join(str(x) for x in ref["provenance"])
    assert "957e556 (unmodified)" in prov and "tensorflow 1.9.0" in prov
    assert len(_cases(ref)) == 11


@pytest.mark.parametrize("case", ["prod_sgd_released", "prod_sgd_alpha", "prod_sgd_l2", "prod_mom", "prod_mom_sym",
                                  "prod_mom_l2"])
def test_prod_matches_tensorflow(ref, case):
    model, batches, _signs, cfg, g = _setup(ref, case)
    hist = _replay(model, batches, cfg)
    assert np.max(np.abs(hist - g("losses"))) < ATOL
    assert _max_param_err(model, g) < ATOL


def test_prod_self_tie_of_treatment_rows_is_exactly_zero_in_tensorflow(ref):
    """S_t rows are tied to themselves (r(k) = k): TF gathers the same row twice and gets exactly 0."""
    for case in ("prod_sgd_alpha", "prod_mom_sym"):
        g = lambda k: ref[f"{case}/{k}"]
        for i in range(10):
            treat = g(f"batch{i}_products") >= N_PRODUCTS
            assert np.all(g(f"batch{i}_tie_diff")[treat] == 0.0)


@pytest.mark.parametrize("case", ["avg_sgd_released", "avg_sgd_alpha", "avg_sgd_l2", "avg_mom", "avg_mom_sym"])
def test_avg_matches_tensorflow_given_its_tie_signs(ref, case):
    """With TF's own subgradient signs injected, CausE-avg matches TF step for step: the rounding of the pooled
    row's self-tie (next test) is the only difference between the two implementations."""
    model, batches, signs, cfg, g = _setup(ref, case)
    hist = _replay(model, batches, cfg, signs=signs)
    assert np.max(np.abs(hist - g("losses"))) < ATOL
    assert _max_param_err(model, g) < ATOL


def test_avg_pooled_self_tie_is_a_tensorflow_rounding_artifact(ref):
    """TF normalises the pooled vector twice (as batch rows, axis 1, and alone, axis 0) and gets different last
    bits, so |difference| is ~1e-8 instead of 0 and abs() passes a ±1 subgradient. Our port uses the exact 0."""
    seen = 0
    for case in ("avg_sgd_alpha", "avg_mom"):
        g = lambda k: ref[f"{case}/{k}"]
        for i in range(10):
            pooled = g(f"batch{i}_products") == 0
            if not pooled.any():
                continue
            d = g(f"batch{i}_tie_diff")[pooled]
            assert np.max(np.abs(d)) < 1e-6
            seen += int(np.count_nonzero(d))
            model, _b, _s, _c, _g = _setup(ref, case)
            r = torch.as_tensor(g(f"batch{i}_products"), dtype=torch.long)
            ours = (C._tf_l2_normalize(model.item_emb[r], dim=1) - C._tf_l2_normalize(model.item_emb[0], dim=0))
            assert torch.all(ours[torch.as_tensor(pooled)] == 0)
    assert seen > 0


def test_avg_native_matches_tensorflow_until_the_first_pooled_batch(ref):
    """Without injected signs our CausE-avg agrees with TF exactly up to the first batch with randomized rows."""
    model, batches, _signs, cfg, g = _setup(ref, "avg_sgd_alpha")
    first_pooled = next(i for i, (_u, r, _y) in enumerate(batches) if bool((r == 0).any()))
    hist = _replay(model, batches[: first_pooled + 1], cfg)
    assert np.max(np.abs(hist - g("losses")[: first_pooled + 1])) < ATOL
