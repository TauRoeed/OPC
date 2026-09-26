"""Policy transforms (models.models: --policy-transform): the linear correction and the
linear + MLP combination start exactly at the identity, compute their formula, learn, and reach
the trainer; ``mlp`` is the older x + MLP(LN(x))."""

import numpy as np
import pytest
import torch

from models.models import (
    POLICY_TRANSFORMS,
    CFModel,
    GlobalLinearCorrection,
    LinearPlusMLPCorrection,
    SingleMLPTransform,
    make_policy_transform,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["linear", "linear+mlp"])
def test_new_transforms_start_exactly_at_the_identity(kind, device):
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")  # TF32 on the GPU: I @ x would not be exact, x + 0 is
    try:
        t = make_policy_transform(kind, 12).to(device)
        x = torch.randn(300, 12, device=device) * 3
        t.train()
        assert torch.equal(t(x), x) and torch.equal(t(x, torch.arange(300, device=device)), x)
        t.eval()
        assert torch.equal(t(x), x)
    finally:
        torch.set_float32_matmul_precision(prev)


def test_linear_formula_and_learning():
    torch.manual_seed(0)
    t = GlobalLinearCorrection(5)
    with torch.no_grad():
        t.delta.copy_(torch.randn(5, 5))
        t.bias.copy_(torch.randn(5))
    x = torch.randn(7, 5)
    torch.testing.assert_close(t(x), x @ (torch.eye(5) + t.delta).T + t.bias)
    fresh = GlobalLinearCorrection(5)
    opt = torch.optim.Adam(fresh.parameters(), lr=0.1)
    target = x @ torch.randn(5, 5) + 1.0
    loss0 = ((fresh(x) - target) ** 2).mean().item()
    for _ in range(50):
        opt.zero_grad()
        ((fresh(x) - target) ** 2).mean().backward()
        opt.step()
    assert ((fresh(x) - target) ** 2).mean().item() < 0.2 * loss0
    assert fresh.delta.abs().sum() > 0 and fresh.bias.abs().sum() > 0


def test_linear_plus_mlp_formula_and_both_parts_learn():
    torch.manual_seed(1)
    t = LinearPlusMLPCorrection(6)
    # the MLP sees the raw vector: no LayerNorm, no dropout
    assert not any(isinstance(m, (torch.nn.LayerNorm, torch.nn.Dropout)) for m in t.modules())
    x = torch.randn(40, 6)
    last = t.mlp[2]
    assert last.weight.abs().sum() == 0 and last.bias.abs().sum() == 0
    opt = torch.optim.Adam(t.parameters(), lr=0.05)
    t.train()
    for _ in range(5):
        opt.zero_grad()
        ((t(x) - torch.sin(x)) ** 2).mean().backward()
        opt.step()
    assert last.weight.abs().sum() > 0 and t.linear.delta.abs().sum() > 0
    with torch.no_grad():
        want = x + x @ t.linear.delta.T + t.linear.bias + t.mlp[2](torch.nn.functional.gelu(t.mlp[0](x)))
        torch.testing.assert_close(t(x), want)
        trained = t(x)
        t.eval()
        assert torch.equal(t(x), trained)  # no dropout: the same output in training and evaluation mode
    # the MLP term sees the vector's norm (after LayerNorm, x and 3x would look alike)
    with torch.no_grad():
        assert not torch.allclose(t(3 * x) - t.linear(3 * x), t(x) - t.linear(x), atol=1e-4)


def test_factory_and_the_older_transform():
    assert POLICY_TRANSFORMS == ("linear", "linear+mlp", "mlp")
    assert isinstance(make_policy_transform("mlp", 4), SingleMLPTransform)
    assert isinstance(make_policy_transform("LINEAR", 4), GlobalLinearCorrection)
    x = torch.randn(10, 4)
    m = make_policy_transform("mlp", 4).eval()
    assert not torch.equal(m(x), x)  # random init: starts off the identity
    with pytest.raises(ValueError):
        make_policy_transform("quadratic", 4)


@pytest.mark.parametrize("pop", [False, True])
def test_cf_model_with_linear_transform_starts_at_the_logger(pop):
    rng = np.random.default_rng(0)
    U = torch.tensor(rng.normal(size=(30, 6)), dtype=torch.float32)
    V = torch.tensor(rng.normal(size=(20, 6)), dtype=torch.float32)
    kw = {"item_popularity": rng.normal(size=20).astype(np.float32), "pop_weight": 1.5} if pop else {}
    m = CFModel(30, 20, 6, initial_user_embeddings=U.clone(), initial_actions_embeddings=V.clone(),
                user_transform=make_policy_transform("linear", 6), action_transform=make_policy_transform("linear", 6),
                temperature=0.7, **kw)
    ex, ea = (t.detach() for t in m.get_params())
    assert torch.equal(ex[:, :6], U) and torch.equal(ea[:, :6], V)
    with torch.no_grad():
        prob = m(torch.arange(30))[:, :, 0]
        logits = U @ V.T + (1.5 * torch.as_tensor(kw["item_popularity"]) if pop else 0.0)
        torch.testing.assert_close(prob, torch.softmax(logits / 0.7, dim=1), rtol=1e-5, atol=1e-7)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert {"user_transform.delta", "user_transform.bias", "action_transform.delta", "action_transform.bias"} <= trainable
    assert "user_embeddings.weight" not in trainable  # the biased vectors themselves stay fixed


def test_runner_passes_the_transform(tmp_path, monkeypatch):
    import pandas as pd

    import training.trainer_trials as tt
    from test_reproducibility import _toy_embeddings
    from training.run_full_study import _run_condition

    _toy_embeddings(tmp_path)
    seen = []
    real = tt.make_policy_transform

    def spy(kind, d):
        seen.append(kind)
        return real(kind, d)

    monkeypatch.setattr(tt, "make_policy_transform", spy)
    for kind in ("linear+mlp", "linear"):
        run_dir = tmp_path / kind.replace("+", "_")
        run_dir.mkdir()
        *_, meta = _run_condition(dataset_name="toy", emb_dir=tmp_path, bias="low", ctr=0.05, seed=0, train_sizes=[1000],
                                  n_trials=1, batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None,
                                  policy_reward_mode="exact", policy_reward_mc_sim=8, run_dir=run_dir, slim=True,
                                  shared_regression_size=2000, policy_transform=kind)
        assert meta["policy_transform"] == kind
        assert pd.read_csv(run_dir / "trials_long.csv")["actual_reward"].notna().all()
    assert seen == ["linear+mlp"] * 4 + ["linear"] * 4  # users and items, for both methods
    with pytest.raises(ValueError):
        tt.regression_trainer_trial(train_sizes=[1000], dataset={}, batch_size=None, policy_transform="quadratic")
