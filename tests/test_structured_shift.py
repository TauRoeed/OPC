"""The structured-shift world and the matched rank-4 adapter (docs/structured_scenario_shift_study.md §12)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.special import expit

from models.lowrank_adapter import LowRankAnchor, SharedLowRankModel
from models.shared_objectives import SharedCorrectionModel
from utils.representation_bias import WorldConfig, build_world
from utils.seeding import seed_everything
from utils.structured_shift import (
    RESPONSE_LEVELS,
    SHIFT_LEVELS,
    build_structured_world,
    is_structured_label,
    parse_structured,
    structured_label,
)

D64 = torch.float64


def _vectors(seed=0, n_users=700, n_items=600, k=12):
    """Anisotropic Gaussian vectors with a positive mean (as BPR factors), so principal directions are well defined."""
    rng = np.random.default_rng(seed)
    scales = np.linspace(1.0, 0.25, k)
    x = rng.normal(size=(n_users, k)) * scales + 0.3
    a = rng.normal(size=(n_items, k)) * scales[::-1] + 0.2
    return x.astype(np.float32), a.astype(np.float32)


_CACHE: dict = {}


def world(label: str, seed: int = 3, z_clip: float = 3.0):
    key = (label, seed, z_clip)
    if key not in _CACHE:
        x, a = _vectors()
        seed_everything(seed)
        _CACHE[key] = build_structured_world(x, a, label, seed=seed, config=WorldConfig(target_ctr=0.05), z_clip=z_clip)
    return _CACHE[key]


def _target_scores(ds):
    st = ds["structured"]
    x, a = ds["our_x"].astype(np.float64), ds["our_a"].astype(np.float64)
    return (x + x @ st["delta"]) @ a.T


def test_labels():
    assert structured_label("moderate", "strong") == "s-moderate.r-strong"
    assert parse_structured("s-none.r-moderate.gated") == {"shift": "none", "response": "moderate", "gated": True}
    assert is_structured_label("s-strong.r-none") and not is_structured_label("high/none/none")


def test_the_shift_hits_its_correlation_targets_and_has_rank_four():
    for level, target in SHIFT_LEVELS.items():
        w = world(f"s-{level}.r-none")["world"]
        assert w["shift_corr"] == pytest.approx(target, abs=0.005)
        assert (np.array(w["shift_singular_values"]) > 0).sum() == (0 if level == "none" else 4)


def test_the_augmented_environment_is_the_formula_and_preserves_the_ranking():
    ds = world("s-strong.r-strong")
    st = ds["structured"]
    sB = _target_scores(ds)
    q = expit(st["kappa"] * st["alpha"][:, None] * (sB - st["mu_B"]) / st["sd_B"] + st["c"] + st["beta"][:, None])
    env_q = ds["env"].reward_prob_block(np.arange(ds["n_users"]), 0, ds["n_actions"]).astype(np.float64)
    np.testing.assert_allclose(env_q, q, atol=2e-6)
    assert (env_q.argmax(axis=1) == sB.argmax(axis=1)).mean() > 0.999  # ties in float32 aside
    assert np.std(np.log(st["alpha"])) > 0 and np.std(st["beta"]) > 0


def test_click_calibration_per_response_level():
    kappas = set()
    for level in RESPONSE_LEVELS:
        w = world(f"s-moderate.r-{level}")["world"]
        assert w["reference_ctr"] == pytest.approx(0.05, rel=1e-6)  # the reference policy's CTR (§3, §13.1)
        kappas.add(round(w["kappa"], 12))
        if level == "none":
            assert w["best_item_ctr"] == pytest.approx(0.30, abs=1e-3)
        assert w["log_alpha"]["sd"] == pytest.approx(RESPONSE_LEVELS[level][0], abs=0.02)
        assert w["beta_stats"]["sd"] == pytest.approx(RESPONSE_LEVELS[level][1], abs=0.02)
        assert w["alpha_stats"]["mean"] == pytest.approx(1.0, abs=1e-9)
    assert len(kappas) == 1  # κ of the homogeneous world at every response level (§13.1)


def test_no_shift_no_heterogeneity_is_the_legacy_clean_world():
    x, a = _vectors()
    seed_everything(3)
    legacy = build_world(x, a, "none", seed=3, config=WorldConfig(target_ctr=0.05, reference_bias=("none",) * 3))
    ds = world("s-none.r-none")
    u = np.arange(ds["n_users"])
    np.testing.assert_allclose(ds["env"].reward_prob_block(u, 0, ds["n_actions"]),
                               legacy["env"].reward_prob_block(u, 0, legacy["n_actions"]), atol=2e-6)
    assert ds["world"]["kappa"] == legacy["world"]["alpha"] and ds["world"]["click_intercept"] == legacy["world"]["b"]
    assert ds["policy_temperature"] == pytest.approx(legacy["policy_temperature"], rel=1e-6)
    np.testing.assert_array_equal(ds["our_x"], legacy["our_x"])
    np.testing.assert_array_equal(ds["user_prior"], legacy["user_prior"])


def _model(ds, mode="policy"):
    t = lambda z: torch.as_tensor(np.asarray(z, dtype=np.float32))
    return SharedLowRankModel(ds["n_users"], ds["n_actions"], ds["emb_dim"], initial_user_embeddings=t(ds["our_x"]),
                              initial_actions_embeddings=t(ds["our_a"]), temperature=ds["policy_temperature"],
                              seed=ds["structured"]["seed"], mode=mode).double()


def test_the_adapter_starts_at_the_source_with_zero_displacement():
    ds = world("s-moderate.r-moderate")
    m = _model(ds)
    u = torch.arange(ds["n_users"])
    np.testing.assert_allclose(m.score(u).detach().numpy(), ds["our_x"].astype(np.float64) @ ds["our_a"].T.astype(np.float64),
                               rtol=1e-5, atol=1e-5)
    assert float(LowRankAnchor(ds["our_a"]).double()(m)) == 0.0
    V = m.action_transform.V.detach().numpy()
    np.testing.assert_allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-6)


def test_the_truth_adapter_reproduces_the_target_score_and_ranking():
    ds = world("s-strong.r-moderate")
    st = ds["structured"]
    m = _model(ds)
    with torch.no_grad():
        m.action_transform.U.copy_(torch.as_tensor(st["directions"]["U"] * (st["gamma"] * st["directions"]["signs"])))
        m.action_transform.V.copy_(torch.as_tensor(st["directions"]["V"]))
    s = m.score(torch.arange(ds["n_users"])).detach().numpy()
    sB = _target_scores(ds)
    np.testing.assert_allclose(s, sB, rtol=1e-5, atol=1e-5)
    assert (s.argmax(axis=1) == sB.argmax(axis=1)).all()


def test_the_gated_truth_adapter_reproduces_the_gated_target_where_the_gate_is_not_clipped():
    from utils.structured_shift import gate_affine

    ds = world("s-moderate.r-moderate.gated")
    st = ds["structured"]
    assert ds["world"]["shift_corr"] == pytest.approx(SHIFT_LEVELS["moderate"], abs=0.005)
    x = ds["our_x"].astype(np.float64)
    v, d, clipped = gate_affine(x, ds["user_prior"], st["seed"])
    inside = np.abs(expit(x @ v + d) - st["gate"]) < 1e-6  # float32 vectors
    assert inside.mean() > 0.97 and clipped < 0.03
    t = lambda z: torch.as_tensor(np.asarray(z, dtype=np.float32))
    m = SharedLowRankModel(ds["n_users"], ds["n_actions"], ds["emb_dim"], initial_user_embeddings=t(ds["our_x"]),
                           initial_actions_embeddings=t(ds["our_a"]), temperature=ds["policy_temperature"],
                           seed=st["seed"], mode="policy", gated=True).double()
    with torch.no_grad():
        m.action_transform.U.copy_(torch.as_tensor(st["directions"]["U"] * (st["gamma"] * st["directions"]["signs"])))
        m.action_transform.V.copy_(torch.as_tensor(st["directions"]["V"]))
        m.gate_w.copy_(torch.as_tensor(v))
        m.gate_b.copy_(torch.as_tensor(d))
    s = m.score(torch.arange(ds["n_users"])).detach().numpy()
    a = ds["our_a"].astype(np.float64)
    sB = (x + st["gate"][:, None] * (x @ st["delta"])) @ a.T
    np.testing.assert_allclose(s[inside], sB[inside], rtol=1e-5, atol=1e-5)
    assert (s.argmax(axis=1) == sB.argmax(axis=1)).mean() > 0.99


def test_the_calibration_aware_head_starts_as_the_ordinary_head_and_represents_the_truth():
    ds = world("s-moderate.r-strong", z_clip=50.0)  # no clipping: the nuisance is exactly linear in x
    st = ds["structured"]
    x = ds["our_x"].astype(np.float64)
    click, calib = _model(ds, "click"), _model(ds, "calib")
    u = torch.arange(ds["n_users"])
    torch.testing.assert_close(calib(u), click(u))
    # the truth in the head's parameters: log α = X w_α + b_α and β = X w_β + b_β exactly (least squares)
    X1 = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    wa = np.linalg.lstsq(X1, np.log(st["alpha"]), rcond=None)[0]
    wb = np.linalg.lstsq(X1, st["beta"], rcond=None)[0]
    np.testing.assert_allclose(X1 @ wa, np.log(st["alpha"]), atol=1e-8)
    T, k, mu, sd = ds["policy_temperature"], st["kappa"], st["mu_B"], st["sd_B"]
    with torch.no_grad():
        calib.action_transform.U.copy_(torch.as_tensor(st["directions"]["U"] * (st["gamma"] * st["directions"]["signs"])))
        calib.action_transform.V.copy_(torch.as_tensor(st["directions"]["V"]))
        calib.w_alpha.copy_(torch.as_tensor(wa[:-1]))
        calib.log_logit_scale.fill_(float(np.log(T * k * np.exp(wa[-1]) / sd) / 30.0))
        calib.gamma.fill_(float(-k * np.exp(wa[-1]) * mu / sd))
        calib.w_beta.copy_(torch.as_tensor(wb[:-1]))
        calib.click_intercept.fill_(float(st["c"] + wb[-1]))
    z = calib(u)[:, :, 0].detach().numpy()
    truth = k * st["alpha"][:, None] * (_target_scores(ds) - mu) / sd + st["c"] + st["beta"][:, None]
    np.testing.assert_allclose(z, truth, rtol=1e-6, atol=1e-6)
    pairs = (np.arange(50), np.arange(50) % ds["n_actions"])
    np.testing.assert_allclose(calib.pair_click_logits(*pairs), truth[pairs], rtol=1e-6, atol=1e-6)


def test_the_low_rank_model_is_the_affine_model_with_a_low_rank_item_map():
    """Mechanical equivalence with the old configuration: U Vᵀ as the affine model's item map D_a (no user map, no
    biases) gives the same logits, policy, click logits and exported vectors."""
    ds = world("s-moderate.r-none")
    t = lambda z: torch.as_tensor(np.asarray(z, dtype=np.float32))
    lr = _model(ds, "click")
    old = SharedCorrectionModel(ds["n_users"], ds["n_actions"], ds["emb_dim"], initial_user_embeddings=t(ds["our_x"]),
                                initial_actions_embeddings=t(ds["our_a"]), temperature=ds["policy_temperature"],
                                mode="click").double()
    rng = np.random.default_rng(1)
    with torch.no_grad():
        lr.action_transform.U.copy_(torch.as_tensor(rng.normal(size=lr.action_transform.U.shape) * 0.1))
        old.action_transform.delta.copy_(lr.action_transform.matrix())
        for m in (lr, old):
            m.log_logit_scale.fill_(0.02)
            m.click_intercept.fill_(-2.0)
    u = torch.arange(ds["n_users"])
    torch.testing.assert_close(lr(u), old(u))
    torch.testing.assert_close(lr.policy_logits(u), old.policy_logits(u))
    for p, q in zip(lr.get_params(), old.get_params()):
        torch.testing.assert_close(p, q)
    xs, as_ = (v.detach().numpy() for v in lr.get_params())
    users, items = np.arange(40), (np.arange(40) * 7) % ds["n_actions"]
    legacy = (xs[users] * as_[items]).sum(axis=1) / ds["policy_temperature"] - 2.0  # the trainer's legacy formula
    np.testing.assert_allclose(lr.pair_click_logits(users, items), legacy, rtol=1e-9, atol=1e-9)


def test_the_population_gradient_of_the_adapter_class_matches_finite_differences():
    from training.opc_gradients import WorldTensors, build_model, exact_value, get_theta, set_theta

    ds = world("s-moderate.r-moderate")
    wt = WorldTensors(ds, "cpu", dtype=D64)
    m = build_model(ds, device="cpu", dtype=D64)
    assert isinstance(m, SharedLowRankModel)
    theta = get_theta(m) + 0.05 * np.random.default_rng(2).normal(size=get_theta(m).size)
    set_theta(m, theta)
    _, g = exact_value(m, ds, world=wt)
    for j in (0, 5, 60, len(theta) - 1):
        e = np.zeros_like(theta)
        e[j] = 1e-6
        set_theta(m, theta + e)
        vp, _ = exact_value(m, ds, world=wt, grad=False)
        set_theta(m, theta - e)
        vm, _ = exact_value(m, ds, world=wt, grad=False)
        assert g[j] == pytest.approx((vp - vm) / 2e-6, rel=1e-5, abs=1e-9)


def test_the_new_arms_share_the_search_and_old_options_are_unchanged():
    import argparse

    from training.run_state import arm_config
    from utils.representation_bias import add_world_arguments, resolve_bias_configs, world_options_from_args

    cfg = {"n_trials": 20, "shared_options": None, "search_space": None}
    base = arm_config("shared_likelihood", cfg)
    for arm in ("shared_lr_likelihood", "shared_lr_likelihood_calib", "shared_lr_opc", "shared_lr_opc_raw", "shared_lr_opc_oq"):
        c = arm_config(arm, cfg)
        assert c["search_space"] == base["search_space"] and c["shared"]["lambdas"] == base["shared"]["lambdas"]
        assert c["shared"]["shared_adapter"] == "lowrank"
    assert "shared_adapter" not in base["shared"]  # the existing arms' configuration keys are unchanged
    ap = argparse.ArgumentParser()
    add_world_arguments(ap)
    assert "world_family" not in world_options_from_args(ap.parse_args([]))
    assert world_options_from_args(ap.parse_args(["--world-family", "structured_shift"]))["world_family"] == "structured_shift"
    assert resolve_bias_configs(["high/none/none", "s-moderate.r-strong"]) == ["w-high.g-none.v-none", "s-moderate.r-strong"]
