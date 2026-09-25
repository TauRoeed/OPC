"""Popularity in the simulated world and in the models that see it.

BPR's item bias b enters the true score as beta_true·b (``pop_strength``) and the logger's score
as beta_log·b (``logger_pop_strength``, default beta_true), carried as one extra column: users
[x, 1], items [a, w·b]. With both weights 0 (the default) the world is the taste-only one.
"""

import argparse
import json

import numpy as np
import pytest
import torch
from scipy.special import expit

import utils.representation_bias as rb
from models.models import CFModel, LinearCFModel, SingleMLPTransform
from training.run_full_study import _condition_run_key, _load_item_bias
from training.trainer_trials import (
    _build_regression_logged_split,
    _cf_model_inputs,
    _policy_pop_weight,
    _reward_action_context,
    _taste_vectors,
    fit_shared_regression_bundle,
)
from utils.noise_snr import dataset_snr_report
from utils.policies import Policy
from utils.representation_bias import describe_world, world_options_from_args, world_run_key_suffix
from utils.simulation_utils import calc_reward, calc_uniform_reward, generate_dataset

N_USERS, N_ITEMS, DIM = 1500, 1200, 12


def _toy(seed=0):
    # BPR-like taste vectors (a shared mean direction, uneven item norms) plus an item bias
    rng = np.random.default_rng(seed)
    X = rng.normal(size=DIM) + 0.6 * rng.normal(size=(N_USERS, DIM))
    A = rng.normal(size=DIM) + rng.gamma(2.0, 0.4, size=(N_ITEMS, 1)) * rng.normal(size=(N_ITEMS, DIM))
    b = 2.5 * rng.normal(size=N_ITEMS)
    return X.astype(np.float32), A.astype(np.float32), b.astype(np.float32)


@pytest.fixture(scope="module")
def toy():
    return _toy()


def _world(toy, bias="medium", item_bias=True, **params):
    X, A, b = toy
    return generate_dataset({"bias": bias, "ctr": 0.05, **params}, seed=0, emb_x=X, emb_a=A,
                            item_bias=b if item_bias else None)


@pytest.fixture(scope="module")
def worlds(toy):
    out = {
        "taste": _world(toy, item_bias=False),
        "off": _world(toy),  # item bias given, both weights 0 (the default)
        "pop1": _world(toy, pop_strength=1.0),
        "pop1_log3": _world(toy, pop_strength=1.0, logger_pop_strength=3.0),
        "pop1_log0": _world(toy, pop_strength=1.0, logger_pop_strength=0.0),
        "pop0_log2": _world(toy, logger_pop_strength=2.0),
    }
    rb._CALIBRATION_CACHE.clear()
    return out


def _logger(ds):
    return Policy(
        n_users=ds["n_users"], n_items=ds["n_actions"], user_emb=ds["our_x"], item_emb=ds["our_a"],
        emb_dim=ds["emb_dim"], temperature=ds["policy_temperature"], rng=np.random.default_rng(0),
    )


# ------------------------------------------------------------------ world
def test_default_world_ignores_the_item_bias(worlds):
    taste, off = worlds["taste"], worlds["off"]
    assert not off["pop_column"] and off["item_popularity"] is None
    for k in ("emb_x", "emb_a", "our_x", "our_a", "user_prior"):
        np.testing.assert_array_equal(off[k], taste[k])
    assert (off["env"].scale, off["env"].offset) == (taste["env"].scale, taste["env"].offset)
    assert json.dumps(off["world"], sort_keys=True) == json.dumps(taste["world"], sort_keys=True)
    assert describe_world(off["world"]).endswith("popularity: off")


def test_popularity_column_layout(toy, worlds):
    _, _, b = toy
    ds, taste = worlds["pop1_log3"], worlds["taste"]
    d = ds["emb_dim"]
    assert d == DIM and ds["pop_column"]
    for k in ("emb_x", "emb_a", "our_x", "our_a", "original_x", "original_a"):
        assert ds[k].shape[1] == DIM + 1 and ds[k].dtype == np.float32
    for k in ("emb_x", "our_x"):
        np.testing.assert_array_equal(ds[k][:, d], 1.0)
    np.testing.assert_allclose(ds["emb_a"][:, d], 1.0 * b, rtol=1e-6)  # truth: beta_true = 1
    np.testing.assert_allclose(ds["our_a"][:, d], 3.0 * b, rtol=1e-6)  # logger: beta_log = 3
    np.testing.assert_array_equal(ds["item_popularity"], b)
    # the taste parts are the popularity-free world's vectors: same draws, same bias levels
    for k in ("emb_x", "emb_a", "our_x", "our_a"):
        np.testing.assert_array_equal(ds[k][:, :d], taste[k])
    assert ds["world"]["eps_table"] == taste["world"]["eps_table"]
    assert ds["world"]["signal_kept"] == taste["world"]["signal_kept"]
    assert ds["world"]["cosine_to_clean"] == taste["world"]["cosine_to_clean"]
    assert dataset_snr_report(ds) == dataset_snr_report(taste)  # taste parts only
    assert (ds["world"]["pop_strength"], ds["world"]["logger_pop_strength"]) == (1.0, 3.0)
    json.dumps(ds["world"])  # the record stays JSON-able
    assert describe_world(ds["world"]).endswith("popularity: truth 1, logger 3")


def test_truth_targets_hold_with_popularity(worlds):
    ds = worlds["pop1"]
    w = ds["world"]
    assert w["best_item_ctr"] == pytest.approx(0.30, abs=1e-9)  # calibration sample, exact
    assert w["reference_ctr"] == pytest.approx(0.05, abs=1e-9)
    # the reference policy is this world's medium logger, popularity weight included: on the
    # calibration users their CTRs differ only by item-sampling noise (~2e-4)
    assert w["logging_ctr"] == pytest.approx(w["reference_ctr"], abs=1e-3)
    # full population: the calibration sees 1,000 prior-weighted users, so on this 1,500-user toy
    # the values differ by sampling noise (as much as in the taste-only world)
    q = ds["env"].reward_prob_block(np.arange(ds["n_users"]), 0, ds["n_actions"])
    prior = ds["user_prior"].astype(np.float64) / ds["user_prior"].sum()
    assert float(prior @ q.max(axis=1)) == pytest.approx(0.30, abs=0.02)  # best item per user
    assert calc_reward(ds, _logger(ds)) == pytest.approx(0.05, abs=0.003)  # medium logger, beta_log = beta_true
    scores = ds["emb_x"].astype(np.float64) @ ds["emb_a"].astype(np.float64).T
    assert rb._effective_items(scores, ds["policy_temperature"]) / ds["n_actions"] == pytest.approx(0.5, abs=0.02)
    # the clean score is taste + popularity
    X, A = ds["emb_x"][:, :DIM].astype(np.float64), ds["emb_a"][:, :DIM].astype(np.float64)
    np.testing.assert_allclose(scores, X @ A.T + ds["item_popularity"].astype(np.float64), atol=1e-4)
    share = ds["world"]["popularity"]["share_of_score_variation"]
    assert 0.05 < share < 0.9, share


def test_logger_weight_moves_only_the_logger(worlds):
    ref = worlds["pop1"]
    for name in ("pop1_log3", "pop1_log0"):
        ds = worlds[name]
        for k in ("emb_x", "emb_a", "user_prior", "our_x"):
            np.testing.assert_array_equal(ds[k], ref[k])
        np.testing.assert_array_equal(ds["our_a"][:, :DIM], ref["our_a"][:, :DIM])
        assert (ds["env"].scale, ds["env"].offset) == (ref["env"].scale, ref["env"].offset)
        assert ds["policy_temperature"] == ref["policy_temperature"]
        for k in ("alpha", "b", "reference_ctr", "uniform_ctr", "eps_table"):
            assert ds["world"][k] == ref["world"][k]
    # a heavier logger weight over-exposes popular items
    users = np.arange(300)
    b = ref["item_popularity"].astype(np.float64)
    exposure = {name: float((_logger(worlds[name])._probs_block(users) @ b).mean())
                for name in ("pop1_log0", "pop1", "pop1_log3")}
    assert exposure["pop1_log0"] < exposure["pop1"] < exposure["pop1_log3"], exposure
    np.testing.assert_array_equal(worlds["pop1_log0"]["our_a"][:, DIM], 0.0)


def test_truth_without_popularity_logger_with_it(worlds):
    # beta_true = 0, beta_log = 2: the truth is the taste-only one; only the logger weighs b
    taste, ds = worlds["taste"], worlds["pop0_log2"]
    assert ds["pop_column"]
    np.testing.assert_array_equal(ds["emb_a"][:, DIM], 0.0)
    for k in ("alpha", "b", "logging_temperature", "reference_ctr", "uniform_ctr", "best_item_ctr"):
        assert ds["world"][k] == pytest.approx(taste["world"][k], rel=1e-9)
    users = np.arange(200)
    np.testing.assert_allclose(ds["env"].reward_prob_block(users, 0, N_ITEMS),
                               taste["env"].reward_prob_block(users, 0, N_ITEMS), atol=1e-6)
    assert calc_uniform_reward(ds) == pytest.approx(calc_uniform_reward(taste), abs=1e-7)


def test_popularity_needs_the_item_bias_and_valid_weights(toy):
    X, A, b = toy
    for params in ({"pop_strength": 0.5}, {"logger_pop_strength": 1.0}):
        with pytest.raises(ValueError, match="item bias"):
            generate_dataset({"bias": "low", "ctr": 0.05, **params}, seed=0, emb_x=X, emb_a=A)
    for params in ({"pop_strength": -1.0}, {"pop_strength": float("nan")}, {"logger_pop_strength": -0.5},
                   {"pop_strength": 1.0, "logger_pop_strength": float("inf")}):
        with pytest.raises(ValueError, match=">= 0"):
            generate_dataset({"bias": "low", "ctr": 0.05, **params}, seed=0, emb_x=X, emb_a=A, item_bias=b)
    with pytest.raises(ValueError, match="entries"):
        generate_dataset({"bias": "low", "ctr": 0.05, "pop_strength": 1.0}, seed=0, emb_x=X, emb_a=A, item_bias=b[:-1])


def test_click_model_paths_agree_with_popularity(worlds, monkeypatch):
    ds = worlds["pop1_log3"]
    env = ds["env"]
    users = np.arange(150)
    block = env.reward_prob_block(users, 0, N_ITEMS)
    pairs = env.reward_prob(np.repeat(users, N_ITEMS), np.tile(np.arange(N_ITEMS), len(users)))
    np.testing.assert_allclose(pairs.reshape(block.shape), block, atol=1e-6)
    oracle = fit_shared_regression_bundle(ds, {}, reward_model="oracle")
    q = oracle["regression_model"].predict_user_action_block(oracle["user_context"][users], 0, N_ITEMS)
    np.testing.assert_allclose(q[:, :, 0], block, atol=1e-6)
    # logging_score: the env's link on the logger's view, its popularity weight included
    ls = fit_shared_regression_bundle(ds, {}, reward_model="logging_score")["regression_model"]
    z = ds["our_x"][users].astype(np.float64) @ ds["our_a"].astype(np.float64).T
    np.testing.assert_allclose(ls.predict_user_action_block(ds["our_x"][users], 0, N_ITEMS)[:, :, 0],
                               expit(env.scale * z + env.offset), atol=1e-6)

    monkeypatch.setenv("OPC_EXACT_REWARD_DEVICE", "cpu")
    cpu = (calc_uniform_reward(ds), calc_reward(ds, _logger(ds)))
    if torch.cuda.is_available():
        monkeypatch.setenv("OPC_EXACT_REWARD_DEVICE", "auto")
        assert calc_uniform_reward(ds) == pytest.approx(cpu[0], abs=1e-7)
        assert calc_reward(ds, _logger(ds)) == pytest.approx(cpu[1], abs=1e-7)


# ------------------------------------------------------------------ CLI, run keys, runner
def test_cli_options_and_run_keys():
    p = argparse.ArgumentParser()
    rb.add_world_arguments(p)
    opts = world_options_from_args(p.parse_args([]))
    assert opts["pop_strength"] == 0.0 and opts["centering"] == 0.0 and "logger_pop_strength" not in opts
    assert world_run_key_suffix(opts) == ""
    opts = world_options_from_args(p.parse_args(["--pop-strength", "1", "--logger-pop-strength", "2.5"]))
    assert (opts["pop_strength"], opts["logger_pop_strength"]) == (1.0, 2.5)
    assert world_run_key_suffix(opts) == "__pop=1__logpop=2.5"

    assert world_run_key_suffix({}) == ""
    assert world_run_key_suffix({"pop_strength": 1.0}) == "__pop=1"
    assert world_run_key_suffix({"pop_strength": 1.0, "logger_pop_strength": 1.0}) == "__pop=1"  # = truth
    assert world_run_key_suffix({"pop_strength": 1.0, "logger_pop_strength": 0.0}) == "__pop=1__logpop=0"
    assert world_run_key_suffix({"logger_pop_strength": 2.0}) == "__logpop=2"
    assert (world_run_key_suffix({"centering": 0.8, "group_source": "metadata", "ctr_reference": "uniform"})
            == "__center=0.8__groups=metadata__ref=uniform")
    assert world_run_key_suffix({"ctr_reference": "uniform"}, defaults={"ctr_reference": "uniform"}) == ""

    # default worlds keep the folder names of earlier runs (--skip-completed finds them)
    assert _condition_run_key("ml", "low", 0.05, 0, {}) == "dataset=ml__bias=low__ctr=0.05__seed=0"
    assert (_condition_run_key("ml", "low", 0.05, 0, {"pop_strength": 1.0}, "5000")
            == "dataset=ml__bias=low__ctr=0.05__seed=0__pop=1__val=5000")


def test_runner_loads_the_item_bias_only_when_used(tmp_path):
    assert _load_item_bias(tmp_path, "toy", {"pop_strength": 0.0}) is None  # no file needed
    with pytest.raises(FileNotFoundError, match="toy_item_bias.npy"):
        _load_item_bias(tmp_path, "toy", {"pop_strength": 0.5})
    with pytest.raises(FileNotFoundError, match="toy_item_bias.npy"):
        _load_item_bias(tmp_path, "toy", {"pop_strength": 0.0, "logger_pop_strength": 1.0})
    np.save(tmp_path / "toy_item_bias.npy", np.arange(3, dtype=np.float32))
    assert _load_item_bias(tmp_path, "toy", {"pop_strength": 0.0}) is None
    np.testing.assert_array_equal(_load_item_bias(tmp_path, "toy", {"pop_strength": 0.5}), np.arange(3))


# ------------------------------------------------------------------ policy model
def _cf_inputs(seed=0, n_users=40, n_items=30, d=6):
    rng = np.random.default_rng(seed)
    U = torch.tensor(rng.normal(size=(n_users, d)), dtype=torch.float32)
    V = torch.tensor(rng.normal(size=(n_items, d)), dtype=torch.float32)
    return U, V, rng.normal(size=n_items).astype(np.float32)


def test_cfmodel_popularity_term_matches_its_exported_vectors():
    U, V, b = _cf_inputs()
    (nU, d), nA, T = U.shape, V.shape[0], 0.7
    m = CFModel(nU, nA, d, initial_user_embeddings=U.clone(), initial_actions_embeddings=V.clone(),
                user_transform=SingleMLPTransform(d), action_transform=SingleMLPTransform(d),
                temperature=T, item_popularity=b, pop_weight=1.5)
    m.eval()
    with torch.no_grad():
        prob = m(torch.arange(nU))[:, :, 0].numpy()
        ex, ea = (t.numpy() for t in m.get_params())
    assert ex.shape == (nU, d + 1) and ea.shape == (nA, d + 1)
    np.testing.assert_array_equal(ex[:, d], 1.0)
    np.testing.assert_allclose(ea[:, d], 1.5 * b, rtol=1e-6)
    pol = Policy(n_users=nU, n_items=nA, user_emb=ex, item_emb=ea, temperature=T, rng=np.random.default_rng(0))
    np.testing.assert_allclose(pol._probs_block(np.arange(nU)), prob, atol=1e-6)
    # the popularity term is w * b: remove it and the probabilities change
    with torch.no_grad():
        m.pop_weight.fill_(0.0)
        assert not np.allclose(m(torch.arange(nU))[:, :, 0].numpy(), prob, atol=1e-4)


def test_cfmodel_learns_and_clones_its_popularity_weight():
    U, V, b = _cf_inputs(1)
    (nU, d), nA = U.shape, V.shape[0]
    m = CFModel(nU, nA, d, initial_user_embeddings=U.clone(), initial_actions_embeddings=V.clone(),
                item_popularity=b, pop_weight=0.0)
    assert m.pop_weight.requires_grad and any(p is m.pop_weight for p in m.parameters())
    assert "pop_weight" in m.state_dict() and "item_popularity" not in m.state_dict()
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(20):  # push the policy towards high-b items: the weight must grow
        opt.zero_grad()
        loss = -(m(torch.arange(nU))[:, :, 0] @ torch.as_tensor(b)).mean()
        loss.backward()
        opt.step()
    w = float(m.pop_weight.detach())
    assert w > 0.5
    np.testing.assert_allclose(m.get_params()[1][:, d].detach().numpy(), w * b, rtol=1e-5)
    c = m.clone()
    assert float(c.pop_weight.detach()) == w and c.pop_weight is not m.pop_weight
    assert c.item_popularity.data_ptr() != m.item_popularity.data_ptr()
    np.testing.assert_array_equal(c.item_popularity.numpy(), b)
    with torch.no_grad():
        np.testing.assert_allclose(c(torch.arange(nU)).numpy(), m(torch.arange(nU)).numpy(), atol=1e-6)


def test_cfmodel_without_popularity_is_unchanged():
    U, V, _ = _cf_inputs(2)
    (nU, d), nA = U.shape, V.shape[0]
    m = CFModel(nU, nA, d, initial_user_embeddings=U.clone(), initial_actions_embeddings=V.clone(), temperature=0.5)
    assert m.pop_weight is None and m.item_popularity is None
    assert set(m.state_dict()) == {"user_embeddings.weight", "actions_embeddings.weight"}
    ex, ea = m.get_params()
    assert ex.shape == (nU, d) and ea.shape == (nA, d)
    with torch.no_grad():
        expected = torch.softmax((U @ V.T) / 0.5, dim=1)
        torch.testing.assert_close(m(torch.arange(nU))[:, :, 0], expected, rtol=0, atol=0)
    assert m.clone().pop_weight is None


def test_linear_cfmodel_passes_popularity_through():
    U, V, b = _cf_inputs(3)
    (nU, d), nA = U.shape, V.shape[0]
    m = LinearCFModel(nU, nA, d, initial_user_embeddings=U.clone(), initial_actions_embeddings=V.clone(),
                      temperature=0.8, item_popularity=b, pop_weight=2.0)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {"user_transform.delta", "action_transform.delta", "cfmodel.pop_weight"}
    ex, ea = m.get_params()
    np.testing.assert_allclose(ea[:, d].detach().numpy(), 2.0 * b, rtol=1e-6)
    assert float(m.clone().cfmodel.pop_weight.detach()) == 2.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cfmodel_popularity_follows_the_device():
    U, V, b = _cf_inputs(4)
    (nU, d), nA = U.shape, V.shape[0]
    m = CFModel(nU, nA, d, initial_user_embeddings=U.clone(), initial_actions_embeddings=V.clone(),
                user_transform=SingleMLPTransform(d), action_transform=SingleMLPTransform(d),
                item_popularity=b, pop_weight=1.0).to("cuda")
    assert m.item_popularity.is_cuda and m.pop_weight.is_cuda
    m.eval()
    with torch.no_grad():
        gpu = m(torch.arange(nU, device="cuda")).cpu().numpy()
        c = m.clone()
        assert c.item_popularity.is_cuda and c.pop_weight.is_cuda
        m.to("cpu")
        np.testing.assert_allclose(m(torch.arange(nU)).numpy(), gpu, atol=1e-3)  # GPU matmuls may use TF32


# ------------------------------------------------------------------ trainer helpers
def test_trainer_helpers_split_the_popularity_column(worlds):
    ds = worlds["pop1_log3"]
    x, a, kw = _cf_model_inputs(ds, ds["our_x"], ds["our_a"])
    np.testing.assert_array_equal(x, ds["our_x"][:, :DIM])
    np.testing.assert_array_equal(a, ds["our_a"][:, :DIM])
    np.testing.assert_array_equal(kw["item_popularity"], ds["item_popularity"])
    assert kw["pop_weight"] == pytest.approx(3.0, rel=1e-6)
    assert _policy_pop_weight(ds, ds["our_a"]) == pytest.approx(3.0, rel=1e-6)
    assert _policy_pop_weight(ds, ds["emb_a"]) == pytest.approx(1.0, rel=1e-6)
    np.testing.assert_array_equal(_taste_vectors(ds, ds["our_x"]), ds["our_x"][:, :DIM])

    ctx = _reward_action_context(ds, ds["our_a"])
    np.testing.assert_array_equal(ctx[:, DIM], ds["item_popularity"])  # raw b, not the logger's 3·b
    np.testing.assert_array_equal(ctx[:, :DIM], ds["our_a"][:, :DIM])
    np.testing.assert_allclose(ds["our_a"][:, DIM], 3.0 * ds["item_popularity"], rtol=1e-6)  # input untouched

    with pytest.raises(ValueError, match="popularity column"):
        _cf_model_inputs(ds, ds["our_x"][:, :DIM], ds["our_a"][:, :DIM])
    # a CF model built from the split reproduces the logger at initialisation
    m = CFModel(ds["n_users"], ds["n_actions"], DIM, initial_user_embeddings=torch.as_tensor(x),
                initial_actions_embeddings=torch.as_tensor(a), temperature=ds["policy_temperature"], **kw)
    with torch.no_grad():
        ex, ea = (t.numpy() for t in m.get_params())
    np.testing.assert_array_equal(ex, ds["our_x"])
    np.testing.assert_allclose(ea, ds["our_a"], rtol=1e-6)

    taste = worlds["taste"]
    assert _cf_model_inputs(taste, taste["our_x"], taste["our_a"])[2] == {}
    assert _reward_action_context(taste, taste["our_a"]) is taste["our_a"]
    assert _policy_pop_weight(taste, taste["our_a"]) is None
    assert _taste_vectors(taste, taste["our_x"]) is taste["our_x"]


def test_regression_reward_model_sees_raw_popularity(worlds):
    ds = worlds["pop1_log0"]  # the truth rewards popular items, the logger ignores b (zero column)
    split = _build_regression_logged_split(ds, ds["our_x"], ds["our_a"], 1000, 1000, 0, split_seed=1,
                                           regression_size=40_000)
    for features in ("concat", "interaction"):
        model = fit_shared_regression_bundle(ds, split["reg_data"], reward_model="regression",
                                             reward_features=features)["regression_model"]
        np.testing.assert_array_equal(model.action_context[:, DIM], ds["item_popularity"])
        coef = model.base_model_list[0].coef_.reshape(-1)
        # features [x, 1 | a, b (| x*a, 1*b)]: b is item feature 2d+1, and with interaction also the
        # last one (the user's constant column times b), so the two share its weight
        assert coef.shape[0] == (2 if features == "concat" else 3) * (DIM + 1)
        b_weight = coef[2 * DIM + 1] + (coef[-1] if features == "interaction" else 0.0)
        assert b_weight > 0.1, (features, coef)  # clicks rise with popularity
