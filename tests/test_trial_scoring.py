"""Trial scoring on the training device (``_split_dr_vec_and_ess``): the same values as the
numpy reference it replaced, on the CPU and on the GPU, for every reward-model kind."""

import numpy as np
import pytest
import torch

import training.trainer_trials as tt
from training.trainer_trials import (
    RegressionScoresLookup,
    _batched_pi_at_logged_actions,
    _build_regression_logged_split,
    _dm_reward_rows_chunked,
    _policy_row_values,
    _scoring_device,
    _split_dr_vec_and_ess,
    fit_shared_regression_bundle,
    uses_importance_weighting,
)
from utils.importance_weights import transform_weights
from utils.simulation_utils import generate_dataset

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
N_USERS, N_ITEMS, DIM = 1500, 1200, 12  # the popularity tests' toy: calibrates at the defaults


def _reference_dr_vec_and_ess(split_data, trial_x, trial_a, score_lookup, dataset, *, propensity_mode="logged", dr_clip_m=None,
                              weights=None):
    """The numpy implementation before the device path (kept here as the reference)."""
    pscore = np.asarray(split_data["pscore"], dtype=np.float32)
    users = np.asarray(split_data["x_idx"], dtype=np.int64)
    reward = np.asarray(split_data["r"], dtype=np.float32)
    actions = np.asarray(split_data["a"], dtype=np.int64)
    pt = tt._policy_temperature(dataset)
    pi = _batched_pi_at_logged_actions(trial_x, trial_a, users, actions, chunk_size=score_lookup.user_chunk,
                                       action_chunk=score_lookup.action_chunk, policy_temperature=pt)
    if not uses_importance_weighting(propensity_mode):
        v = (reward * pi).astype(np.float32)
        return v, float(len(v)), float(len(v))
    q_f = np.asarray(score_lookup.regression_model.predict_pairs(score_lookup.user_context[users], actions), dtype=np.float32)
    dm = _dm_reward_rows_chunked(users, trial_x, trial_a, score_lookup, policy_temperature=pt).astype(np.float32)
    raw = pi / (pscore + 1e-12)
    if weights is None:
        weights = ("clip", dr_clip_m) if dr_clip_m is not None and np.isfinite(float(dr_clip_m)) else "none"
    iw = transform_weights(raw, weights)
    ess = lambda w: float((w.sum() ** 2) / ((w**2).sum() + 1e-12))
    return dm + iw * (reward - q_f), ess(iw), ess(raw)


def _world(pop=False):
    rng = np.random.default_rng(0)
    X = (rng.normal(size=DIM) + 0.6 * rng.normal(size=(N_USERS, DIM))).astype(np.float32)
    A = (rng.normal(size=DIM) + rng.gamma(2.0, 0.4, size=(N_ITEMS, 1)) * rng.normal(size=(N_ITEMS, DIM))).astype(np.float32)
    params = {"bias": "medium", "ctr": 0.05}
    if pop:
        params.update(pop_strength=1.0, logger_pop_strength=2.0)
    return generate_dataset(params, seed=0, emb_x=X, emb_a=A, item_bias=2.5 * rng.normal(size=N_ITEMS).astype(np.float32))


@pytest.fixture(scope="module", params=[False, True], ids=["taste", "popularity"])
def setup(request):
    ds = _world(pop=request.param)
    split = _build_regression_logged_split(ds, ds["our_x"], ds["our_a"], 3000, 1500, 0, split_seed=1, regression_size=20_000)
    rng = np.random.default_rng(5)
    trial_x = ds["our_x"] + 0.3 * rng.normal(size=ds["our_x"].shape).astype(np.float32)  # a trained-looking policy
    trial_a = ds["our_a"] + 0.3 * rng.normal(size=ds["our_a"].shape).astype(np.float32)
    return ds, split, trial_x, trial_a


_LOOKUPS: dict = {}


def _lookups(ds, split, device):
    """(name, lookup) for every reward-model kind the trainer can score with (cached per world)."""
    key = (id(ds), str(device))
    if key in _LOOKUPS:
        return _LOOKUPS[key]
    out = []
    # no dense cache: the toy catalog is small enough to materialize q_hat, the large ones are not,
    # and those take the reward models' closed forms on the device
    for features in ("interaction", "concat"):
        b = fit_shared_regression_bundle(ds, split["reg_data"], reward_model="regression", reward_features=features,
                                         materialize_qhat="never")
        assert b["regression_model"].features == features
        out.append((f"regression-{features}", tt._scores_lookup_from_bundle(b, device)))
    for kind in ("logging_score", "oracle"):
        b = fit_shared_regression_bundle(ds, split["reg_data"], reward_model=kind, materialize_qhat="never")
        out.append((kind, tt._scores_lookup_from_bundle(b, device)))
    b = fit_shared_regression_bundle(ds, {}, reward_model="oracle", q_error=0.3, q_bad_value=0.08, materialize_qhat="never")
    out.append(("bounded", tt._scores_lookup_from_bundle(b, device)))
    b = fit_shared_regression_bundle(ds, split["reg_data"], reward_model="regression", materialize_qhat="always")
    assert b.get("q_hat_all") is not None
    out.append(("dense", tt._scores_lookup_from_bundle(b, device)))
    assert all(lk.q_hat_all is None for _, lk in out[:-1])
    _LOOKUPS[key] = out
    return out


@pytest.mark.parametrize("device", DEVICES)
def test_qhat_blocks_match_numpy(setup, device):
    ds, split, _, _ = setup
    users = np.array([0, 5, 5, 17, N_USERS - 1])
    for name, lk in _lookups(ds, split, device):
        fn = lk.qhat_block_fn(device)
        for a0, a1 in ((0, 64), (300, N_ITEMS)):
            with torch.no_grad():
                prev = torch.get_float32_matmul_precision()
                torch.set_float32_matmul_precision("highest")
                got = fn(users, a0, a1).cpu().numpy()
                torch.set_float32_matmul_precision(prev)
            np.testing.assert_allclose(got, lk.qhat_block_numpy(users, a0, a1), rtol=0, atol=2e-7, err_msg=name)


@pytest.mark.parametrize("device", DEVICES)
def test_row_values_match_numpy_and_do_not_depend_on_blocks(setup, device):
    ds, split, trial_x, trial_a = setup
    data = split["train_data"]
    users, actions = data["x_idx"], data["a"]
    lk = dict(_lookups(ds, split, "cpu"))["regression-interaction"]
    pt = ds["policy_temperature"]
    ref_pi = _batched_pi_at_logged_actions(trial_x, trial_a, users, actions, action_chunk=512, policy_temperature=pt)
    ref_dm = _dm_reward_rows_chunked(users, trial_x, trial_a, lk, policy_temperature=pt)
    pi, dm = _policy_row_values(trial_x, trial_a, users, actions, policy_temperature=pt, device=device,
                                qhat_fn=lk.qhat_block_fn(device))
    np.testing.assert_allclose(pi, ref_pi, rtol=2e-5)
    np.testing.assert_allclose(dm, ref_dm, rtol=2e-5)
    small = _policy_row_values(trial_x, trial_a, users, actions, policy_temperature=pt, device=device,
                               qhat_fn=lk.qhat_block_fn(device), action_chunk=97, block_cells=97 * 400)
    np.testing.assert_allclose(small[0], pi, rtol=1e-6)
    np.testing.assert_allclose(small[1], dm, rtol=1e-6)
    np.testing.assert_allclose(_policy_row_values(trial_x, trial_a, users, actions, policy_temperature=pt, device=device)[0], pi, rtol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_split_scores_match_the_numpy_reference(setup, device, monkeypatch):
    ds, split, trial_x, trial_a = setup
    for name, lk in _lookups(ds, split, device):
        for data in (split["train_data"], split["val_data"]):
            for mode, clip, weights in (("logged", 1.0, None), ("logged", None, None), ("uniform", None, None),
                                        ("logged", None, "clip:5"), ("logged", None, "shrink:25")):
                got_v, got_ess, got_raw = _split_dr_vec_and_ess(data, trial_x, trial_a, lk, ds, propensity_mode=mode,
                                                                dr_clip_m=clip, weights=weights)
                ref_v, ref_ess, ref_raw = _reference_dr_vec_and_ess(data, trial_x, trial_a, lk, ds, propensity_mode=mode,
                                                                    dr_clip_m=clip, weights=weights)
                np.testing.assert_allclose(got_v, ref_v, rtol=1e-4, atol=1e-6, err_msg=f"{name} {mode} {clip} {weights}")
                assert got_ess == pytest.approx(ref_ess, rel=1e-4) and got_raw == pytest.approx(ref_raw, rel=1e-4)
                assert float(got_v.mean()) == pytest.approx(float(ref_v.mean()), rel=1e-5, abs=1e-8)


def test_cpu_and_gpu_agree(setup):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    ds, split, trial_x, trial_a = setup
    for (name, cpu), (_, gpu) in zip(_lookups(ds, split, "cpu"), _lookups(ds, split, "cuda")):
        c = _split_dr_vec_and_ess(split["val_data"], trial_x, trial_a, cpu, ds, dr_clip_m=1.0)
        g = _split_dr_vec_and_ess(split["val_data"], trial_x, trial_a, gpu, ds, dr_clip_m=1.0)
        np.testing.assert_allclose(g[0], c[0], rtol=1e-5, atol=1e-7, err_msg=name)


def test_scoring_device_follows_lookup_and_env(monkeypatch):
    class L:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    assert _scoring_device(L()).type == ("cuda" if torch.cuda.is_available() else "cpu")
    monkeypatch.setenv(tt.SCORING_DEVICE_ENV, "cpu")
    assert _scoring_device(L()).type == "cpu"


def test_model_without_closed_form_uses_the_numpy_block(setup):
    ds, split, trial_x, trial_a = setup
    b = fit_shared_regression_bundle(ds, split["reg_data"], reward_model="regression")

    class Opaque:  # a reward model the device path does not know
        def __init__(self, m):
            self.m, self.len_list, self.n_actions, self.action_context = m, 1, m.n_actions, m.action_context

        def predict_pairs(self, c, a, pos=0):
            return self.m.predict_pairs(c, a, pos)

        def predict_user_action_block(self, c, a0, a1):
            return self.m.predict_user_action_block(c, a0, a1)

    lk = RegressionScoresLookup(Opaque(b["regression_model"]), b["user_context"], DEVICES[-1])
    got = _split_dr_vec_and_ess(split["val_data"], trial_x, trial_a, lk, ds, dr_clip_m=1.0)
    ref = _reference_dr_vec_and_ess(split["val_data"], trial_x, trial_a, lk, ds, dr_clip_m=1.0)
    np.testing.assert_allclose(got[0], ref[0], rtol=1e-4, atol=1e-6)


def test_trainer_selection_matches_the_numpy_reference(tmp_path, monkeypatch):
    """A toy condition scored with the device path and with the numpy reference: the same trial
    values up to rounding, hence the same selected trial."""
    from test_reproducibility import _run, _toy_embeddings

    _toy_embeddings(tmp_path)
    new_summary, new_trials = _run(tmp_path, seed=0, tag="new")
    monkeypatch.setattr(tt, "_split_dr_vec_and_ess", _reference_dr_vec_and_ess)
    ref_summary, ref_trials = _run(tmp_path, seed=0, tag="ref")
    for col in ("r_hat", "r_hat_train", "value", "ess"):
        np.testing.assert_allclose(new_trials[col], ref_trials[col], rtol=1e-4, atol=1e-7, err_msg=col)
    np.testing.assert_array_equal(new_trials["is_best_in_run"], ref_trials["is_best_in_run"])
    np.testing.assert_allclose(new_summary["policy_rewards"], ref_summary["policy_rewards"], rtol=1e-6)
