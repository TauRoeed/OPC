"""Policy probabilities are a proper distribution and both samplers draw from it."""

import os

import numpy as np
import pytest
import torch
from scipy import stats

from utils.policies import SAMPLER_DEVICE_ENV, Policy

N_USERS, N_ITEMS, DIM = 50, 25, 6


def _policy(seed=0, uniform_mix=0.0, action_chunk=8192):
    rng = np.random.default_rng(123)
    return Policy(
        n_users=N_USERS,
        n_items=N_ITEMS,
        user_emb=rng.standard_normal((N_USERS, DIM)).astype(np.float32),
        item_emb=rng.standard_normal((N_ITEMS, DIM)).astype(np.float32),
        emb_dim=DIM,
        temperature=0.7,
        action_chunk=action_chunk,
        uniform_mix=uniform_mix,
        rng=np.random.default_rng(seed),
    )


def _reference_probs(pol, users):
    logits = (pol.user_emb[users].astype(np.float64) @ pol.item_emb.astype(np.float64).T) / pol.temperature
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def test_probs_block_is_normalized_over_all_items():
    users = np.arange(10)
    for chunk in (8192, 7):  # one block, and several blocks (the old per-block bug)
        probs = _policy(action_chunk=chunk)._probs_block(users)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-12)
        np.testing.assert_allclose(probs, _reference_probs(_policy(), users), rtol=1e-5)


_devices = ["cpu"] + (["auto"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("uniform_mix", [0.0, 0.3])
def test_sampler_matches_policy_probabilities(device, uniform_mix, monkeypatch):
    monkeypatch.setenv(SAMPLER_DEVICE_ENV, device)
    pol = _policy(seed=5, uniform_mix=uniform_mix)
    user, n = 4, 40_000
    actions, pscore = pol.sample_actions(np.full(n, user))

    expected = _reference_probs(pol, [user])[0]
    expected = (1 - uniform_mix) * expected + uniform_mix / N_ITEMS
    counts = np.bincount(actions, minlength=N_ITEMS)
    assert stats.chisquare(counts, expected * n).pvalue > 1e-4  # fixed seed: deterministic

    np.testing.assert_allclose(pscore, pol.prob_actions(np.full(n, user), actions), rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("uniform_mix", [0.0, 0.3])
def test_cpu_and_gpu_draw_identical_samples(uniform_mix, monkeypatch):
    users = np.arange(N_USERS).repeat(400)
    out = {}
    for device in ("cpu", "auto"):
        monkeypatch.setenv(SAMPLER_DEVICE_ENV, device)
        out[device] = _policy(seed=3, uniform_mix=uniform_mix).sample_actions(users)
    assert np.array_equal(out["cpu"][0], out["auto"][0])
    np.testing.assert_allclose(out["cpu"][1], out["auto"][1], rtol=1e-12)


@pytest.mark.parametrize("device", _devices)
def test_sampler_reproducible_per_seed(device, monkeypatch):
    monkeypatch.setenv(SAMPLER_DEVICE_ENV, device)
    users = np.arange(N_USERS).repeat(20)
    a1, _ = _policy(seed=9).sample_actions(users)
    a2, _ = _policy(seed=9).sample_actions(users)
    a3, _ = _policy(seed=10).sample_actions(users)
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, a3)
