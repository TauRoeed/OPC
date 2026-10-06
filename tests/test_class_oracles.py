"""training/class_oracles.py: every class starts at the logger, the ranking vectors reproduce f up to user terms, and
both objectives improve on the toy world."""
from pathlib import Path

import numpy as np
import pytest
import torch

from training.class_oracles import CLASSES, ScoreClass, fit_class_oracle
from utils.simulation_utils import calc_greedy_reward


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    from test_reproducibility import _toy_embeddings
    from training.run_full_study import build_condition_world
    from utils.seeding import seed_everything

    root = tmp_path_factory.mktemp("class_oracles")
    _toy_embeddings(root)
    seed_everything(0)
    dataset, *_ = build_condition_world("toy", Path(root), "medium", 0.05, 0, world_options={})
    return dataset


@pytest.mark.parametrize("cls", CLASSES)
def test_every_class_starts_at_the_logger(toy, cls):
    from training.trainer_trials import _policy_greedy_reward_from_embeddings, _policy_temperature

    m = ScoreClass(toy["our_x"], toy["our_a"], cls, _policy_temperature(toy))
    ux, ia = m.ranking_vectors()
    assert calc_greedy_reward(toy, ux, ia) == pytest.approx(
        _policy_greedy_reward_from_embeddings(toy, toy["our_x"], toy["our_a"]), abs=1e-7)


@pytest.mark.parametrize("cls", CLASSES)
def test_ranking_vectors_equal_scores_up_to_user_terms(toy, cls):
    m = ScoreClass(toy["our_x"], toy["our_a"], cls, 0.5)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_()
    users = torch.arange(5)
    f = m.scores(users).detach().numpy()
    ux, ia = m.ranking_vectors()
    g = ux[:5] @ ia.T
    d = f - g  # constant in a for every user
    np.testing.assert_allclose(d - d[:, :1], 0.0, atol=1e-4)


def test_value_oracle_does_not_lose_and_likelihood_oracle_improves_its_objective(toy):
    from training.trainer_trials import _policy_greedy_reward_from_embeddings, _policy_temperature

    logger = _policy_greedy_reward_from_embeddings(toy, toy["our_x"], toy["our_a"])
    m, trace, obj = fit_class_oracle(toy, "blob", "value", lr=3e-3, steps=60, fit_users=200, batch_users=50, device="cpu")
    ux, ia = m.ranking_vectors()
    assert calc_greedy_reward(toy, ux, ia) >= logger - 0.02
    assert trace[-1] <= trace[0] + 1e-6  # the loss is -value
    m2, trace2, obj2 = fit_class_oracle(toy, "affine_bilinear", "likelihood", lr=3e-3, steps=60, fit_users=200,
                                        batch_users=50, device="cpu")
    assert trace2[-1] < trace2[0] and np.isfinite(obj2)


@pytest.mark.parametrize("cls", CLASSES)
def test_click_offset_completes_the_ranking_vectors_to_the_scores(toy, cls):
    m = ScoreClass(toy["our_x"], toy["our_a"], cls, 0.5)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_()
    ux, ia = m.ranking_vectors()
    f = m.scores(torch.arange(toy["n_users"])).detach().numpy()
    np.testing.assert_allclose(ux @ ia.T + m.click_offset()[:, None], f, atol=1e-4)


def test_the_weighted_likelihood_objectives_weight_items_as_defined():
    """uniform_likelihood weights every item 1/P; clip10_likelihood min(1/P, 10 π0) (docs/shared_objective_study.md
    §6); likelihood the logger's probabilities."""
    from training.class_oracles import _logger_probs, likelihood_weights

    g = torch.Generator().manual_seed(0)
    x, a = torch.randn(5, 4, generator=g), 3 * torch.randn(50, 4, generator=g)
    pi = _logger_probs(x, a, 0.5)
    torch.testing.assert_close(likelihood_weights("likelihood", x, a, 0.5), pi)
    torch.testing.assert_close(likelihood_weights("uniform_likelihood", x, a, 0.5), torch.full((5, 50), 1 / 50))
    clip = likelihood_weights("clip10_likelihood", x, a, 0.5)
    torch.testing.assert_close(clip, torch.minimum(torch.full_like(pi, 1 / 50), 10 * pi))
    assert bool((clip <= 1 / 50 + 1e-12).all()) and bool((clip < 1 / 50).any())  # a sharp logger: some items clipped
    with pytest.raises(ValueError):
        likelihood_weights("value", x, a, 0.5)


@pytest.mark.parametrize("objective", ["uniform_likelihood", "clip10_likelihood"])
def test_the_weighted_likelihood_oracles_improve_their_objective(toy, objective):
    from training.class_oracles import fit_class_oracle

    start, _, obj0 = fit_class_oracle(toy, "bilinear", objective, lr=1e-2, steps=1, fit_users=512, batch_users=256)
    model, trace, obj = fit_class_oracle(toy, "bilinear", objective, lr=1e-2, steps=200, fit_users=512,
                                         batch_users=256)
    assert obj < obj0 and np.isfinite(obj)
