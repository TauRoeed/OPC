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
