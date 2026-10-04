"""CausE-warm and CausE-capacity-matched (docs/cause_fair_comparison_25k.md §1): the warm start, the capacity match
with OPC's policy class, rho = 0 behavior of the tie, batched training, and both arms end to end with tempering."""
import numpy as np
import pytest
import torch

from models.cause import (ALPHA_INIT, CausELayout, CausELinModel, CausEModel, fit_cause, fit_cause_batch,
                          warm_start_)
from models.models import CFModel, GlobalLinearCorrection

D = 8


def _source(n_users=30, n_items=40, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_users, D)).astype(np.float32),
            rng.standard_normal((n_items, D)).astype(np.float32))


def _data(n_users, n_items, n=400, treated_share=0.3, seed=3):
    rng = np.random.default_rng(seed)
    users = rng.integers(0, n_users, n)
    actions = rng.integers(0, n_items, n)
    labels = (rng.random(n) < 0.3).astype(np.float32)
    return users, actions, rng.random(n) < treated_share, labels


def _warm(x, a, layout):
    return warm_start_(CausEModel.for_layout(layout, len(x), D, generator=torch.Generator().manual_seed(0)), x, a)


def test_warm_start_puts_users_and_both_item_tables_at_the_source():
    x, a = _source()
    m = _warm(x, a, CausELayout("prod", len(a)))
    assert torch.equal(m.user_emb, torch.as_tensor(x))
    assert torch.equal(m.item_emb[:len(a)], torch.as_tensor(a)) and torch.equal(m.item_emb[len(a):], torch.as_tensor(a))
    assert float(m.alpha) == pytest.approx(ALPHA_INIT)
    assert not m.user_bias.any() and not m.item_bias.any() and not m.global_bias.any()
    with pytest.raises(ValueError):
        warm_start_(CausEModel.for_layout(CausELayout("avg", len(a)), len(x), D), x, a)


def test_capacity_matched_logits_equal_opc_policy_logits():
    """OPC's policy (CFModel with a GlobalLinearCorrection per side, logit scale s, logger temperature T) and the
    capacity-matched treatment prediction with the same maps and alpha = s / T give the same logits."""
    x, a = _source()
    temperature, scale = 0.7, 2.5
    opc = CFModel(len(x), len(a), D, initial_user_embeddings=torch.as_tensor(x),
                  initial_actions_embeddings=torch.as_tensor(a), user_transform=GlobalLinearCorrection(D),
                  action_transform=GlobalLinearCorrection(D), temperature=temperature, logit_scale=scale)
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for mod in (opc.user_transform, opc.action_transform):
            mod.delta.copy_(0.3 * torch.randn(mod.delta.shape, generator=gen))
            mod.bias.copy_(0.3 * torch.randn(mod.bias.shape, generator=gen))
        ex, ea = opc.get_params()
        opc_logits = (ex @ ea.T / temperature).numpy()
    cap = CausELinModel(x, a)
    with torch.no_grad():
        cap.user_map.load_state_dict(opc.user_transform.state_dict())
        cap.treatment_map.load_state_dict(opc.action_transform.state_dict())
        cap.alpha.fill_(scale / temperature)
    rows = CausELayout("prod", len(a)).prediction_rows("treatment")
    users = np.repeat(np.arange(len(x)), len(a))
    with torch.no_grad():
        z = cap.logits(torch.as_tensor(users), torch.as_tensor(np.tile(rows, len(x)))).numpy().reshape(len(x), len(a))
    np.testing.assert_allclose(z, opc_logits, rtol=1e-5, atol=1e-5)
    ux, ia = cap.policy_vectors(rows)  # what the exact evaluation scores
    np.testing.assert_allclose(ux @ ia.T, opc_logits, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("family", ["warm", "cap"])
def test_without_uniform_rows_only_the_symmetric_tie_moves_the_treatment_side(family):
    """rho = 0: the treatment side gets no data. The released one-way tie sends it no gradient either, so it stays at
    the source; the symmetric tie (eq. 18) pulls it toward the control side."""
    x, a = _source()
    layout = CausELayout("prod", len(a))
    users, actions, _treated, labels = _data(len(x), len(a))
    rows = layout.train_rows(actions, np.zeros(len(actions), bool))

    def treatment(m):
        if family == "cap":
            return torch.cat([m.treatment_map.delta.reshape(-1), m.treatment_map.bias]).detach().clone()
        return torch.cat([m.item_emb[len(a):].reshape(-1), m.item_bias[len(a):]]).detach().clone()

    for symmetric in (False, True):
        m = CausELinModel(x, a) if family == "cap" else _warm(x, a, layout)
        before = treatment(m)
        fit_cause(m, users, rows, labels, epochs=3, batch_size=64, lr=0.5, l2_pen=0.0, cf_pen=1.0, symmetric=symmetric)
        assert (not torch.equal(treatment(m), before)) == symmetric


def test_capacity_matched_trials_trained_together_equal_separate_runs():
    x, a = _source()
    layout = CausELayout("prod", len(a))
    users, actions, treated, labels = _data(len(x), len(a))
    rows = layout.train_rows(actions, treated)
    configs = [(0.5, 0.0, 1.0), (0.2, 1e-3, 0.1), (0.05, 0.0, 10.0)]
    singles = []
    for lr, l2, cf in configs:
        m = CausELinModel(x, a)
        fit_cause(m, users, rows, labels, epochs=2, batch_size=64, lr=lr, l2_pen=l2, cf_pen=cf, seed=7)
        singles.append(m)
    batch = [CausELinModel(x, a) for _ in configs]
    info = fit_cause_batch(batch, users, rows, labels, epochs=2, batch_size=64, lrs=[c[0] for c in configs],
                           l2_pens=[c[1] for c in configs], cf_pens=[c[2] for c in configs], order_seed=7)
    assert info["finite"].all()
    for s, b in zip(singles, batch):
        for ps, pb in zip(s.parameters(), b.parameters()):
            torch.testing.assert_close(pb, ps, rtol=1e-5, atol=1e-6)
        assert not torch.equal(s.treatment_map.delta, torch.zeros_like(s.treatment_map.delta))  # it trained


@pytest.fixture(scope="module")
def fair_runs(tmp_path_factory):
    from test_reproducibility import _toy_embeddings
    from training.run_full_study import _run_condition

    root = tmp_path_factory.mktemp("cause_fair")
    _toy_embeddings(root)
    out = {}
    for family in ("warm", "cap"):
        run_dir = root / family
        run_dir.mkdir()
        options = {"rhos": [0.0, 0.25], "dim": D, "batch_size": 128, "n_trials": 4, "family": family,
                   "ties": ["one_way", "symmetric"], "temper": True}
        out[family] = _run_condition(dataset_name="toy", emb_dir=root, bias="medium", ctr=0.05, seed=0,
                                     train_sizes=[1000], n_trials=2, batch_size=None, val_size=1000, val_frac=0.15,
                                     val_min=1000, val_max=None, policy_reward_mode="exact", policy_reward_mc_sim=8,
                                     run_dir=run_dir, slim=True, shared_regression_size=2000, methods=("cause",),
                                     sampler="random", return_extra=True, cause_options=options)
    return out


@pytest.mark.parametrize("family", ["warm", "cap"])
def test_both_families_end_to_end_with_tempering(fair_runs, family):
    from training.cause_trials import cause_method_label

    extra = fair_runs[family][5]
    expected = {cause_method_label(p, r, family) for p in ("c", "t") for r in (0.0, 0.25)}
    assert set(extra) == expected
    for label in expected:
        summary, trials = extra[label]
        row = summary.loc[1000]
        assert row["cause_family"] == family and row["n_total"] == 1000 and row["n_trials"] == 4 == len(trials)
        assert set(trials["tie"]) <= {"one_way", "symmetric"}
        for col in ("policy_rewards", "policy_rewards_greedy", "policy_rewards_tempered", "val_dr_greedy",
                    "val_dr_tempered_low"):
            assert np.isfinite(row[col]), col
        assert row["temper_scale"] > 0 and row["val_dr_tempered_low"] <= row["val_dr_tempered"]
        assert row["oracle_selected_value_greedy"] >= row["policy_rewards_greedy"] - 1e-12
