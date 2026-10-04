"""What the logging bug of 69fffab..dbc401b generated, computed exactly per world.

From 69fffab (2026-09-24) to the fix (dbc401b on CRM), ``_simulate_from_embedding_policy`` seeded the
logger's generator with the simulation's ``random_state``. ``create_simulation_data_from_policy`` drew
row i's user with ``default_rng(random_state).choice(p=user_prior)``, which consumes the i-th uniform
U_i, and ``Policy.sample_actions`` drew row i's action from the same U_i (one uniform per row, inverse CDF
over the items in index order). The pair (user, action) was therefore a function of one uniform:

    user(U)   = the u with  C(u-1) <= U < C(u),          C = normalized cumulative user prior
    action(U) = the a with  F_u(a-1) <= U < F_u(a),      F_u = cumulative pi0(.|u) in item-index order

so the generated conditional is

    P_eff(a | u) = | [C(u-1), C(u)) ∩ [F_u(a-1), F_u(a)) | / prior(u),

while the stored pscore was pi0(a | u). Rewards used later uniforms of the simulation stream, so they were
Bernoulli(q(u, a)) given (u, a), independent of the action draw.

This script computes P_eff exactly over every user of a world and reports, weighted by the user prior:
how many actions P_eff spreads over, how far it is from pi0, the propensity that was recorded vs the one
that generated the row, the logger's value under each, and the expectation of the importance-weighted
estimate of a few target policies under the generated distribution (what IPS converged to with the
recorded pscores). With ``--sample`` it also draws the old and the fixed logs of one split and reports
the per-user repeat rate and corr(user index, action index) on the rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.run_full_study import build_condition_world
from training.trainer_trials import (
    LOGGED_RUN_IDX,
    _logging_uniform_mix,
    _policy_temperature,
    _simulate_from_embedding_policy,
    _split_seed_for_condition,
)
from utils.policies import Policy
from utils.representation_bias import add_world_arguments, resolve_bias_configs, world_options_from_args
from utils.seeding import seed_everything
from utils.simulation_utils import create_simulation_data_from_policy

USER_ROWS = 1024  # users per block of the exact computation
TARGETS = {"logger_x2": 2.0, "logger_x0.5": 0.5, "uniform": 0.0}  # target policy: logits of pi0 times the factor


def _logits(dataset, users):
    x = np.asarray(dataset["our_x"], dtype=np.float64)[users]
    a = np.asarray(dataset["our_a"], dtype=np.float64)
    return (x @ a.T) / max(_policy_temperature(dataset), 1e-8)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def exact_coupling(dataset) -> dict:
    """Prior-weighted summaries of the generated conditional P_eff against pi0 (see the module docstring)."""
    n_users, n_items = int(dataset["n_users"]), int(dataset["n_actions"])
    prior = np.asarray(dataset["user_prior"], dtype=np.float64)
    cdf_u = np.cumsum(prior)
    cdf_u /= cdf_u[-1]  # as Generator.choice normalizes it
    lo_u = np.concatenate([[0.0], cdf_u[:-1]])
    alpha = _logging_uniform_mix(dataset)
    env = dataset["env"]
    acc = {k: 0.0 for k in ("support", "pmax", "coll_eff", "coll_pi0", "tv", "pscore_recorded", "pscore_true",
                            "v_pi0", "v_eff")}
    for name in TARGETS:
        acc[f"{name}_true"] = acc[f"{name}_ips"] = acc[f"{name}_meanw"] = 0.0
    marg_eff = np.zeros(n_items)
    marg_pi0 = np.zeros(n_items)
    for s in range(0, n_users, USER_ROWS):
        users = np.arange(s, min(n_users, s + USER_ROWS))
        z = _logits(dataset, users)
        pi0 = _softmax(z)
        if alpha > 0.0:
            pi0 = (1.0 - alpha) * pi0 + alpha / n_items
        cdf = np.cumsum(pi0, axis=1)
        cdf /= cdf[:, -1:]  # the sampler compares U * cdf[-1] with cdf: the same as U with cdf / cdf[-1]
        prev = np.concatenate([np.zeros((len(users), 1)), cdf[:, :-1]], axis=1)
        lo, hi = lo_u[users][:, None], cdf_u[users][:, None]
        overlap = np.clip(np.minimum(cdf, hi) - np.maximum(prev, lo), 0.0, None)
        w_u = prior[users]
        p_eff = overlap / np.maximum(hi - lo, 1e-300)
        q = env.reward_prob_block(users, 0, n_items).astype(np.float64)
        acc["support"] += float(w_u @ (p_eff > 0).sum(axis=1))
        acc["pmax"] += float(w_u @ p_eff.max(axis=1))
        acc["coll_eff"] += float(w_u @ (p_eff ** 2).sum(axis=1))
        acc["coll_pi0"] += float(w_u @ (pi0 ** 2).sum(axis=1))
        acc["tv"] += float(w_u @ (0.5 * np.abs(p_eff - pi0).sum(axis=1)))
        acc["pscore_recorded"] += float(w_u @ (p_eff * pi0).sum(axis=1))  # E[stored pscore] of a generated row
        acc["pscore_true"] += float(w_u @ (p_eff ** 2).sum(axis=1))  # E[P_eff(a|u)] of a generated row
        acc["v_pi0"] += float(w_u @ (pi0 * q).sum(axis=1))
        acc["v_eff"] += float(w_u @ (p_eff * q).sum(axis=1))
        marg_eff += w_u @ p_eff
        marg_pi0 += w_u @ pi0
        for name, factor in TARGETS.items():
            pe = _softmax(z * factor) if factor > 0 else np.full_like(pi0, 1.0 / n_items)
            w = pe / pi0
            acc[f"{name}_true"] += float(w_u @ (pe * q).sum(axis=1))
            acc[f"{name}_ips"] += float(w_u @ (p_eff * w * q).sum(axis=1))  # E_generated[w r] with the stored pscore
            acc[f"{name}_meanw"] += float(w_u @ (p_eff * w).sum(axis=1))  # E_generated[w]; 1 for a correct log
    out = dict(acc)
    out["marginal_tv"] = float(0.5 * np.abs(marg_eff - marg_pi0).sum())
    out["marginal_items_eff"] = int((marg_eff > 0).sum())
    out["marginal_items_pi0"] = int((marg_pi0 > 0).sum())
    out["marginal_effective_items_eff"] = float(1.0 / (marg_eff ** 2).sum())
    out["marginal_effective_items_pi0"] = float(1.0 / (marg_pi0 ** 2).sum())
    return out


def _sample_without_guard(dataset, policy, n, random_state):
    """``create_simulation_data_from_policy`` as it was before the guard (same draws, same order)."""
    rng = np.random.default_rng(random_state)
    n_users = int(dataset["n_users"])
    users = rng.choice(np.arange(n_users), size=n, p=dataset["user_prior"], replace=True)
    out = {"users": np.zeros(n, np.int32), "actions": np.zeros(n, np.int32), "pscore": np.zeros(n), "reward": np.zeros(n)}
    for s in range(0, n, 100_000):
        e = min(n, s + 100_000)
        a, p = policy.sample_actions(users[s:e])
        qq = dataset["env"].reward_prob(users[s:e], a)
        out["users"][s:e], out["actions"][s:e], out["pscore"][s:e] = users[s:e], a, p
        out["reward"][s:e] = (rng.random(size=e - s) < qq).astype(float)
    return out


def row_statistics(sim) -> dict:
    users, actions = np.asarray(sim["users"], np.int64), np.asarray(sim["actions"], np.int64)
    df = pd.DataFrame({"u": users, "a": actions})
    g = df.groupby("u")["a"]
    multi = g.size() >= 2
    same = (g.nunique() == 1)[multi]
    return {"rows": int(len(users)), "users": int(df["u"].nunique()), "users_ge2": int(multi.sum()),
            "share_users_one_action": float(same.mean()) if len(same) else float("nan"),
            "corr_user_action": float(np.corrcoef(users, actions)[0, 1]), "distinct_actions": int(df["a"].nunique()),
            "mean_reward": float(np.mean(sim["reward"]))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", default=["ml", "kuairand", "anime"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[100])
    ap.add_argument("--ctr", type=float, default=0.05)
    ap.add_argument("--emb-dir", default="BPR/embeddings")
    ap.add_argument("--train-size", type=int, default=25_000, help="split whose seed the row statistics use")
    ap.add_argument("--val-size", type=int, default=20_000)
    ap.add_argument("--regression-size", type=int, default=50_000)
    ap.add_argument("--sample", action="store_true", help="also draw the old and the fixed logs of that split")
    ap.add_argument("--out", required=True)
    add_world_arguments(ap, bias_default=("none", "w-high.g-none.v-none", "high"))
    args = ap.parse_args(argv)
    world_options = world_options_from_args(args)
    rows = []
    for ds in args.datasets:
        for bias in resolve_bias_configs(args.bias_configs):
            for seed in args.seeds:
                seed_everything(seed)
                dataset, _, _, label = build_condition_world(ds, Path(args.emb_dir), bias, args.ctr, seed,
                                                             world_options=world_options)
                row = {"dataset": ds, "bias": label, "seed": seed,
                       "logger_greedy_share": world_options.get("logger_greedy_share"),
                       "n_users": int(dataset["n_users"]), "n_items": int(dataset["n_actions"]),
                       **exact_coupling(dataset)}
                if args.sample:
                    n = args.regression_size + args.train_size + args.val_size
                    rs = _split_seed_for_condition(seed, args.train_size, LOGGED_RUN_IDX)
                    old = _sample_without_guard(dataset, Policy(
                        n_users=int(dataset["n_users"]), n_items=int(dataset["n_actions"]), user_emb=dataset["our_x"],
                        item_emb=dataset["our_a"], emb_dim=int(dataset["our_x"].shape[1]),
                        temperature=_policy_temperature(dataset), user_chunk=10_000, action_chunk=10_000,
                        rng=np.random.default_rng(int(rs) % (2**31 - 1)), uniform_mix=_logging_uniform_mix(dataset)),
                        n, rs)
                    new = _simulate_from_embedding_policy(dataset, dataset["our_x"], dataset["our_a"], n, rs)
                    row.update({f"old_{k}": v for k, v in row_statistics(old).items()})
                    row.update({f"new_{k}": v for k, v in row_statistics(new).items()})
                rows.append(row)
                print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
