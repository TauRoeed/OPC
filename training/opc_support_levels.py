"""The logging-support levels of docs/opc_gradient_regime_study.md §5, from propensity and overlap diagnostics only.

For each world and candidate logger greedy share, the world is rebuilt with that share. It must keep the click model
bit for bit; otherwise the run stops. Then, exact over users (prior) and items:
- the logger's effective number of items per user, exp(entropy), prior-weighted;
- the quantiles of the logged propensities (a prior-drawn user sample, one action each from the logger);
- for two fixed targets that do not depend on the logger (the Stage 8A value-optimum and mid policies, as
  distributions over items), the population ESS share 1 / E_π0[(π/π0)²] and the target's mass on items with
  π0 < 1e-4.

The rule (§5): poor support is the candidate whose geometric-mean effective items over the worlds is closest to 1/4 of
the current share's (0.8), better support the one closest to 4×.

    python -m training.opc_support_levels --states-run artifacts/full_study/run_opc_gradients_8a --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training.opc_gradient_benchmark import build_world
from training.opc_gradients import WorldTensors, _policy, build_model, set_theta
from utils.seeding import derive_seed

CANDIDATES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)
CURRENT = 0.8
FACTOR = 4.0
TARGETS = ("value", "mid", "mid_greedy")


def same_click_model(a: dict, b: dict, users: np.ndarray) -> bool:
    ea, eb = a["env"], b["env"]
    if float(ea.scale) != float(eb.scale) or float(ea.offset) != float(eb.offset):
        return False
    if not (np.array_equal(ea.emb_x, eb.emb_x) and np.array_equal(ea.emb_a, eb.emb_a)):
        return False
    n = int(a["n_actions"])
    return bool(np.array_equal(ea.reward_prob_block(users, 0, n), eb.reward_prob_block(users, 0, n)))


@torch.no_grad()
def logger_diagnostics(dataset: dict, world: WorldTensors, targets: dict, *, seed: int, chunk: int = 2048) -> dict:
    """The logger's effective items and propensity quantiles, and each target's overlap with it."""
    from training.opc_gradients import _prior

    prior = _prior(dataset)
    n_users = int(dataset["n_users"])
    eff, second = 0.0, {k: 0.0 for k in targets}
    low = {k: 0.0 for k in targets}
    for s in range(0, n_users, chunk):
        u = torch.arange(s, min(s + chunk, n_users), device=world.device)
        p0 = world.logger(u).double()
        pr = torch.as_tensor(prior[s:s + chunk], device=world.device)
        ent = -(p0 * torch.log(p0.clamp(min=1e-300))).sum(dim=1)
        eff += float((torch.exp(ent) * pr).sum())
        for k, model in targets.items():
            pi = _policy(model, u).double()
            second[k] += float(((pi ** 2 / p0.clamp(min=1e-300)).sum(dim=1) * pr).sum())
            low[k] += float(((pi * (p0 < 1e-4)).sum(dim=1) * pr).sum())
    rng = np.random.default_rng(derive_seed(int(seed), "support_levels_sample"))
    users = rng.choice(n_users, size=20_000, p=prior)
    ps = []
    for s in range(0, len(users), chunk):
        u = torch.as_tensor(users[s:s + chunk], device=world.device)
        p0 = world.logger(u).double()
        a = torch.multinomial(p0, 1, generator=torch.Generator(device=world.device).manual_seed(s)).squeeze(1)
        ps.append(p0[torch.arange(len(u), device=world.device), a].cpu().numpy())
    ps = np.concatenate(ps)
    out = {"effective_items": eff, **{f"p_q{q * 100:g}": float(np.quantile(ps, q)) for q in (0.01, 0.1, 0.5, 0.9)}}
    for k in targets:
        out[f"ess_share_{k}"] = 1.0 / second[k]
        out[f"low_p0_mass_{k}"] = low[k]
    return out


def main(argv=None) -> None:
    from utils.representation_bias import add_world_arguments, resolve_bias_configs, world_options_from_args

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states-run", required=True, help="the Stage 8A benchmark run (its states at the current share)")
    ap.add_argument("--datasets", nargs="+", default=["ml", "kuairand", "anime"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[100])
    ap.add_argument("--candidates", nargs="+", type=float, default=list(CANDIDATES))
    ap.add_argument("--emb-dir", default="BPR/embeddings")
    ap.add_argument("--out", required=True)
    add_world_arguments(ap, bias_default=("none", "high/none/none", "none/high/none", "none/none/high", "high"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = world_options_from_args(args)
    rows = []
    for seed in args.seeds:
        for ds in args.datasets:
            for bias in resolve_bias_configs(args.bias_configs):
                current, label = build_world(ds, bias, seed, Path(args.emb_dir), {**base, "logger_greedy_share": CURRENT})
                wdir = Path(args.states_run) / f"dataset={ds}__bias={label}__seed={seed}__lgs={CURRENT:g}"
                states = np.load(wdir / "states.npz")
                world_cur = WorldTensors(current)
                targets = {}
                for k in TARGETS:  # the target policies at the current parameterization (T of the current logger)
                    m = build_model(current, device=world_cur.device)
                    set_theta(m, states[k])
                    targets[k] = m
                check_users = np.arange(min(2000, int(current["n_users"])))
                for share in args.candidates:
                    ds_s, _ = build_world(ds, bias, seed, Path(args.emb_dir), {**base, "logger_greedy_share": share})
                    if not same_click_model(current, ds_s, check_users):
                        raise SystemExit(f"STOP: logger share {share} changes the click model of {wdir.name} (§5, §11)")
                    world_s = WorldTensors(ds_s, world_cur.device)
                    row = {"dataset": ds, "bias": label, "seed": int(seed), "share": float(share),
                           "logging_temperature": float(ds_s["policy_temperature"]), "same_click_model": True,
                           **logger_diagnostics(ds_s, world_s, targets, seed=seed)}
                    rows.append(row)
                    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
                    del ds_s, world_s
                del current, targets
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(out / "support_diagnostics.csv", index=False, float_format="%.6g")
    geo = df.groupby("share").apply(lambda g: pd.Series({
        "effective_items_geomean": float(np.exp(np.log(g["effective_items"]).mean())),
        **{f"{c}_geomean": float(np.exp(np.log(g[c].clip(lower=1e-300)).mean())) for c in df.columns if c.startswith("ess_share_")},
        "p_q50_median": float(g["p_q50"].median())})).reset_index()
    cur = float(geo.loc[np.isclose(geo["share"], CURRENT), "effective_items_geomean"].iloc[0])
    pick = lambda target: float(geo.loc[(np.log(geo["effective_items_geomean"]) - np.log(target)).abs().idxmin(), "share"])
    levels = {"poor": pick(cur / FACTOR), "current": CURRENT, "better": pick(cur * FACTOR)}
    geo.to_csv(out / "support_summary.csv", index=False, float_format="%.6g")
    (out / "support_levels.json").write_text(json.dumps({"levels": levels, "current_effective_items": cur,
                                                          "factor": FACTOR}, indent=2))
    print(json.dumps({"levels": levels}))


if __name__ == "__main__":
    main()
