"""Validation of the Stage 1 oracle repair bounds (development diagnostic).

The Stage 1 oracle (``training.oracle_repair``) fits the learner's linear repair class to the true click model
with Adam at learning rates {1e-3, 3e-3, 1e-2} for 3,000 cosine-annealed steps. The best rate of the scale-fixed
``linear`` class was the top of that grid in 35 of 36 worlds. The bounds may therefore be conservative.

This module refits the same class with the same objective, user sample, batch and seeds (``fit_oracle``,
``exact_values``), and records every candidate:

  - learning rate beyond the old edge: each class starts from its Stage 1 winner (reproduced exactly) and
    extends past whichever grid edge won. ``linear``: 1e-2, 3e-2, 1e-1, 3e-1, then larger while the top one
    wins. ``linear+scale``: 3e-3, plus 3e-4 where 1e-3 won;
  - budget: each class's best rate refit at ``budget_factor`` × the steps (the cosine schedule stretched), and
    at the factor squared when the value still moves by ``plateau_tol`` or more.

Nothing here uses logged data: this is the true-reward oracle, as in Stage 1. The bound is the best exact value
found; every candidate is a policy of the class scored exactly, so a larger search only tightens it.

Usage: python -m training.oracle_validation --stage1 artifacts/full_study/run_oracle_repair_20260927 --out DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch

from training.oracle_repair import DEFAULT_BATCH_USERS, DEFAULT_FIT_USERS, DEFAULT_LRS as STAGE1_LRS, DEFAULT_STEPS, _git_commit, exact_values, fit_oracle
from training.run_full_study import build_condition_world
from utils.representation_bias import add_world_arguments, resolve_bias_configs, world_options_from_args
from utils.seeding import enable_determinism, seed_everything

REPAIR_CLASSES = ("linear", "linear+scale")
LINEAR_LRS = (1e-2, 3e-2, 1e-1, 3e-1)
MAX_EXTENSIONS = 3  # beyond the top of LINEAR_LRS: 1, 3, 10
BUDGET_FACTOR = 3
PLATEAU_TOL = 2e-4  # 0.02 CTR points (stochastic or greedy)


def _next_up(lr: float) -> float:
    """The next point of the half-decade grid above ``lr`` (1e-2 -> 3e-2 -> 1e-1 -> ...)."""
    e = math.floor(math.log10(lr) + 1e-9)
    return float(f"{(3 if lr < 2.5 * 10**e else 10) * 10**e:.3g}")


def _next_down(lr: float) -> float:
    """The next point of the half-decade grid below ``lr`` (1e-3 -> 3e-4 -> 1e-4 -> ...)."""
    e = math.floor(math.log10(lr) + 1e-9)
    return float(f"{(10**e / 10 * 3 if lr < 2.5 * 10**e else 10**e):.3g}")


def _fit(dataset, cls, lr, steps, *, seed, fit_users, batch_users, device=None) -> dict:
    t0 = time.time()
    model, trace = fit_oracle(dataset, cls, lr=lr, steps=steps, fit_users=fit_users, batch_users=batch_users,
                              seed=seed, device=device)
    row = dict(cls=cls, lr=float(lr), steps=int(steps), **exact_values(dataset, model), trace_first=trace[0],
               trace_prev=trace[-2], trace_last=trace[-1], seconds=time.time() - t0)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def validate_world(dataset: dict, *, stage1: dict, seed: int, steps: int = DEFAULT_STEPS,
                   fit_users: int = DEFAULT_FIT_USERS, batch_users: int = DEFAULT_BATCH_USERS,
                   linear_lrs=LINEAR_LRS, max_extensions: int = MAX_EXTENSIONS, budget_factor: int = BUDGET_FACTOR,
                   plateau_tol: float = PLATEAU_TOL, device=None) -> list[dict]:
    """Every candidate for one world. ``stage1``: that world's Stage 1 row (its winning rates per class)."""
    kw = dict(seed=seed, fit_users=fit_users, batch_users=batch_users, device=device)
    rows = []
    for cls in REPAIR_CLASSES:
        won = float(stage1[f"oracle_{cls}_lr"])
        if cls == "linear":
            lrs = sorted({won, *linear_lrs})
        else:  # its Stage 1 winner, plus one point past a Stage 1 grid edge that won
            edge = {min(STAGE1_LRS): _next_down(won), max(STAGE1_LRS): _next_up(won)}
            lrs = sorted({won} | ({edge[won]} if won in edge else set()))
        cands = [_fit(dataset, cls, lr, steps, **kw) for lr in lrs]
        extensions = 0
        while cls == "linear" and max(cands, key=lambda r: r["value"])["lr"] == max(lrs) and extensions < max_extensions:
            lrs.append(_next_up(max(lrs)))
            cands.append(_fit(dataset, cls, lrs[-1], steps, **kw))
            extensions += 1
        best = max(cands, key=lambda r: r["value"])
        longer = _fit(dataset, cls, best["lr"], steps * budget_factor, **kw)
        cands.append(longer)
        if max(abs(longer["value"] - best["value"]), abs(longer["greedy"] - best["greedy"])) >= plateau_tol:
            cands.append(_fit(dataset, cls, best["lr"], steps * budget_factor**2, **kw))
        for r in cands:
            r["stage1_lr"] = won
        rows += cands
    return rows


def load_stage1(root) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(glob.glob(str(Path(root) / "**" / "oracle_repair.csv"), recursive=True))]
    return pd.concat(frames, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage1", required=True, help="the Stage 1 oracle folder (its winning rates per world)")
    ap.add_argument("--datasets", nargs="+", default=["ml", "kuairand", "anime"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101])
    ap.add_argument("--ctr", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--budget-factor", type=int, default=BUDGET_FACTOR)
    ap.add_argument("--fit-users", type=int, default=DEFAULT_FIT_USERS)
    ap.add_argument("--batch-users", type=int, default=DEFAULT_BATCH_USERS)
    ap.add_argument("--emb-dir", default="BPR/embeddings")
    ap.add_argument("--out", required=True)
    add_world_arguments(ap, bias_default=("medium", "high", "high/none/none", "none/high/none", "none/none/high"))
    args = ap.parse_args(argv)
    enable_determinism(True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world_options = world_options_from_args(args)
    stage1 = load_stage1(args.stage1).set_index(["dataset", "bias", "seed"])
    (out / "validation_settings.json").write_text(json.dumps(dict(
        stage1=str(args.stage1), linear_lrs=list(LINEAR_LRS), max_extensions=MAX_EXTENSIONS, steps=args.steps,
        budget_factor=args.budget_factor, plateau_tol=PLATEAU_TOL, fit_users=args.fit_users,
        batch_users=args.batch_users, ctr=args.ctr, world_options=world_options, stage="development",
        commit=_git_commit()), indent=2))
    path = out / "oracle_candidates.csv"
    done = set()
    if path.exists():
        prev = pd.read_csv(path)
        done = set(zip(prev["dataset"], prev["bias"], prev["seed"]))
    for seed in args.seeds:
        for ds in args.datasets:
            for bias in resolve_bias_configs(args.bias_configs):
                if (ds, bias, seed) in done:
                    continue
                t0 = time.time()
                seed_everything(seed)
                dataset, _, _, label = build_condition_world(ds, Path(args.emb_dir), bias, args.ctr, seed, world_options=world_options)
                rows = validate_world(dataset, stage1=stage1.loc[(ds, bias, seed)].to_dict(), seed=seed, steps=args.steps,
                                      fit_users=args.fit_users, budget_factor=args.budget_factor, batch_users=args.batch_users)
                frame = pd.DataFrame(rows).assign(dataset=ds, bias=bias, bias_label=label, seed=seed,
                                                  world_seconds=time.time() - t0, commit=_git_commit())
                frame.to_csv(path, mode="a", header=not path.exists(), index=False)
                best = frame.loc[frame.groupby("cls")["value"].idxmax(), ["cls", "lr", "steps", "value", "greedy"]]
                print(json.dumps({"dataset": ds, "bias": bias, "seed": seed, "fits": len(frame),
                                  "best": best.round(5).to_dict("records")}), flush=True)
                del dataset
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
