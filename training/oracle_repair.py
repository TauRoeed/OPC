"""Structural recoverability: the oracle repair bound of the learner's policy class (development experiment).

Before asking whether a learner recovers value from biased logs, measure how much value the representation-repair
policy class can recover at all, with perfect information. The oracle repair policy is the learner's own class,
trained against the simulator's true reward surface:

  - Class: ``CFModel`` on the frozen biased vectors (``our_x``, ``our_a``), at the logger's temperature, with the
    study's ``--policy-transform`` correction on each side (default ``linear``: (I + D) x + b, starting exactly at
    the logger). This is the same construction ``regression_trainer_trial`` uses for OPC, DM-only and no-propensity.
  - Objective: the exact true value V(pi) = sum_u prior(u) sum_a pi(a|u) q(u, a), with q the simulator's click
    probability (clean vectors). Users are drawn from the prior; every item enters every step. No logged data, no
    reward model, no propensities, no selection.
  - Optimizer: Adam from the logger for a fixed number of steps, at several prespecified learning rates. Each
    candidate is scored exactly on all users by the study's own value functions (``calc_reward``,
    ``calc_greedy_reward``, the ones that grade every learned policy), and the best is kept. The value is exact,
    so taking the best of the candidates adds no selection bias. It is a lower bound on the class optimum, and
    the report says whether the objective had flattened out by the end.

Classes:
  - ``linear``: the repair with the logit scale fixed at 1 (the logger's sharpness): ranking repair only.
  - ``linear+scale``: the repair plus a learnable logit scale (the learner's class under ``--learn-logit-scale``).
  - ``scale``: the logit scale alone (the tempered logger's class): sharpening without repair.

Greedy values (the true CTR of each user's top item under the policy) compare rankings without sharpness.
Usage: python -m training.oracle_repair --datasets ml --bias-configs none medium high --seeds 100 101 --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.models import POLICY_TRANSFORMS, CFModel, make_policy_transform
from training.run_full_study import build_condition_world
from training.trainer_trials import _cf_model_inputs, _policy_greedy_reward_from_embeddings, _policy_reward_from_embeddings, _policy_temperature
from utils.representation_bias import add_world_arguments, resolve_bias_configs, world_options_from_args
from utils.seeding import derive_seed, enable_determinism, seed_everything
from utils.simulation_utils import _normalized_prior

ORACLE_CLASSES = ("linear", "linear+scale", "scale")
DEFAULT_LRS = (1e-3, 3e-3, 1e-2)
DEFAULT_STEPS = 3000
DEFAULT_FIT_USERS = 20_000
DEFAULT_BATCH_USERS = 2048


def true_q_rows(dataset: dict, users: torch.Tensor, clean_items: torch.Tensor) -> torch.Tensor:
    """q(u, a) for the given users and every item, float32 on the items' device: the simulator's click model
    sigmoid(scale * x_u . a + offset) on the clean vectors (``SyntheticBanditEnv.reward_prob_block``)."""
    env = dataset["env"]
    x = torch.as_tensor(np.asarray(env.emb_x)[users.cpu().numpy()], dtype=torch.float32, device=clean_items.device)
    return torch.sigmoid(float(env.scale) * (x @ clean_items.T) + float(env.offset))


def build_oracle_model(dataset: dict, cls: str, policy_transform: str = "linear") -> CFModel:
    """The learner's policy class for ``cls``, starting exactly at the logger (as OPC's trials do)."""
    if cls not in ORACLE_CLASSES:
        raise ValueError(f"oracle class must be one of {ORACLE_CLASSES}, got {cls!r}")
    cf_x, cf_a, cf_pop = _cf_model_inputs(dataset, dataset["our_x"], dataset["our_a"])
    d = int(dataset["emb_dim"])
    t = lambda z: torch.as_tensor(np.asarray(z, dtype=np.float32))
    repair = cls != "scale"
    model = CFModel(
        int(dataset["n_users"]), int(dataset["n_actions"]), d,
        initial_user_embeddings=t(cf_x), initial_actions_embeddings=t(cf_a),
        user_transform=make_policy_transform(policy_transform, d) if repair else None,
        action_transform=make_policy_transform(policy_transform, d) if repair else None,
        temperature=_policy_temperature(dataset), logit_scale=1.0, learn_logit_scale=cls != "linear", **cf_pop,
    )
    # the biased vectors stay frozen in every class, as in the learner (CFModel freezes them only when a
    # transform is present; the tempered logger's class has none and trains nothing but its scale)
    for emb in (model.user_embeddings, model.actions_embeddings):
        emb.weight.requires_grad_(False)
    return model


def fit_oracle(dataset: dict, cls: str, *, lr: float, steps: int = DEFAULT_STEPS, fit_users: int = DEFAULT_FIT_USERS,
               batch_users: int = DEFAULT_BATCH_USERS, seed: int = 0, policy_transform: str = "linear", device=None):
    """Adam ascent of the true value from the logger. Returns (model, trace): trace holds the minibatch objective
    averaged over each tenth of the run (to check it has flattened out)."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(derive_seed(seed, "oracle_repair", cls, f"{lr:g}"))
    model = build_oracle_model(dataset, cls, policy_transform).to(device)
    model.train()
    prior = _normalized_prior(dataset)
    rng = np.random.default_rng(derive_seed(seed, "oracle_repair_users", cls))
    users = rng.choice(int(dataset["n_users"]), size=int(fit_users), p=prior)  # prior-weighted sample, equal weights
    clean_items = torch.as_tensor(np.asarray(dataset["env"].emb_a), dtype=torch.float32, device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(steps))
    per_chunk, trace = max(1, int(steps) // 10), []
    order, pos, running = rng.permutation(len(users)), 0, []
    for step in range(int(steps)):
        if pos + batch_users > len(order):
            order, pos = rng.permutation(len(users)), 0
        batch = torch.as_tensor(users[order[pos:pos + batch_users]], device=device)
        pos += batch_users
        prob = model(batch)[:, :, 0]
        q = true_q_rows(dataset, batch, clean_items)
        loss = -(prob * q).sum(dim=1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running.append(-float(loss.detach()))
        if len(running) == per_chunk:
            trace.append(float(np.mean(running)))
            running = []
    return model, trace


def exact_values(dataset: dict, model: CFModel) -> dict:
    """The study's exact true values of a trained CFModel: stochastic (softmax at the logger's temperature,
    the learned scale folded into the vectors) and greedy (each user's top item)."""
    x, a = model.get_params()
    x, a = x.detach().cpu().numpy(), a.detach().cpu().numpy()
    return {"value": float(np.asarray(_policy_reward_from_embeddings(dataset, x, a)).reshape(-1)[0]),
            "greedy": float(_policy_greedy_reward_from_embeddings(dataset, x, a)), "logit_scale": float(model.logit_scale)}


def logger_values(dataset: dict) -> dict:
    """The logger's own exact values (its biased vectors at its temperature) and the ceiling (each user's truly
    best item: the greedy value of the clean ranking)."""
    x, a = dataset["our_x"], dataset["our_a"]
    env, prior = dataset["env"], _normalized_prior(dataset)
    ceiling = 0.0
    items = np.asarray(env.emb_a)
    for u0 in range(0, int(dataset["n_users"]), 4096):
        users = np.arange(u0, min(u0 + 4096, int(dataset["n_users"])))
        ceiling += float(np.dot(prior[users], env.reward_prob_block(users, 0, items.shape[0]).max(axis=1)))
    return {"value": float(np.asarray(_policy_reward_from_embeddings(dataset, x, a)).reshape(-1)[0]),
            "greedy": float(_policy_greedy_reward_from_embeddings(dataset, x, a)), "ceiling": ceiling}


def oracle_repair(dataset: dict, *, classes=ORACLE_CLASSES, lrs=DEFAULT_LRS, steps: int = DEFAULT_STEPS,
                  fit_users: int = DEFAULT_FIT_USERS, batch_users: int = DEFAULT_BATCH_USERS, seed: int = 0,
                  policy_transform: str = "linear", device=None) -> dict:
    """Per class: the best exact value over the learning rates (stochastic), its greedy value and logit scale,
    the learning rate that won, and whether its objective flattened out (last tenth vs the one before)."""
    out = {f"logger_{k}": v for k, v in logger_values(dataset).items()}
    for cls in classes:
        best = None
        for lr in lrs:
            t0 = time.time()
            model, trace = fit_oracle(dataset, cls, lr=lr, steps=steps, fit_users=fit_users, batch_users=batch_users,
                                      seed=seed, policy_transform=policy_transform, device=device)
            vals = exact_values(dataset, model)
            vals.update(lr=float(lr), seconds=time.time() - t0, trace_last=trace[-1], trace_prev=trace[-2],
                        trace_first=trace[0])
            if best is None or vals["value"] > best["value"]:
                best = vals
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for k, v in best.items():
            out[f"oracle_{cls}_{k}"] = v
        out[f"oracle_{cls}_flat"] = bool(abs(best["trace_last"] - best["trace_prev"]) < 1e-4)
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", default=["ml", "kuairand", "anime"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101])
    ap.add_argument("--ctr", type=float, default=0.05)
    ap.add_argument("--classes", nargs="+", choices=list(ORACLE_CLASSES), default=list(ORACLE_CLASSES))
    ap.add_argument("--lrs", nargs="+", type=float, default=list(DEFAULT_LRS))
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--fit-users", type=int, default=DEFAULT_FIT_USERS)
    ap.add_argument("--batch-users", type=int, default=DEFAULT_BATCH_USERS)
    ap.add_argument("--policy-transform", choices=list(POLICY_TRANSFORMS), default="linear")
    ap.add_argument("--emb-dir", default="BPR/embeddings")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["development", "confirmatory"], default="development")
    add_world_arguments(ap, bias_default=("none", "medium", "high", "high/none/none", "none/high/none", "none/none/high"))
    args = ap.parse_args(argv)
    enable_determinism(True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world_options = world_options_from_args(args)
    settings = dict(classes=args.classes, lrs=args.lrs, steps=args.steps, fit_users=args.fit_users,
                    batch_users=args.batch_users, policy_transform=args.policy_transform, ctr=args.ctr,
                    world_options=world_options, stage=args.stage, commit=_git_commit())
    (out / "oracle_settings.json").write_text(json.dumps(settings, indent=2))
    rows_path = out / "oracle_repair.csv"
    done = set()
    if rows_path.exists():
        prev = pd.read_csv(rows_path)
        done = set(zip(prev["dataset"], prev["bias"], prev["seed"]))
    for seed in args.seeds:
        for ds in args.datasets:
            for bias in resolve_bias_configs(args.bias_configs):
                if (ds, bias, seed) in done:
                    continue
                t0 = time.time()
                seed_everything(seed)
                dataset, _, _, label = build_condition_world(ds, Path(args.emb_dir), bias, args.ctr, seed, world_options=world_options)
                row = {"dataset": ds, "bias": bias, "bias_label": label, "seed": seed,
                       "logger_share": float(dataset["world"].get("logger_greedy_share", 0.0)),
                       "temperature": _policy_temperature(dataset), "n_users": int(dataset["n_users"]),
                       "n_items": int(dataset["n_actions"])}
                row.update(oracle_repair(dataset, classes=args.classes, lrs=args.lrs, steps=args.steps,
                                         fit_users=args.fit_users, batch_users=args.batch_users, seed=seed,
                                         policy_transform=args.policy_transform))
                row.update(seconds=time.time() - t0, commit=settings["commit"], stage=args.stage)
                pd.DataFrame([row]).to_csv(rows_path, mode="a", header=not rows_path.exists(), index=False)
                print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()
                                  if k.endswith(("value", "greedy", "ceiling", "flat", "scale")) or k in ("dataset", "bias", "seed")}), flush=True)
                del dataset
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
