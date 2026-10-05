"""Truth-trained oracles for the score-function classes of the BLOB comparison (docs/blob_controlled_integration.md §3).

The classes act on the logger's frozen vectors x (users) and a (items):

    affine_bilinear   f(u, a) = xᵀ M a + wᵀ a + vᵀ x + c      OPC's and CausE-capacity-matched's family
    blob              f(u, a) = xᵀ M a + κ_a + c              BLOB-supplied-source with free per-item intercepts
    bilinear          f(u, a) = xᵀ M a + c                    BLOB with its intercepts pinned

Two objectives, both exact on the simulator's truth (no logged data, no reward model, no selection):

    value        maximize Σ_u prior(u) Σ_a softmax(f(u, ·))_a q(u, a)        (the Stage 1 recipe; terms constant in a
                                                                           cancel, the scale of M sharpens)
    likelihood   minimize Σ_u prior(u) Σ_a π0(a|u) CE(q(u, a), σ(f(u, a)))  (the infinite-data maximum likelihood under
                                                                           the logger's sampling)

The ``bilinear`` fit starts at the logger (M = I / T; c = 0 for value, the logit of the logged click rate for
likelihood); the superset classes start at its solution with their extra terms at 0. Each fit is optimized with Adam and a cosine schedule on a prior-weighted sample of users, every item in every
step, at a few learning rates, as training/oracle_repair.py does. The value oracle keeps the learning rate with the
best exact value. The likelihood oracle keeps the one with the lowest exact objective: choosing it by greedy value
would turn it into a value oracle. Each kept fit is graded by the study's exact functions: the greedy value of
argmax_a f, and for the value oracle also the value of its softmax.

Usage: python -m training.class_oracles --datasets ml --seeds 100 101 --bias-configs none high/none/none ... --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from training.oracle_repair import _git_commit, true_q_rows
from training.run_full_study import build_condition_world
from training.trainer_trials import _policy_temperature
from utils.representation_bias import add_world_arguments, resolve_bias_configs, world_options_from_args
from utils.seeding import derive_seed, enable_determinism, seed_everything
from utils.simulation_utils import _normalized_prior, calc_greedy_reward, calc_reward

CLASSES = ("affine_bilinear", "blob", "bilinear")
OBJECTIVES = ("value", "likelihood")
DEFAULT_LRS = (3e-3, 1e-2, 3e-2)
WARM_LR_SCALE = 1.0 / 3.0  # a superset class starts at the subset's optimum: one step lower on the lr grid
DEFAULT_STEPS = 3000
DEFAULT_FIT_USERS = 20_000
DEFAULT_BATCH_USERS = 2048


class ScoreClass(torch.nn.Module):
    """f(u, a) for one class on the frozen logger vectors."""

    def __init__(self, x: np.ndarray, a: np.ndarray, cls: str, temperature: float, c0: float = 0.0):
        super().__init__()
        if cls not in CLASSES:
            raise ValueError(f"class must be one of {CLASSES}, got {cls!r}")
        self.cls = cls
        K = x.shape[1]
        self.register_buffer("x", torch.as_tensor(np.asarray(x, dtype=np.float32)))
        self.register_buffer("a", torch.as_tensor(np.asarray(a, dtype=np.float32)))
        self.M = torch.nn.Parameter(torch.eye(K) / float(temperature))
        self.c = torch.nn.Parameter(torch.tensor(float(c0)))
        if cls == "affine_bilinear":
            self.w = torch.nn.Parameter(torch.zeros(K))
            self.v = torch.nn.Parameter(torch.zeros(K))
        if cls == "blob":
            self.kappa = torch.nn.Parameter(torch.zeros(self.a.shape[0]))

    def item_term(self) -> torch.Tensor:
        if self.cls == "affine_bilinear":
            return self.a @ self.w
        if self.cls == "blob":
            return self.kappa
        return torch.zeros(self.a.shape[0], device=self.a.device)

    def scores(self, users: torch.Tensor) -> torch.Tensor:
        """f(u, ·) for a batch of users (B × P)."""
        xu = self.x[users]
        f = (xu @ self.M) @ self.a.T + self.item_term()[None, :] + self.c
        if self.cls == "affine_bilinear":
            f = f + (xu @ self.v)[:, None]
        return f

    @torch.no_grad()
    def ranking_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """(user, item) vectors whose dot product is f(u, a) minus terms constant in a: argmax and softmax."""
        ux = torch.cat([self.x @ self.M, torch.ones(self.x.shape[0], 1, device=self.x.device)], dim=1)
        ia = torch.cat([self.a, self.item_term()[:, None]], dim=1)
        return ux.float().cpu().numpy(), ia.float().cpu().numpy()


def _logger_probs(x_users: torch.Tensor, a: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax((x_users @ a.T) / float(temperature), dim=1)


def fit_class_oracle(dataset: dict, cls: str, objective: str, *, lr: float, steps: int = DEFAULT_STEPS,
                     fit_users: int = DEFAULT_FIT_USERS, batch_users: int = DEFAULT_BATCH_USERS, seed: int = 0,
                     device=None, start: "ScoreClass | None" = None):
    """(model, trace, exact objective on the fit users). ``start``: a fitted ``bilinear`` model whose M and c this fit
    starts from (its extra terms at 0), so a superset class starts at the subset's solution."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(derive_seed(seed, "class_oracle", cls, objective, f"{lr:g}"))
    T = float(_policy_temperature(dataset))
    x, a = np.asarray(dataset["our_x"], np.float32), np.asarray(dataset["our_a"], np.float32)
    prior = _normalized_prior(dataset)
    rng = np.random.default_rng(derive_seed(seed, "class_oracle_users", objective))  # the same users for every class
    users = rng.choice(int(dataset["n_users"]), size=int(fit_users), p=prior)
    clean_items = torch.as_tensor(np.asarray(dataset["env"].emb_a), dtype=torch.float32, device=device)
    c0 = 0.0
    model = ScoreClass(x, a, cls, T).to(device)
    if start is not None:
        with torch.no_grad():
            model.M.copy_(start.M.detach())
            model.c.copy_(start.c.detach())
    elif objective == "likelihood":  # start the intercept at the logged click rate
        with torch.no_grad():
            num = den = 0.0
            for s in range(0, len(users), batch_users):
                b = torch.as_tensor(users[s:s + batch_users], device=device)
                pi = _logger_probs(model.x[b], model.a, T)
                num += float((pi * true_q_rows(dataset, b, clean_items)).sum())
                den += float(len(b))
            rate = num / den
            c0 = float(np.log(rate / (1 - rate)))
            model.c.fill_(c0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(steps))
    per_chunk, trace, running = max(1, int(steps) // 10), [], []
    order, pos = rng.permutation(len(users)), 0

    def loss_of(batch):
        f = model.scores(batch)
        q = true_q_rows(dataset, batch, clean_items)
        if objective == "value":
            return -(torch.softmax(f, dim=1) * q).sum(dim=1).mean()
        pi = _logger_probs(model.x[batch], model.a, T)
        ce = torch.nn.functional.binary_cross_entropy_with_logits(f, q, reduction="none")
        return (pi * ce).sum(dim=1).mean()

    for _ in range(int(steps)):
        if pos + batch_users > len(order):
            order, pos = rng.permutation(len(users)), 0
        batch = torch.as_tensor(users[order[pos:pos + batch_users]], device=device)
        pos += batch_users
        loss = loss_of(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running.append(float(loss.detach()))
        if len(running) == per_chunk:
            trace.append(float(np.mean(running)))
            running = []
    with torch.no_grad():  # the exact objective over the whole fit sample
        total = 0.0
        for s in range(0, len(users), batch_users):
            b = torch.as_tensor(users[s:s + batch_users], device=device)
            total += float(loss_of(b)) * len(b)
    return model, trace, total / len(users)


def class_oracles(dataset: dict, *, classes=CLASSES, objectives=OBJECTIVES, lrs=DEFAULT_LRS, steps: int = DEFAULT_STEPS,
                  fit_users: int = DEFAULT_FIT_USERS, batch_users: int = DEFAULT_BATCH_USERS, seed: int = 0,
                  device=None) -> list[dict]:
    """Per objective: ``bilinear`` first, from the logger; then each superset class (``affine_bilinear``, ``blob``)
    from the best ``bilinear`` solution, so a larger class never ends below a smaller one by optimization alone."""
    rows = []
    order = sorted(classes, key=lambda c: c != "bilinear")
    for objective in objectives:
        base_model = None
        for cls in order:
            best = None
            for lr in (lrs if cls == "bilinear" else [x * WARM_LR_SCALE for x in lrs]):
                t0 = time.time()
                model, trace, obj = fit_class_oracle(dataset, cls, objective, lr=lr, steps=steps, fit_users=fit_users,
                                                     batch_users=batch_users, seed=seed, device=device,
                                                     start=base_model if cls != "bilinear" else None)
                ux, ia = model.ranking_vectors()
                rec = {"class": cls, "objective": objective, "lr": float(lr), "fit_objective": obj,
                       "greedy": float(calc_greedy_reward(dataset, ux, ia)),
                       "trace_first": trace[0], "trace_prev": trace[-2], "trace_last": trace[-1],
                       "seconds": time.time() - t0, "M_norm": float(model.M.detach().norm()),
                       "item_term_sd": float(model.item_term().detach().std())}
                if objective == "value":
                    rec["value"] = float(calc_reward(dataset, SimpleNamespace(user_emb=ux, item_emb=ia, temperature=1.0,
                                                                               action_chunk=8192)))
                rec["start"] = "bilinear" if (cls != "bilinear" and base_model is not None) else "logger"
                # value: the best exact value; likelihood: the lowest exact objective (never the greedy value)
                key = rec["value"] if objective == "value" else -obj
                if best is None or key > best[0]:
                    if cls == "bilinear":
                        base_model = model
                    best = (key, rec)
                if base_model is not model:
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            row = dict(best[1])
            row["flat"] = bool(abs(row["trace_last"] - row["trace_prev"]) < 1e-4)
            rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", default=["ml", "kuairand", "anime"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101])
    ap.add_argument("--ctr", type=float, default=0.05)
    ap.add_argument("--classes", nargs="+", choices=list(CLASSES), default=list(CLASSES))
    ap.add_argument("--objectives", nargs="+", choices=list(OBJECTIVES), default=list(OBJECTIVES))
    ap.add_argument("--lrs", nargs="+", type=float, default=list(DEFAULT_LRS))
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--fit-users", type=int, default=DEFAULT_FIT_USERS)
    ap.add_argument("--batch-users", type=int, default=DEFAULT_BATCH_USERS)
    ap.add_argument("--emb-dir", default="BPR/embeddings")
    ap.add_argument("--out", required=True)
    add_world_arguments(ap, bias_default=("none", "high/none/none", "none/high/none", "none/none/high", "high"))
    args = ap.parse_args(argv)
    enable_determinism(True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world_options = world_options_from_args(args)
    settings = dict(classes=args.classes, objectives=args.objectives, lrs=args.lrs, steps=args.steps,
                    fit_users=args.fit_users, batch_users=args.batch_users, ctr=args.ctr, world_options=world_options,
                    commit=_git_commit())
    (out / "class_oracle_settings.json").write_text(json.dumps(settings, indent=2))
    rows_path = out / "class_oracles.csv"
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
                dataset, _, _, label = build_condition_world(ds, Path(args.emb_dir), bias, args.ctr, seed,
                                                             world_options=world_options)
                rows = class_oracles(dataset, classes=args.classes, objectives=args.objectives, lrs=args.lrs,
                                     steps=args.steps, fit_users=args.fit_users, batch_users=args.batch_users, seed=seed)
                frame = pd.DataFrame(rows).assign(dataset=ds, bias=bias, bias_label=label, seed=seed,
                                                  world_seconds=time.time() - t0, commit=settings["commit"])
                frame.to_csv(rows_path, mode="a", header=not rows_path.exists(), index=False)
                print(json.dumps({"dataset": ds, "bias": bias, "seed": seed,
                                  **{f"{r['class']}/{r['objective']}": round(r["greedy"], 5) for r in rows}}), flush=True)
                del dataset
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
