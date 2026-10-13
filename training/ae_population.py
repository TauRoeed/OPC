"""Population objectives of the adaptive-estimator study (docs/opc_adaptive_estimator_study.md §9), per world, in the
matched rank-4 adapter class, exact over items with a prior-drawn user sample (as ``ss_population``):

- the base optima of ``ss_population.world_population``: the source, the truth adapter, θ_value*, θ_lik*, θ_calib*,
  θ_harm*(λ = 0.1) with q̂_∞ (the logistic reward model's population fit);
- θ_harm*(λ) for λ ∈ {0.005, 0.01, 0.03, 0.2} with q̂_∞ (family A's fixed-λ targets);
- θ_harm*(0.1) with the D1 and D2 reward models' population fits (family D's targets);
- the value–S2 frontier by Lagrangian relaxation (§5.2a): θ*(ν) = argmax V(θ) − ν (S2(π_θ) − 1) for ν in a grid that
  contains family B-soft's β values (so θ*(β) is B-soft's population target), by continuation from θ_value*; the
  S2-capped value optimum at C (family B-hard's target) is the best frontier point with S2 ≤ C (θ_value* included), a
  monotone inner approximation. (An augmented Lagrangian with ρ = 1 on the raw S2 − C stalled Adam: §5.2a.)

Each extra fit uses the learning rate the corresponding base fit chose (θ_harm*(0.1)'s for the harmonic and D fits,
θ_value*'s for the frontier). The harmonic and D optima go to ``ae_population.json``, the frontier to
``ae_population_s2.json``. Per optimum: greedy and stochastic values, exact S2 of the softmax and of the greedy
policy, and M = V_g(θ_value*) − V_g(·) in points.

    python -m training.ae_population --datasets ml --seeds 300 301 --labels s-moderate.r-strong ... --out RUN
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from training.opc_gradients import (
    FIT_STEPS,
    FIT_USERS,
    WorldTensors,
    _fit_users,
    _policy,
    _prior,
    build_model,
    exact_greedy,
    exact_value,
    get_theta,
    policy_params,
    set_theta,
)
from utils.seeding import derive_seed, seed_everything

HARMONIC_LAMBDAS = (0.005, 0.01, 0.03, 0.2)
S2_CAPS = (5.0, 10.0, 20.0, 50.0)
NU_GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)  # contains B-soft's β values
PATH_STEPS = 1500
D_KINDS = ("lowrank", "hetero")
RHO = 1.0
DUAL_EVERY = 100


@torch.no_grad()
def exact_s2(model, dataset: dict, world: WorldTensors, chunk: int = 2048) -> tuple[float, float]:
    """(S2 of the softmax, S2 of its greedy policy) over the prior and the catalog."""
    prior = torch.as_tensor(_prior(dataset), dtype=torch.float64, device=world.device)
    s2 = s2g = 0.0
    for s in range(0, int(dataset["n_users"]), chunk):
        u = torch.arange(s, min(s + chunk, int(dataset["n_users"])), device=world.device)
        pi, mu = _policy(model, u).double(), world.logger(u).double().clamp(min=1e-300)
        s2 += float((prior[u] * (pi * pi / mu).sum(dim=1)).sum())
        top = pi.argmax(dim=1)
        s2g += float((prior[u] / mu[torch.arange(len(u), device=world.device), top]).sum())
    return s2, s2g


def _adam_path(dataset, world, objective, *, lr: float, seed: int, label: str, steps: int = FIT_STEPS,
               batch_users: int = 2048, dual=None, start=None):
    """Adam ascent of ``objective(model, users)`` from the source on a fixed prior-drawn user sample with a cosine
    schedule (the population fits' path). ``dual``: called every ``DUAL_EVERY`` steps with the model (the multiplier
    update of a constrained fit)."""
    users = _fit_users(dataset, seed, label, FIT_USERS)
    seed_everything(derive_seed(int(seed), "ae_population", label, f"{lr:g}"))
    model = build_model(dataset, device=world.device)
    if start is not None:  # continuation from a given θ
        set_theta(model, start)
    rng = np.random.default_rng(derive_seed(int(seed), "ae_population_order", label, f"{lr:g}"))
    opt = torch.optim.Adam(policy_params(model), lr=float(lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(steps))
    order, pos = rng.permutation(len(users)), 0
    for step in range(1, int(steps) + 1):
        if pos + batch_users > len(order):
            order, pos = rng.permutation(len(users)), 0
        u = torch.as_tensor(users[order[pos:pos + batch_users]], device=world.device)
        pos += batch_users
        loss = -objective(model, u)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if dual is not None and step % DUAL_EVERY == 0:
            dual(model, users)
    return model


def harmonic_objective(world, qrows, lam: float):
    """V_h(θ) = mean_u Σ_j [π q̂ + π0 h_λ(π/π0)(q − q̂)] with a fixed q̂ (``qrows(u)``)."""
    def vh(model, u):
        pi, p0 = _policy(model, u), world.logger(u)
        qh = qrows(u)
        wj = pi / p0.clamp(min=1e-10)
        return (pi * qh + p0 * (wj / (1.0 - lam + lam * wj)) * (world.q(u) - qh)).sum(dim=1).mean()
    return vh


def penalized_value_objective(world, nu: float):
    """V(θ) − ν (S2 − 1), S2 over the step's users (enumerated): B-soft's population objective at β = ν."""
    def obj(model, u):
        pi, p0 = _policy(model, u), world.logger(u)
        v = (pi * world.q(u)).sum(dim=1).mean()
        s2 = (pi * pi / p0.clamp(min=1e-30)).sum(dim=1).mean()
        return v - nu * (s2 - 1.0)
    return obj


def population_d_model(dataset, world, users, kind: str, seed: int, chunk: int = 2048):
    """The D reward model fit to the population under the logger: min Σ_u Σ_j π0(j|u) CE(q, σ(f)) over a prior-drawn
    user sample and every item (as ``population_reward_model`` for D0), L-BFGS in float64, unpenalized."""
    from models.reward_models import BilinearRewardModel

    X, A = world.our_x.double(), world.our_a.double()
    m = BilinearRewardModel(X.shape[1], kind, seed=seed).to(world.device)
    blocks = [torch.as_tensor(users[s:s + chunk], device=world.device) for s in range(0, len(users), chunk)]
    opt = torch.optim.LBFGS(list(m.parameters()), max_iter=300, tolerance_grad=1e-10, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe", history_size=50)
    bce = lambda z, q: torch.nn.functional.binary_cross_entropy_with_logits(z, q, reduction="none")

    def closure():
        opt.zero_grad()
        total = 0.0
        for u in blocks:
            loss = (world.logger(u).double() * bce(m.logits(X[u], A), world.q(u).double())).sum() / len(users)
            loss.backward()
            total += float(loss.detach())
        return torch.tensor(total)

    opt.step(closure)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return lambda u: torch.sigmoid(m.logits(X[u], A)).float()


def extra_optima(dataset: dict, *, seed: int, base: dict, log=print) -> dict:
    """The extra optima of one world, given its ``ss_population`` result (``base``: the chosen learning rates and the
    value optimum's greedy value)."""
    from training.opc_empirical_objective import population_reward_model, qinf_rows

    world = WorldTensors(dataset, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    vg_star = float(base["value"]["greedy"])
    lr_h, lr_v = float(base["harmonic"]["lr"]), float(base["value"]["lr"])
    out = {}

    def record(name, model, **extra):
        v, _ = exact_value(model, dataset, world=world, grad=False)
        g = exact_greedy(model, dataset)
        s2, s2g = exact_s2(model, dataset, world)
        out[name] = {"greedy": g, "V": v, "S2": s2, "S2_greedy": s2g, "M": 100.0 * (vg_star - g), **extra}
        log(f"[ae population] {name}: greedy {g:.5f} M {out[name]['M']:+.3f} S2 {s2:.2f} S2_g {s2g:.1f}")
        return get_theta(model)

    thetas = {}
    qinf = population_reward_model(dataset, world, _fit_users(dataset, seed, "qinf", FIT_USERS))
    q0 = lambda u: qinf_rows(qinf, world, u)
    for lam in HARMONIC_LAMBDAS:
        m = _adam_path(dataset, world, harmonic_objective(world, q0, lam), lr=lr_h, seed=seed, label=f"harmonic_{lam:g}")
        thetas[f"harmonic_{lam:g}"] = record(f"harmonic_{lam:g}", m, lr=lr_h, lam=lam)
    for kind in D_KINDS:
        qd = population_d_model(dataset, world, _fit_users(dataset, seed, "qinf", FIT_USERS), kind,
                                seed=derive_seed(int(seed), "ae_population_d", kind))
        m = _adam_path(dataset, world, harmonic_objective(world, qd, 0.1), lr=lr_h, seed=seed, label=f"harmonic_{kind}")
        thetas[f"harmonic_{kind}"] = record(f"harmonic_{kind}", m, lr=lr_h, reward_model=kind)
    return {"result": out, "thetas": thetas}


def s2_frontier(dataset: dict, *, seed: int, base: dict, value_theta, log=print) -> dict:
    """§5.2a: θ*(ν) for ν in ``NU_GRID`` by continuation from θ_value* (each ν from the previous solution, ``PATH_STEPS``
    Adam steps at θ_value*'s learning rate), with exact values and S2; per cap C, the best greedy value among θ_value*
    and the path points with S2 ≤ C."""
    world = WorldTensors(dataset, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    vg_star = float(base["value"]["greedy"])
    lr_v = float(base["value"]["lr"])
    m0 = build_model(dataset, device=world.device)
    set_theta(m0, value_theta)
    s2, s2g = exact_s2(m0, dataset, world)
    v0, _ = exact_value(m0, dataset, world=world, grad=False)
    points = [{"nu": 0.0, "greedy": float(base["value"]["greedy"]), "V": v0, "S2": s2, "S2_greedy": s2g, "M": 0.0}]
    thetas, start = {}, value_theta
    for nu in NU_GRID:
        m = _adam_path(dataset, world, penalized_value_objective(world, nu), lr=lr_v, seed=seed, label=f"s2path_{nu:g}",
                       steps=PATH_STEPS, start=start)
        v, _ = exact_value(m, dataset, world=world, grad=False)
        g = exact_greedy(m, dataset)
        s2, s2g = exact_s2(m, dataset, world)
        points.append({"nu": nu, "greedy": g, "V": v, "S2": s2, "S2_greedy": s2g, "M": 100.0 * (vg_star - g)})
        log(f"[ae population] s2 path ν={nu:g}: greedy {g:.5f} M {points[-1]['M']:+.3f} S2 {s2:.2f} S2_g {s2g:.1f}")
        start = get_theta(m)
        thetas[f"s2path_{nu:g}"] = start
    caps = {}
    for cap in S2_CAPS:
        ok = [p for p in points if p["S2"] <= cap]
        best = max(ok, key=lambda p: p["greedy"]) if ok else None
        caps[f"s2cap_{cap:g}"] = None if best is None else {**best, "cap": cap}
    return {"path": points, "caps": caps, "thetas": thetas}


def main(argv=None) -> None:
    from training.run_full_study import build_condition_world
    from training.ss_population import world_dir_name, world_population
    from utils.seeding import enable_determinism, pin_cpu_threads

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--emb-dir", default="BPR/embeddings")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu-threads", type=int, default=4)
    args = ap.parse_args(argv)
    enable_determinism(True)
    pin_cpu_threads(int(args.cpu_threads))
    out = Path(args.out)
    for seed in args.seeds:
        for ds in args.datasets:
            for label in args.labels:
                wdir = out / world_dir_name(ds, label, seed, 0.8)
                if (wdir / "ae_population.json").exists() and (wdir / "ae_population_s2.json").exists():
                    continue
                wdir.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                seed_everything(int(seed))
                dataset, *_ = build_condition_world(ds, Path(args.emb_dir), label, 0.05, int(seed),
                                                    world_options={"world_family": "structured_shift"})
                print(f"=== {wdir.name}", flush=True)
                if not (wdir / "population.json").exists():  # the base optima (structured study §6)
                    pop = world_population(dataset, seed=int(seed))
                    np.savez(wdir / "population_thetas.npz", **pop["thetas"])
                    (wdir / "world.json").write_text(json.dumps(dataset["world"], indent=2, default=float))
                    (wdir / "population.json").write_text(json.dumps(pop["result"], indent=2, default=float))
                base = json.loads((wdir / "population.json").read_text())
                if not (wdir / "ae_population.json").exists():
                    ex = extra_optima(dataset, seed=int(seed), base=base)
                    np.savez(wdir / "ae_population_thetas.npz", **ex["thetas"])
                    ex["result"]["seconds"] = time.time() - t0
                    (wdir / "ae_population.json").write_text(json.dumps(ex["result"], indent=2, default=float))
                    print(json.dumps({"world": wdir.name, **{k: round(v["M"], 3) for k, v in ex["result"].items()
                                                             if isinstance(v, dict)}}), flush=True)
                t1 = time.time()
                fr = s2_frontier(dataset, seed=int(seed), base=base,
                                 value_theta=np.load(wdir / "population_thetas.npz")["value"])
                np.savez(wdir / "ae_population_s2_thetas.npz", **fr["thetas"])
                (wdir / "ae_population_s2.json").write_text(json.dumps(
                    {"path": fr["path"], "caps": fr["caps"], "seconds": time.time() - t1}, indent=2, default=float))
                print(json.dumps({"world": wdir.name, "caps_M": {k: None if v is None else round(v["M"], 3)
                                                                 for k, v in fr["caps"].items()}}), flush=True)


if __name__ == "__main__":
    main()
