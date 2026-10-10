"""Population analysis of the structured-shift worlds (docs/structured_scenario_shift_study.md §6), before any
finite-sample run: per world, in the matched rank-4 adapter class, exact over users (prior) and items with q_B:

- the truth adapter (U* D*, V*): the representability reference, its greedy value the target greedy value;
- θ_value*: the value path (Adam on the exact value, the best of three learning rates);
- θ_lik*: the logging-weighted population NLL with the ordinary head (σ(s ŝ/T + c));
- θ_calib*: the same with the calibration-aware head (exp(w_αᵀx)(s ŝ/T + γ) + w_βᵀx + c);
- θ_harm*: harmonic DR's population objective with q̂_∞ (the reward model's logistic features fit to the population);
- raw DR's population objective is the value itself.

Mismatches in greedy CTR points: M_L = V_g(θ_value*) − V_g(θ_lik*), M_calib, M_harm, and the class optimizer's gap
V_g(truth) − V_g(θ_value*). The sanity checks A-C of §6 are evaluated per world.

    python -m training.ss_population --datasets ml --seeds 100 101 --labels s-none.r-none ... --out RUN
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from training.opc_gradients import (
    FIT_USERS,
    VALUE_LRS,
    WorldTensors,
    _fit_users,
    _policy,
    _prior,
    build_model,
    exact_greedy,
    exact_likelihood,
    exact_value,
    fit_likelihood_population,
    fit_value_path,
    get_theta,
    population_head,
    set_theta,
    value_optimal_scale,
)

WARN_POINTS = 0.25  # §6: the sanity checks' warning threshold, in greedy CTR points


def world_dir_name(ds: str, label: str, seed: int, share: float) -> str:
    return f"dataset={ds}__bias={label}__seed={seed}__lgs={share:g}"


def truth_adapter(dataset: dict, device=None):
    """The adapter set to the truth: U = U* diag(γ ε), V = V* (U Vᵀ = Δ*), logit scale 1."""
    st = dataset["structured"]
    if st["levels"]["gated"]:
        raise NotImplementedError("the gated truth adapter is set in Stage 2")
    model = build_model(dataset, device=device)
    d = st["directions"]
    with torch.no_grad():
        model.action_transform.U.copy_(torch.as_tensor(d["U"] * (st["gamma"] * d["signs"]), dtype=torch.float32))
        model.action_transform.V.copy_(torch.as_tensor(d["V"], dtype=torch.float32))
    return model


@torch.no_grad()
def qhat_errors(qinf: dict, model, dataset: dict, world: WorldTensors, chunk: int = 2048) -> dict:
    """q̂_∞'s RMSE against q_B, weighted by the logger and by the given policy (prior over users)."""
    from training.opc_empirical_objective import qinf_rows

    prior = _prior(dataset)
    out = {"logging": 0.0, "target": 0.0}
    for s in range(0, int(dataset["n_users"]), chunk):
        u = torch.arange(s, min(s + chunk, int(dataset["n_users"])), device=world.device)
        e2 = (qinf_rows(qinf, world, u).double() - world.q(u).double()) ** 2
        w = torch.as_tensor(prior[s:s + chunk], device=world.device)
        out["logging"] += float(((world.logger(u).double() * e2).sum(dim=1) * w).sum())
        out["target"] += float(((_policy(model, u).double() * e2).sum(dim=1) * w).sum())
    return {k: float(np.sqrt(v)) for k, v in out.items()}


def _likelihood_optimum(dataset, world, seed, mode, head, log) -> tuple:
    best = None
    for lr in VALUE_LRS:
        m = fit_likelihood_population(dataset, lr=lr, seed=seed, world=world, head=head, mode=mode)
        nll = exact_likelihood(m, dataset, world=world)
        log(f"[population] {mode} lr={lr:g}: L_log={nll:.6f}")
        if best is None or nll < best[0]:
            best = (nll, lr, m)
    return best


def world_population(dataset: dict, *, seed: int, log=print) -> dict:
    """Every population optimum of §6 for one world, with greedy and stochastic values and the mismatches."""
    from training.opc_empirical_objective import fit_harmonic_population, population_reward_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world = WorldTensors(dataset, device)
    res, thetas = {}, {}

    def record(name, model, **extra):
        mode = getattr(model, "mode", "policy")
        model.mode = "policy"  # the policy of the maps (a click model's intercept / nuisance play no part)
        v, _ = exact_value(model, dataset, world=world, grad=False)
        res[name] = {"greedy": exact_greedy(model, dataset), "V": v, **extra}
        thetas[name] = get_theta(model)
        model.mode = mode

    record("source", build_model(dataset, device=device))
    record("truth", truth_adapter(dataset, device=device))
    best = None
    for lr in VALUE_LRS:
        m, _ = fit_value_path(dataset, lr=lr, seed=seed, world=world)
        v, _ = exact_value(m, dataset, world=world, grad=False)
        log(f"[population] value lr={lr:g}: V={v:.6f}")
        if best is None or v > best[0]:
            best = (v, lr, m)
    record("value", best[2], lr=best[1])
    value_model = best[2]

    head = population_head(dataset, _fit_users(dataset, seed, "likelihood", FIT_USERS), world=world)
    for name, mode in (("likelihood", "click"), ("calib", "calib")):
        nll, lr, m = _likelihood_optimum(dataset, world, seed, mode, head, log)
        extra = {"lr": lr, "L_log": nll, "click_intercept": float(m.click_intercept.detach())}
        if mode == "calib":
            extra.update(m.correction_norms())
        m.mode = "policy"
        extra["theta_s_value_optimal"] = value_optimal_scale(m, dataset, world=world,
                                                             users=_fit_users(dataset, seed, "scale", FIT_USERS))
        record(name, m, **extra)

    qinf = population_reward_model(dataset, world, _fit_users(dataset, seed, "qinf", FIT_USERS))
    h = fit_harmonic_population(dataset, qinf, world=world, seed=seed, log=log)
    harm = build_model(dataset, device=device)
    set_theta(harm, h["theta"])
    record("harmonic", harm, lr=h["lr"], V_h_fit_users=h["V_h_fit_users"])
    res["qhat_inf_rmse"] = qhat_errors(qinf, value_model, dataset, world)

    pts = lambda a, b: 100.0 * (res[a]["greedy"] - res[b]["greedy"])
    res["mismatch"] = {"M_L": pts("value", "likelihood"), "M_calib": pts("value", "calib"),
                       "M_harm": pts("value", "harmonic"), "optimizer_gap": pts("truth", "value"),
                       "available_gain": pts("truth", "source")}
    resp = dataset["structured"]["levels"]["response"]
    m = res["mismatch"]
    res["checks"] = {
        "A_M_L_homogeneous": None if resp != "none" else bool(m["M_L"] < WARN_POINTS),
        "B_M_calib": None if resp == "none" else bool(m["M_calib"] < WARN_POINTS),
        "C_truth_is_target": bool(abs(res["truth"]["greedy"] - dataset["world"]["target_greedy_ctr"]) < 0.02),
        "C_optimizer_gap": bool(m["optimizer_gap"] < WARN_POINTS),
    }
    return {"result": res, "thetas": thetas}


def main(argv=None) -> None:
    from training.run_full_study import build_condition_world
    from utils.seeding import enable_determinism, pin_cpu_threads, seed_everything

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--logger-greedy-share", type=float, default=0.8)
    ap.add_argument("--emb-dir", default="BPR/embeddings")
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    enable_determinism(True)
    pin_cpu_threads(int(args.cpu_threads))
    out = Path(args.out)
    for seed in args.seeds:
        for ds in args.datasets:
            for label in args.labels:
                wdir = out / world_dir_name(ds, label, seed, args.logger_greedy_share)
                if (wdir / "population.json").exists():
                    continue
                wdir.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                seed_everything(int(seed))
                dataset, *_ = build_condition_world(
                    ds, Path(args.emb_dir), label, 0.05, int(seed),
                    world_options={"world_family": "structured_shift", "logger_greedy_share": args.logger_greedy_share})
                print(f"=== {wdir.name}", flush=True)
                pop = world_population(dataset, seed=int(seed))
                np.savez(wdir / "population_thetas.npz", **pop["thetas"])
                (wdir / "world.json").write_text(json.dumps(dataset["world"], indent=2, default=float))
                pop["result"]["seconds"] = time.time() - t0
                (wdir / "population.json").write_text(json.dumps(pop["result"], indent=2, default=float))
                print(json.dumps({"world": wdir.name, **pop["result"]["mismatch"], "checks": pop["result"]["checks"],
                                  "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
