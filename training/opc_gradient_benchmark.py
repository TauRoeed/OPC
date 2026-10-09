"""The gradient benchmark of docs/opc_gradient_regime_study.md §3-§4 (Stage 8A, and the regime cells of 8B).

For each world: the policy states (fit once, cached), the exact gradient g* at each state with the states' population
overlap with the logger, then R independent logged datasets of N rows. Each dataset gets its cross-fitted reward model,
the estimators G1-G5, the conditional bias of G5 and the minibatch gradients at every state. Every replicate is one file;
a rerun skips the replicates already written (resume).

    python -m training.opc_gradient_benchmark --datasets ml --bias-configs high/none/none --seeds 100 \
        --train-size 25000 --replicates 200 --out artifacts/full_study/run_opc_gradients_8a

Layout: OUT/<world>/states.npz, states.json (values, fits), gstar.npz (g*, V and overlap per state), rep/rep_NNNN.npz
(per state: the estimators' gradients, bias5, the minibatch errors) and rep/rep_NNNN.json (diagnostics), config.json.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from training.opc_gradients import (
    ESTIMATOR_NAMES,
    STATES,
    WorldTensors,
    build_model,
    crossfit_qhat,
    estimator_gradients,
    exact_value,
    get_theta,
    logged_rows,
    minibatch_gradients,
    policy_states,
    set_theta,
    study_user_fold,
)
from utils.seeding import derive_seed, enable_determinism, pin_cpu_threads, seed_everything

MINIBATCH_ESTIMATORS = ("G3", "G5")


def world_dir_name(ds: str, label: str, seed: int, share) -> str:
    return f"dataset={ds}__bias={label}__seed={seed}__lgs={share:g}"


def build_world(ds: str, bias: str, seed: int, emb_dir: Path, world_options: dict):
    from training.run_full_study import build_condition_world

    seed_everything(int(seed))
    with contextlib.redirect_stdout(io.StringIO()):
        dataset, _, _, label = build_condition_world(ds, Path(emb_dir), bias, 0.05, int(seed),
                                                     world_options=world_options)
    return dataset, label


def _states_from_cache(path: Path) -> dict | None:
    if not (path / "states.npz").exists() or not (path / "states.json").exists():
        return None
    arr = np.load(path / "states.npz")
    meta = json.loads((path / "states.json").read_text())
    out = {k: dict(meta[k], theta=arr[k]) for k in meta if k in arr.files}
    out["value_path"] = arr["value_path"]
    return out


def ensure_states(dataset: dict, wdir: Path, seed: int, *, steps: int, log=print) -> dict:
    """The policy states of the world, from the cache or fit and cached."""
    cached = _states_from_cache(wdir)
    if cached is not None:
        return cached
    t0 = time.time()
    states = policy_states(dataset, seed=int(seed), steps=steps, log=log)
    np.savez(wdir / "states.npz", value_path=states["value_path"],
             **{k: v["theta"] for k, v in states.items() if isinstance(v, dict)})
    meta = {k: {kk: vv for kk, vv in v.items() if kk != "theta"} for k, v in states.items() if isinstance(v, dict)}
    meta["fit_seconds"] = time.time() - t0
    (wdir / "states.json").write_text(json.dumps(meta, indent=2, default=float))
    log(f"[states] fit in {meta['fit_seconds']:.0f}s: " + ", ".join(
        f"{k} V={v['value']:.5f} greedy={v['greedy']:.5f}" for k, v in meta.items() if isinstance(v, dict)))
    return _states_from_cache(wdir)


@torch.no_grad()
def population_overlap(model, dataset: dict, world: WorldTensors, chunk: int = 2048) -> dict:
    """The state's softmax policy π against the logger π0, exact over users (prior) and items: the population ESS
    share 1 / E_π0[(π/π0)²], the target's mass on items with π0 < 1e-4, and the maximum ratio π/π0."""
    from training.opc_gradients import _policy, _prior

    prior = _prior(dataset)
    second, tail, wmax = 0.0, 0.0, 0.0
    for s in range(0, int(dataset["n_users"]), chunk):
        u = torch.arange(s, min(s + chunk, int(dataset["n_users"])), device=world.device)
        pi, p0 = _policy(model, u).double(), world.logger(u).double()
        pr = torch.as_tensor(prior[s:s + chunk], device=world.device)
        second += float(((pi ** 2 / p0.clamp(min=1e-300)).sum(dim=1) * pr).sum())
        tail += float(((pi * (p0 < 1e-4)).sum(dim=1) * pr).sum())
        wmax = max(wmax, float((pi / p0.clamp(min=1e-300)).max()))
    return {"pop_ess_share": 1.0 / second, "target_mass_low_p0": tail, "pop_w_max": wmax}


def ensure_gstar(dataset: dict, wdir: Path, states: dict, world: WorldTensors) -> dict:
    path = wdir / "gstar.npz"
    if path.exists():
        z = np.load(path)
        meta = json.loads((wdir / "gstar.json").read_text())
        return {k: dict(meta[k], g=z[k]) for k in meta}
    model = build_model(dataset, device=world.device)
    out, meta = {}, {}
    for name in STATES + ("likelihood_click",):
        set_theta(model, states[name]["theta"])
        v, g = exact_value(model, dataset, world=world)
        meta[name] = {"V": v, "gstar_norm": float(np.linalg.norm(g)), **population_overlap(model, dataset, world)}
        out[name] = g
    np.savez(path, **out)
    (wdir / "gstar.json").write_text(json.dumps(meta, indent=2))
    return {k: dict(meta[k], g=out[k]) for k in meta}


def run_replicate(dataset: dict, world: WorldTensors, model, states: dict, gstar: dict, *, seed: int, n: int,
                  r: int, support: float, user_fold, estimators, state_names, minibatches: int, batch: int,
                  chunk: int) -> tuple[dict, dict]:
    rep_seed = derive_seed(int(seed), "gradient_benchmark", int(n), f"{support:g}", int(r))
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        rows = logged_rows(dataset, n, rep_seed)
    t_sim = time.time() - t0
    qhat = crossfit_qhat(dataset, rows, user_fold, device=world.device) if any(
        e in ("G3", "G5") for e in estimators) else None
    t_fit = time.time() - t0 - t_sim
    arrays, diag = {}, {"seed": int(rep_seed), "clicks": float(np.sum(rows["r"])), "sim_seconds": t_sim,
                        "qhat_seconds": t_fit, "states": {}}
    for name in state_names:
        set_theta(model, states[name]["theta"])
        d = {}
        g = estimator_gradients(model, rows, world=world, qhat=qhat, estimators=estimators, chunk=chunk,
                                conditional_bias=qhat is not None, diagnostics=d)
        arrays[f"{name}__grads"] = np.stack([g[e] for e in estimators]).astype(np.float32)
        if "bias5" in g:
            arrays[f"{name}__bias5"] = g["bias5"].astype(np.float32)
        if minibatches and qhat is not None:
            mb_est = tuple(e for e in MINIBATCH_ESTIMATORS if e in estimators)
            mb = minibatch_gradients(model, rows, world=world, qhat=qhat, estimators=mb_est, batch=batch,
                                     count=minibatches, seed=derive_seed(rep_seed, name))
            gs = gstar[name]["g"]
            for e in mb_est:
                full = g[e]
                m = np.stack(mb[e])
                d[f"mb_{e}_sq_to_full"] = float(np.mean(((m - full) ** 2).sum(axis=1)))
                d[f"mb_{e}_sq_to_gstar"] = float(np.mean(((m - gs) ** 2).sum(axis=1)))
                denom = np.linalg.norm(m, axis=1) * max(np.linalg.norm(gs), 1e-300)
                d[f"mb_{e}_cos_gstar"] = float(np.mean((m @ gs) / np.maximum(denom, 1e-300)))
        diag["states"][name] = d
    diag["seconds"] = time.time() - t0
    return arrays, diag


def main(argv=None) -> None:
    from training.trainer_trials import batch_schedule
    from utils.representation_bias import add_world_arguments, resolve_bias_configs, world_options_from_args

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[100])
    ap.add_argument("--train-size", type=int, default=25_000)
    ap.add_argument("--replicates", type=int, required=True)
    ap.add_argument("--states", nargs="+", default=list(STATES), choices=list(STATES) + ["likelihood_click"])
    ap.add_argument("--estimators", nargs="+", default=list(ESTIMATOR_NAMES), choices=list(ESTIMATOR_NAMES))
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--state-steps", type=int, default=3000)
    ap.add_argument("--emb-dir", default="BPR/embeddings")
    ap.add_argument("--cpu-threads", type=int, default=4, help="CPU threads per process (the reward model's fits)")
    ap.add_argument("--out", required=True)
    add_world_arguments(ap, bias_default=("high/none/none",))
    args = ap.parse_args(argv)
    enable_determinism(True)
    pin_cpu_threads(int(args.cpu_threads))
    world_options = world_options_from_args(args)
    share = float(world_options.get("logger_greedy_share", 0.8))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    batch = int(batch_schedule(int(args.train_size))[0])
    config = {"train_size": int(args.train_size), "replicates": int(args.replicates), "states": args.states,
              "estimators": args.estimators, "minibatches": int(args.minibatches), "minibatch_size": batch,
              "world_options": world_options, "state_steps": int(args.state_steps)}
    for seed in args.seeds:
        for ds in args.datasets:
            for bias in resolve_bias_configs(args.bias_configs):
                dataset, label = build_world(ds, bias, seed, Path(args.emb_dir), world_options)
                wdir = out / world_dir_name(ds, label, seed, share)
                (wdir / "rep").mkdir(parents=True, exist_ok=True)
                prev = json.loads((wdir / "config.json").read_text()) if (wdir / "config.json").exists() else None
                if prev is not None and {k: v for k, v in prev.items() if k != "replicates"} != \
                        {k: v for k, v in config.items() if k != "replicates"}:
                    raise ValueError(f"{wdir}: written with another configuration {prev}")
                (wdir / "config.json").write_text(json.dumps(config, indent=2))
                print(f"=== {wdir.name}", flush=True)
                states = ensure_states(dataset, wdir, seed, steps=int(args.state_steps))
                world = WorldTensors(dataset)
                gstar = ensure_gstar(dataset, wdir, states, world)
                user_fold = study_user_fold(dataset, seed)
                model = build_model(dataset)
                for r in range(int(args.replicates)):
                    stem = wdir / "rep" / f"rep_{r:04d}"
                    if stem.with_suffix(".json").exists():
                        continue
                    arrays, diag = run_replicate(dataset, world, model, states, gstar, seed=seed,
                                                 n=int(args.train_size), r=r, support=share, user_fold=user_fold,
                                                 estimators=tuple(args.estimators), state_names=tuple(args.states),
                                                 minibatches=int(args.minibatches), batch=batch, chunk=int(args.chunk))
                    np.savez(stem.with_suffix(".npz"), **arrays)
                    stem.with_suffix(".json").write_text(json.dumps(diag))  # written last: the completion marker
                    print(json.dumps({"world": wdir.name, "rep": r, "seconds": round(diag["seconds"], 2),
                                      "qhat_seconds": round(diag["qhat_seconds"], 2)}), flush=True)
                del dataset, world
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
