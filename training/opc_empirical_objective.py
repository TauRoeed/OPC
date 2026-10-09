"""The 25k decomposition's references (docs/opc_gradient_regime_study.md §6).

- ``empirical_reference``: an arm's objective on the study's own fixed training rows of a world (the same simulation,
  split and cross-fitted q̂ as ``run_shared_main_25k``), λ = 0, optimized by full-batch Adam from the source at
  several learning rates with a cosine schedule. The restart with the best empirical objective is kept (true value
  never chooses it); the true value along its path is recorded (the oracle-stopped best is a diagnostic only).
- ``population_reward_model`` and ``fit_harmonic_population``: q̂_∞, the reward model's population fit (the logistic
  model on [x, a, x ⊙ a] under the logging distribution, exact over items, unpenalized), and θ_harm*, the optimum of
  harmonic DR's population objective V_h(θ) = Σ_u prior Σ_j [π q̂_∞ + π0 h(π/π0)(q − q̂_∞)].

    python -m training.opc_empirical_objective --datasets ml --bias-configs high/none/none --seeds 100 --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from models.custom_losses import dr_sndr_surrogate
from training.opc_gradients import (
    FIT_STEPS,
    FIT_USERS,
    VALUE_LRS,
    WorldTensors,
    _bce,
    _fit_users,
    _policy,
    build_model,
    exact_greedy,
    exact_value,
    get_theta,
    policy_params,
    sample_greedy,
    set_theta,
    study_user_fold,
)
from utils.importance_weights import parse_weight_spec
from utils.seeding import derive_seed, seed_everything

# objective -> (kind, weight spec, reward source)
OBJECTIVES = {"likelihood": ("likelihood", None, None),
              "dr_raw": ("dr", "none", "qhat"), "dr_harmonic": ("dr", "harmonic:0.1", "qhat"),
              "dr_raw_oq": ("dr", "none", "oracle"), "dr_harmonic_oq": ("dr", "harmonic:0.1", "oracle")}
EMPIRICAL_LRS = (3e-4, 1e-3, 3e-3)
EMPIRICAL_STEPS = 3000
PATH_EVERY = 50


# ------------------------------------------------------------------------------------- the study's training rows
def study_training_rows(dataset: dict, seed: int, n: int, *, val_size: int = 20_000, val_min: int = 5000,
                        regression_size: int = 50_000) -> dict:
    """The training rows of the world at size n, as ``run_full_study._run_condition`` builds them."""
    from training.trainer_trials import LOGGED_RUN_IDX, LazyRegressionSplitCache

    cache = LazyRegressionSplitCache(dataset, [int(n)], val_size=val_size, val_frac=0.15, val_min=val_min, val_max=None,
                                     condition_seed=int(seed), regression_size=int(regression_size))
    with contextlib.redirect_stdout(io.StringIO()):
        return cache[(int(n), LOGGED_RUN_IDX)]["train_data"]


def study_crossfit_lookup(dataset: dict, rows: dict, seed: int, device):
    """The training loss's cross-fitted q̂ of those rows, as the trainer builds it (fold models over the study's user
    folds, the full-data model as the lookup's base)."""
    from training.run_full_study import _crossfit_bundles
    from training.trainer_trials import CrossFitScoresLookup, _scores_lookup_from_bundle, fit_shared_regression_bundle

    user_fold = study_user_fold(dataset, seed)
    with contextlib.redirect_stdout(io.StringIO()):
        full = fit_shared_regression_bundle(dataset, rows, reward_model="regression", reward_features="interaction")
        folds = _crossfit_bundles(dataset, rows, user_fold, 5, reward_model="regression", reward_features="interaction")
    full_lookup = _scores_lookup_from_bundle(full, device)
    return CrossFitScoresLookup([_scores_lookup_from_bundle(b, device) for b in folds], user_fold, full_lookup)


# --------------------------------------------------------------------------------------- the empirical objective
class _Rows:
    """The training rows on the device, with the reward rows (q̂ or q) the DR objectives read, precomputed."""

    def __init__(self, rows: dict, world: WorldTensors, lookup=None, source: str | None = None, chunk: int = 4096):
        dev = world.device
        self.u = torch.as_tensor(np.asarray(rows["x_idx"], dtype=np.int64), device=dev)
        self.a = torch.as_tensor(np.asarray(rows["a"], dtype=np.int64), device=dev)
        self.r = torch.as_tensor(np.asarray(rows["r"]), dtype=torch.float64, device=dev)
        self.p = torch.as_tensor(np.asarray(rows["pscore"]), dtype=torch.float64, device=dev)
        self.n = len(self.u)
        self.chunk = max(1, min(int(chunk), (48 * 1024 * 1024) // max(world.n_items, 1)))
        self.q = None
        if source is not None:
            parts = []
            with torch.no_grad():
                for s in range(0, self.n, self.chunk):
                    u = self.u[s:s + self.chunk]
                    parts.append((world.q(u) if source == "oracle" else lookup[u].to(dev)).float())
            self.q = torch.cat(parts)


def _objective_backward(model, rows: _Rows, kind: str, spec: str | None, backward: bool = True) -> float:
    """Accumulate the gradient of the full-data loss (the training loss's mean over all rows) and return its value
    (``backward=False``: the value only)."""
    with torch.set_grad_enabled(backward):
        return _objective(model, rows, kind, spec, backward)


def _objective(model, rows: _Rows, kind: str, spec: str | None, backward: bool) -> float:
    total = 0.0
    if kind == "likelihood":
        u = model.user_transform(model.user_embeddings(rows.u), rows.u)
        a = model.action_transform(model.actions_embeddings(rows.a), rows.a)
        scale = model._scale()
        z = (u * a).sum(dim=1) * (scale if scale is not None else 1.0) / max(model.temperature, 1e-8)
        z = z + model.click_intercept
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, rows.r.to(z.dtype))
        if backward:
            loss.backward()
        return float(loss.detach())
    mode, param = parse_weight_spec(spec)
    for s in range(0, rows.n, rows.chunk):
        sl = slice(s, s + rows.chunk)
        prob = _policy(model, rows.u[sl])
        per_row = dr_sndr_surrogate(rows.q[sl], prob, rows.a[sl], rows.r[sl], rows.p[sl], use_iw=True,
                                    use_log_trick=False, iw_mode=mode, iw_param=param, normalizer="none")
        loss = -(per_row.sum() / rows.n)
        if backward:
            loss.backward()
        total += float(loss.detach())
    return total


def empirical_fit(dataset: dict, rows: _Rows, objective: str, *, lr: float, steps: int, seed: int, world: WorldTensors,
                  eval_users: np.ndarray, head=None) -> dict:
    """Full-batch Adam from the source on one empirical objective (λ = 0, no gradient clipping), cosine to 0."""
    kind, spec, _ = OBJECTIVES[objective]
    seed_everything(derive_seed(int(seed), "opc_empirical", objective, f"{lr:g}"))
    model = build_model(dataset, mode="click" if kind == "likelihood" else "policy", device=world.device)
    params = policy_params(model)
    if kind == "likelihood":
        with torch.no_grad():
            model.log_logit_scale.fill_(head[0])
            model.click_intercept.fill_(head[1])
        params = params + [model.click_intercept]
    opt = torch.optim.Adam(params, lr=float(lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(steps))
    path = []

    def record(step):
        mode = model.mode
        model.mode = "policy"
        v, _ = exact_value(model, dataset, world=world, grad=False, users=eval_users)
        path.append((step, v, sample_greedy(model, dataset, world, eval_users)))
        model.mode = mode

    record(0)
    obj = float("nan")
    for step in range(1, int(steps) + 1):
        opt.zero_grad(set_to_none=True)
        obj = _objective_backward(model, rows, kind, spec)
        opt.step()
        sched.step()
        if step % PATH_EVERY == 0:
            record(step)
    opt.zero_grad(set_to_none=True)
    final_obj = _objective_backward(model, rows, kind, spec)  # the objective and its gradient norm at the end
    gnorm = float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in params if p.grad is not None)))
    model.mode = "policy"
    return {"model": model, "objective": final_obj, "last_step_objective": obj, "grad_norm": gnorm,
            "path": np.array(path), "lr": float(lr)}


def empirical_reference(dataset: dict, rows_dict: dict, objective: str, *, seed: int, world: WorldTensors,
                        lookup=None, lrs=EMPIRICAL_LRS, steps: int = EMPIRICAL_STEPS, log=print,
                        compare_thetas: dict | None = None) -> dict:
    from models.shared_objectives import fit_click_head
    from training.trainer_trials import _policy_temperature

    kind, _, source = OBJECTIVES[objective]
    rows = _Rows(rows_dict, world, lookup=lookup, source=source)
    head = None
    if kind == "likelihood":
        users, items = np.asarray(rows_dict["x_idx"]), np.asarray(rows_dict["a"])
        g0 = (np.asarray(dataset["our_x"], np.float64)[users] * np.asarray(dataset["our_a"], np.float64)[items]).sum(1)
        head = fit_click_head(g0 / _policy_temperature(dataset), rows_dict["r"])
    eval_users = _fit_users(dataset, seed, "empirical_eval", FIT_USERS)
    fits = []
    for lr in lrs:
        f = empirical_fit(dataset, rows, objective, lr=lr, steps=steps, seed=seed, world=world, eval_users=eval_users,
                          head=head)
        log(f"[empirical] {objective} lr={lr:g}: objective={f['objective']:.6f} grad_norm={f['grad_norm']:.2e}")
        fits.append(f)
    best = min(fits, key=lambda f: f["objective"])  # the empirical objective (a loss), never the true value
    v, _ = exact_value(best["model"], dataset, world=world, grad=False)
    at = {}  # the same empirical objective at given population states: does the sample prefer its own solution?
    for name, (theta, intercept) in (compare_thetas or {}).items():
        m = build_model(dataset, mode="click" if kind == "likelihood" else "policy", device=world.device)
        set_theta(m, theta)
        if kind == "likelihood" and intercept is not None:
            with torch.no_grad():
                m.click_intercept.fill_(float(intercept))
        at[name] = _objective_backward(m, rows, kind, OBJECTIVES[objective][1], backward=False)
    path = best["path"]
    k = int(np.argmax(path[:, 2]))
    probe = build_model(dataset, device=world.device)
    return {"objective_name": objective, "lr": best["lr"], "empirical_objective": best["objective"],
            "grad_norm": best["grad_norm"], "V": v, "greedy": exact_greedy(best["model"], dataset),
            "logit_scale": float(best["model"].logit_scale), "theta": get_theta(best["model"]),
            "path": path, "path_best_step": int(path[k, 0]), "path_best_greedy_sample": float(path[k, 2]),
            "restarts": [{"lr": f["lr"], "objective": f["objective"], "grad_norm": f["grad_norm"]} for f in fits],
            "objective_at": at, "_probe": probe}


# ------------------------------------------------------------------------------- harmonic DR at infinite data
def population_reward_model(dataset: dict, world: WorldTensors, users: np.ndarray, chunk: int = 2048) -> dict:
    """q̂_∞: the logistic reward model on [x, a, x ⊙ a] (the study's features) fit to the population under the
    logging distribution: min Σ_u Σ_j π0(j|u) CE(q(u, j), σ(f(u, j))) over a prior-drawn user sample, every item,
    unpenalized (sklearn's L2 term vanishes as n grows). LBFGS in float64."""
    k = int(world.our_x.shape[1])
    dev = world.device
    w = {name: torch.zeros(k, dtype=torch.float64, device=dev, requires_grad=True) for name in ("x", "a", "xa")}
    b = torch.tensor(math.log(0.05 / 0.95), dtype=torch.float64, device=dev, requires_grad=True)
    params = [w["x"], w["a"], w["xa"], b]
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=200, tolerance_grad=1e-10, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")
    blocks = [torch.as_tensor(users[s:s + chunk], device=dev) for s in range(0, len(users), chunk)]
    A = world.our_a.double()

    def logits(u):
        x = world.our_x[u].double()
        return (x @ w["x"])[:, None] + (A @ w["a"])[None, :] + (x * w["xa"]) @ A.T + b

    def closure():
        opt.zero_grad()
        total = 0.0
        for u in blocks:
            loss = (world.logger(u).double() * _bce(logits(u), world.q(u).double())).sum() / len(users)
            loss.backward()
            total += float(loss.detach())
        return torch.tensor(total)

    opt.step(closure)
    return {"w_x": w["x"].detach(), "w_a": w["a"].detach(), "w_xa": w["xa"].detach(), "b": b.detach()}


def qinf_rows(qinf: dict, world: WorldTensors, u: torch.Tensor) -> torch.Tensor:
    x = world.our_x[u].double()
    A = world.our_a.double()
    f = (x @ qinf["w_x"])[:, None] + (A @ qinf["w_a"])[None, :] + (x * qinf["w_xa"]) @ A.T + qinf["b"]
    return torch.sigmoid(f).float()


def fit_harmonic_population(dataset: dict, qinf: dict, *, world: WorldTensors, seed: int, lrs=VALUE_LRS,
                            steps: int = FIT_STEPS, batch_users: int = 2048, log=print,
                            compare_thetas: dict | None = None) -> dict:
    """θ_harm*: Adam ascent of V_h(θ) = Σ_u Σ_j [π q̂_∞ + π0 h(π/π0)(q − q̂_∞)] from the source (exact over items, a
    prior-drawn user sample, as the value fit), the best of ``lrs`` by V_h on the fit users."""
    mode, lam = parse_weight_spec("harmonic:0.1")
    users = _fit_users(dataset, seed, "harmonic", FIT_USERS)

    def vh(model, u):
        pi, p0 = _policy(model, u), world.logger(u)
        qh = qinf_rows(qinf, world, u)
        wj = pi / p0.clamp(min=1e-10)
        return (pi * qh + p0 * (wj / (1.0 - lam + lam * wj)) * (world.q(u) - qh)).sum(dim=1).mean()

    best = None
    for lr in lrs:
        seed_everything(derive_seed(int(seed), "opc_harmonic_population", f"{lr:g}"))
        model = build_model(dataset, device=world.device)
        rng = np.random.default_rng(derive_seed(int(seed), "opc_harmonic_population_order", f"{lr:g}"))
        opt = torch.optim.Adam(policy_params(model), lr=float(lr))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(steps))
        order, pos = rng.permutation(len(users)), 0
        for _ in range(int(steps)):
            if pos + batch_users > len(order):
                order, pos = rng.permutation(len(users)), 0
            u = torch.as_tensor(users[order[pos:pos + batch_users]], device=world.device)
            pos += batch_users
            loss = -vh(model, u)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            score = float(np.mean([float(vh(model, torch.as_tensor(users[s:s + 2048], device=world.device)))
                                   for s in range(0, len(users), 2048)]))
        log(f"[harmonic population] lr={lr:g}: V_h={score:.6f}")
        if best is None or score > best[0]:
            best = (score, lr, model)
    score, lr, model = best
    v, _ = exact_value(model, dataset, world=world, grad=False)
    at = {}
    for name, theta in (compare_thetas or {}).items():  # V_h at other states (θ_value*: is θ_harm* V_h's optimum?)
        m = build_model(dataset, device=world.device)
        set_theta(m, theta)
        with torch.no_grad():
            at[name] = float(np.mean([float(vh(m, torch.as_tensor(users[s:s + 2048], device=world.device)))
                                      for s in range(0, len(users), 2048)]))
    return {"theta": get_theta(model), "V_h_fit_users": score, "lr": lr, "V": v, "greedy": exact_greedy(model, dataset),
            "V_h_at": at}


# ------------------------------------------------------------------------------------------------------- the CLI
def main(argv=None) -> None:
    from training.opc_gradient_benchmark import build_world, world_dir_name
    from utils.representation_bias import add_world_arguments, resolve_bias_configs, world_options_from_args
    from utils.seeding import enable_determinism, pin_cpu_threads

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[100])
    ap.add_argument("--train-size", type=int, default=25_000)
    ap.add_argument("--objectives", nargs="+", default=list(OBJECTIVES), choices=list(OBJECTIVES))
    ap.add_argument("--no-harmonic-population", action="store_true")
    ap.add_argument("--steps", type=int, default=EMPIRICAL_STEPS)
    ap.add_argument("--emb-dir", default="BPR/embeddings")
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--states-run", default=None, help="a gradient-benchmark run whose states (θ_value*, θ_log*) the "
                    "objectives are also evaluated at")
    add_world_arguments(ap, bias_default=("high/none/none",))
    args = ap.parse_args(argv)
    enable_determinism(True)
    pin_cpu_threads(int(args.cpu_threads))
    world_options = world_options_from_args(args)
    share = float(world_options.get("logger_greedy_share", 0.8))
    for seed in args.seeds:
        for ds in args.datasets:
            for bias in resolve_bias_configs(args.bias_configs):
                dataset, label = build_world(ds, bias, seed, Path(args.emb_dir), world_options)
                wdir = Path(args.out) / world_dir_name(ds, label, seed, share)
                wdir.mkdir(parents=True, exist_ok=True)
                print(f"=== {wdir.name}", flush=True)
                world = WorldTensors(dataset)
                rows = study_training_rows(dataset, seed, args.train_size)
                lookup = None
                states = None
                if args.states_run:
                    sdir = Path(args.states_run) / wdir.name
                    z = np.load(sdir / "states.npz")
                    meta = json.loads((sdir / "states.json").read_text())
                    states = {"value": z["value"], "source": z["source"], "likelihood_click": z["likelihood_click"],
                              "likelihood_click_intercept": meta["likelihood_click"]["click_intercept"]}
                for objective in args.objectives:
                    path = wdir / f"empirical_{objective}.json"
                    if path.exists():
                        continue
                    if OBJECTIVES[objective][2] == "qhat" and lookup is None:
                        lookup = study_crossfit_lookup(dataset, rows, seed, world.device)
                    t0 = time.time()
                    compare = None
                    if states is not None:
                        compare = ({"theta_log_star": (states["likelihood_click"], states["likelihood_click_intercept"])}
                                   if OBJECTIVES[objective][0] == "likelihood" else
                                   {"theta_value_star": (states["value"], None), "source": (states["source"], None)})
                    ref = empirical_reference(dataset, rows, objective, seed=seed, world=world, lookup=lookup,
                                              steps=int(args.steps), compare_thetas=compare)
                    np.savez(wdir / f"empirical_{objective}.npz", theta=ref.pop("theta"), path=ref.pop("path"))
                    ref.pop("_probe", None)
                    ref.update(seconds=time.time() - t0, train_size=int(args.train_size),
                               train_click_sum=float(np.sum(rows["r"])))
                    path.write_text(json.dumps(ref, indent=2))
                    print(json.dumps({k: ref[k] for k in ("objective_name", "lr", "V", "greedy", "grad_norm")}), flush=True)
                hpath = wdir / "population_harmonic.json"
                if not args.no_harmonic_population and not hpath.exists():
                    t0 = time.time()
                    users = _fit_users(dataset, seed, "qinf", FIT_USERS)
                    qinf = population_reward_model(dataset, world, users)
                    h = fit_harmonic_population(dataset, qinf, world=world, seed=seed, compare_thetas=None if states is None
                                                else {"theta_value_star": states["value"], "source": states["source"]})
                    np.savez(wdir / "population_harmonic.npz", theta=h.pop("theta"),
                             **{k: v.cpu().numpy() for k, v in qinf.items()})
                    h["seconds"] = time.time() - t0
                    hpath.write_text(json.dumps(h, indent=2))
                    print(json.dumps({"harmonic_population": h}), flush=True)
                del dataset, world
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
