"""Oracle gradient diagnostics of the adaptive-estimator study (docs/opc_adaptive_estimator_study.md §10, §10a).

At a fixed policy θ of the rank-4 adapter class and a fixed reward model q̂ (a world's selection-time full-data model,
fit on its training rows, so independent of the fresh datasets):

- the true gradient g* = ∇V(θ) (exact over users and items);
- per estimator, the expected gradient. For an estimator linear in the rows with fixed q̂ and weight transform g, it is
  the gradient of its exact expectation Σ_u p(u) [DM_π(u) + Σ_a μ g(π/μ)(q − q̂)], so its bias is exact. The adaptive
  ones (A2, A3) and the CRM objective take the mean over the replicates;
- the variance: R fresh datasets of N rows (users from the prior, actions from the logger, clicks from q), the
  per-dataset gradient of the estimate.

Metrics (regime study §4): ‖bias‖, ‖bias‖ / ‖g*(source)‖, the variance (trace), MSE = ‖bias‖² + variance, the
signal-to-noise ratio ‖E ĝ‖ / sqrt(variance), cos(E ĝ, g*) and the mean cosine of single-dataset gradients with g*.

States (§10a): the source; the selected policy (θ recovered from its exported vectors: the scale from the user side,
M = VUᵀ by least squares from the item side, a balanced SVD factorization); the intermediate point on the straight
path (t M, log s scaled by t) whose true value is halfway between the source's and the selected policy's.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from training.opc_gradients import (
    WorldTensors,
    _flat_grad,
    _policy,
    _prior,
    build_model,
    exact_value,
    get_theta,
    policy_params,
    set_theta,
)
from utils.seeding import derive_seed

CRM_GAMMA = 1.0


def theta_from_export(dataset: dict, ux, ia, device) -> np.ndarray:
    """θ of the rank-4 adapter (U, V, the scale) whose exported vectors are (ux, ia): ux = s x, ia = a + (aV)Uᵀ."""
    model = build_model(dataset, device=device)
    x = np.asarray(dataset["our_x"], np.float64)
    a = np.asarray(dataset["our_a"], np.float64)
    s = float((np.asarray(ux, np.float64) * x).sum() / (x * x).sum())
    M, *_ = np.linalg.lstsq(a, np.asarray(ia, np.float64) - a, rcond=None)  # ia − a = a M, M = V Uᵀ
    P, sv, Qt = np.linalg.svd(M)
    r = model.action_transform.U.shape[1]
    V = P[:, :r] * np.sqrt(sv[:r])
    U = Qt[:r].T * np.sqrt(sv[:r])
    with torch.no_grad():
        model.action_transform.U.copy_(torch.as_tensor(U, dtype=torch.float32))
        model.action_transform.V.copy_(torch.as_tensor(V, dtype=torch.float32))
        model.log_logit_scale.copy_(torch.as_tensor(math.log(max(s, 1e-12)) / 30.0, dtype=torch.float32))
    return get_theta(model)


def line_state(dataset: dict, theta_final: np.ndarray, world: WorldTensors, iters: int = 20) -> tuple[np.ndarray, float]:
    """The point θ(t) = (√t U, √t V, t log s) (so M(t) = t M) whose true value is halfway between t = 0 (the source;
    U = 0 there, V the final's) and t = 1, by bisection on t."""
    m = build_model(dataset, device=world.device)
    set_theta(m, theta_final)
    U0, V0 = m.action_transform.U.detach().clone(), m.action_transform.V.detach().clone()
    ls0 = m.log_logit_scale.detach().clone()

    def at(t):
        with torch.no_grad():
            m.action_transform.U.copy_(U0 * math.sqrt(t))
            m.action_transform.V.copy_(V0 * math.sqrt(t) if t > 0 else V0)
            m.log_logit_scale.copy_(ls0 * t)
        return exact_value(m, dataset, world=world, grad=False)[0]

    v0, v1 = at(0.0), at(1.0)
    target, lo, hi = 0.5 * (v0 + v1), 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if (at(mid) - target) * (v1 - v0) < 0:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    at(t)
    return get_theta(m), t


def _transform(w, lam: float):
    return w if lam <= 0 else w / (1.0 - lam + lam * w)


def expected_gradient(model, dataset, world, qrows, lam: float, chunk: int = 2048) -> np.ndarray:
    """∇ of Σ_u p(u) [Σ_j π q̂ + Σ_a μ g_λ(π/μ)(q − q̂)]: the expected gradient of the linear DR estimator with fixed q̂."""
    params = policy_params(model)
    prior = torch.as_tensor(_prior(dataset), dtype=torch.float64, device=world.device)
    total = None
    for s in range(0, int(dataset["n_users"]), chunk):
        u = torch.arange(s, min(s + chunk, int(dataset["n_users"])), device=world.device)
        pi, mu, q, qh = _policy(model, u).double(), world.logger(u).double(), world.q(u).double(), qrows(u).double()
        w = pi / mu.clamp(min=1e-300)
        val = (prior[u] * ((pi * qh).sum(dim=1) + (mu * _transform(w, lam) * (q - qh)).sum(dim=1))).sum()
        g = _flat_grad(torch.autograd.grad(val, params))
        total = g if total is None else total + g
    return total


def true_gradient(model, dataset, world) -> np.ndarray:
    return exact_value(model, dataset, world=world, grad=True)[1]


def sample_rows(world, dataset, n: int, seed: int) -> dict:
    """n i.i.d. logged rows (users from the prior, actions from the logger, clicks from q)."""
    g = torch.Generator(device=world.device)
    g.manual_seed(int(seed) % (2 ** 63))
    prior = torch.as_tensor(_prior(dataset), dtype=torch.float32, device=world.device)
    users = torch.multinomial(prior, n, replacement=True, generator=g)
    acts = torch.empty(n, dtype=torch.long, device=world.device)
    for s in range(0, n, 4096):
        acts[s:s + 4096] = torch.multinomial(world.logger(users[s:s + 4096]), 1, generator=g)[:, 0]
    rows_q = torch.empty(n, dtype=torch.float64, device=world.device)
    p = torch.empty(n, dtype=torch.float64, device=world.device)
    for s in range(0, n, 4096):
        u, a = users[s:s + 4096], acts[s:s + 4096]
        idx = torch.arange(len(u), device=world.device)
        rows_q[s:s + 4096] = world.q(u)[idx, a].double()
        p[s:s + 4096] = world.logger(u).double()[idx, a]
    r = (torch.rand(n, generator=g, device=world.device, dtype=torch.float64) < rows_q).double()
    return {"u": users, "a": acts, "r": r, "p": p}


def dataset_gradients(model, rows: dict, world, qmodels: dict, estimators: list, n: int, chunk: int = 2048) -> dict:
    """The gradient of each estimator's estimate on one dataset. ``estimators``: (name, q̂ key, kind, param) with
    kind 'lam' (fixed λ), 'a2' (λ*₂ from the rows' enumerated S2), 'a3' (the MSE-minimizing λ of the grid on these
    rows) or 'crm' (raw DR minus γ times its standard error)."""
    from models.adaptive_losses import A3_GRID, a2_lambda

    params = policy_params(model)
    out = {}
    # per-row quantities that do not need gradients: the A2 / A3 choices
    with torch.no_grad():
        s2_rows, fq = [], {}
        for s in range(0, n, chunk):
            u = rows["u"][s:s + chunk]
            pi, mu = _policy(model, u).double(), world.logger(u).double()
            s2_rows.append((pi * pi / mu.clamp(min=1e-300)).sum(dim=1))
        s2 = float(torch.cat(s2_rows).mean())
    lam_a2 = a2_lambda(s2, n)
    for name, qkey, kind, param in estimators:
        qrows = qmodels[qkey]
        lam = param if kind == "lam" else lam_a2 if kind == "a2" else 0.0
        if kind == "a3":  # the λ minimizing the estimated MSE on these rows (no gradient)
            with torch.no_grad():
                fs = {l_: [] for l_ in A3_GRID}
                for s in range(0, n, chunk):
                    u, a = rows["u"][s:s + chunk], rows["a"][s:s + chunk]
                    idx = torch.arange(len(u), device=world.device)
                    pi, qh = _policy(model, u).double(), qrows(u).double()
                    w = pi[idx, a] / rows["p"][s:s + chunk]
                    dm, res = (pi * qh).sum(dim=1), rows["r"][s:s + chunk] - qh[idx, a]
                    for l_ in A3_GRID:
                        fs[l_].append(dm + _transform(w, l_) * res)
                f0 = torch.cat(fs[0.0])
                best = None
                for l_ in A3_GRID:
                    fl = torch.cat(fs[l_])
                    d = fl - f0
                    mse = max(float(d.mean()) ** 2 - float(d.var()) / n, 0.0) + float(fl.var()) / n
                    if best is None or mse < best[0]:
                        best = (mse, l_)
                lam = best[1]
        total, f_all = None, []
        for s in range(0, n, chunk):
            u, a = rows["u"][s:s + chunk], rows["a"][s:s + chunk]
            idx = torch.arange(len(u), device=world.device)
            pi, qh = _policy(model, u).double(), qrows(u).double()
            w = pi[idx, a] / rows["p"][s:s + chunk]
            f = (pi * qh).sum(dim=1) + _transform(w, lam) * (rows["r"][s:s + chunk] - qh[idx, a])
            if kind == "crm":
                f_all.append(f)
                continue
            g = _flat_grad(torch.autograd.grad(f.sum() / n, params))
            total = g if total is None else total + g
        if kind == "crm":  # J = mean f − γ sqrt(var f / n): one graph over the whole dataset
            f = torch.cat(f_all)
            J = f.mean() - float(param) * torch.sqrt(f.var() / n + 1e-12)
            total = _flat_grad(torch.autograd.grad(J, params))
        out[name] = total
    out["_lam_a2"], out["_lam_a3"] = lam_a2, None
    return out


def metrics(g_star: np.ndarray, g_mean: np.ndarray, grads: np.ndarray, ref_norm: float) -> dict:
    bias = g_mean - g_star
    dev = grads - grads.mean(axis=0, keepdims=True)
    var = float((dev ** 2).sum(axis=1).mean())
    cos = lambda a, b: float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))
    return {"bias_norm": float(np.linalg.norm(bias)), "rel_bias": float(np.linalg.norm(bias) / max(ref_norm, 1e-300)),
            "variance": var, "mse": float(bias @ bias + var), "snr": float(np.linalg.norm(g_mean) / max(math.sqrt(var), 1e-300)),
            "cos_mean": cos(g_mean, g_star), "cos_single": float(np.mean([cos(g, g_star) for g in grads])),
            "g_star_norm": float(np.linalg.norm(g_star))}


ESTIMATORS = [("dr_raw", "d0", "lam", 0.0), ("dr_h0.1", "d0", "lam", 0.1), ("dr_h0.2", "d0", "lam", 0.2),
              ("dr_hA2", "d0", "a2", None), ("dr_hA3", "d0", "a3", None), ("crm1", "d0", "crm", CRM_GAMMA),
              ("dr_h0.1_D1", "d1", "lam", 0.1), ("dr_h0.1_D2", "d2", "lam", 0.1), ("dr_h0.1_oracle", "q", "lam", 0.1)]


def main(argv=None) -> None:
    """§10 on the representative subset: per world, the source, the intermediate point and harmonic OPC's (B3)
    selected policy of the pilot; every estimator's gradient metrics."""
    import argparse
    import json
    import time
    from pathlib import Path

    from models.reward_models import BilinearRewardLookup, fit_bilinear_reward
    from training.opc_empirical_objective import study_crossfit_lookup, study_training_rows
    from training.run_full_study import build_condition_world
    from training.ss_population import world_dir_name
    from utils.seeding import enable_determinism, pin_cpu_threads, seed_everything

    ap = argparse.ArgumentParser(description="Stage B/C oracle gradient diagnostics (§10)")
    ap.add_argument("--datasets", nargs="+", default=["ml", "kuairand"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[300])
    ap.add_argument("--labels", nargs="+", default=["s-small.r-strong", "s-moderate.r-strong"])
    ap.add_argument("--run", default="artifacts/full_study/run_ae_pilot_A")
    ap.add_argument("--arm", default="shared_lr_opc")
    ap.add_argument("--n", type=int, default=25_000)
    ap.add_argument("--replicates", type=int, default=100)
    ap.add_argument("--emb-dir", default="BPR/embeddings")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    enable_determinism(True)
    pin_cpu_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in args.seeds:
        for ds in args.datasets:
            for label in args.labels:
                name = world_dir_name(ds, label, seed, 0.8)
                f_out = Path(args.out) / name / f"gradients_{args.arm}.json"
                if f_out.exists():
                    continue
                f_out.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                seed_everything(int(seed))
                dataset, *_ = build_condition_world(ds, Path(args.emb_dir), label, 0.05, int(seed),
                                                    world_options={"world_family": "structured_shift"})
                world = WorldTensors(dataset, device)
                rows = study_training_rows(dataset, int(seed), 25_000)
                d0 = study_crossfit_lookup(dataset, rows, int(seed), device).full_lookup
                X, A = np.asarray(dataset["our_x"]), np.asarray(dataset["our_a"])
                u_, a_ = np.asarray(rows["x_idx"], np.int64), np.asarray(rows["a"], np.int64)
                qm = {"d0": lambda u, lk=d0: torch.as_tensor(lk[u], device=device).float(), "q": world.q}
                for key, kind in (("d1", "lowrank"), ("d2", "hetero")):
                    rm_seed = derive_seed(int(seed), "reward_model", kind, 25_000)
                    lk = BilinearRewardLookup(fit_bilinear_reward(X[u_], A[a_], np.asarray(rows["r"], np.float64), kind=kind,
                                                                  seed=rm_seed * 31 + 5, device=device), X, A, device)
                    qm[key] = lambda u, lk=lk: lk[u]
                cond = next(Path(args.run).glob(f"dataset={ds}__bias={label}__ctr=0.05__seed={seed}__*"))
                z = np.load(cond / f"{args.arm}_n25000_r0_selected_policy.npz")
                th_final = theta_from_export(dataset, z["ux"], z["ia"], device)
                th_mid, t_mid = line_state(dataset, th_final, world)
                th_src = get_theta(build_model(dataset, device=device))
                res = {"world": name, "arm": args.arm, "t_mid": t_mid, "states": {}}
                ref = None
                for state, th in (("source", th_src), ("intermediate", th_mid), ("final", th_final)):
                    m = build_model(dataset, device=device)
                    set_theta(m, th)
                    g_star = true_gradient(m, dataset, world)
                    ref = float(np.linalg.norm(g_star)) if state == "source" else ref
                    exp_g = {e[0]: expected_gradient(m, dataset, world, qm[e[1]], e[3]) for e in ESTIMATORS
                             if e[2] == "lam"}
                    reps = {e[0]: [] for e in ESTIMATORS}
                    lam_a2 = []
                    for r in range(int(args.replicates)):
                        data = sample_rows(world, dataset, args.n, derive_seed(int(seed), ds, label, "gradients", state, r))
                        gr = dataset_gradients(m, data, world, qm, ESTIMATORS, args.n)
                        for e in ESTIMATORS:
                            reps[e[0]].append(gr[e[0]])
                        lam_a2.append(gr["_lam_a2"])
                    st = {"V": exact_value(m, dataset, world=world, grad=False)[0], "lam_a2_median": float(np.median(lam_a2))}
                    for e in ESTIMATORS:
                        G = np.stack(reps[e[0]])
                        mean = exp_g.get(e[0], G.mean(axis=0))
                        st[e[0]] = {**metrics(g_star, mean, G, ref), "expected": "exact" if e[0] in exp_g else "replicates"}
                    res["states"][state] = st
                res["seconds"] = time.time() - t0
                f_out.write_text(json.dumps(res, indent=1, default=float))
                print(json.dumps({"world": name, "seconds": round(res["seconds"]),
                                  **{s_: {e: round(v[e]["cos_mean"], 3) for e in ("dr_raw", "dr_h0.1", "dr_h0.1_D2")}
                                     for s_, v in res["states"].items()}}), flush=True)


if __name__ == "__main__":
    main()
