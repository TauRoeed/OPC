"""Fixed-policy oracle evaluator diagnostics (docs/opc_adaptive_estimator_study.md §8, Stage A).

A world, a fixed reward model q̂ and fixed policies (each learned policy's softmax and its greedy counterpart, the
source, θ_value*). Every estimator here is a mean of per-row contributions over n logged rows (x, a, r, μ(a|x)):

    f(x, a, r) = DM_π(x) + g(w) (r − q̂(x, a)),   DM_π(x) = Σ_j π(j|x) q̂(x, j),   w = π(a|x) / μ(a|x),

with g the weight transform: raw (g = w), harmonic h_λ(w) = w / (1 − λ + λ w), clip min(w, M), DM (g = 0), and the
oracle-q DR (q̂ = q, a reference). For a fixed q̂ (fit on other data) the rows are i.i.d., so the estimate's expectation
and variance are exact sums over users (the prior) and items (the logger), with E[(r − q̂)²] = (q − q̂)² + q(1 − q):

    E[f | x]  = DM_π(x) + Σ_a μ g(w) (q − q̂),
    E[f² | x] = Σ_a μ [ (DM_π(x) + g(w)(q − q̂))² + g(w)² q (1 − q) ],      Var(V̂) = (E f² − (E f)²) / n.

The same holds for the paired improvement f_π − f_π0 on the same rows. ``exact_world`` computes these; ``mc_world``
draws R fresh logs of n rows (users from the prior, actions from the logger, clicks from q) and computes every
estimator, also the nonlinear ones (self-normalized DR, the adaptive harmonic rules A2 and A3) and the plug-in
intervals, on the same logs for every policy (paired). The exact sums are the reference the replicates are checked
against.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from utils.seeding import derive_seed

HARMONIC_GRID = (0.005, 0.01, 0.03, 0.1, 0.2)  # λ = 0 is raw DR
CLIP_M = 10.0  # the native and common selectors' clip
A2_DELTA = 0.05  # δ of the concentration-optimal λ*₂ (Metelli, Russo and Restelli 2021, Theorem 5.1)
LOW_PROPENSITY = (1e-4, 1e-3)
HIGH_WEIGHT = (10.0, 100.0)


def linear_estimators() -> dict:
    """{name: (kind, parameter, reward source)} of the estimators with exact moments."""
    est = {"dm": ("dm", 0.0, "qhat"), "dr_raw": ("raw", 0.0, "qhat")}
    est.update({f"dr_h{lam:g}": ("harmonic", lam, "qhat") for lam in HARMONIC_GRID})
    est["dr_clip10"] = ("clip", CLIP_M, "qhat")
    est["dr_oracle"] = ("raw", 0.0, "q")
    return est


def transform(w: torch.Tensor, kind: str, param: float) -> torch.Tensor:
    if kind == "raw":
        return w
    if kind == "harmonic":
        return w / (1.0 - param + param * w)
    if kind == "clip":
        return torch.clamp(w, max=param)
    if kind == "dm":
        return torch.zeros_like(w)
    raise ValueError(kind)


def a2_lambda(s2: float, n: int, delta: float = A2_DELTA) -> float:
    """λ*₂ = sqrt(2 log(1/δ) / (3 I₂ n)), I₂ = E_μ[w²] = S2 (Metelli, Russo and Restelli 2021, Theorem 5.1)."""
    return float(min(1.0, math.sqrt(2.0 * math.log(1.0 / delta) / (3.0 * max(s2, 1.0) * n))))


# --------------------------------------------------------------------------------------------------- the world
class DenseWorld:
    """q, q̂ and the logger μ as dense float32 matrices on one device (users × items), the prior in float64, and the
    logger's float64 log-normalizers (exact propensities of sampled rows)."""

    def __init__(self, dataset: dict, qhat_lookup, device, chunk: int = 2048):
        from training.opc_gradients import WorldTensors, _prior

        self.world = WorldTensors(dataset, device)
        self.device = self.world.device
        self.n_users, self.n_items = int(dataset["n_users"]), int(dataset["n_actions"])
        self.prior = torch.as_tensor(_prior(dataset), dtype=torch.float64, device=self.device)
        self.T = self.world.temperature
        q, qh, mu, logz = [], [], [], []
        with torch.no_grad():
            for s in range(0, self.n_users, chunk):
                u = torch.arange(s, min(s + chunk, self.n_users), device=self.device)
                q.append(self.world.q(u).float())
                qh.append(torch.as_tensor(qhat_lookup[u], device=self.device).float())
                lg = (self.world.our_x[u].double() @ self.world.our_a.double().T) / self.T
                lz = torch.logsumexp(lg, dim=1)
                logz.append(lz)
                mu.append(torch.exp(lg - lz[:, None]).float())
        self.q, self.qhat, self.mu = torch.cat(q), torch.cat(qh), torch.cat(mu)
        self.logz_mu = torch.cat(logz)
        if self.world.mix > 0.0:
            raise NotImplementedError("a uniform-mixed logger is not used in these worlds")

    def mu_at(self, users, actions) -> torch.Tensor:
        """μ(a|x) in float64 at logged pairs."""
        lg = (self.world.our_x[users].double() * self.world.our_a[actions].double()).sum(dim=1) / self.T
        return torch.exp(lg - self.logz_mu[users])


class FixedPolicy:
    """A fixed policy over the catalog: ``softmax`` of exported vectors (ux · ia / T, as ``utils.policies.Policy``)
    or ``greedy`` (each user's top item of ux · ia, ties to the first)."""

    def __init__(self, name: str, ux, ia, temperature: float, kind: str, device):
        self.name, self.kind, self.T = name, kind, float(temperature)
        self.ux = torch.as_tensor(np.asarray(ux), dtype=torch.float32, device=device)
        self.ia = torch.as_tensor(np.asarray(ia), dtype=torch.float32, device=device)
        self._logz = None
        self._pick = None

    def prepare(self, n_users: int, chunk: int = 2048):
        logz, pick = [], []
        with torch.no_grad():
            for s in range(0, n_users, chunk):
                sc = self.ux[s:s + chunk].double() @ self.ia.double().T
                logz.append(torch.logsumexp(sc / self.T, dim=1))
                pick.append(sc.float().argmax(dim=1))
        self._logz, self._pick = torch.cat(logz), torch.cat(pick)

    def probs(self, users) -> torch.Tensor:
        if self.kind == "greedy":
            out = torch.zeros(len(users), self.ia.shape[0], device=self.ux.device)
            out[torch.arange(len(users), device=self.ux.device), self._pick[users]] = 1.0
            return out
        sc = (self.ux[users].double() @ self.ia.double().T) / self.T
        return torch.exp(sc - self._logz[users][:, None]).float()

    def prob_at(self, users, actions) -> torch.Tensor:
        if self.kind == "greedy":
            return (self._pick[users] == actions).double()
        sc = (self.ux[users].double() * self.ia[actions].double()).sum(dim=1) / self.T
        return torch.exp(sc - self._logz[users])


# ----------------------------------------------------------------------------------------------- exact moments
@torch.no_grad()
def exact_world(world: DenseWorld, policies: list, *, n: int, source: str, chunk: int = 1024) -> dict:
    """Per policy: the true value, support statistics, the reward model's error under it, and every linear
    estimator's exact expectation, bias, per-row variance and variance at n rows; the same for the paired improvement
    over ``source`` (one of the policies, of the same kind as each policy compared with it)."""
    est = linear_estimators()
    pol = {p.name: p for p in policies}
    acc = {p.name: {"V": 0.0, "S2": 0.0, "dm_err2": 0.0, "qerr2": 0.0, **{f"lowprop_{t:g}": 0.0 for t in LOW_PROPENSITY},
                    **{f"mass_w_gt_{t:g}": 0.0 for t in HIGH_WEIGHT}, "Ef": {e: 0.0 for e in est},
                    "Ef2": {e: 0.0 for e in est}, "dEf": {e: 0.0 for e in est}, "dEf2": {e: 0.0 for e in est}}
           for p in policies}
    acc_log = {"qerr2_logger": 0.0}
    for s in range(0, world.n_users, chunk):
        u = torch.arange(s, min(s + chunk, world.n_users), device=world.device)
        pr = world.prior[u]
        q, qh, mu = world.q[u].double(), world.qhat[u].double(), world.mu[u].double()
        mu = mu / mu.sum(dim=1, keepdim=True)  # the float32 rows renormalized in float64: Σ_a μ = 1 in the sums
        err = q - qh
        acc_log["qerr2_logger"] += float((pr * (mu * err ** 2).sum(dim=1)).sum())
        cache = {}
        for p in policies:
            pi = p.probs(u).double()
            cache[p.name] = pi
        for p in policies:
            pi = cache[p.name]
            a = acc[p.name]
            w = pi / mu.clamp(min=1e-300)
            a["V"] += float((pr * (pi * q).sum(dim=1)).sum())
            a["S2"] += float((pr * (pi * w).sum(dim=1)).sum())
            a["qerr2"] += float((pr * (pi * err ** 2).sum(dim=1)).sum())
            for t in LOW_PROPENSITY:
                a[f"lowprop_{t:g}"] += float((pr * (pi * (mu < t)).sum(dim=1)).sum())
            for t in HIGH_WEIGHT:
                a[f"mass_w_gt_{t:g}"] += float((pr * (pi * (w > t)).sum(dim=1)).sum())
            src = cache[source[p.kind]] if isinstance(source, dict) else cache[source]
            w0 = src / mu.clamp(min=1e-300)
            for e, (kind, param, rsrc) in est.items():
                base = q if rsrc == "q" else qh
                res = q - base
                dm = (pi * base).sum(dim=1, keepdim=True)
                g = transform(w, kind, param)
                m1 = dm[:, 0] + (mu * g * res).sum(dim=1)
                m2 = (mu * ((dm + g * res) ** 2 + g ** 2 * q * (1 - q))).sum(dim=1)
                a["Ef"][e] += float((pr * m1).sum())
                a["Ef2"][e] += float((pr * m2).sum())
                # the paired improvement over the source on the same rows
                dm0 = (src * base).sum(dim=1, keepdim=True)
                gd = g - transform(w0, kind, param)
                d1 = dm[:, 0] - dm0[:, 0] + (mu * gd * res).sum(dim=1)
                d2 = (mu * ((dm - dm0 + gd * res) ** 2 + gd ** 2 * q * (1 - q))).sum(dim=1)
                a["dEf"][e] += float((pr * d1).sum())
                a["dEf2"][e] += float((pr * d2).sum())
    out = {"logger": {"qhat_rmse": math.sqrt(acc_log["qerr2_logger"])}}
    for name, a in acc.items():
        p = pol[name]
        src_name = source[p.kind] if isinstance(source, dict) else source
        V0 = acc[src_name]["V"]
        row = {"kind": p.kind, "V": a["V"], "S2": a["S2"], "ess_fraction": 1.0 / a["S2"],
               "qhat_rmse": math.sqrt(max(a["qerr2"], 0.0)), "source": src_name, "true_gain": a["V"] - V0}
        row.update({k: v for k, v in a.items() if k.startswith(("lowprop_", "mass_w_gt_"))})
        row["a2_lambda_population"] = a2_lambda(a["S2"], n)
        ests = {}
        for e in est:
            m, m2 = a["Ef"][e], a["Ef2"][e]
            var_row = max(m2 - m * m, 0.0)
            dm_, dm2 = a["dEf"][e], a["dEf2"][e]
            dvar_row = max(dm2 - dm_ * dm_, 0.0)
            ests[e] = {"mean": m, "bias": m - a["V"], "var_row": var_row, "var": var_row / n,
                       "mse": (m - a["V"]) ** 2 + var_row / n,
                       "delta_mean": dm_, "delta_bias": dm_ - row["true_gain"], "delta_var_row": dvar_row,
                       "delta_var": dvar_row / n, "delta_mse": (dm_ - row["true_gain"]) ** 2 + dvar_row / n}
        row["estimators"] = ests
        out[name] = row
    # covariance of the two absolute estimates on the same rows, and the variance of the improvement from two
    # independent logs (each estimate's own variance)
    for name in acc:
        if name == "logger":
            continue
        src_name = out[name]["source"]
        for e, r in out[name]["estimators"].items():
            v1, v0 = r["var_row"], out[src_name]["estimators"][e]["var_row"]
            r["cov_with_source_row"] = 0.5 * (v1 + v0 - r["delta_var_row"])
            r["delta_var_independent"] = (v1 + v0) / n
    return out


# ------------------------------------------------------------------------------------------------- Monte Carlo
@torch.no_grad()
def sample_log(world: DenseWorld, n: int, seed: int, chunk: int = 4096) -> dict:
    """n i.i.d. rows: x ~ prior, a ~ μ(·|x), r ~ Bernoulli(q(x, a)); propensities in float64."""
    g = torch.Generator(device=world.device)
    g.manual_seed(int(seed) % (2 ** 63))
    users = torch.multinomial(world.prior.float(), n, replacement=True, generator=g)
    acts = torch.empty(n, dtype=torch.long, device=world.device)
    for s in range(0, n, chunk):
        acts[s:s + chunk] = torch.multinomial(world.mu[users[s:s + chunk]], 1, generator=g)[:, 0]
    qa = world.q[users, acts].double()
    r = (torch.rand(n, generator=g, device=world.device, dtype=torch.float64) < qa).double()
    return {"u": users, "a": acts, "r": r, "p": world.mu_at(users, acts), "q": qa,
            "qhat": world.qhat[users, acts].double()}


def _row_terms(world: DenseWorld, policy: FixedPolicy, log: dict, cache: dict) -> tuple:
    """Per-row DM_π(x_i) (with q̂ and with q), w_i, and S2_π(x_i) = Σ_a π(a|x_i)² / μ(a|x_i) (enumerated)."""
    if policy.name not in cache:
        dmh, dmq, s2 = [], [], []
        for s in range(0, world.n_users, 2048):
            u = torch.arange(s, min(s + 2048, world.n_users), device=world.device)
            pi = policy.probs(u).double()
            dmh.append((pi * world.qhat[u].double()).sum(dim=1))
            dmq.append((pi * world.q[u].double()).sum(dim=1))
            s2.append((pi * pi / world.mu[u].double().clamp(min=1e-300)).sum(dim=1))
        cache[policy.name] = (torch.cat(dmh), torch.cat(dmq), torch.cat(s2))
    dmh, dmq, s2 = cache[policy.name]
    w = policy.prob_at(log["u"], log["a"]) / log["p"]
    return dmh[log["u"]], dmq[log["u"]], w, s2[log["u"]]


def _estimates(dm_h, dm_q, w, s2_rows, log, n) -> dict:
    """{estimator: (estimate, per-row contributions)} on one log (contributions None for nonlinear ones)."""
    out = {}
    res_h = log["r"] - log["qhat"]
    for e, (kind, param, rsrc) in linear_estimators().items():
        dm, res = (dm_q, log["r"] - log["q"]) if rsrc == "q" else (dm_h, res_h)
        f = dm + transform(w, kind, param) * res
        out[e] = (float(f.mean()), f)
    # self-normalized DR (raw weights): DM mean + Σ w (r − q̂) / Σ w
    out["sndr"] = (float(dm_h.mean() + (w * res_h).sum() / w.sum().clamp(min=1e-300)), None)
    # A2: λ*₂ = sqrt(2 log(1/δ) / (3 n Ŝ2)) (Metelli, Russo and Restelli 2021, Theorem 5.1), with I₂ = S2 estimated by
    # its conditional value over the logged users, enumerated over every action (the logger is known): no sampled
    # action enters it, so it has none of the empirical w²'s heavy tail
    lam = a2_lambda(float(s2_rows.mean()), n)
    f = dm_h + transform(w, "harmonic", lam) * res_h
    out["dr_hA2"] = (float(f.mean()), f)
    out["_a2_lambda"] = (lam, None)
    # A3: the λ of the grid (with 0) minimizing an estimate of the MSE on this log: the squared difference to raw DR
    # less its own sampling variance (floored at 0), plus the shrunk estimate's variance
    f0 = dm_h + w * res_h
    best = None
    for lam_ in (0.0,) + HARMONIC_GRID:
        fl = f0 if lam_ == 0 else dm_h + transform(w, "harmonic", lam_) * res_h
        d = fl - f0
        b2 = max(float(d.mean()) ** 2 - float(d.var()) / n, 0.0)
        mse = b2 + float(fl.var()) / n
        if best is None or mse < best[0]:
            best = (mse, lam_, fl)
    out["dr_hA3"] = (float(best[2].mean()), best[2])
    out["_a3_lambda"] = (best[1], None)
    return out


@torch.no_grad()
def mc_world(world: DenseWorld, policies: list, *, n: int, replicates: int, seed: int, source: dict,
             z: float = 1.959963984540054) -> list:
    """R fresh logs; per log, policy and estimator: the estimate, its plug-in SE (row sd / √n; None for the
    nonlinear ones) and for the paired improvement over the source policy of the same kind the same; rows of
    dicts."""
    rows = []
    dm_cache = {}
    for rep in range(int(replicates)):
        log = sample_log(world, n, derive_seed(int(seed), "oracle_eval", "replicate", rep))
        per = {}
        for p in policies:
            dm_h, dm_q, w, s2_rows = _row_terms(world, p, log, dm_cache)
            per[p.name] = (_estimates(dm_h, dm_q, w, s2_rows, log, n), w)
        for p in policies:
            est, w = per[p.name]
            src_est, _ = per[source[p.kind]]
            base = {"replicate": rep, "policy": p.name, "kind": p.kind, "w_max": float(w.max()),
                    "w_q99": float(torch.quantile(w.float(), 0.99)), "w_q999": float(torch.quantile(w.float(), 0.999)),
                    "ess_kish": float(w.sum() ** 2 / (w ** 2).sum()) / n, "a2_lambda": est["_a2_lambda"][0],
                    "a3_lambda": est["_a3_lambda"][0]}
            for e, (val, f) in est.items():
                if e.startswith("_"):
                    continue
                se = float(f.std(unbiased=True)) / math.sqrt(n) if f is not None else float("nan")
                d_val = val - src_est[e][0]
                fs = src_est[e][1]
                d_se = (float((f - fs).std(unbiased=True)) / math.sqrt(n)) if f is not None and fs is not None \
                    else float("nan")
                rows.append({**base, "estimator": e, "estimate": val, "se": se, "delta": d_val, "delta_se": d_se})
    return rows


def summarize_mc(rows, exact: dict, z: float = 1.959963984540054):
    """Per policy and estimator: MC bias (± its SE), variance with a 95% chi-square interval, MSE, RMSE, median and
    upper-tail |error|, coverage of the plug-in 95% interval, against the exact truth; the same for the paired
    improvement; and the exact moments next to them."""
    import pandas as pd
    from scipy import stats

    d = pd.DataFrame(rows)
    out = []
    for (pol, e), g in d.groupby(["policy", "estimator"]):
        V = exact[pol]["V"]
        G = exact[pol]["true_gain"]
        R = len(g)
        err = g["estimate"].to_numpy() - V
        derr = g["delta"].to_numpy() - G
        var = float(np.var(g["estimate"], ddof=1))
        lo, hi = (R - 1) * var / stats.chi2.ppf(0.975, R - 1), (R - 1) * var / stats.chi2.ppf(0.025, R - 1)
        cover = float(np.mean(np.abs(err) <= z * g["se"].to_numpy())) if g["se"].notna().all() else float("nan")
        dcover = float(np.mean(np.abs(derr) <= z * g["delta_se"].to_numpy())) if g["delta_se"].notna().all() \
            else float("nan")
        r = {"policy": pol, "kind": g["kind"].iloc[0], "estimator": e, "replicates": R, "V": V, "true_gain": G,
             "mc_bias": float(err.mean()), "mc_bias_se": float(err.std(ddof=1) / math.sqrt(R)), "mc_var": var,
             "mc_var_lo": lo, "mc_var_hi": hi, "mc_mse": float(np.mean(err ** 2)),
             "mc_rmse": float(np.sqrt(np.mean(err ** 2))), "mc_abs_err_median": float(np.median(np.abs(err))),
             "mc_abs_err_q90": float(np.quantile(np.abs(err), 0.9)), "mc_coverage": cover,
             "mc_delta_bias": float(derr.mean()), "mc_delta_bias_se": float(derr.std(ddof=1) / math.sqrt(R)),
             "mc_delta_var": float(np.var(g["delta"], ddof=1)), "mc_delta_mse": float(np.mean(derr ** 2)),
             "mc_delta_coverage": dcover}
        ex = exact[pol]["estimators"].get(e)
        if ex is not None:
            r.update({f"exact_{k}": v for k, v in ex.items()})
            # the replicates against the exact moments: bias in MC standard errors, variance ratio
            r["bias_z"] = (r["mc_bias"] - ex["bias"]) / max(r["mc_bias_se"], 1e-300)
            r["var_ratio"] = var / ex["var"] if ex["var"] > 0 else float("nan")
            r["var_in_ci"] = bool(lo <= ex["var"] <= hi) if ex["var"] > 0 else True
        out.append(r)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------- Stage A driver
STAGE_A_ARMS = ("shared_lr_likelihood", "shared_lr_likelihood_calib", "shared_lr_opc", "shared_lr_opc_raw",
                "shared_lr_opc_oq")


def world_policies(dataset: dict, run_dir, population_dir, device, arms=STAGE_A_ARMS, n: int = 25_000) -> list:
    """The fixed policies of one world: each arm's selected policy (``--save-policies``; ``run_dir`` is one condition
    folder or a list of them, searched in order), the source (the logger's own vectors: π = μ) and θ_value* (the
    population value optimum, ``population_thetas.npz``, when it exists), each as its softmax and its greedy
    counterpart."""
    from pathlib import Path

    from training.opc_gradients import build_model, set_theta
    from training.trainer_trials import _policy_temperature

    T = float(_policy_temperature(dataset))
    dirs = [Path(d) for d in (run_dir if isinstance(run_dir, (list, tuple)) else [run_dir])]
    specs = [("source", dataset["our_x"], dataset["our_a"])]
    for arm in arms:
        f = next((d / f"{arm}_n{n}_r0_selected_policy.npz" for d in dirs
                  if (d / f"{arm}_n{n}_r0_selected_policy.npz").exists()), None)
        if f is None:
            raise FileNotFoundError(f"no selected policy of {arm} in {dirs}")
        z = np.load(f)
        specs.append((arm.replace("shared_lr_", "").replace("shared_lrg_", ""), z["ux"], z["ia"]))
    if population_dir is not None and (Path(population_dir) / "population_thetas.npz").exists():
        m = build_model(dataset, device=torch.device("cpu"))
        set_theta(m, np.load(Path(population_dir) / "population_thetas.npz")["value"])
        with torch.no_grad():
            ux, ia = m.get_params()
        specs.append(("value_star", ux.detach().cpu().numpy(), ia.detach().cpu().numpy()))
    pols = []
    for name, ux, ia in specs:
        for kind in ("softmax", "greedy"):
            p = FixedPolicy(name if kind == "softmax" else f"{name}_g", ux, ia, T, kind, device)
            p.prepare(int(dataset["n_users"]))
            pols.append(p)
    return pols


def main(argv=None) -> None:
    """Stage A: per world, the selection-time q̂ (the full-data reward model of the world's 25k training rows), the
    fixed policies, the exact moments and R fresh logs of n rows."""
    import argparse
    import json
    import time
    from pathlib import Path

    import pandas as pd

    from training.opc_empirical_objective import study_crossfit_lookup, study_training_rows
    from training.run_full_study import build_condition_world
    from training.ss_population import world_dir_name
    from utils.seeding import enable_determinism, pin_cpu_threads, seed_everything

    ap = argparse.ArgumentParser(description="Stage A fixed-policy oracle evaluator diagnostics")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--run", nargs="+", default=["artifacts/full_study/run_ss_main_25k"],
                    help="run folders with the selected policies (searched in order)")
    ap.add_argument("--arms", nargs="+", default=list(STAGE_A_ARMS))
    ap.add_argument("--stage", default="stage_a", help="the replicate seeds' label")
    ap.add_argument("--population", default="artifacts/full_study/run_ss_population")
    ap.add_argument("--n", type=int, default=25_000)
    ap.add_argument("--replicates", type=int, default=50)
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
                wdir = Path(args.out) / name
                if (wdir / f"summary_R{args.replicates}.csv").exists():
                    continue
                wdir.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                seed_everything(int(seed))
                dataset, *_ = build_condition_world(ds, Path(args.emb_dir), label, 0.05, int(seed),
                                                    world_options={"world_family": "structured_shift"})
                rows = study_training_rows(dataset, int(seed), 25_000)
                qhat = study_crossfit_lookup(dataset, rows, int(seed), device).full_lookup
                run_dirs = [d for run in args.run
                            for d in Path(run).glob(f"dataset={ds}__bias={label}__ctr=0.05__seed={seed}__*")]
                world = DenseWorld(dataset, qhat, device)
                pols = world_policies(dataset, run_dirs, Path(args.population) / name, device, arms=args.arms)
                t1 = time.time()
                src = {"softmax": "source", "greedy": "source_g"}
                ex = exact_world(world, pols, n=args.n, source=src)
                (wdir / "exact.json").write_text(json.dumps(ex, indent=1, default=float))
                t2 = time.time()
                mc = mc_world(world, pols, n=args.n, replicates=args.replicates,
                              seed=derive_seed(int(seed), ds, label, args.stage), source=src)
                pd.DataFrame(mc).to_csv(wdir / f"mc_rows_R{args.replicates}.csv.gz", index=False, float_format="%.8g")
                summ = summarize_mc(mc, ex)
                summ.to_csv(wdir / f"summary_R{args.replicates}.csv", index=False, float_format="%.8g")
                t3 = time.time()
                info = {"world": name, "n": args.n, "replicates": args.replicates, "train_rows": int(len(rows["r"])),
                        "train_click_sum": float(np.sum(rows["r"])), "seconds_setup": t1 - t0,
                        "seconds_exact": t2 - t1, "seconds_mc": t3 - t2}
                (wdir / f"info_R{args.replicates}.json").write_text(json.dumps(info, indent=1))
                print(json.dumps(info), flush=True)
                del world
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
