"""Structured scenario shift (docs/structured_scenario_shift_study.md §1-§3): a matched low-rank source-to-target
preference shift and a ranking-preserving response heterogeneity.

Source score s_A(u, i) = x_uᵀ a_i: the learner's and the logger's representation, the BPR vectors as they are.
Target score s_B(u, i) = x_uᵀ (I + g_u Δ*) a_i, Δ* = U* D* Vᵀ* of rank 4 (g_u = 1, or a smooth gate in the gated
family of §7). Target clicks q_B(u, i) = σ(κ α_u s̃_B(u, i) + c + β_u), with s̃_B the globally standardized target
score, α_u > 0 and β_u smooth in x_u: argmax_i q_B = argmax_i s_B.

The world reuses the legacy calibration (``utils.representation_bias.calibrate_world``) for the user prior, the spread
temperature and the calibration users, and carries the truth in a ``SyntheticBanditEnv`` with augmented vectors
[(κ α_u / σ_B) y_u, c + β_u − κ α_u μ_B / σ_B] · [a_i, 1] (y_u = (I + g_u Δ*)ᵀ x_u), scale 1 and offset 0, so every
existing consumer of the click model reads q_B. Labels: ``s-<shift>.r-<response>[.gated]``.
"""
from __future__ import annotations

import copy
import dataclasses
import re

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from utils.seeding import derive_seed

FAMILY = "structured_shift"
SHIFT_LEVELS = {"none": 1.0, "moderate": 0.90, "strong": 0.75}  # mean user-level score correlation (§2)
# sd(log α), sd(β) (§3 as amended in §13.1: half the first targets, which the legacy CTR targets could not carry)
RESPONSE_LEVELS = {"none": (0.0, 0.0), "moderate": (0.125, 0.25), "strong": (0.25, 0.5)}
PATHOLOGY = {"best_q": 0.95, "best_q_excess": 0.10, "best_q_max": 0.15, "low_q": 1e-3, "low_q_share": 0.05}  # §13.1
SHIFT_RANK = 4
SHIFT_PCS = 8  # the dominant source subspaces the shift directions mix (§2)
RESPONSE_PCS = 4  # the user subspace of the response directions (§3)
Z_CLIP = 3.0  # standardized projections are clipped here (§3; the one-time sanity adjustment would make it 2)
GATE_SLOPE = 2.0  # g(x) = σ(2 z_g(x)) in the gated family (§7)
RHO_TOL = 0.005
TOP_K = 10
_LABEL = re.compile(r"^s-(none|moderate|strong)\.r-(none|moderate|strong)(\.gated)?$")


class StructuredCalibrationError(RuntimeError):
    """A pre-registered target cannot be met (the study stops, §11)."""


def is_structured_label(label) -> bool:
    return isinstance(label, str) and bool(_LABEL.match(label.strip()))


def parse_structured(label: str) -> dict:
    m = _LABEL.match(str(label).strip())
    if not m:
        raise ValueError(f"not a structured-shift label: {label!r} (expected s-<level>.r-<level>[.gated])")
    return {"shift": m.group(1), "response": m.group(2), "gated": bool(m.group(3))}


def structured_label(shift: str, response: str, gated: bool = False) -> str:
    label = f"s-{shift}.r-{response}" + (".gated" if gated else "")
    parse_structured(label)
    return label


# ------------------------------------------------------------------------------------------------- source geometry
def principal_directions(V: np.ndarray, k: int, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The top-k eigenvectors (columns) and eigenvalues of the (weighted) covariance of the rows of V, each sign fixed
    so that its largest-magnitude entry is positive (deterministic)."""
    V = np.asarray(V, dtype=np.float64)
    w = np.full(V.shape[0], 1.0 / V.shape[0]) if weights is None else np.asarray(weights, np.float64) / np.sum(weights)
    mu = w @ V
    C = (V - mu).T @ ((V - mu) * w[:, None])
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1][:k]
    vals, vecs = vals[order], vecs[:, order]
    signs = np.sign(vecs[np.abs(vecs).argmax(axis=0), np.arange(k)])
    return vecs * signs, vals


def _orthonormal(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, k)))
    return q * np.sign(np.diag(r))  # unique: positive diagonal of r


def shift_directions(X: np.ndarray, A: np.ndarray, prior: np.ndarray, seed: int, rank: int = SHIFT_RANK,
                     pcs: int = SHIFT_PCS) -> dict:
    """U* = P_x Q_u, V* = P_a Q_v (orthonormal columns inside the top-``pcs`` source subspaces) and the signs ε of
    D* = γ diag(ε) (§2)."""
    Px, _ = principal_directions(X, pcs, prior)
    Pa, _ = principal_directions(A, pcs)
    Qu = _orthonormal(np.random.default_rng(derive_seed(seed, "structured_shift", "directions", "users")), pcs, rank)
    Qv = _orthonormal(np.random.default_rng(derive_seed(seed, "structured_shift", "directions", "items")), pcs, rank)
    signs = np.random.default_rng(derive_seed(seed, "structured_shift", "directions", "signs")).choice([-1.0, 1.0], rank)
    return {"U": Px @ Qu, "V": Pa @ Qv, "signs": signs}


def shift_matrix(directions: dict, gamma: float) -> np.ndarray:
    """Δ* = U* diag(γ ε) V*ᵀ."""
    return (directions["U"] * (float(gamma) * directions["signs"])) @ directions["V"].T


def gate_values(X: np.ndarray, prior: np.ndarray, seed: int) -> np.ndarray:
    """g(x_u) = σ(2 z_g(x_u)), z_g the standardized (prior) projection on a seeded direction in the top-4 user
    subspace, clipped at ±3 (§7)."""
    z = _standardized_projection(X, prior, _response_coeffs(X, prior, seed, "gate"))
    return expit(GATE_SLOPE * z)


def target_user_vectors(X: np.ndarray, delta: np.ndarray, gate: np.ndarray | None = None) -> np.ndarray:
    """y_u = (I + g_u Δ)ᵀ x_u, so that s_B(u, i) = y_u · a_i (row vectors: y = x + g ⊙ (x Δ))."""
    shift = X @ delta
    return X + (shift if gate is None else gate[:, None] * shift)


def _row_corr(S: np.ndarray, T: np.ndarray) -> np.ndarray:
    S = S - S.mean(axis=1, keepdims=True)
    T = T - T.mean(axis=1, keepdims=True)
    return (S * T).sum(axis=1) / np.sqrt((S * S).sum(axis=1) * (T * T).sum(axis=1))


def score_agreement(X: np.ndarray, Y: np.ndarray, A: np.ndarray, users: np.ndarray, top_k: int = TOP_K) -> dict:
    """On ``users``: the per-user correlation across items between x_u·a and y_u·a, top-1 agreement and top-k overlap
    (equal weights: ``users`` are prior draws, so these are prior-weighted means)."""
    corr, top1, topk = [], [], []
    for s in range(0, len(users), 128):
        u = users[s:s + 128]
        SA, SB = X[u] @ A.T, Y[u] @ A.T
        corr.append(_row_corr(SA, SB))
        top1.append(SA.argmax(axis=1) == SB.argmax(axis=1))
        ka = np.argpartition(-SA, top_k, axis=1)[:, :top_k]
        kb = np.argpartition(-SB, top_k, axis=1)[:, :top_k]
        topk.append(np.array([len(np.intersect1d(p, q)) / top_k for p, q in zip(ka, kb)]))
    corr, top1, topk = (np.concatenate(v) for v in (corr, top1, topk))
    return {"score_corr": float(corr.mean()), "score_corr_q10": float(np.quantile(corr, 0.1)),
            "score_corr_q50": float(np.median(corr)), "score_corr_min": float(corr.min()),
            "top1_agreement": float(top1.mean()), f"top{top_k}_overlap": float(topk.mean())}


def calibrate_gamma(X, A, directions, users, target: float, gate=None) -> tuple[float, float]:
    """The smallest γ ≥ 0 whose mean user-level score correlation is ``target`` (a log grid, then brentq; §2)."""
    if target >= 1.0:
        return 0.0, 1.0
    Xu = X[users]
    g = None if gate is None else gate[users]

    def rho(gamma):
        Y = target_user_vectors(Xu, shift_matrix(directions, gamma), g)
        out = []
        for s in range(0, len(users), 128):
            out.append(_row_corr(Xu[s:s + 128] @ A.T, Y[s:s + 128] @ A.T))
        return float(np.concatenate(out).mean())

    grid = np.logspace(-3, 3, 61)
    prev = 0.0
    for gam in grid:
        r = rho(gam)
        if r <= target:
            g_star = float(brentq(lambda t: rho(t) - target, prev, gam, xtol=1e-10))
            achieved = rho(g_star)
            if abs(achieved - target) > RHO_TOL:
                raise StructuredCalibrationError(f"shift: correlation {achieved:.4f} for target {target:.2f}")
            return g_star, achieved
        prev = gam
    raise StructuredCalibrationError(f"shift: the correlation never falls to {target:.2f} (minimum {r:.3f} at γ=1e3)")


# ------------------------------------------------------------------------------------------- response heterogeneity
def _response_coeffs(X, prior, seed, label: str) -> np.ndarray:
    P, lam = principal_directions(X, RESPONSE_PCS, prior)
    c = np.random.default_rng(derive_seed(seed, "structured_shift", "response", label)).normal(size=RESPONSE_PCS)
    if label == "beta":  # orthogonal to alpha's direction in the eigenvalue metric: uncorrelated projections
        ca = np.random.default_rng(derive_seed(seed, "structured_shift", "response", "alpha")).normal(size=RESPONSE_PCS)
        ca /= np.linalg.norm(ca)
        c = c - (ca @ (lam * c)) / (ca @ (lam * ca)) * ca
    return P @ (c / np.linalg.norm(c))


def _standardized_projection(X, prior, w, clip: float = Z_CLIP) -> np.ndarray:
    p = np.asarray(prior, np.float64) / np.sum(prior)
    z = X @ w
    z = (z - p @ z) / np.sqrt(p @ (z - p @ z) ** 2)
    z = np.clip(z, -clip, clip)
    z = z - p @ z
    return z / np.sqrt(p @ z ** 2)


def response_factors(X, prior, seed, level: str, clip: float = Z_CLIP) -> dict:
    """α_u and β_u of §3: log α = σ_α z_α − log E_prior exp(σ_α z_α), β = σ_β z_β, z standardized seeded projections
    in the top-4 user subspace (uncorrelated under the prior), clipped at ``clip``."""
    s_alpha, s_beta = RESPONSE_LEVELS[level]
    p = np.asarray(prior, np.float64) / np.sum(prior)
    za = _standardized_projection(X, prior, _response_coeffs(X, prior, seed, "alpha"), clip)
    zb = _standardized_projection(X, prior, _response_coeffs(X, prior, seed, "beta"), clip)
    log_alpha = s_alpha * za - np.log(p @ np.exp(s_alpha * za))
    return {"alpha": np.exp(log_alpha), "beta": s_beta * zb, "z_alpha": za, "z_beta": zb,
            "sigma_alpha": s_alpha, "sigma_beta": s_beta, "clip": float(clip)}


# ----------------------------------------------------------------------------------------------- click calibration
def _solve_c(L: np.ndarray, target: float, c0: float | None = None) -> float:
    """c with mean(σ(L + c)) = target (Newton with a bisection fallback), as the legacy ``_solve_b``."""
    reach = float(np.max(np.abs(L))) + 60.0
    lo, hi = -reach, reach
    c = float(np.log(target / (1 - target)) - float(np.mean(L))) if c0 is None else float(c0)
    c = min(max(c, lo), hi)
    for _ in range(200):
        q = expit(L + c)
        f = float(q.mean()) - target
        if abs(f) < 1e-12:
            return c
        if f > 0:
            hi = c
        else:
            lo = c
        d = float((q * (1 - q)).mean())
        nc = c - f / d if d > 1e-15 else (lo + hi) / 2
        c = nc if lo < nc < hi else (lo + hi) / 2
    return c


def calibrate_click(zs, zmax, alpha_u, beta_u, target_ctr: float, best_ctr: float) -> tuple[float, float]:
    """κ and c: the reference policy's CTR (``zs``: its sampled items' standardized target scores per calibration user)
    is ``target_ctr`` and the best item's (``zmax``) averages ``best_ctr`` (§3; the legacy procedure, the truth q_B)."""
    a, b = np.asarray(alpha_u, np.float64)[:, None], np.asarray(beta_u, np.float64)[:, None]
    warm = {"c": None}

    def best_of(kappa):
        warm["c"] = _solve_c(kappa * a * zs + b, target_ctr, warm["c"])
        return float(expit(kappa * a[:, 0] * zmax + warm["c"] + b[:, 0]).mean())

    lo, hi = 1e-4, 60.0
    if best_of(hi) < best_ctr or best_of(lo) > best_ctr:
        raise StructuredCalibrationError(f"best-item CTR {best_ctr:.0%} unreachable with reference CTR {target_ctr:.1%}")
    kappa = float(brentq(lambda k: best_of(k) - best_ctr, lo, hi, xtol=1e-12))
    return kappa, _solve_c(kappa * a * zs + b, target_ctr)


def click_sanity(q_best: np.ndarray, q_logger: np.ndarray, q_best_homogeneous: np.ndarray) -> dict:
    """The pathology criteria of §13.1 on the calibration users, relative to the homogeneous world of the same dataset,
    seed and shift: the share of near-deterministic best items (q > 0.95) more than 10 points above the homogeneous
    world's or above 15%, or more than 5% of users with a logger click probability below 0.001."""
    P = PATHOLOGY
    out = {"share_best_q_gt_0.95": float(np.mean(q_best > P["best_q"])),
           "share_best_q_gt_0.95_homogeneous": float(np.mean(q_best_homogeneous > P["best_q"])),
           "share_logger_q_lt_0.001": float(np.mean(q_logger < P["low_q"]))}
    out["pathological"] = bool(
        out["share_best_q_gt_0.95"] > out["share_best_q_gt_0.95_homogeneous"] + P["best_q_excess"]
        or out["share_best_q_gt_0.95"] > P["best_q_max"] or out["share_logger_q_lt_0.001"] > P["low_q_share"])
    return out


def _quantiles(v, w=None) -> dict:
    v = np.asarray(v, np.float64)
    if w is None:
        return {f"q{int(q * 100)}": float(np.quantile(v, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)} | {
            "mean": float(v.mean()), "sd": float(v.std())}
    w = np.asarray(w, np.float64) / np.sum(w)
    order = np.argsort(v)
    cw = np.cumsum(w[order])
    q = {f"q{int(p * 100)}": float(v[order][np.searchsorted(cw, p)]) for p in (0.01, 0.1, 0.5, 0.9, 0.99)}
    m = float(w @ v)
    return q | {"mean": m, "sd": float(np.sqrt(w @ (v - m) ** 2))}


# --------------------------------------------------------------------------------------------------------- world
def build_structured_world(emb_x, emb_a, label: str, *, seed: int, config=None, logging_uniform_mix: float = 0.0,
                           logger_greedy_share=0.8, z_clip: float = Z_CLIP) -> dict:
    """The dataset dict of one structured-shift world (the trainers' keys) with ``dataset['structured']`` (the truth's
    pieces, for the population tools and tests) and a JSON-able ``world`` record (§1-§3)."""
    from utils.representation_bias import (
        WorldConfig,
        _pair_scores,
        _sample_policy_items,
        _score_stats,
        calibrate_world,
        parse_logger_greedy_share,
        sharpen_logger,
    )
    from utils.simulation_utils import SyntheticBanditEnv

    levels = parse_structured(label)
    config = dataclasses.replace(config or WorldConfig(), reference_bias=("none", "none", "none"), pop_strength=0.0)
    share = parse_logger_greedy_share(logger_greedy_share)
    cal = calibrate_world(emb_x, emb_a, seed=seed, config=config)  # prior, spread temperature, calibration users
    X = cal["clean_x"].astype(np.float64)
    A = cal["clean_a"].astype(np.float64)
    nU, nA, K = X.shape[0], A.shape[0], X.shape[1]
    prior = cal["user_prior"].astype(np.float64)
    T = float(cal["logging_temperature"])

    # the calibration users and the reference policy's items, drawn exactly as the legacy calibration draws them
    crng = np.random.default_rng(derive_seed(seed, "world", "ctr"))
    cu = crng.choice(nU, size=min(config.ctr_users, nU), replace=True, p=prior / prior.sum())
    assert np.array_equal(cu, cal["_ctr_users"])
    ref_items = _sample_policy_items(X, A, cu, T, int(config.ctr_samples_per_user), crng)

    # the decision-relevant shift (§2)
    directions = shift_directions(X, A, prior, seed)
    gate = gate_values(X, prior, seed) if levels["gated"] else None
    gamma, rho = calibrate_gamma(X, A, directions, cu, SHIFT_LEVELS[levels["shift"]], gate)
    delta = shift_matrix(directions, gamma)
    Y = target_user_vectors(X, delta, gate)
    mu_B, sd_B = _score_stats(Y, A)
    agreement = score_agreement(X, Y, A, cu)

    # response heterogeneity (§3) and click calibration
    resp = response_factors(X, prior, seed, levels["response"], z_clip)
    alpha, beta = resp["alpha"], resp["beta"]
    zs = (_pair_scores(Y, A, cu, ref_items) - mu_B) / sd_B
    zmax = np.empty(len(cu))
    for s in range(0, len(cu), 128):
        zmax[s:s + 128] = ((Y[cu[s:s + 128]] @ A.T).max(axis=1) - mu_B) / sd_B
    # §13.1: κ and c of the homogeneous world (the legacy targets); with heterogeneity κ stays and c restores the
    # reference CTR, the best item's CTR becoming an outcome
    kappa, c0 = calibrate_click(zs, zmax, np.ones(len(cu)), np.zeros(len(cu)), float(config.target_ctr),
                                float(config.best_ctr))
    c = c0 if levels["response"] == "none" else _solve_c(kappa * alpha[cu][:, None] * zs + beta[cu][:, None],
                                                          float(config.target_ctr))
    q_best_homogeneous = expit(kappa * zmax + c0)

    ex = np.concatenate([(kappa * alpha / sd_B)[:, None] * Y, (c + beta - kappa * alpha * mu_B / sd_B)[:, None]], axis=1)
    ea = np.concatenate([A, np.ones((nA, 1))], axis=1)
    env = SyntheticBanditEnv(emb_x=ex.astype(np.float32), emb_a=ea.astype(np.float32), scale=1.0, offset=0.0,
                             ctr=float(config.target_ctr))
    our_x, our_a = X.astype(np.float32), A.astype(np.float32)

    # the logger: softmax over the source scores, sharpened to its greedy share under q_B (the legacy procedure)
    mix = float(np.clip(logging_uniform_mix, 0.0, 1.0))
    sharp = sharpen_logger(our_x, our_a, cu, T, env, share)
    T_log = sharp["temperature"] if share > 0.0 else T
    lrng = np.random.default_rng(derive_seed(seed, "world", "logging_ctr"))
    items = _sample_policy_items(X, A, cu, T_log, int(config.ctr_samples_per_user), lrng)
    q_log_rows = env.reward_prob(np.repeat(cu, items.shape[1]), items.reshape(-1)).reshape(items.shape)
    logging_ctr = float(q_log_rows.mean())
    q_best = np.empty(len(cu))
    src_pick, tgt_pick = np.empty(len(cu), np.int64), np.empty(len(cu), np.int64)
    for s in range(0, len(cu), 128):
        u = cu[s:s + 128]
        q = env.reward_prob_block(u, 0, nA)
        q_best[s:s + 128] = q.max(axis=1)
        src_pick[s:s + 128] = (X[u] @ A.T).argmax(axis=1)
        tgt_pick[s:s + 128] = (Y[u] @ A.T).argmax(axis=1)
    q_src = env.reward_prob(cu, src_pick).astype(np.float64)
    q_tgt = env.reward_prob(cu, tgt_pick).astype(np.float64)
    sanity = click_sanity(q_best, q_log_rows.mean(axis=1), q_best_homogeneous)
    if sanity["pathological"]:
        raise StructuredCalibrationError(f"{label}: pathological click distribution {sanity} (§13.1; the study stops)")

    sv = np.linalg.svd(delta, compute_uv=False)[:SHIFT_RANK]
    world = copy.deepcopy({k: v for k, v in cal.items()
                           if k in ("config", "user_prior_kind", "logging_temperature", "clean_logger_effective_items",
                                    "uniform_ctr")})
    world.pop("logging_temperature", None)
    world.update(
        family=FAMILY, structured=levels, structured_label=label, bias=levels, bias_label=label, rank=SHIFT_RANK,
        gamma=float(gamma), shift_corr_target=SHIFT_LEVELS[levels["shift"]], shift_corr=float(rho),
        shift_singular_values=[float(v) for v in sv], shift_frobenius=float(np.linalg.norm(delta)),
        **agreement,
        source_greedy_ctr=float(q_src.mean()), target_greedy_ctr=float(q_tgt.mean()),
        available_gain=float(q_tgt.mean() - q_src.mean()),
        sigma_alpha=resp["sigma_alpha"], sigma_beta=resp["sigma_beta"], z_clip=float(z_clip),
        log_alpha=_quantiles(np.log(alpha), prior), alpha_stats=_quantiles(alpha, prior), beta_stats=_quantiles(beta, prior),
        kappa=float(kappa), click_intercept=float(c), click_intercept_homogeneous=float(c0),
        best_item_ctr_homogeneous=float(q_best_homogeneous.mean()), score_mean_B=float(mu_B), score_sd_B=float(sd_B),
        reference_ctr=float(np.mean(expit(kappa * alpha[cu][:, None] * zs + c + beta[cu][:, None]))),
        best_item_ctr=float(q_best.mean()), best_item_q=_quantiles(q_best), user_logger_q=_quantiles(q_log_rows.mean(axis=1)),
        click_sanity=sanity,
        logging_ctr=(1.0 - mix) * logging_ctr + mix * float(cal["uniform_ctr"]), logging_uniform_mix=mix,
        logging_temperature=float(T_log), spread_temperature=T, logger_greedy_share=share,
        logger_sharpness=T / float(T_log), logger_greedy_ctr=sharp["greedy_ctr"], spread_logger_ctr=sharp["spread_ctr"],
        logger_softmax_ctr=sharp["softmax_ctr"], logger_share_achieved=sharp["share_achieved"],
        logger_effective_items=sharp["effective_items"], gated=bool(levels["gated"]),
        gate_stats=None if gate is None else _quantiles(gate, prior),
        pop_strength=0.0, logger_pop_strength=0.0, popularity={"item_bias": False, "pop_strength": 0.0},
    )
    structured = {"label": label, "levels": levels, "seed": int(seed), "rank": SHIFT_RANK, "directions": directions,
                  "gamma": float(gamma), "delta": delta, "gate": gate, "alpha": alpha, "beta": beta,
                  "mu_B": float(mu_B), "sd_B": float(sd_B), "kappa": float(kappa), "c": float(c), "z_clip": float(z_clip)}
    return {
        "emb_x": env.emb_x, "emb_a": env.emb_a, "our_x": our_x, "our_a": our_a,
        "original_x": our_x.copy(), "original_a": our_a.copy(),
        "n_users": int(nU), "n_actions": int(nA), "emb_dim": int(K), "pop_column": False, "item_popularity": None,
        "pop_strength": 0.0, "logger_pop_strength": 0.0, "env": env, "user_prior": cal["user_prior"].copy(),
        "policy_temperature": float(T_log), "logging_uniform_mix": mix, "world": world, "structured": structured,
    }


def describe_structured_world(world: dict) -> str:
    return (f"[world] structured {world['structured_label']}: rank {world['rank']}, γ={world['gamma']:.4g}, score corr "
            f"{world['shift_corr']:.3f}, top-1 agree {world['top1_agreement']:.2f}, gain {world['available_gain']:.2%} | "
            f"sd log α {world['log_alpha']['sd']:.2f}, sd β {world['beta_stats']['sd']:.2f}, κ={world['kappa']:.3g}, "
            f"c={world['click_intercept']:.3g} | logger T={world['logging_temperature']:.4g} CTR {world['logging_ctr']:.2%} | "
            f"reference {world['reference_ctr']:.2%}, best item {world['best_item_ctr']:.1%}, target greedy "
            f"{world['target_greedy_ctr']:.2%} | sanity {'PATHOLOGICAL' if world['click_sanity']['pathological'] else 'ok'}")
