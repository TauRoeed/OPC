"""Simulated world: clean vectors (BPR taste, optionally a popularity term), representation
bias, logging temperature and a logistic click model, all calibrated per dataset (and per seed).

Build order for one condition (``build_world``):
  1. Clean vectors: BPR's user / item factors, optionally centered (``x - lambda * mean(x)``,
     ``WorldConfig.centering``, off by default). With BPR's item bias b_i and a popularity weight
     beta_true (``pop_strength``, default 0 = taste only) the clean score is x·a + beta_true·b_i,
     carried as one extra column: users [x, 1], items [a, beta_true·b_i]. The clean vectors define
     the true click model.
  2. Draw three representation-bias types per side (users, items) from the seed, applied to the
     taste part in this order, each scale-matched to the clean vectors:
       - warp:   one shared random linear map for the whole side (``x -> x @ W``)
       - group:  one Gaussian offset per group (k-means cluster, or metadata group)
       - vector: one Gaussian offset per user / item
     A type at level L mixes ``(1 - eps) * current + eps * bias`` with eps calibrated so that
     ALL THREE types at level L together keep ``LEVEL_SIGNAL_KEPT[L]`` of the taste signal
     (per-user correlation between biased and clean scores across items). The logger adds its
     own popularity term beta_log·b_i (``logger_pop_strength``, default beta_true).
  3. Logging temperature T: the clean softmax logger spreads over ``logging_spread`` of the
     catalog (effective number of items = exp(entropy)).
  4. True click model ``q(u, a) = sigmoid(alpha * z(u, a) + b)``, z = standardized clean score:
     alpha so the best item per user averages ``best_ctr``; b so the reference policy (the
     spread-temperature logger at ``reference_bias`` levels, or the uniform policy) has
     ``target_ctr``.
  5. Logger sharpness (per condition): the biased logger's temperature is lowered from T until
     its CTR is ``logger_greedy_share`` (default 0.8) of its own greedy CTR, so it mostly
     exploits its ranking and explores near the top. ``off`` keeps T (the logger before
     2026-09-26, spread over half the catalog). The click model of step 4 does not change.
Draws depend only on the seed, so bias levels are nested and the truth (alpha, b, T) is
identical across bias configurations for a given dataset and seed; the sharpened logger's
temperature depends on the bias configuration. Calibration runs in float64 numpy (same result
with or without a GPU) on user / item samples.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import check_random_state

from utils.seeding import derive_seed

BIAS_TYPES = ("warp", "group", "vector")  # applied in this order
BIAS_LEVELS = ("none", "low", "medium", "high")
LEVEL_SIGNAL_KEPT = {"none": 1.0, "low": 0.90, "medium": 0.75, "high": 0.50}
GROUP_SOURCES = ("cluster", "metadata")
CTR_REFERENCES = ("logger", "uniform")
DEFAULT_LOGGER_GREEDY_SHARE = 0.8  # the logger earns this share of its own greedy CTR (0 = off)


def parse_logger_greedy_share(value) -> float:
    """``off`` / ``none`` / 0 / None -> 0.0 (the spread-temperature logger); else a share in (0, 1)."""
    if value is None:
        return 0.0
    text = str(value).strip().lower()
    if text in ("off", "none"):
        return 0.0
    share = float(text)
    if share == 0.0:
        return 0.0
    if not 0.0 < share < 1.0:
        raise ValueError(f"logger greedy share must be in (0, 1) or off, got {value!r}")
    return share


class WorldCalibrationError(RuntimeError):
    """The requested targets cannot be met for this dataset (see the message for values)."""


@dataclass(frozen=True)
class WorldConfig:
    centering: float = 0.0            # lambda: share of the mean vector removed (0 = off)
    pop_strength: float = 0.0         # beta_true: weight of the BPR item bias in the true score (0 = taste only)
    logging_spread: float = 0.5       # clean logger's effective items / catalog size
    target_ctr: float = 0.05          # CTR of the reference policy
    ctr_reference: str = "logger"     # "logger" (at reference_bias) or "uniform" (random policy)
    reference_bias: tuple = ("medium", "medium", "medium")  # warp / group / vector
    best_ctr: float = 0.30            # best item per user, averaged over users
    group_source: str = "cluster"     # "cluster" or "metadata" (falls back to cluster per side)
    calib_rows: int = 50_000          # rows per side used to calibrate bias levels (signal kept)
    temp_users: int = 300             # users for the logging-temperature calibration
    ctr_users: int = 1000             # users (drawn from the user prior) for CTR calibration
    ctr_samples_per_user: int = 2048
    strict: bool = True               # raise WorldCalibrationError when a target is missed

    def validate(self) -> None:
        if not 0.0 <= self.centering <= 1.0:
            raise ValueError(f"centering must be in [0, 1], got {self.centering}")
        if not (np.isfinite(self.pop_strength) and self.pop_strength >= 0.0):
            raise ValueError(f"pop_strength must be >= 0, got {self.pop_strength}")
        if not 0.0 < self.logging_spread <= 1.0:
            raise ValueError(f"logging_spread must be in (0, 1], got {self.logging_spread}")
        if not 0.0 < self.target_ctr < self.best_ctr < 1.0:
            raise ValueError(f"need 0 < target_ctr < best_ctr < 1, got {self.target_ctr}, {self.best_ctr}")
        if self.ctr_reference not in CTR_REFERENCES:
            raise ValueError(f"ctr_reference must be one of {CTR_REFERENCES}")
        if self.group_source not in GROUP_SOURCES:
            raise ValueError(f"group_source must be one of {GROUP_SOURCES}")
        parse_bias(self.reference_bias)


# --------------------------------------------------------------------------- bias configs
def parse_bias(spec) -> dict[str, str]:
    """Bias levels per type from 'medium' (all three), 'high/medium/low' (warp/group/vector),
    a label from ``bias_label`` ('w-high.g-medium.v-low'), a (warp, group, vector) tuple, or a
    {type: level} dict (missing types = none)."""
    if isinstance(spec, dict):
        levels = {k: str(spec.get(k, "none")).lower() for k in BIAS_TYPES}
        unknown = set(spec) - set(BIAS_TYPES)
        if unknown:
            raise ValueError(f"unknown bias types {sorted(unknown)}; expected {BIAS_TYPES}")
    elif isinstance(spec, str) and "-" in spec:  # a bias_label such as 'w-high.g-none.v-low'
        short = {k[0]: k for k in BIAS_TYPES}
        levels = {k: "none" for k in BIAS_TYPES}
        for part in spec.strip().lower().split("."):
            key, _, lvl = part.partition("-")
            if key not in short:
                raise ValueError(f"bias label {spec!r}: unknown type {key!r}")
            levels[short[key]] = lvl
    else:
        parts = list(spec) if isinstance(spec, (tuple, list)) else str(spec).strip().lower().split("/")
        parts = [str(p).strip().lower() for p in parts]
        if len(parts) == 1:
            parts = parts * 3
        if len(parts) != 3:
            raise ValueError(f"bias spec {spec!r}: use a level or 'warp/group/vector' levels")
        levels = dict(zip(BIAS_TYPES, parts))
    for k, v in levels.items():
        if v not in BIAS_LEVELS:
            raise ValueError(f"bias level {v!r} for {k}: expected one of {BIAS_LEVELS}")
    return levels


def bias_label(spec) -> str:
    """Folder / column label: 'medium' when all types share a level, else 'w-high.g-none.v-low'."""
    levels = parse_bias(spec)
    vals = [levels[k] for k in BIAS_TYPES]
    if len(set(vals)) == 1:
        return vals[0]
    return ".".join(f"{k[0]}-{v}" for k, v in zip(BIAS_TYPES, vals))


# --------------------------------------------------------------------------- bias draws
def _rms(M: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(M))))


def _group_ids(features: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    n = int(features.shape[0])
    k = int(np.clip(round(np.sqrt(n)), 8, 64))
    k = max(1, min(k, n))
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=int(seed) % (2**31 - 1),
        n_init=3,
        batch_size=min(10_000, max(256, n)),
        reassignment_ratio=0.0,
    )
    ids = km.fit_predict(np.asarray(features, dtype=np.float64)).astype(np.int64)
    return ids, k


def _metadata_features(meta: np.ndarray | None, n_rows: int) -> np.ndarray | None:
    if meta is None:
        return None
    M = np.asarray(meta, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != n_rows or M.shape[1] == 0:
        return None
    sd = M.std(axis=0)
    keep = sd > 1e-12
    if not keep.any():
        return None
    M = M[:, keep]
    return (M - M.mean(axis=0)) / sd[keep]


class SideBias:
    """Clean vectors of one side plus their seed-fixed bias draws.

    Bias targets, each rescaled to the RMS of the clean rows: warp ``x @ W`` (of the clean
    vector), group ``template[group(x)]``, vector ``noise(x)``.
    """

    def __init__(self, X: np.ndarray, *, side: str, seed: int, group_features=None):
        self.X = np.asarray(X, dtype=np.float64)
        n, d = self.X.shape
        rng = lambda kind: np.random.default_rng(derive_seed(seed, "bias", side, kind))
        self.W = rng("warp").standard_normal((d, d))
        feats = self.X if group_features is None else group_features
        self.groups, self.n_groups = _group_ids(feats, derive_seed(seed, "bias", side, "groups"))
        self.group_source = "cluster" if group_features is None else "metadata"
        self.templates = rng("group").standard_normal((self.n_groups, d))
        self.per_vector = rng("vector").standard_normal((n, d), dtype=np.float32)
        self._targets = None  # scaled targets, kept only for calibration subsets

    def subset(self, rows) -> "SideBias":
        """Same draws restricted to ``rows``, with the scaled targets precomputed (calibration)."""
        sub = object.__new__(SideBias)
        sub.X = self.X[rows]
        sub.W, sub.templates, sub.n_groups, sub.group_source = self.W, self.templates, self.n_groups, self.group_source
        sub.groups = self.groups[rows]
        sub.per_vector = self.per_vector[rows]
        s = _rms(sub.X)
        sub._targets = {k: sub._target(k, s) for k in BIAS_TYPES}
        return sub

    def _target(self, kind: str, s: float) -> np.ndarray:
        if kind == "warp":
            t = self.X @ self.W
        elif kind == "group":
            t = self.templates[self.groups]
        else:
            t = self.per_vector.astype(np.float64)
        return t * (s / _rms(t))

    def biased(self, eps: dict[str, float]) -> np.ndarray:
        """Sequential warp -> group -> vector mixes, each rescaled to the clean RMS."""
        s = _rms(self.X)
        out = self.X
        for kind in BIAS_TYPES:
            e = float(eps.get(kind, 0.0))
            if e <= 0.0:
                continue
            target = self._targets[kind] if self._targets is not None else self._target(kind, s)
            out = (1.0 - e) * out + e * target
            out = out * (s / _rms(out))
        return out


class SignalKept:
    """Mean over users of corr_items(biased scores, clean scores), on row samples of each side.

    Centering a user's scores across items equals centering the item vectors, so with
    Ac = A - mean(A) the per-user covariance is ``bx_u' (Bc' Ac) x_u`` and the variances are
    ``bx_u' (Bc' Bc) bx_u`` and ``x_u' (Ac' Ac) x_u``: exact, and O(rows * d^2) per call
    instead of O(users * items * d).
    """

    def __init__(self, X: np.ndarray, A: np.ndarray):
        self.X = X
        self.Ac = A - A.mean(axis=0)
        self.ref_var = np.sum((X @ (self.Ac.T @ self.Ac)) * X, axis=1)

    def __call__(self, bx: np.ndarray, ba: np.ndarray) -> float:
        bc = ba - ba.mean(axis=0)
        cov = np.sum((bx @ (bc.T @ self.Ac)) * self.X, axis=1)
        var = np.sum((bx @ (bc.T @ bc)) * bx, axis=1)
        den = np.sqrt(np.maximum(var, 0.0) * np.maximum(self.ref_var, 0.0))
        return float(np.mean(cov / np.maximum(den, 1e-300)))


def with_popularity(X: np.ndarray, A: np.ndarray, item_bias, weight: float):
    """Append the popularity column: [x, 1] · [a, weight · b] = x · a + weight · b.

    Without an item bias the vectors are returned unchanged."""
    if item_bias is None:
        return X, A
    b = np.asarray(item_bias, dtype=np.float64).reshape(-1, 1)
    return (np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)]),
            np.hstack([A, (float(weight) * b).astype(A.dtype)]))


# --------------------------------------------------------------------------- calibration
BLOCK_CELLS = 4_000_000  # float64 cells per (users x items) block; changes results only by rounding


def _block_rows(n_cols: int, cap: int = 256) -> int:
    return int(max(1, min(cap, BLOCK_CELLS // max(int(n_cols), 1))))


def _effective_items_rows(scores: np.ndarray, T: float) -> np.ndarray:
    """exp(entropy) of softmax(scores / T), per row."""
    lg = scores.astype(np.float64) / T
    m = lg.max(axis=1, keepdims=True)
    e = np.exp(lg - m)
    Z = e.sum(axis=1, keepdims=True)
    p = e / Z
    return np.exp(np.log(Z[:, 0]) + m[:, 0] - (p * lg).sum(axis=1))


def _effective_items(scores: np.ndarray, T: float) -> float:
    vals = []
    chunk = _block_rows(scores.shape[1])
    for s in range(0, scores.shape[0], chunk):
        vals.append(_effective_items_rows(scores[s : s + chunk], T))
    return float(np.mean(np.concatenate(vals)))


SHARPEN_ROWS = 128  # users per block in sharpen_logger (fixed: rows are independent, results exact)


def sharpen_logger(bx: np.ndarray, ba: np.ndarray, users: np.ndarray, T: float, env, share: float) -> dict:
    """Logit factor f (logger temperature T / f) at which the softmax logger on (bx, ba) earns
    ``share`` of its own greedy CTR on ``users``, exact over the catalog (scores and sums in float64,
    click probabilities kept in float32 to halve the memory of large catalogs).

    The logger's CTR at factor f runs from the uniform policy's (f -> 0) to the greedy logger's
    (f -> inf). From f = 1 the search steps by sqrt(2) until the share is crossed, then brentq
    refines f inside that bracket. ``share = 0`` keeps f = 1 and only reports the logger.
    """
    users = np.asarray(users, dtype=np.int64)
    n_items = int(ba.shape[0])
    ba64 = ba.astype(np.float64)
    blocks = []
    for s in range(0, len(users), SHARPEN_ROWS):
        u = users[s : s + SHARPEN_ROWS]
        scores = (bx[u].astype(np.float64) @ ba64.T) / float(T)
        blocks.append((scores, env.reward_prob_block(u, 0, n_items).astype(np.float32)))
    greedy_rows = np.concatenate([q[np.arange(len(sc)), sc.argmax(axis=1)] for sc, q in blocks])
    greedy = float(greedy_rows.mean())
    if not greedy > 0.0:
        raise WorldCalibrationError(f"the greedy logger has CTR {greedy}: nothing to sharpen toward")

    def ctr(f: float) -> float:
        rows = []
        for scores, q in blocks:
            lg = scores * f
            lg -= lg.max(axis=1, keepdims=True)
            e = np.exp(lg)
            rows.append((e * q).sum(axis=1) / e.sum(axis=1))
        return float(np.concatenate(rows).mean())

    f = 1.0
    spread_ctr = ctr(1.0)
    if share > 0.0:
        gap = lambda log_f: (ctr(float(np.exp(log_f))) if log_f != 0.0 else spread_ctr) / greedy - share
        step, lo, hi = 0.5 * np.log(2.0), 0.0, 0.0
        g0 = gap(0.0)
        if g0 < 0.0:  # sharpen: raise f until the share is reached
            while gap(hi) < 0.0:
                lo, hi = hi, hi + step
                if hi > np.log(1e6):
                    raise WorldCalibrationError(
                        f"logger greedy share {share} unreachable: even logits x1e6 earn "
                        f"{ctr(1e6) / greedy:.4f} of the greedy CTR {greedy:.4f}")
        elif g0 > 0.0:  # the spread logger already earns more: flatten it
            while gap(lo) > 0.0:
                hi, lo = lo, lo - step
                if lo < np.log(1e-6):
                    raise WorldCalibrationError(
                        f"logger greedy share {share} unreachable: even logits x1e-6 earn "
                        f"{ctr(1e-6) / greedy:.4f} of the greedy CTR {greedy:.4f}")
        if g0 != 0.0:
            f = float(np.exp(brentq(gap, lo, hi, xtol=1e-13)))
    softmax_ctr = ctr(f)
    eff = np.concatenate([_effective_items_rows(sc, 1.0 / f) for sc, _ in blocks])
    return {
        "share": float(share),
        "factor": f,
        "temperature": float(T) / f,
        "greedy_ctr": greedy,
        "spread_ctr": spread_ctr,
        "softmax_ctr": softmax_ctr,
        "share_achieved": softmax_ctr / greedy,
        "effective_items": float(eff.mean()),
        "n_users": int(len(users)),
    }


def _solve_b(zs: np.ndarray, alpha: float, target: float, b0: float | None = None) -> float:
    """b with mean(sigmoid(alpha * zs + b)) = target (Newton with a bisection fallback)."""
    reach = float(alpha) * float(np.max(np.abs(zs))) + 60.0  # sigmoid(+-60) saturates: root bracketed
    lo, hi = -reach, reach
    b = float(np.log(target / (1 - target)) - alpha * float(np.mean(zs))) if b0 is None else float(b0)
    b = min(max(b, lo), hi)
    for _ in range(200):
        q = expit(alpha * zs + b)
        f = float(q.mean()) - target
        if abs(f) < 1e-12:
            return b
        if f > 0:
            hi = b
        else:
            lo = b
        d = float((q * (1 - q)).mean())
        nb = b - f / d if d > 1e-15 else (lo + hi) / 2
        b = nb if lo < nb < hi else (lo + hi) / 2
    return b


def _sample_policy_items(bx: np.ndarray, ba: np.ndarray, users: np.ndarray, T: float, k: int, rng) -> np.ndarray:
    """k items per user from softmax(bx[u] @ ba.T / T) (inverse CDF, float64)."""
    out = np.empty((len(users), k), dtype=np.int64)
    step = _block_rows(ba.shape[0], cap=64)
    for s in range(0, len(users), step):
        u = users[s : s + step]
        lg = (bx[u] @ ba.T) / T
        lg -= lg.max(axis=1, keepdims=True)
        p = np.exp(lg)
        cdf = np.cumsum(p, axis=1)
        cdf /= cdf[:, -1:]
        r = rng.random((len(u), k))
        idx = np.empty((len(u), k), dtype=np.int64)
        for i in range(len(u)):
            idx[i] = np.searchsorted(cdf[i], r[i], side="right")
        out[s : s + len(u)] = np.minimum(idx, ba.shape[0] - 1)
    return out


def _pair_scores(X: np.ndarray, A: np.ndarray, users: np.ndarray, items: np.ndarray) -> np.ndarray:
    """x_u · a_i for users[j] and each items[j, :] (chunked to bound memory)."""
    out = np.empty(items.shape, dtype=np.float64)
    for s in range(0, len(users), 64):
        out[s : s + 64] = np.einsum("ud,ukd->uk", X[users[s : s + 64]], A[items[s : s + 64]])
    return out


def _score_stats(X: np.ndarray, A: np.ndarray) -> tuple[float, float]:
    """Exact mean and sd of x·a over all user-item pairs."""
    mean = float(X.mean(axis=0) @ A.mean(axis=0))
    second = float(np.sum((X.T @ X / X.shape[0]) * (A.T @ A / A.shape[0])))
    return mean, float(np.sqrt(max(second - mean**2, 1e-300)))


def _digest(*arrays) -> str:
    h = hashlib.sha1()
    for a in arrays:
        if a is None:
            h.update(b"none")
        else:
            a = np.ascontiguousarray(a)
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
    return h.hexdigest()[:16]


_CALIBRATION_CACHE: dict = {}


def calibrate_world(emb_x, emb_a, *, seed: int, config: WorldConfig, metadata_x=None, metadata_a=None,
                    item_bias=None) -> dict:
    """Everything that depends on (dataset, seed, config) but not on the run's bias levels.

    With ``item_bias`` (BPR's b), the clean score is x·a + pop_strength·b, carried as one extra
    column of the clean vectors. Representation bias and signal kept act on the taste part x·a.
    """
    config.validate()
    key = (_digest(emb_x, emb_a, metadata_x, metadata_a, item_bias), int(seed), tuple(sorted(asdict(config).items())))
    if key in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[key]

    EX = np.asarray(emb_x, dtype=np.float64)
    EA = np.asarray(emb_a, dtype=np.float64)
    X = EX - config.centering * EX.mean(axis=0)
    A = EA - config.centering * EA.mean(axis=0)
    nU, nA = X.shape[0], A.shape[0]
    item_b = None
    if item_bias is not None:
        item_b = np.asarray(item_bias, dtype=np.float64).reshape(-1)
        if item_b.shape[0] != nA:
            raise ValueError(f"item_bias has {item_b.shape[0]} entries for {nA} items")
    Xp, Ap = with_popularity(X, A, item_b, config.pop_strength)  # clean vectors of the truth
    user_prior = check_random_state(seed).exponential(scale=1.0, size=(nU,)).astype(np.float32)
    user_prior = user_prior / user_prior.sum()

    fx = _metadata_features(metadata_x, nU) if config.group_source == "metadata" else None
    fa = _metadata_features(metadata_a, nA) if config.group_source == "metadata" else None
    if config.group_source == "metadata":
        for side, feats in (("users", fx), ("items", fa)):
            if feats is None:
                print(f"[world] no usable {side} metadata: {side} groups fall back to k-means clusters", flush=True)
    sides = {"users": SideBias(X, side="users", seed=seed, group_features=fx),
             "items": SideBias(A, side="items", seed=seed, group_features=fa)}

    rng = np.random.default_rng(derive_seed(seed, "world", "calibration"))
    pop_u = rng.permutation(nU)[: min(config.calib_rows, nU)]
    pop_a = rng.permutation(nA)[: min(config.calib_rows, nA)]
    signal_kept = SignalKept(X[pop_u], A[pop_a])
    sub_u, sub_a = sides["users"].subset(pop_u), sides["items"].subset(pop_a)

    def kept(eps):
        return signal_kept(sub_u.biased(eps), sub_a.biased(eps))

    grid = np.linspace(0.0, 1.0, 26)
    curves = {}
    for k in BIAS_TYPES:
        c = np.array([kept({k: e}) for e in grid])
        curves[k] = np.minimum.accumulate(c) - np.arange(len(c)) * 1e-12  # monotone for inversion

    def eps_for(k, kappa):
        return float(np.interp(-kappa, -curves[k], grid))

    eps_table = {k: {"none": 0.0} for k in BIAS_TYPES}
    kappas, achieved = {}, {"none": 1.0}
    for lvl in ("low", "medium", "high"):
        target = LEVEL_SIGNAL_KEPT[lvl]
        # per-type share kappa in [target, 1]: the combination keeps < target at kappa = target
        gap = lambda kap: kept({k: eps_for(k, kap) for k in BIAS_TYPES}) - target
        kappas[lvl] = float(brentq(gap, target, 1.0, xtol=1e-10)) if gap(target) < 0 else target
        for k in BIAS_TYPES:
            eps_table[k][lvl] = eps_for(k, kappas[lvl])
        achieved[lvl] = kept({k: eps_table[k][lvl] for k in BIAS_TYPES})
        if config.strict and abs(achieved[lvl] - target) > 0.02:
            raise WorldCalibrationError(
                f"bias level {lvl!r}: all three types keep {achieved[lvl]:.3f} of the signal, "
                f"target {target:.2f} (per-type reach at eps=1: "
                + ", ".join(f"{k} {curves[k][-1]:.2f}" for k in BIAS_TYPES) + ")"
            )

    # logging temperature from the clean logger's spread
    tu = pop_u[: min(config.temp_users, len(pop_u))]
    clean_scores = (Xp[tu] @ Ap.T).astype(np.float32)
    target_eff = config.logging_spread * nA
    if config.logging_spread >= 1.0:
        T = 1e6  # effectively uniform
    else:
        # effective items rise with T from ~1 (argmax) to nA (uniform)
        spread_gap = lambda lt: _effective_items(clean_scores, float(np.exp(lt))) - target_eff
        lo_t, hi_t = np.log(1e-6), np.log(1e6)
        if spread_gap(lo_t) > 0 or spread_gap(hi_t) < 0:
            raise WorldCalibrationError(
                f"logging spread {config.logging_spread} ({target_eff:.1f} effective items of {nA}) "
                "is outside what a softmax temperature in [1e-6, 1e6] can give"
            )
        T = float(np.exp(brentq(spread_gap, lo_t, hi_t, xtol=1e-10)))
    eff_clean = _effective_items(clean_scores, T)

    # click model: alpha (best item), b (reference policy CTR); users drawn from the prior
    s_mean, s_sd = _score_stats(Xp, Ap)
    crng = np.random.default_rng(derive_seed(seed, "world", "ctr"))
    p64 = user_prior.astype(np.float64)
    cu = crng.choice(nU, size=min(config.ctr_users, nU), replace=True, p=p64 / p64.sum())
    zmax = np.empty(len(cu))
    step = _block_rows(nA)
    for s in range(0, len(cu), step):
        zmax[s : s + step] = ((Xp[cu[s : s + step]] @ Ap.T).max(axis=1) - s_mean) / s_sd
    ref_eps = {k: eps_table[k][lvl] for k, lvl in parse_bias(config.reference_bias).items()}
    K = int(config.ctr_samples_per_user)
    if config.ctr_reference == "logger":
        # the reference logger weighs popularity like the truth does
        rbx, rba = with_popularity(sides["users"].biased(ref_eps), sides["items"].biased(ref_eps), item_b, config.pop_strength)
        items = _sample_policy_items(rbx, rba, cu, T, K, crng)
    else:
        items = crng.integers(0, nA, size=(len(cu), K))
    zs = (_pair_scores(Xp, Ap, cu, items) - s_mean) / s_sd
    target = float(config.target_ctr)
    warm = {"b": None}  # b from the previous alpha: Newton then needs a few steps

    def best_of(a):
        warm["b"] = _solve_b(zs, a, target, warm["b"])
        return float(expit(a * zmax + warm["b"]).mean())

    lo, hi = 1e-4, 60.0
    if best_of(hi) < config.best_ctr or best_of(lo) > config.best_ctr:
        best_lo, best_hi = best_of(lo), best_of(hi)
        msg = (f"best-item CTR {config.best_ctr:.0%} unreachable with reference CTR {target:.1%}: "
               f"achievable range {best_lo:.1%}..{best_hi:.1%}. The reference policy is too close to "
               f"the best items; widen --logging-spread or lower --best-ctr.")
        if config.strict:
            raise WorldCalibrationError(msg)
        alpha = hi if best_of(hi) < config.best_ctr else lo
    else:
        alpha = float(brentq(lambda a: best_of(a) - config.best_ctr, lo, hi, xtol=1e-12))
    b = _solve_b(zs, alpha, target)
    ref_ctr = float(expit(alpha * zs + b).mean())
    best = float(expit(alpha * zmax + b).mean())
    missed = [f"{name} {got:.6g} (target {want:.6g})" for name, got, want in (
        ("reference CTR", ref_ctr, target), ("best-item CTR", best, config.best_ctr),
        ("clean logger effective items", eff_clean, target_eff)) if abs(got - want) > 1e-4 * want]
    if missed and config.strict:
        raise WorldCalibrationError("world calibration missed: " + "; ".join(missed))
    if config.ctr_reference == "uniform":
        uniform_ctr = ref_ctr
    else:
        uz = (_pair_scores(Xp, Ap, cu, crng.integers(0, nA, size=(len(cu), K))) - s_mean) / s_sd
        uniform_ctr = float(expit(alpha * uz + b).mean())

    popularity = {"item_bias": item_b is not None, "pop_strength": float(config.pop_strength)}
    if item_b is not None:
        pop_scores = config.pop_strength * item_b
        popularity.update(
            item_bias_sd=float(np.std(item_b)),
            # share of a user's clean-score variation (across items) that the popularity term carries
            share_of_score_variation=float(np.var(pop_scores) / np.mean(np.var(clean_scores.astype(np.float64), axis=1))),
        )
    result = {
        "config": asdict(config) | {"reference_bias": list(config.reference_bias)},
        "sides": sides,
        "clean_x": Xp.astype(np.float32),
        "clean_a": Ap.astype(np.float32),
        "item_bias": None if item_b is None else item_b.astype(np.float32),
        "taste_dim": int(X.shape[1]),
        "popularity": popularity,
        "user_prior": user_prior,
        "eps_table": eps_table,
        "per_type_signal_kept": kappas,
        "level_signal_kept": achieved,
        "logging_temperature": T,
        "clean_logger_effective_items": eff_clean,
        "alpha": float(alpha),
        "b": float(b),
        "score_mean": s_mean,
        "score_sd": s_sd,
        "scale": float(alpha / s_sd),
        "offset": float(b - alpha * s_mean / s_sd),
        "reference_ctr": ref_ctr,
        "best_item_ctr": best,
        "uniform_ctr": uniform_ctr,
        "groups": {"users": sides["users"].group_source, "items": sides["items"].group_source,
                   "n_user_groups": sides["users"].n_groups, "n_item_groups": sides["items"].n_groups},
        "_kept_fn": kept,
        "_ctr_users": cu,
    }
    _CALIBRATION_CACHE.clear()  # one entry: conditions of a worker usually share dataset and seed
    _CALIBRATION_CACHE[key] = result
    return result


def build_world(emb_x, emb_a, bias, *, seed: int, config: WorldConfig | None = None,
                metadata_x=None, metadata_a=None, logging_uniform_mix: float = 0.0,
                item_bias=None, logger_pop_strength: float | None = None,
                logger_greedy_share=DEFAULT_LOGGER_GREEDY_SHARE) -> dict:
    """Dataset dict for one condition (same keys the trainers use) plus a JSON-able ``world`` record.

    ``logger_greedy_share`` (default 0.8; 0 / 'off' = the spread temperature T): the logger's
    temperature becomes T / f, f chosen so its CTR on the calibration users is that share of its
    own greedy CTR (``sharpen_logger``). ``policy_temperature`` and ``world['logging_temperature']``
    are the logger's temperature; ``world['spread_temperature']`` is T. A uniform mix
    (``logging_uniform_mix``) is applied on top of the sharpened softmax.

    When either popularity weight is positive (``config.pop_strength`` for the truth,
    ``logger_pop_strength`` for the logger, default: the same), ``item_bias`` is required and every
    vector carries a popularity column: users [x, 1], items [a, w·b], w being the truth's weight
    for the clean vectors and the logger's for the biased ones. With both weights 0 the item bias
    is ignored and the world is the taste-only one. ``emb_dim`` stays the taste dimension;
    ``item_popularity`` holds b for the models that learn their own popularity weight.
    """
    from utils.simulation_utils import SyntheticBanditEnv

    config = config or WorldConfig()
    config.validate()
    share = parse_logger_greedy_share(logger_greedy_share)
    beta_true = float(config.pop_strength)
    beta_log = beta_true if logger_pop_strength is None else float(logger_pop_strength)
    if not (np.isfinite(beta_log) and beta_log >= 0.0):
        raise ValueError(f"logger_pop_strength must be >= 0, got {beta_log}")
    if beta_true == 0.0 and beta_log == 0.0:
        item_bias = None  # popularity off: the taste-only world, no popularity column
    elif item_bias is None:
        raise ValueError(
            f"popularity weights (truth {beta_true:g}, logger {beta_log:g}) need BPR's item bias: "
            "{dataset}_item_bias.npy, written by BPR/generate_artifacts.py"
        )
    cal = calibrate_world(emb_x, emb_a, seed=seed, config=config, metadata_x=metadata_x, metadata_a=metadata_a,
                          item_bias=item_bias)
    levels = parse_bias(bias)
    eps = {k: cal["eps_table"][k][levels[k]] for k in BIAS_TYPES}
    b = cal["item_bias"]
    taste_x = cal["sides"]["users"].biased(eps)
    taste_a = cal["sides"]["items"].biased(eps)
    our_x, our_a = with_popularity(taste_x, taste_a, b, beta_log)
    our_x, our_a = our_x.astype(np.float32), our_a.astype(np.float32)
    X, A = cal["clean_x"].copy(), cal["clean_a"].copy()  # own copies: the calibration is cached
    d = cal["taste_dim"]
    env = SyntheticBanditEnv(emb_x=X, emb_a=A, scale=cal["scale"], offset=cal["offset"], ctr=float(config.target_ctr))

    # the logger of this bias configuration: sharpened toward its own greedy ranking
    mix = float(np.clip(logging_uniform_mix, 0.0, 1.0))
    cu = cal["_ctr_users"]
    sharp = sharpen_logger(our_x, our_a, cu, cal["logging_temperature"], env, share)
    T_log = sharp["temperature"] if share > 0.0 else float(cal["logging_temperature"])
    crng = np.random.default_rng(derive_seed(seed, "world", "logging_ctr"))
    items = _sample_policy_items(our_x.astype(np.float64), our_a.astype(np.float64), cu, T_log,
                                 int(config.ctr_samples_per_user), crng)
    logging_ctr_softmax = float(env.reward_prob(np.repeat(cu, items.shape[1]), items.reshape(-1)).mean())
    world = copy.deepcopy({k: v for k, v in cal.items()
                           if k not in ("sides", "clean_x", "clean_a", "user_prior", "item_bias") and not k.startswith("_")})
    Xt, At = X[:, :d], A[:, :d]  # taste parts
    world.update(
        bias=levels,
        bias_label=bias_label(levels),
        eps=eps,
        signal_kept=float(cal["_kept_fn"](eps)),
        pop_strength=beta_true,
        logger_pop_strength=beta_log,
        logging_ctr=(1.0 - mix) * logging_ctr_softmax + mix * cal["uniform_ctr"],
        logging_uniform_mix=mix,
        logging_temperature=T_log,
        spread_temperature=float(cal["logging_temperature"]),
        logger_greedy_share=share,
        logger_sharpness=float(cal["logging_temperature"]) / T_log,
        logger_greedy_ctr=sharp["greedy_ctr"],
        spread_logger_ctr=sharp["spread_ctr"],
        logger_softmax_ctr=sharp["softmax_ctr"],
        logger_share_achieved=sharp["share_achieved"],
        logger_effective_items=sharp["effective_items"],
        vector_rms={"users": {"clean": _rms(Xt), "biased": _rms(taste_x)}, "items": {"clean": _rms(At), "biased": _rms(taste_a)}},
        cosine_to_clean={"users": _mean_cosine(Xt, taste_x), "items": _mean_cosine(At, taste_a)},
    )
    return {
        "emb_x": X,
        "emb_a": A,
        "our_x": our_x,
        "our_a": our_a,
        "original_x": our_x.copy(),
        "original_a": our_a.copy(),
        "n_users": int(X.shape[0]),
        "n_actions": int(A.shape[0]),
        "emb_dim": int(d),
        "pop_column": b is not None,
        "item_popularity": None if b is None else b.copy(),
        "pop_strength": beta_true,
        "logger_pop_strength": beta_log,
        "env": env,
        "user_prior": cal["user_prior"].copy(),
        "policy_temperature": float(T_log),
        "logging_uniform_mix": mix,
        "world": world,
    }


def _mean_cosine(C: np.ndarray, B: np.ndarray) -> float:
    num = np.sum(C.astype(np.float64) * B, axis=1)
    den = np.linalg.norm(C, axis=1) * np.linalg.norm(B, axis=1)
    return float(np.mean(num / np.maximum(den, 1e-300)))


def describe_world(world: dict) -> str:
    """One-line summary of a built world (for logs)."""
    b = world["bias"]
    share = world.get("logger_greedy_share", 0.0)
    sharp = (f"x{world['logger_sharpness']:.3g}, {world['logger_share_achieved']:.0%} of its greedy CTR "
             f"{world['logger_greedy_ctr']:.1%}, {world['logger_effective_items']:.3g} effective items"
             if share else "spread temperature, sharpening off")
    return (
        f"[world] bias warp={b['warp']} group={b['group']} vector={b['vector']} "
        f"(signal kept {world['signal_kept']:.2f}) | logger T={world['logging_temperature']:.4g} "
        f"({sharp}) "
        f"CTR {world['logging_ctr']:.2%} | reference CTR {world['reference_ctr']:.2%} "
        f"({world['config']['ctr_reference']}), uniform {world['uniform_ctr']:.2%}, "
        f"best item {world['best_item_ctr']:.1%} | popularity: "
        + (f"truth {world['pop_strength']:g}, logger {world['logger_pop_strength']:g}"
           if world.get("popularity", {}).get("item_bias") else "off")
    )


# --------------------------------------------------------------------------- runner CLI
def add_world_arguments(parser, *, bias_default=("low", "medium", "high"), ctr_reference: bool = True):
    """World options shared by the study runners (``world_options_from_args`` reads them)."""
    d = WorldConfig()
    g = parser.add_argument_group("simulated world (utils/representation_bias.py)")
    g.add_argument(
        "--bias-configs",
        nargs="+",
        default=list(bias_default),
        help="Representation-bias configurations to sweep. Each entry is one level for all three "
        "types (none/low/medium/high) or warp/group/vector levels, e.g. high/none/low. "
        "All three at low/medium/high keep 90/75/50%% of the signal.",
    )
    g.add_argument(
        "--bias-groups",
        choices=list(GROUP_SOURCES),
        default=d.group_source,
        help="Groups for the group bias: k-means clusters of the clean vectors (default) or "
        "clusters of the metadata arrays (sides without metadata fall back to clusters).",
    )
    g.add_argument(
        "--env-centering",
        type=float,
        default=d.centering,
        help="Share of the mean vector removed from the clean BPR vectors (default %(default)s = off; "
        "kept for experiments that also remove the popularity-like direction all users share).",
    )
    g.add_argument(
        "--pop-strength",
        type=float,
        default=d.pop_strength,
        help="Weight of BPR's item bias in the true score, x.a + w.b (default %(default)s: clicks follow "
        "personal taste only; 1 = popularity as BPR learned it). Needs {dataset}_item_bias.npy.",
    )
    g.add_argument(
        "--logger-pop-strength",
        type=float,
        default=None,
        help="The logger's weight on the item bias (default: same as --pop-strength). Above "
        "--pop-strength, the logger over-exposes popular items.",
    )
    g.add_argument(
        "--logging-spread",
        type=float,
        default=d.logging_spread,
        help="Spread temperature T: the clean logger's effective number of items as a share of "
        "the catalog (default %(default)s). It calibrates the click model (the reference logger); "
        "the actual logger is then sharpened, see --logger-greedy-share.",
    )
    g.add_argument(
        "--logger-greedy-share",
        type=parse_logger_greedy_share,
        default=DEFAULT_LOGGER_GREEDY_SHARE,
        help="Logger sharpness: per condition, the logger's temperature is lowered from T until it "
        "earns this share of its own greedy CTR (default %(default)s: it mostly exploits its ranking). "
        "'off' keeps T (the logger before 2026-09-26, spread over half the catalog).",
    )
    g.add_argument(
        "--best-ctr",
        type=float,
        default=d.best_ctr,
        help="Click probability of each user's best item, averaged over users (default %(default)s).",
    )
    if ctr_reference:
        g.add_argument(
            "--ctr-reference",
            choices=list(CTR_REFERENCES),
            default=d.ctr_reference,
            help="Policy whose CTR is set to --ctr-levels: the logger at medium bias (default) "
            "or the uniform random policy.",
        )
    return g


def world_options_from_args(args) -> dict:
    opts = {
        "centering": float(args.env_centering),
        "logging_spread": float(args.logging_spread),
        "best_ctr": float(args.best_ctr),
        "group_source": str(args.bias_groups),
        "pop_strength": float(args.pop_strength),
        "logger_greedy_share": parse_logger_greedy_share(getattr(args, "logger_greedy_share", DEFAULT_LOGGER_GREEDY_SHARE)),
    }
    if getattr(args, "ctr_reference", None) is not None:
        opts["ctr_reference"] = str(args.ctr_reference)
    if getattr(args, "logger_pop_strength", None) is not None:
        opts["logger_pop_strength"] = float(args.logger_pop_strength)
    return opts


WORLD_RUN_KEY_TAGS = (  # world option -> run-key tag, added only when the option is not the default
    ("pop_strength", "pop"), ("logger_pop_strength", "logpop"), ("centering", "center"),
    ("logging_spread", "spread"), ("best_ctr", "best"), ("group_source", "groups"), ("ctr_reference", "ref"),
    ("logger_greedy_share", "lgs"),
)


def world_run_key_suffix(world_options: dict, defaults: dict | None = None) -> str:
    """'__pop=0.5__logpop=1' style suffix for the world options that differ from the defaults, so
    runs of different worlds never share (and skip) each other's condition folders."""
    base = (asdict(WorldConfig()) | {"logger_pop_strength": None, "logger_greedy_share": DEFAULT_LOGGER_GREEDY_SHARE}
            | dict(defaults or {}))
    opts = dict(world_options or {})
    if opts.get("logger_pop_strength") is not None and opts["logger_pop_strength"] == opts.get("pop_strength", base["pop_strength"]):
        opts.pop("logger_pop_strength")  # same as the truth: the default
    parts = []
    for key, tag in WORLD_RUN_KEY_TAGS:
        if key in opts and opts[key] is not None and opts[key] != base.get(key):
            val = opts[key]
            parts.append(f"__{tag}={val:g}" if isinstance(val, float) else f"__{tag}={val}")
    return "".join(parts)


def resolve_bias_configs(specs) -> list[str]:
    """Validated, de-duplicated bias labels in the given order."""
    out = []
    for spec in specs:
        label = bias_label(spec)
        if label not in out:
            out.append(label)
    if not out:
        raise ValueError("at least one bias configuration is required")
    return out

