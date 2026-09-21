#!/usr/bin/env python3
"""Offline DR selection-clip grid with logging_score q̂ (no fit, no training).

Simulates logged bandit data, builds policy shifts via emb mix, scores DR with
ci_low = mean − 1.96·SE under clip M, bootstrap-resamples the log, picks M by
mean Spearman(ci_low, true V) then regret.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.trainer_trials import AnalyticRewardModel, _build_regression_logged_split
from utils.noise_levels import resolve_noise_spec
from utils.policies import Policy
from utils.simulation_utils import calc_reward, ensure_exact_env_q_cache, generate_dataset


def _load_ml(emb_dir: Path):
    emb_x = np.load(emb_dir / "ml_user_factors.npy")
    emb_a = np.load(emb_dir / "ml_item_factors.npy")
    meta_x_p = emb_dir / "ml_user_metadata.npy"
    meta_a_p = emb_dir / "ml_item_metadata.npy"
    meta_x = np.load(meta_x_p) if meta_x_p.exists() else None
    meta_a = np.load(meta_a_p) if meta_a_p.exists() else None
    return emb_x, emb_a, meta_x, meta_a


def _build_dataset(
    emb_dir: Path,
    *,
    noise_axis: str,
    noise_component: str,
    noise_level: str,
    seed: int,
    ctr: float,
) -> dict:
    emb_x, emb_a, meta_x, meta_a = _load_ml(emb_dir)
    spec = resolve_noise_spec(noise_level, axis=noise_axis, component=noise_component)
    params = {
        "n_users": int(emb_x.shape[0]),
        "n_actions": int(emb_a.shape[0]),
        "emb_dim": int(emb_x.shape[1]),
        "n_clusters": max(8, min(64, int(np.sqrt(emb_a.shape[0])))),
        "eps1": float(spec["eps1"]),
        "eps2": float(spec["eps2"]),
        "eps_meta": float(spec["eps_meta"]),
        "sigma1": 1.0,
        "sigma2": 1.0,
        "sigma_meta": 1.0,
        "noise_mode": "kmeans_templates",
        "noise_apply_user": bool(spec["apply_user"]),
        "noise_apply_item": bool(spec["apply_item"]),
        "ctr": float(ctr),
        "policy_temperature": 1.0,
        "logging_uniform_mix": 0.0,
    }
    return generate_dataset(
        params,
        seed=seed,
        emb_a=emb_a,
        emb_x=emb_x,
        metadata_a=meta_a,
        metadata_x=meta_x,
        store_original=True,
    )


def _policy_probs_at_actions(policy: Policy, users: np.ndarray, actions: np.ndarray) -> np.ndarray:
    users = np.asarray(users, dtype=np.int64)
    actions = np.asarray(actions, dtype=np.int64)
    out = np.empty(users.shape[0], dtype=np.float64)
    chunk = 2048
    for s in range(0, len(users), chunk):
        e = min(s + chunk, len(users))
        u = users[s:e]
        a = actions[s:e]
        uniq, inv = np.unique(u, return_inverse=True)
        probs = policy._probs_block(uniq)
        out[s:e] = probs[inv, a]
    return out


def _eval_policy_rows(policy: Policy, users: np.ndarray) -> np.ndarray:
    users = np.asarray(users, dtype=np.int64)
    chunk = 2048
    rows = []
    for s in range(0, len(users), chunk):
        e = min(s + chunk, len(users))
        uniq, inv = np.unique(users[s:e], return_inverse=True)
        probs = policy._probs_block(uniq)
        rows.append(probs[inv])
    return np.concatenate(rows, axis=0)


def _make_candidate_policy(dataset: dict, *, alpha: float, seed: int) -> Policy:
    rng = np.random.default_rng(seed)
    our_x = np.asarray(dataset["our_x"], dtype=np.float64)
    our_a = np.asarray(dataset["our_a"], dtype=np.float64)
    emb_x = np.asarray(dataset["emb_x"], dtype=np.float64)
    emb_a = np.asarray(dataset["emb_a"], dtype=np.float64)
    ux = (1.0 - alpha) * our_x + alpha * emb_x + 0.02 * rng.standard_normal(our_x.shape)
    ua = (1.0 - alpha) * our_a + alpha * emb_a + 0.02 * rng.standard_normal(our_a.shape)
    return Policy(
        n_users=int(dataset["n_users"]),
        n_items=int(dataset["n_actions"]),
        user_emb=ux.astype(np.float32),
        item_emb=ua.astype(np.float32),
        emb_dim=int(ux.shape[1]),
        temperature=1.0,
        user_chunk=4096,
        action_chunk=4096,
        rng=np.random.default_rng(seed + 7),
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def _metrics_for_clip(
    *,
    rewards: np.ndarray,
    q_f: np.ndarray,
    q_pis: list[np.ndarray],
    raw_iws: list[np.ndarray],
    true_vs: np.ndarray,
    idx: np.ndarray,
    clip_m: float,
) -> tuple[float, float, float, float, float]:
    """Return spearman_ci, spearman_r, regret_ci, regret_r, mse_r on row subset idx."""
    r_hats = []
    ci_lows = []
    n = int(len(idx))
    for iw, q_pi in zip(raw_iws, q_pis):
        w = np.asarray(iw[idx], dtype=np.float64)
        if np.isfinite(clip_m):
            w = np.minimum(w, float(clip_m))
        vec = q_pi[idx] + w * (rewards[idx] - q_f[idx])
        mean = float(np.mean(vec))
        se = float(np.std(vec, ddof=1) / np.sqrt(max(n, 2)))
        r_hats.append(mean)
        ci_lows.append(mean - 1.96 * se)
    r_hats = np.asarray(r_hats, dtype=np.float64)
    ci_lows = np.asarray(ci_lows, dtype=np.float64)
    oracle = int(np.argmax(true_vs))
    pick_r = int(np.argmax(r_hats))
    pick_ci = int(np.argmax(ci_lows))
    return (
        _spearman(ci_lows, true_vs),
        _spearman(r_hats, true_vs),
        float(true_vs[oracle] - true_vs[pick_ci]),
        float(true_vs[oracle] - true_vs[pick_r]),
        float(np.mean((r_hats - true_vs) ** 2)),
    )


def run_cell(
    dataset: dict,
    *,
    n_log: int,
    seed: int,
    clips: list[float],
    n_policies: int,
    n_boot: int,
    cell_name: str,
) -> pd.DataFrame:
    our_x = dataset["our_x"]
    our_a = dataset["our_a"]
    ctr = float(dataset["env"].ctr)
    pt = float(dataset.get("policy_temperature", 1.0))

    # Logged split only — no regression fit / no CF train.
    split = _build_regression_logged_split(
        dataset,
        our_x,
        our_a,
        int(n_log),
        int(max(5_000, n_log // 5)),
        0,
        regression_size=1,  # unused; logging_score needs no reg data
    )
    log_data = split["train_data"]

    qhat = AnalyticRewardModel(
        our_a, ctr=ctr, temperature=pt, kind="logging_score"
    )

    users = np.asarray(log_data["x_idx"], dtype=np.int64)
    actions = np.asarray(log_data["a"], dtype=np.int64)
    rewards = np.asarray(log_data["r"], dtype=np.float64)
    pscore = np.asarray(log_data["pscore"], dtype=np.float64).reshape(-1)
    contexts = np.asarray(log_data["x"], dtype=np.float32)
    q_f = np.asarray(qhat.predict_pairs(contexts, actions), dtype=np.float64)

    ensure_exact_env_q_cache(dataset)
    # q̂(x,·) once per unique logged user (logging_score, closed form).
    uniq_u, inv_u = np.unique(users, return_inverse=True)
    _, first_pos = np.unique(users, return_index=True)
    uniq_ctx = contexts[first_pos]
    q_uniq = np.asarray(qhat.predict(uniq_ctx), dtype=np.float64)
    if q_uniq.ndim == 3:
        q_uniq = q_uniq.squeeze(-1)
    _ = uniq_u

    alphas = np.linspace(0.0, 1.0, int(n_policies))
    true_vs = []
    raw_iws = []
    q_pis = []
    for i, alpha in enumerate(alphas):
        pol = _make_candidate_policy(dataset, alpha=float(alpha), seed=seed * 1000 + i)
        true_vs.append(float(calc_reward(dataset, pol)))
        print(f"  policy {i+1}/{n_policies} α={alpha:.2f} true_v={true_vs[-1]:.6f}", flush=True)
        pi_e_a = _policy_probs_at_actions(pol, users, actions)
        raw_iws.append(pi_e_a / np.maximum(pscore, 1e-12))
        pi_rows = _eval_policy_rows(pol, users)
        q_pis.append(np.sum(q_uniq[inv_u] * pi_rows, axis=1))

    true_vs = np.asarray(true_vs, dtype=np.float64)
    n = len(rewards)
    rng = np.random.default_rng(seed + 99)
    full_idx = np.arange(n)

    rows = []
    for clip_m in clips:
        label = "inf" if not np.isfinite(clip_m) else f"{clip_m:g}"
        # Full-sample point estimate
        sp_ci, sp_r, rg_ci, rg_r, mse = _metrics_for_clip(
            rewards=rewards,
            q_f=q_f,
            q_pis=q_pis,
            raw_iws=raw_iws,
            true_vs=true_vs,
            idx=full_idx,
            clip_m=clip_m,
        )
        boot_sp_ci = np.empty(n_boot, dtype=np.float64)
        boot_rg_ci = np.empty(n_boot, dtype=np.float64)
        boot_mse = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            b_sp_ci, _, b_rg_ci, _, b_mse = _metrics_for_clip(
                rewards=rewards,
                q_f=q_f,
                q_pis=q_pis,
                raw_iws=raw_iws,
                true_vs=true_vs,
                idx=idx,
                clip_m=clip_m,
            )
            boot_sp_ci[b] = b_sp_ci
            boot_rg_ci[b] = b_rg_ci
            boot_mse[b] = b_mse

        rows.append(
            {
                "cell": cell_name,
                "seed": int(seed),
                "clip_m": None if not np.isfinite(clip_m) else float(clip_m),
                "clip_label": label,
                "spearman_ci_low": sp_ci,
                "spearman_r_hat": sp_r,
                "regret_ci_low": rg_ci,
                "regret_r_hat": rg_r,
                "mse_r_hat": mse,
                "boot_mean_spearman_ci": float(np.nanmean(boot_sp_ci)),
                "boot_lo_spearman_ci": float(np.nanpercentile(boot_sp_ci, 2.5)),
                "boot_hi_spearman_ci": float(np.nanpercentile(boot_sp_ci, 97.5)),
                "boot_mean_regret_ci": float(np.nanmean(boot_rg_ci)),
                "boot_lo_regret_ci": float(np.nanpercentile(boot_rg_ci, 2.5)),
                "boot_hi_regret_ci": float(np.nanpercentile(boot_rg_ci, 97.5)),
                "boot_mean_mse": float(np.nanmean(boot_mse)),
                "n_policies": int(n_policies),
                "n_log": int(n),
                "n_boot": int(n_boot),
                "max_raw_iw": float(max(float(np.max(w)) for w in raw_iws)),
                "reward_model": "logging_score",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb-dir", type=Path, default=Path("BPR/embeddings"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/oom_smoke/dr_score_clip_logging_score"),
    )
    p.add_argument("--n-log", type=int, default=50_000)
    p.add_argument("--n-policies", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--ctr", type=float, default=0.05)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--clips",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, float("inf")],
    )
    args = p.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    cells = [
        ("context_cluster_high", "context", "cluster", "high"),
        ("action_linear_high", "action", "linear", "high"),
        ("context_cluster_medium", "context", "cluster", "medium"),
    ]
    frames = []
    for name, axis, comp, level in cells:
        for seed in args.seeds:
            print(f"=== {name} seed={seed} (logging_score, no fit) ===", flush=True)
            ds = _build_dataset(
                args.emb_dir,
                noise_axis=axis,
                noise_component=comp,
                noise_level=level,
                seed=int(seed),
                ctr=float(args.ctr),
            )
            frames.append(
                run_cell(
                    ds,
                    n_log=int(args.n_log),
                    seed=int(seed),
                    clips=list(args.clips),
                    n_policies=int(args.n_policies),
                    n_boot=int(args.n_boot),
                    cell_name=name,
                )
            )

    df = pd.concat(frames, ignore_index=True)
    csv_path = out / "dr_score_clip_sweep.csv"
    df.to_csv(csv_path, index=False)

    g = (
        df.groupby("clip_label", dropna=False)
        .agg(
            mean_spearman_ci=("spearman_ci_low", "mean"),
            mean_boot_spearman_ci=("boot_mean_spearman_ci", "mean"),
            mean_boot_lo_spearman=("boot_lo_spearman_ci", "mean"),
            mean_boot_hi_spearman=("boot_hi_spearman_ci", "mean"),
            mean_regret_ci=("regret_ci_low", "mean"),
            mean_boot_regret_ci=("boot_mean_regret_ci", "mean"),
            mean_mse=("mse_r_hat", "mean"),
            mean_spearman_r=("spearman_r_hat", "mean"),
            clip_m=("clip_m", "first"),
        )
        .reset_index()
        .sort_values(
            by=["mean_boot_spearman_ci", "mean_boot_regret_ci", "mean_mse"],
            ascending=[False, True, True],
        )
    )
    summary_path = out / "dr_score_clip_summary.csv"
    g.to_csv(summary_path, index=False)

    best = g.iloc[0]
    winner_m = best["clip_m"]
    winner_m = float("inf") if pd.isna(winner_m) else float(winner_m)
    rec = {
        "winner_clip_m": None if not np.isfinite(winner_m) else winner_m,
        "winner_clip_label": str(best["clip_label"]),
        "mean_boot_spearman_ci_low": float(best["mean_boot_spearman_ci"]),
        "mean_boot_spearman_ci_lo95": float(best["mean_boot_lo_spearman"]),
        "mean_boot_spearman_ci_hi95": float(best["mean_boot_hi_spearman"]),
        "mean_boot_regret_ci_low": float(best["mean_boot_regret_ci"]),
        "mean_mse": float(best["mean_mse"]),
        "reward_model": "logging_score",
        "metric": (
            "maximize mean bootstrap Spearman(ci_low, true_V); "
            "tie-break bootstrap regret then mse"
        ),
        "decision": "use_fixed_dr_score_clip",
        "note": "No fit / no CF train. Analytic logging_score q̂. OPC scoring only.",
    }
    rec_path = out / "recommendation.json"
    rec_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    print(g.to_string(index=False))
    print(f"\nrecommendation: {rec}")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {rec_path}")


if __name__ == "__main__":
    main()
