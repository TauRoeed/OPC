#!/usr/bin/env python3
"""Offline grid: DR selection clip M vs true V(π) ranking — pick fixed OPC score clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from models.models import RegressionModel
from training.trainer_trials import _build_regression_logged_split
from utils.policies import Policy
from utils.simulation_utils import calc_reward, ensure_exact_env_q_cache, generate_dataset


def _build_dataset(emb_dir: Path, *, bias: str, seed: int, ctr: float) -> dict:
    """ML world at one representation-bias configuration (utils/representation_bias.py)."""
    emb_x = np.load(emb_dir / "ml_user_factors.npy")
    emb_a = np.load(emb_dir / "ml_item_factors.npy")
    return generate_dataset({"bias": bias, "ctr": float(ctr)}, seed=seed, emb_a=emb_a, emb_x=emb_x)


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


def _make_candidate_policy(
    dataset: dict,
    *,
    alpha: float,
    seed: int,
) -> Policy:
    """Mix clean vs noisy embeddings + small noise → candidate π_e."""
    rng = np.random.default_rng(seed)
    our_x = np.asarray(dataset["our_x"], dtype=np.float64)
    our_a = np.asarray(dataset["our_a"], dtype=np.float64)
    emb_x = np.asarray(dataset["emb_x"], dtype=np.float64)
    emb_a = np.asarray(dataset["emb_a"], dtype=np.float64)
    ux = (1.0 - alpha) * our_x + alpha * emb_x
    ua = (1.0 - alpha) * our_a + alpha * emb_a
    ux = ux + 0.02 * rng.standard_normal(ux.shape)
    ua = ua + 0.02 * rng.standard_normal(ua.shape)
    return Policy(
        n_users=int(dataset["n_users"]),
        n_items=int(dataset["n_actions"]),
        user_emb=ux.astype(np.float32),
        item_emb=ua.astype(np.float32),
        emb_dim=int(ux.shape[1]),
        temperature=float(dataset["policy_temperature"]),
        user_chunk=4096,
        action_chunk=4096,
        rng=np.random.default_rng(seed + 7),
    )


def _dr_vec(
    reward: np.ndarray,
    q_f: np.ndarray,
    q_pi: np.ndarray,
    iw: np.ndarray,
    clip_m: float,
) -> np.ndarray:
    w = np.asarray(iw, dtype=np.float64)
    if np.isfinite(clip_m):
        w = np.minimum(w, float(clip_m))
    return (q_pi + w * (reward - q_f)).astype(np.float64)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def run_cell(
    dataset: dict,
    *,
    n_log: int,
    reg_size: int,
    seed: int,
    clips: list[float],
    n_policies: int,
    cell_name: str,
) -> pd.DataFrame:
    our_x = dataset["our_x"]
    our_a = dataset["our_a"]
    n_actions = int(dataset["n_actions"])

    split = _build_regression_logged_split(
        dataset,
        our_x,
        our_a,
        int(n_log),
        int(max(5_000, n_log // 5)),
        0,
        regression_size=int(reg_size),
    )
    log_data = split["train_data"]
    reg_data = split["reg_data"]

    reg = RegressionModel(
        n_actions=n_actions,
        action_context=our_a,
        base_model=LogisticRegression(random_state=12345, max_iter=200),
    )
    reg.fit(reg_data["x"], reg_data["a"], reg_data["r"])

    users = np.asarray(log_data["x_idx"], dtype=np.int64)
    actions = np.asarray(log_data["a"], dtype=np.int64)
    rewards = np.asarray(log_data["r"], dtype=np.float64)
    pscore = np.asarray(log_data["pscore"], dtype=np.float64).reshape(-1)
    contexts = np.asarray(log_data["x"], dtype=np.float32)
    q_all = np.asarray(reg.predict(contexts), dtype=np.float64)
    if q_all.ndim == 3:
        q_all = q_all.squeeze(-1)
    q_f = q_all[np.arange(len(actions)), actions]

    alphas = np.linspace(0.0, 1.0, int(n_policies))
    true_vs = []
    raw_iws = []
    q_pis = []
    ensure_exact_env_q_cache(dataset)
    for i, alpha in enumerate(alphas):
        pol = _make_candidate_policy(dataset, alpha=float(alpha), seed=seed * 1000 + i)
        true_vs.append(float(calc_reward(dataset, pol)))
        print(f"  policy {i+1}/{n_policies} true_v={true_vs[-1]:.6f}", flush=True)
        pi_e_a = _policy_probs_at_actions(pol, users, actions)
        raw_iws.append(pi_e_a / np.maximum(pscore, 1e-12))
        pi_rows = _eval_policy_rows(pol, users)
        q_pis.append(np.sum(q_all * pi_rows, axis=1))

    true_vs = np.asarray(true_vs, dtype=np.float64)
    oracle_best = int(np.argmax(true_vs))

    rows = []
    for clip_m in clips:
        r_hats = []
        ci_lows = []
        for iw, q_pi in zip(raw_iws, q_pis):
            vec = _dr_vec(rewards, q_f, q_pi, iw, clip_m)
            mean = float(np.mean(vec))
            se = float(np.std(vec, ddof=1) / np.sqrt(len(vec)))
            r_hats.append(mean)
            ci_lows.append(mean - 1.96 * se)
        r_hats = np.asarray(r_hats)
        ci_lows = np.asarray(ci_lows)
        pick_r = int(np.argmax(r_hats))
        pick_ci = int(np.argmax(ci_lows))
        label = "inf" if not np.isfinite(clip_m) else f"{clip_m:g}"
        rows.append(
            {
                "cell": cell_name,
                "seed": int(seed),
                "clip_m": None if not np.isfinite(clip_m) else float(clip_m),
                "clip_label": label,
                "spearman_r_hat": _spearman(r_hats, true_vs),
                "spearman_ci_low": _spearman(ci_lows, true_vs),
                "regret_r_hat": float(true_vs[oracle_best] - true_vs[pick_r]),
                "regret_ci_low": float(true_vs[oracle_best] - true_vs[pick_ci]),
                "mse_r_hat": float(np.mean((r_hats - true_vs) ** 2)),
                "mean_abs_bias": float(np.mean(np.abs(r_hats - true_vs))),
                "n_policies": int(n_policies),
                "n_log": int(len(rewards)),
                "max_raw_iw": float(max(float(np.max(w)) for w in raw_iws)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb-dir", type=Path, default=Path("BPR/embeddings"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/oom_smoke/dr_score_clip_sweep"),
    )
    p.add_argument("--n-log", type=int, default=50_000)
    p.add_argument("--reg-size", type=int, default=25_000)
    p.add_argument("--n-policies", type=int, default=8)
    p.add_argument("--ctr", type=float, default=0.05)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--clips",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, float("inf")],
    )
    args = p.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # bias applies to users and items; group = the old cluster cells, warp = the old linear cells
    cells = [
        ("group_high", "none/high/none"),
        ("warp_high", "high/none/none"),
        ("group_medium", "none/medium/none"),
    ]
    frames = []
    for name, bias in cells:
        for seed in args.seeds:
            print(f"=== {name} seed={seed} ===", flush=True)
            ds = _build_dataset(
                args.emb_dir,
                bias=bias,
                seed=int(seed),
                ctr=float(args.ctr),
            )
            frames.append(
                run_cell(
                    ds,
                    n_log=int(args.n_log),
                    reg_size=int(args.reg_size),
                    seed=int(seed),
                    clips=list(args.clips),
                    n_policies=int(args.n_policies),
                    cell_name=name,
                )
            )

    df = pd.concat(frames, ignore_index=True)
    csv_path = out / "dr_score_clip_sweep.csv"
    df.to_csv(csv_path, index=False)

    # Aggregate: prefer high Spearman(ci_low), then low regret_ci_low, then low mse
    g = (
        df.groupby("clip_label", dropna=False)
        .agg(
            mean_spearman_ci=("spearman_ci_low", "mean"),
            mean_spearman_r=("spearman_r_hat", "mean"),
            mean_regret_ci=("regret_ci_low", "mean"),
            mean_regret_r=("regret_r_hat", "mean"),
            mean_mse=("mse_r_hat", "mean"),
            mean_abs_bias=("mean_abs_bias", "mean"),
            clip_m=("clip_m", "first"),
        )
        .reset_index()
        .sort_values(
            by=["mean_spearman_ci", "mean_regret_ci", "mean_mse"],
            ascending=[False, True, True],
        )
    )
    summary_path = out / "dr_score_clip_summary.csv"
    g.to_csv(summary_path, index=False)

    best = g.iloc[0]
    winner_m = best["clip_m"]
    if pd.isna(winner_m):
        winner_m = float("inf")
    else:
        winner_m = float(winner_m)

    rec = {
        "winner_clip_m": None if not np.isfinite(winner_m) else winner_m,
        "winner_clip_label": str(best["clip_label"]),
        "mean_spearman_ci_low": float(best["mean_spearman_ci"]),
        "mean_regret_ci_low": float(best["mean_regret_ci"]),
        "mean_mse": float(best["mean_mse"]),
        "metric": "maximize mean Spearman(ci_low, true_V); tie-break regret then mse",
        "decision": "use_fixed_dr_score_clip",
        "note": (
            "Fixed clip for OPC DR trial scoring only. No-propensity stays pure naive."
        ),
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
