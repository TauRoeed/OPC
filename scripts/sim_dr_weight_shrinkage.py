#!/usr/bin/env python3
"""Sweep DR weight forms (raw / clip / Su shrink) vs true V(π) on ML noise sim."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from models.models import RegressionModel
from training.trainer_trials import (
    _build_regression_logged_split,
)
from utils.noise_levels import resolve_noise_spec
from utils.policies import Policy
from utils.simulation_utils import calc_reward, generate_dataset


def _load_ml(emb_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
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


def _iw_transform(w: np.ndarray, form: str, lam: float) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    if form == "raw":
        return w
    if form == "clip":
        return np.minimum(w, float(lam))
    if form == "shrink":
        # Su et al.: ŵ = (λ w) / (λ + w²)
        return (float(lam) * w) / (float(lam) + w * w)
    raise ValueError(f"unknown form {form}")


def _dr_estimate(
    reward: np.ndarray,
    q_factual: np.ndarray,
    q_pi: np.ndarray,
    iw_hat: np.ndarray,
) -> float:
    return float(np.mean(q_pi + iw_hat * (reward - q_factual)))


def _ess(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64)
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    if s2 <= 0:
        return 0.0
    return (s1 * s1) / s2


def _policy_probs_at_actions(policy: Policy, users: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """π(a|u) via chunked full softmax (exact)."""
    users = np.asarray(users, dtype=np.int64)
    actions = np.asarray(actions, dtype=np.int64)
    out = np.empty(users.shape[0], dtype=np.float64)
    # Policy._probs_block is public-ish; batch users for speed
    chunk = 2048
    for s in range(0, len(users), chunk):
        e = min(s + chunk, len(users))
        u = users[s:e]
        a = actions[s:e]
        # unique users in chunk → full probs then gather
        uniq, inv = np.unique(u, return_inverse=True)
        probs = policy._probs_block(uniq)  # (n_uniq, n_items)
        out[s:e] = probs[inv, a]
    return out


def _qhat_for_logged(
    reg: RegressionModel,
    contexts: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (q_all [n, A], q_factual [n])."""
    q_all = np.asarray(reg.predict(contexts), dtype=np.float64)
    if q_all.ndim == 3:
        q_all = q_all.squeeze(-1)
    n = actions.shape[0]
    q_f = q_all[np.arange(n), actions]
    return q_all, q_f


def _q_pi(q_all: np.ndarray, pi_rows: np.ndarray) -> np.ndarray:
    return np.sum(q_all * pi_rows, axis=1)


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


def run_cell(
    dataset: dict,
    *,
    n_log: int,
    reg_size: int,
    seed: int,
    lambdas: list[float],
    cell_name: str,
) -> pd.DataFrame:
    our_x = dataset["our_x"]
    our_a = dataset["our_a"]
    emb_x = dataset["emb_x"]
    emb_a = dataset["emb_a"]
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
    # Use train slice as OPE logged data
    log_data = split["train_data"]
    reg_data = split["reg_data"]

    reg = RegressionModel(
        n_actions=n_actions,
        action_context=our_a,
        base_model=LogisticRegression(random_state=12345, max_iter=200),
    )
    reg.fit(reg_data["x"], reg_data["a"], reg_data["r"])

    # π_b = logging (noisy); π_e = clean-embedding softmax (shift)
    logging_pol = Policy(
        n_users=int(dataset["n_users"]),
        n_items=n_actions,
        user_emb=our_x,
        item_emb=our_a,
        emb_dim=int(our_x.shape[1]),
        temperature=1.0,
        user_chunk=4096,
        action_chunk=4096,
        rng=np.random.default_rng(seed),
    )
    eval_pol = Policy(
        n_users=int(dataset["n_users"]),
        n_items=n_actions,
        user_emb=emb_x,
        item_emb=emb_a,
        emb_dim=int(emb_x.shape[1]),
        temperature=1.0,
        user_chunk=4096,
        action_chunk=4096,
        rng=np.random.default_rng(seed + 1),
    )

    true_v = float(calc_reward(dataset, eval_pol))

    users = np.asarray(log_data["x_idx"], dtype=np.int64)
    actions = np.asarray(log_data["a"], dtype=np.int64)
    rewards = np.asarray(log_data["r"], dtype=np.float64)
    pscore = np.asarray(log_data["pscore"], dtype=np.float64).reshape(-1)

    pi_e_a = _policy_probs_at_actions(eval_pol, users, actions)
    w_raw = pi_e_a / np.maximum(pscore, 1e-12)

    contexts = np.asarray(log_data["x"], dtype=np.float32)
    q_all, q_f = _qhat_for_logged(reg, contexts, actions)
    pi_rows = _eval_policy_rows(eval_pol, users)
    q_pi = _q_pi(q_all, pi_rows)

    rows = []
    # raw (λ unused)
    est = _dr_estimate(rewards, q_f, q_pi, w_raw)
    rows.append(
        {
            "cell": cell_name,
            "form": "raw",
            "lambda": np.nan,
            "estimate": est,
            "true_v": true_v,
            "bias": est - true_v,
            "abs_bias": abs(est - true_v),
            "rmse": abs(est - true_v),  # single draw
            "ess": _ess(w_raw),
            "max_w": float(np.max(w_raw)),
            "mean_w": float(np.mean(w_raw)),
            "n": int(len(rewards)),
        }
    )
    for form in ("clip", "shrink"):
        for lam in lambdas:
            wh = _iw_transform(w_raw, form, lam)
            est = _dr_estimate(rewards, q_f, q_pi, wh)
            rows.append(
                {
                    "cell": cell_name,
                    "form": form,
                    "lambda": float(lam),
                    "estimate": est,
                    "true_v": true_v,
                    "bias": est - true_v,
                    "abs_bias": abs(est - true_v),
                    "rmse": abs(est - true_v),
                    "ess": _ess(wh),
                    "max_w": float(np.max(wh)),
                    "mean_w": float(np.mean(wh)),
                    "n": int(len(rewards)),
                }
            )
    _ = logging_pol  # logging used via logged pscore; keep for clarity
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb-dir", type=Path, default=Path("BPR/embeddings"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/oom_smoke/dr_shrink_sim"))
    p.add_argument("--n-log", type=int, default=50_000)
    p.add_argument("--reg-size", type=int, default=25_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ctr", type=float, default=0.05)
    p.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0],
    )
    args = p.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    cells = [
        ("context_cluster_high", "context", "cluster", "high"),
        ("action_linear_high", "action", "linear", "high"),
    ]
    frames = []
    for name, axis, comp, level in cells:
        print(f"=== cell {name} ===", flush=True)
        ds = _build_dataset(
            args.emb_dir,
            noise_axis=axis,
            noise_component=comp,
            noise_level=level,
            seed=int(args.seed),
            ctr=float(args.ctr),
        )
        frames.append(
            run_cell(
                ds,
                n_log=int(args.n_log),
                reg_size=int(args.reg_size),
                seed=int(args.seed),
                lambdas=list(args.lambdas),
                cell_name=name,
            )
        )

    df = pd.concat(frames, ignore_index=True)
    csv_path = out / "dr_shrink_sweep.csv"
    df.to_csv(csv_path, index=False)

    # Pick best by mean abs_bias across cells (raw has one row; clip/shrink by λ)
    summary_rows = []
    for form in ("raw", "clip", "shrink"):
        part = df[df["form"] == form]
        if form == "raw":
            summary_rows.append(
                {
                    "form": form,
                    "lambda": np.nan,
                    "mean_abs_bias": float(part["abs_bias"].mean()),
                    "mean_ess": float(part["ess"].mean()),
                    "mean_max_w": float(part["max_w"].mean()),
                }
            )
        else:
            for lam, g in part.groupby("lambda"):
                summary_rows.append(
                    {
                        "form": form,
                        "lambda": float(lam),
                        "mean_abs_bias": float(g["abs_bias"].mean()),
                        "mean_ess": float(g["ess"].mean()),
                        "mean_max_w": float(g["max_w"].mean()),
                    }
                )
    summary = pd.DataFrame(summary_rows).sort_values("mean_abs_bias")
    summary_path = out / "dr_shrink_summary.csv"
    summary.to_csv(summary_path, index=False)

    best = summary.iloc[0]
    rec = {
        "winner_form": str(best["form"]),
        "winner_lambda": None if pd.isna(best["lambda"]) else float(best["lambda"]),
        "mean_abs_bias": float(best["mean_abs_bias"]),
        "raw_mean_abs_bias": float(
            summary.loc[summary["form"] == "raw", "mean_abs_bias"].iloc[0]
        ),
    }
    rec_path = out / "recommendation.json"
    import json

    rec_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    html = [
        "<html><body><h1>DR weight shrinkage vs GT</h1>",
        f"<p>Winner: <b>{rec['winner_form']}</b>",
    ]
    if rec["winner_lambda"] is not None:
        html.append(f" λ={rec['winner_lambda']:g}")
    html.append(
        f" (mean |bias|={rec['mean_abs_bias']:.6g}; raw={rec['raw_mean_abs_bias']:.6g})</p>"
    )
    html.append("<h2>Summary</h2>")
    html.append(summary.to_html(index=False, float_format=lambda x: f"{x:.6g}"))
    html.append("<h2>Full sweep</h2>")
    html.append(df.to_html(index=False, float_format=lambda x: f"{x:.6g}"))
    html.append("</body></html>")
    html_path = out / "dr_shrink_report.html"
    html_path.write_text("\n".join(html), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"\nrecommendation: {rec}")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {html_path}")
    print(f"wrote {rec_path}")


if __name__ == "__main__":
    main()
