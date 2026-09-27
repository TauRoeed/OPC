"""Recoverability tables for the representation-bias study (development runs).

Stage 1 (structural): from ``training.oracle_repair`` rows, per condition (dataset, bias, seed). The learner's class
(``repair``) is the linear repair with a learnable logit scale; both oracle fits (scale fixed and scale learned) are
policies in it, so its bound is the better of the two (value and greedy value separately):
  V_logger, V_logger_greedy        the biased logger's exact true value and its ranking's (greedy) value
  V_clean, V_clean_greedy          the matched no-bias reference: the same dataset and seed with bias ``none``
                                   (its greedy value is the ceiling: each user's truly best item)
  V_oracle[_greedy] per class      the best exact value the learner's class reaches trained on the truth
  representation_loss              V_clean - V_logger          (and the greedy version)
  oracle_repair_gain               V_oracle - V_logger         (and the greedy version)
  structural_recoverability        gain / loss, not clipped; NaN when |loss| < MIN_LOSS (then read the raw values)

The stochastic values mix ranking and sharpness: the clean reference is the clean logger at its own exploration
level, while the repair class can also sharpen (a linear map can scale the vectors). The greedy values compare
rankings only and are the primary structural measure.

Stage 2 (learned): ``learned_recovery`` joins learned runs to the oracle rows: learned_gain = V_method - V_logger
and fraction_of_oracle_repair = learned_gain / (V_oracle - V_logger) (NaN when the denominator is below MIN_LOSS).
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from utils.representation_bias import bias_label, parse_bias

MIN_LOSS = 1e-3  # CTR points / 100: below this a ratio is not reported
CLASSES = ("repair", "linear", "linear+scale", "scale")
_CONFIGS = (("none", "no bias"), ("medium", "combined medium"), ("high", "combined high"),
            ("high/none/none", "warp only (high)"), ("none/high/none", "group only (high)"), ("none/none/high", "vector only (high)"))
BIAS_ORDER = tuple(bias_label(parse_bias(c)) for c, _ in _CONFIGS)  # the labels the runners record
BIAS_NAMES = {bias_label(parse_bias(c)): name for c, name in _CONFIGS}


def load_oracle(root) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(glob.glob(str(Path(root) / "**" / "oracle_repair.csv"), recursive=True))]
    if not frames:
        raise FileNotFoundError(f"no oracle_repair.csv under {root}")
    return pd.concat(frames, ignore_index=True)


def _ratio(num, den):
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    out = np.full(num.shape, np.nan)
    ok = np.abs(den) >= MIN_LOSS
    out[ok] = num[ok] / den[ok]
    return out


def derive(rows: pd.DataFrame) -> pd.DataFrame:
    """Adds the clean reference (bias none of the same dataset and seed) and the derived quantities."""
    df = rows.copy()
    df["bias"] = [bias_label(parse_bias(b)) for b in df["bias"]]
    if {"oracle_linear_value", "oracle_linear+scale_value"} <= set(df.columns):  # the learner's class: best of both fits
        df["oracle_repair_value"] = df[["oracle_linear_value", "oracle_linear+scale_value"]].max(axis=1)
        df["oracle_repair_greedy"] = df[["oracle_linear_greedy", "oracle_linear+scale_greedy"]].max(axis=1)
    clean = df[df["bias"] == bias_label(parse_bias("none"))].set_index(["dataset", "seed"])[["logger_value", "logger_greedy"]]
    clean = clean.rename(columns={"logger_value": "V_clean", "logger_greedy": "V_clean_greedy"})
    df = df.join(clean, on=["dataset", "seed"])
    df["V_logger"], df["V_logger_greedy"], df["ceiling"] = df["logger_value"], df["logger_greedy"], df["logger_ceiling"]
    df["representation_loss"] = df["V_clean"] - df["V_logger"]
    df["representation_loss_greedy"] = df["V_clean_greedy"] - df["V_logger_greedy"]
    for cls in CLASSES:
        if f"oracle_{cls}_value" not in df:
            continue
        df[f"gain_{cls}"] = df[f"oracle_{cls}_value"] - df["V_logger"]
        df[f"gain_{cls}_greedy"] = df[f"oracle_{cls}_greedy"] - df["V_logger_greedy"]
        df[f"recoverability_{cls}"] = _ratio(df[f"gain_{cls}"], df["representation_loss"])
        df[f"recoverability_{cls}_greedy"] = _ratio(df[f"gain_{cls}_greedy"], df["representation_loss_greedy"])
    return df


def summary(df: pd.DataFrame, cls: str = "repair") -> pd.DataFrame:
    """Per bias configuration: means over datasets and seeds (CTR in %), and the per-dataset greedy recoverability."""
    cols = {"V_logger": "V_logger %", "V_logger_greedy": "V_logger greedy %", "V_clean": "V_clean %",
            "ceiling": "ceiling %", f"oracle_{cls}_value": "V_oracle %", f"oracle_{cls}_greedy": "V_oracle greedy %",
            "representation_loss": "loss %", "representation_loss_greedy": "loss greedy %",
            f"gain_{cls}": "gain %", f"gain_{cls}_greedy": "gain greedy %"}
    g = df.groupby("bias")
    out = pd.DataFrame({new: 100 * g[old].mean() for old, new in cols.items()})
    out["recoverability"] = g[f"recoverability_{cls}"].mean()
    out["recoverability greedy"] = g[f"recoverability_{cls}_greedy"].mean()
    out["recoverability greedy min"] = g[f"recoverability_{cls}_greedy"].min()
    out["recoverability greedy max"] = g[f"recoverability_{cls}_greedy"].max()
    per_ds = df.pivot_table(index="bias", columns="dataset", values=f"recoverability_{cls}_greedy", aggfunc="mean")
    out = out.join(per_ds.add_prefix("rec. greedy "))
    out["n"] = g.size()
    order = [b for b in BIAS_ORDER if b in out.index] + [b for b in out.index if b not in BIAS_ORDER]
    out = out.loc[order]
    out.insert(0, "bias type", [BIAS_NAMES.get(b, b) for b in out.index])
    return out


def learned_recovery(learned: pd.DataFrame, oracle: pd.DataFrame, cls: str = "repair") -> pd.DataFrame:
    """``learned``: one row per (dataset, bias, seed, method, train_size) with columns V_method and V_method_greedy
    (true stochastic and greedy CTR of the selected policy). Joins the oracle rows of the same world."""
    o = derive(oracle)[["dataset", "bias", "seed", "V_logger", "V_logger_greedy", f"oracle_{cls}_value",
                        f"oracle_{cls}_greedy"]]
    learned = learned.assign(bias=[bias_label(parse_bias(b)) for b in learned["bias"]])
    m = learned.merge(o, on=["dataset", "bias", "seed"], how="left", validate="many_to_one")
    m["learned_gain"] = m["V_method"] - m["V_logger"]
    m["learned_gain_greedy"] = m["V_method_greedy"] - m["V_logger_greedy"]
    m["fraction_of_oracle_repair"] = _ratio(m["learned_gain"], m[f"oracle_{cls}_value"] - m["V_logger"])
    m["fraction_of_oracle_repair_greedy"] = _ratio(m["learned_gain_greedy"], m[f"oracle_{cls}_greedy"] - m["V_logger_greedy"])
    return m


def _tags(folder: str) -> dict:
    return dict(part.split("=", 1) for part in folder.split("__") if "=" in part)


def load_learned(*run_dirs) -> pd.DataFrame:
    """One row per (condition, method, train size) of study runs: the selected policy's true stochastic and greedy
    CTR (V_method, V_method_greedy), its learned logit scale, the tempered logger's value in the same cell, and the
    selected trial's diagnostics (raw-weight ESS, share of weights > 10, largest weight, selection-estimate error of
    the DR point estimate and of the lower bound) and the true selection regret over the arm's trials (development
    diagnostic)."""
    out = []
    for run in run_dirs:
        for cond in sorted(Path(run).glob("dataset=*")):
            if not (cond / "summary_metrics.csv").exists():
                continue
            tags = _tags(cond.name)
            s = pd.read_csv(cond / "summary_metrics.csv")
            t = pd.read_csv(cond / "trials_long.csv")
            s = s[s["train_size"] > 0]
            temp = s[s["method"] == "tempered_logger"].set_index("train_size")["policy_rewards"]
            for _, r in s.iterrows():
                g = t[(t["method"] == r["method"]) & (t["train_size"] == r["train_size"])]
                best = g[g["is_best_in_run"].astype(bool)].iloc[0]
                assert abs(float(best["actual_reward"]) - float(r["policy_rewards"])) < 1e-9
                out.append(dict(
                    run=Path(run).name, dataset=tags["dataset"], bias=tags["bias"], seed=int(tags["seed"]),
                    method=r["method"], train_size=int(r["train_size"]), V_method=float(r["policy_rewards"]),
                    V_method_greedy=float(r["policy_rewards_greedy"]), logit_scale=float(best.get("logit_scale", np.nan)),
                    V_tempered=float(temp.get(r["train_size"], np.nan)),
                    ess_raw=float(best.get("ess_raw", np.nan)), w_share_gt10=float(best.get("diag_w_share_gt10", np.nan)),
                    w_max=float(best.get("diag_w_max", np.nan)),
                    sel_error_point=float(best["r_hat"] - best["actual_reward"]),
                    sel_error_lower=float(best["value"] - best["actual_reward"]),
                    regret=float(g["actual_reward"].max() - best["actual_reward"]), n_trials=len(g)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    import sys

    df = derive(load_oracle(sys.argv[1]))
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for cls in CLASSES:
        print(f"\n===== class {cls}")
        print(summary(df, cls).round(3).to_string())
