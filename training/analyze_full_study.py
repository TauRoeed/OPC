import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_LEVEL_ORDER = ["low", "medium", "high"]


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _parse_condition_dirname(name: str):
    out = {}
    for part in name.split("__"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k] = v
    return out


def _load_summary(path: Path):
    df = pd.read_csv(path)
    numeric_cols = [
        "train_size",
        "policy_rewards",
        "conv_dr",
        "seed",
        "ctr",
        "initial_reward",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "noise_axis" not in df.columns:
        df["noise_axis"] = "combined"
    if "ctr" not in df.columns:
        df["ctr"] = np.nan
    return df


def _load_long(run_dir: Path, filename: str):
    frames = []
    for p in sorted(run_dir.glob(f"dataset=*/{filename}")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        tags = _parse_condition_dirname(p.parent.name)
        mapping = {
            "dataset": "dataset",
            "noise": "noise_mode",
            "axis": "noise_axis",
            "level": "noise_level",
            "seed": "seed",
            "ctr": "ctr",
        }
        for src, dst in mapping.items():
            if src in tags and dst not in df.columns:
                df[dst] = tags[src]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for c in ("train_size", "run", "seed", "initial_reward"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "ctr" in out.columns:
        out["ctr"] = pd.to_numeric(out["ctr"], errors="coerce")
    if "noise_axis" not in out.columns:
        out["noise_axis"] = "combined"
    return out


def _load_trials_long_union(run_dir: Path) -> pd.DataFrame:
    """Merge per-condition trial logs (unified or split OPC / no-prop filenames)."""
    parts = []
    for name in ("trials_long.csv", "opc_trials_long.csv", "no_prop_trials_long.csv"):
        df = _load_long(run_dir, name)
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    dedupe_cols = [
        c
        for c in (
            "dataset",
            "noise_mode",
            "noise_axis",
            "noise_level",
            "ctr",
            "seed",
            "method",
            "train_size",
            "run",
            "trial_number",
        )
        if c in out.columns
    ]
    if len(dedupe_cols) >= 5:
        out = out.drop_duplicates(subset=dedupe_cols, keep="first")
    return out


def _load_meta_ctr(run_dir: Path):
    rows = []
    for p in sorted(run_dir.glob("dataset=*/run_meta.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        tags = _parse_condition_dirname(p.parent.name)
        rows.append(
            {
                "dataset": tags.get("dataset", obj.get("dataset")),
                "noise_mode": tags.get("noise", obj.get("noise_mode")),
                "noise_axis": tags.get("axis", obj.get("noise_axis", "combined")),
                "noise_level": tags.get("level", obj.get("noise_level")),
                "seed": pd.to_numeric(tags.get("seed", obj.get("seed")), errors="coerce"),
                "ctr": pd.to_numeric(tags.get("ctr", obj.get("ctr", np.nan)), errors="coerce"),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _pivot_delta(df: pd.DataFrame, metric: str):
    idx_cols = [
        "dataset",
        "noise_mode",
        "noise_axis",
        "noise_level",
        "seed",
        "train_size",
        "ctr",
    ]
    idx_cols = [c for c in idx_cols if c in df.columns]
    if "ctr" in idx_cols and pd.to_numeric(df["ctr"], errors="coerce").notna().sum() == 0:
        idx_cols = [c for c in idx_cols if c != "ctr"]
    tmp = (
        df.pivot_table(index=idx_cols, columns="method", values=metric, aggfunc="mean")
        .reset_index()
        .dropna(subset=["opc", "no_propensity"])
    )
    tmp["delta"] = tmp["opc"] - tmp["no_propensity"]
    return tmp


def _ctr_ref(part: pd.DataFrame):
    vals = pd.to_numeric(part.get("ctr"), errors="coerce").dropna()
    if vals.empty:
        return np.nan
    return float(vals.median())


def _initial_reward_ref(part: pd.DataFrame):
    if "initial_reward" not in part.columns:
        return np.nan
    vals = pd.to_numeric(part["initial_reward"], errors="coerce").dropna()
    if vals.empty:
        return np.nan
    return float(vals.median())


def _ordered_levels(values):
    present = {v for v in values if pd.notna(v)}
    ordered = [level for level in _LEVEL_ORDER if level in present]
    extras = sorted(present - set(_LEVEL_ORDER))
    return ordered + extras


def _fmt_ctr(value):
    return f"{float(value):g}"


def _plot_curves_per_axis(df: pd.DataFrame, out_dir: Path):
    grp = (
        df.groupby(["dataset", "noise_mode", "noise_axis", "ctr", "noise_level", "method", "train_size"])[
            "policy_rewards"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grp["se"] = grp["std"] / np.sqrt(grp["count"].clip(lower=1))
    levels = _LEVEL_ORDER

    for (dataset, noise_mode, noise_axis, ctr), part in grp.groupby(
        ["dataset", "noise_mode", "noise_axis", "ctr"]
    ):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for i, level in enumerate(levels):
            ax = axes[i]
            pp = part[part["noise_level"] == level]
            for method, mm in pp.groupby("method"):
                mm = mm.sort_values("train_size")
                x = pd.to_numeric(mm["train_size"], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(mm["mean"], errors="coerce").to_numpy(dtype=float)
                se = pd.to_numeric(mm["se"], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(se)
                x = x[ok]
                y = y[ok]
                se = se[ok]
                if len(x) == 0:
                    continue
                ax.plot(x, y, marker="o", label=method)
                ax.fill_between(
                    x,
                    y - 1.96 * se,
                    y + 1.96 * se,
                    alpha=0.2,
                )
            if np.isfinite(ctr):
                ax.axhline(ctr, ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_title(level)
            ax.set_xlabel("train_size")
        axes[0].set_ylabel("policy_rewards")
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
        ctr_label = _fmt_ctr(ctr)
        fig.suptitle(f"Curves ({dataset}, {noise_mode}, axis={noise_axis}, ctr={ctr_label})")
        fig.tight_layout()
        fig.savefig(out_dir / f"curves_{dataset}_{noise_mode}_{noise_axis}_ctr_{ctr_label}.png", dpi=180)
        plt.close(fig)


def _selection_with_ess(trials_long: pd.DataFrame, runs_long: pd.DataFrame, qs):
    if trials_long.empty or runs_long.empty:
        return pd.DataFrame()
    key = [
        "dataset",
        "noise_mode",
        "noise_axis",
        "noise_level",
        "ctr",
        "seed",
        "method",
        "train_size",
        "run",
    ]
    key = [c for c in key if c in trials_long.columns and c in runs_long.columns]
    runs = runs_long.copy()
    if "val_size" not in runs.columns:
        runs["val_size"] = np.nan
    runs["val_size"] = pd.to_numeric(runs["val_size"], errors="coerce")

    t = trials_long.copy()
    t["ess"] = pd.to_numeric(t.get("ess"), errors="coerce")
    t["value"] = pd.to_numeric(t.get("value"), errors="coerce")
    t = t.merge(runs[key + ["val_size"]], on=key, how="left")
    t["ess_ratio"] = t["ess"] / t["val_size"].clip(lower=1)

    outs = []
    gcols = [c for c in key if c != "run"]
    for q in qs:
        tt = t[t["ess_ratio"] >= float(q)].copy()
        if tt.empty:
            continue
        sel = tt.sort_values("value", ascending=False).groupby(gcols, as_index=False).first()
        chosen = runs.merge(sel[gcols + ["run"]], on=gcols + ["run"], how="inner")
        chosen["ess_threshold"] = float(q)
        outs.append(chosen)
    if not outs:
        return pd.DataFrame()
    return pd.concat(outs, ignore_index=True)


def _plot_ess_threshold_curves(selection_with_ess: pd.DataFrame, out_dir: Path):
    if selection_with_ess.empty:
        return
    grp = (
        selection_with_ess.groupby(
            ["dataset", "noise_mode", "noise_axis", "ctr", "noise_level", "method", "ess_threshold", "train_size"]
        )["policy_rewards"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grp["se"] = grp["std"] / np.sqrt(grp["count"].clip(lower=1))

    for (dataset, noise_mode, noise_axis, ctr, method), part in grp.groupby(
        ["dataset", "noise_mode", "noise_axis", "ctr", "method"]
    ):
        levels = _ordered_levels(part["noise_level"].dropna().unique().tolist())
        fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4.2), sharey=True)
        if len(levels) == 1:
            axes = [axes]
        for ax, level in zip(axes, levels):
            pp = part[part["noise_level"] == level]
            for q, qq in pp.groupby("ess_threshold"):
                qq = qq.sort_values("train_size")
                x = pd.to_numeric(qq["train_size"], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(qq["mean"], errors="coerce").to_numpy(dtype=float)
                se = pd.to_numeric(qq["se"], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(se)
                x = x[ok]
                y = y[ok]
                se = se[ok]
                if len(x) == 0:
                    continue
                ax.plot(x, y, marker="o", label=f"ess>={q:.1f}")
                ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, alpha=0.18)
            ax.set_xscale("log")
            ax.set_title(level)
            ax.set_xlabel("train_size")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("policy_rewards")
        h, lab = axes[-1].get_legend_handles_labels()
        if h:
            fig.legend(h, lab, loc="upper center", ncol=max(1, len(set(lab))))
        ctr_label = _fmt_ctr(ctr)
        fig.suptitle(
            f"ESS-filtered selection ({dataset}, {noise_mode}, axis={noise_axis}, {method}, ctr={ctr_label})"
        )
        fig.tight_layout()
        fig.savefig(
            out_dir / f"ess_curves_{dataset}_{noise_mode}_{noise_axis}_{method}_ctr_{ctr_label}.png",
            dpi=180,
        )
        plt.close(fig)


def _plot_delta_with_ci(df: pd.DataFrame, out_dir: Path):
    delta = _pivot_delta(df, "policy_rewards")
    agg = (
        delta.groupby(["dataset", "noise_mode", "noise_axis", "ctr", "noise_level", "train_size"])["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["se"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    levels = _LEVEL_ORDER

    for (dataset, noise_mode, noise_axis, ctr), part in agg.groupby(
        ["dataset", "noise_mode", "noise_axis", "ctr"]
    ):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for i, level in enumerate(levels):
            ax = axes[i]
            pp = part[part["noise_level"] == level].sort_values("train_size")
            if not pp.empty:
                ax.errorbar(pp["train_size"], pp["mean"], yerr=1.96 * pp["se"], marker="o")
            ax.axhline(0.0, ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_title(level)
            ax.set_xlabel("train_size")
        axes[0].set_ylabel("opc - no_propensity")
        ctr_label = _fmt_ctr(ctr)
        fig.suptitle(f"Delta with CI ({dataset}, {noise_mode}, axis={noise_axis}, ctr={ctr_label})")
        fig.tight_layout()
        fig.savefig(out_dir / f"delta_{dataset}_{noise_mode}_{noise_axis}_ctr_{ctr_label}.png", dpi=180)
        plt.close(fig)


def _plot_seed_strip(df: pd.DataFrame, out_dir: Path):
    for (dataset, noise_axis), part in df.groupby(["dataset", "noise_axis"]):
        levels = _ordered_levels(part["noise_level"].dropna().unique().tolist())
        modes = sorted(part["noise_mode"].dropna().unique().tolist())
        fig, axes = plt.subplots(
            len(modes), len(levels), figsize=(5 * len(levels), 3.5 * len(modes)), squeeze=False
        )
        for i, mode in enumerate(modes):
            for j, level in enumerate(levels):
                ax = axes[i][j]
                pp = part[(part["noise_mode"] == mode) & (part["noise_level"] == level)]
                for method, mm in pp.groupby("method"):
                    x = np.log10(pd.to_numeric(mm["train_size"], errors="coerce").clip(lower=1))
                    jitter = (
                        pd.to_numeric(mm.get("seed"), errors="coerce").fillna(0).astype(float) % 7 - 3
                    ) * 0.01
                    ax.scatter(x + jitter, mm["policy_rewards"], alpha=0.7, label=method, s=18)
                ax.set_title(f"{mode} | {level}")
                ax.set_xlabel("log10(train_size)")
                if j == 0:
                    ax.set_ylabel("policy_rewards")
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=max(1, len(set(labels))))
        fig.suptitle(f"Per-seed strip ({dataset}, axis={noise_axis})")
        fig.tight_layout()
        fig.savefig(out_dir / f"seeds_{dataset}_{noise_axis}.png", dpi=180)
        plt.close(fig)


def _plot_calibration(df: pd.DataFrame, out_dir: Path):
    for (dataset, noise_axis), part in df.groupby(["dataset", "noise_axis"]):
        plt.figure(figsize=(6, 5))
        for method, p in part.groupby("method"):
            x = pd.to_numeric(p["conv_dr"], errors="coerce").to_numpy()
            y = pd.to_numeric(p["policy_rewards"], errors="coerce").to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            x = x[ok]
            y = y[ok]
            if len(x) == 0:
                continue
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            bins = np.array_split(np.arange(len(x)), min(20, len(x)))
            xb = np.array([x[b].mean() for b in bins if len(b) > 0])
            yb = np.array([y[b].mean() for b in bins if len(b) > 0])
            plt.plot(xb, yb, marker="o", label=method)
        lo, hi = plt.xlim()
        plt.plot([lo, hi], [lo, hi], "k--", alpha=0.6)
        plt.xlabel("estimated (conv_dr)")
        plt.ylabel("true (policy_rewards)")
        plt.title(f"Calibration ({dataset}, axis={noise_axis})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"calibration_{dataset}_{noise_axis}.png", dpi=180)
        plt.close()


def _plot_oracle_best_actual(trials_long: pd.DataFrame, out_dir: Path):
    """Max actual_reward over hyperparam trials per setting; ref lines from logs (initial_reward, ctr)."""
    if trials_long.empty or "actual_reward" not in trials_long.columns:
        return

    t = trials_long.copy()
    for c in (
        "train_size",
        "seed",
        "run",
        "trial_number",
        "actual_reward",
        "initial_reward",
        "ctr",
    ):
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    key = [
        "dataset",
        "noise_mode",
        "noise_axis",
        "noise_level",
        "ctr",
        "seed",
        "method",
        "train_size",
    ]
    key = [c for c in key if c in t.columns]
    if len(key) < 5:
        return

    oracle = (
        t.dropna(subset=["actual_reward"])
        .groupby(key, as_index=False)["actual_reward"]
        .max()
        .rename(columns={"actual_reward": "best_actual_reward"})
    )
    oracle.to_csv(out_dir / "oracle_best_actual_by_setting.csv", index=False)

    gcols = [
        c
        for c in ("dataset", "noise_mode", "noise_axis", "noise_level", "ctr", "method", "train_size")
        if c in oracle.columns
    ]
    agg = (
        oracle.groupby(gcols, as_index=False)
        .agg(mean_best_actual=("best_actual_reward", "mean"), std=("best_actual_reward", "std"), n=("best_actual_reward", "count"))
        .reset_index(drop=True)
    )
    agg["se"] = agg["std"] / np.sqrt(agg["n"].clip(lower=1))
    agg.to_csv(out_dir / "oracle_best_actual_aggregated.csv", index=False)

    for (dataset, noise_mode, axis, ctr), part in agg.groupby(["dataset", "noise_mode", "noise_axis", "ctr"]):
        levels = _ordered_levels(part["noise_level"].dropna().unique().tolist())
        if not levels:
            continue
        fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4.2), sharey=True)
        if len(levels) == 1:
            axes = [axes]

        ref_base = t[
            (t["dataset"] == dataset)
            & (t["noise_mode"] == noise_mode)
            & (t["noise_axis"] == axis)
            & (t["ctr"] == ctr)
        ]

        for i, level in enumerate(levels):
            ax = axes[i]
            pp = part[part["noise_level"] == level]
            ref_sub = ref_base[ref_base["noise_level"] == level]
            if not ref_sub.empty and "initial_reward" in ref_sub.columns:
                ir = float(pd.to_numeric(ref_sub["initial_reward"], errors="coerce").median())
            else:
                ir = float("nan")
            if not ref_sub.empty and "ctr" in ref_sub.columns:
                ctr_line = float(pd.to_numeric(ref_sub["ctr"], errors="coerce").median())
            else:
                ctr_line = float("nan")

            for method, mm in pp.groupby("method"):
                mm = mm.sort_values("train_size")
                x = mm["train_size"].to_numpy(dtype=float)
                y = mm["mean_best_actual"].to_numpy(dtype=float)
                se = mm["se"].fillna(0).to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(se)
                x, y, se = x[ok], y[ok], se[ok]
                if len(x) == 0:
                    continue
                ax.plot(x, y, marker="o", label=method)
                ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, alpha=0.2)
            if np.isfinite(ir):
                ax.axhline(ir, color="gray", ls=":", lw=1.2, label="initial_reward")
            if np.isfinite(ctr_line):
                ax.axhline(ctr_line, color="tab:red", ls="--", lw=1.2, label="max (ctr)")
            ax.set_xscale("log")
            ax.set_title(level)
            ax.set_xlabel("train_size")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("oracle best actual_reward")
        h, lab = axes[-1].get_legend_handles_labels()
        if h:
            fig.legend(h, lab, loc="upper center", ncol=max(1, len(set(lab))))
        ctr_label = _fmt_ctr(ctr)
        fig.suptitle(f"Oracle best actual ({dataset}, {noise_mode}, axis={axis}, ctr={ctr_label})")
        fig.tight_layout()
        fig.savefig(out_dir / f"oracle_best_actual_{dataset}_{noise_mode}_{axis}_ctr_{ctr_label}.png", dpi=180)
        plt.close(fig)


def _plot_selected_policy_reward(df: pd.DataFrame, out_dir: Path):
    """Selected policy reward over train_size; same layout as oracle-best plots."""
    if df.empty or "policy_rewards" not in df.columns:
        return

    selected = df.copy()
    for c in ("train_size", "seed", "policy_rewards", "initial_reward", "ctr"):
        if c in selected.columns:
            selected[c] = pd.to_numeric(selected[c], errors="coerce")
    selected = selected[selected["train_size"] > 0].dropna(subset=["policy_rewards"])

    key = [
        "dataset",
        "noise_mode",
        "noise_axis",
        "noise_level",
        "ctr",
        "seed",
        "method",
        "train_size",
    ]
    key = [c for c in key if c in selected.columns]
    if len(key) < 5:
        return

    setting = (
        selected.groupby(key, as_index=False)["policy_rewards"]
        .mean()
        .rename(columns={"policy_rewards": "selected_policy_reward"})
    )
    setting.to_csv(out_dir / "selected_policy_reward_by_setting.csv", index=False)

    gcols = [
        c
        for c in ("dataset", "noise_mode", "noise_axis", "noise_level", "ctr", "method", "train_size")
        if c in setting.columns
    ]
    agg = (
        setting.groupby(gcols, as_index=False)
        .agg(
            mean_selected_policy_reward=("selected_policy_reward", "mean"),
            std=("selected_policy_reward", "std"),
            n=("selected_policy_reward", "count"),
        )
        .reset_index(drop=True)
    )
    agg["se"] = agg["std"] / np.sqrt(agg["n"].clip(lower=1))
    agg.to_csv(out_dir / "selected_policy_reward_aggregated.csv", index=False)

    for (dataset, noise_mode, axis, ctr), part in agg.groupby(
        ["dataset", "noise_mode", "noise_axis", "ctr"]
    ):
        levels = _ordered_levels(part["noise_level"].dropna().unique().tolist())
        if not levels:
            continue
        fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4.2), sharey=True)
        if len(levels) == 1:
            axes = [axes]

        ref_base = selected[
            (selected["dataset"] == dataset)
            & (selected["noise_mode"] == noise_mode)
            & (selected["noise_axis"] == axis)
            & (selected["ctr"] == ctr)
        ]

        for i, level in enumerate(levels):
            ax = axes[i]
            pp = part[part["noise_level"] == level]
            ref_sub = ref_base[ref_base["noise_level"] == level]
            if not ref_sub.empty and "initial_reward" in ref_sub.columns:
                ir = float(pd.to_numeric(ref_sub["initial_reward"], errors="coerce").median())
            else:
                ir = float("nan")
            if not ref_sub.empty and "ctr" in ref_sub.columns:
                ctr_line = float(pd.to_numeric(ref_sub["ctr"], errors="coerce").median())
            else:
                ctr_line = float("nan")

            for method, mm in pp.groupby("method"):
                mm = mm.sort_values("train_size")
                x = mm["train_size"].to_numpy(dtype=float)
                y = mm["mean_selected_policy_reward"].to_numpy(dtype=float)
                se = mm["se"].fillna(0).to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(se)
                x, y, se = x[ok], y[ok], se[ok]
                if len(x) == 0:
                    continue
                ax.plot(x, y, marker="o", label=method)
                ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, alpha=0.2)
            if np.isfinite(ir):
                ax.axhline(ir, color="gray", ls=":", lw=1.2, label="initial_reward")
            if np.isfinite(ctr_line):
                ax.axhline(ctr_line, color="tab:red", ls="--", lw=1.2, label="max (ctr)")
            ax.set_xscale("log")
            ax.set_title(level)
            ax.set_xlabel("train_size")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("selected policy reward")
        h, lab = axes[-1].get_legend_handles_labels()
        if h:
            fig.legend(h, lab, loc="upper center", ncol=max(1, len(set(lab))))
        ctr_label = _fmt_ctr(ctr)
        fig.suptitle(f"Selected policy reward ({dataset}, {noise_mode}, axis={axis}, ctr={ctr_label})")
        fig.tight_layout()
        fig.savefig(
            out_dir / f"selected_policy_reward_{dataset}_{noise_mode}_{axis}_ctr_{ctr_label}.png",
            dpi=180,
        )
        plt.close(fig)


def _robustness_ranking(df: pd.DataFrame):
    delta = _pivot_delta(df, "policy_rewards")
    return (
        delta.groupby(["dataset", "noise_mode", "noise_axis", "ctr", "noise_level"])["delta"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .sort_values(["dataset", "mean"], ascending=[True, False])
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize full study outputs.")
    parser.add_argument("--run-dir", default=None, help="Run directory containing dataset=*/...")
    parser.add_argument("--summary-csv", default=None, help="Optional explicit summary CSV.")
    parser.add_argument("--out-dir", default=None, help="Optional explicit output directory.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is None and args.summary_csv is None:
        run_dir = Path("artifacts/full_study")
    if args.summary_csv is not None:
        summary_csv = Path(args.summary_csv)
    else:
        summary_csv = run_dir / "all_summary_metrics.csv"
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif run_dir is not None:
        out_dir = run_dir / "figures"
    else:
        out_dir = Path("artifacts/full_study/figures")
    _ensure_dir(out_dir)

    df = _load_summary(summary_csv)
    if run_dir is None:
        run_dir = summary_csv.parent

    meta_ctr = _load_meta_ctr(run_dir)
    if not meta_ctr.empty:
        key_cols = ["dataset", "noise_mode", "noise_axis", "noise_level", "seed"]
        existing = [c for c in key_cols if c in df.columns and c in meta_ctr.columns]
        if existing:
            df = df.merge(meta_ctr[existing + ["ctr"]], on=existing, how="left", suffixes=("", "_meta"))
            if "ctr_meta" in df.columns:
                df["ctr"] = pd.to_numeric(df["ctr"], errors="coerce").fillna(
                    pd.to_numeric(df["ctr_meta"], errors="coerce")
                )
                df = df.drop(columns=["ctr_meta"])

    trials_long = _load_trials_long_union(run_dir)
    runs_long = _load_long(run_dir, "runs_long.csv")
    if runs_long.empty:
        runs_long = _load_long(run_dir, "opc_runs_long.csv")
        r2 = _load_long(run_dir, "no_prop_runs_long.csv")
        if not r2.empty:
            runs_long = pd.concat([runs_long, r2], ignore_index=True) if not runs_long.empty else r2
    selection_with_ess = _selection_with_ess(trials_long, runs_long, qs=(0.0, 0.1, 0.3, 0.5, 0.7))
    if not selection_with_ess.empty:
        selection_with_ess.to_csv(out_dir / "selection_with_ess.csv", index=False)

    _plot_curves_per_axis(df, out_dir)
    _plot_delta_with_ci(df, out_dir)
    _plot_seed_strip(df, out_dir)
    _plot_calibration(df, out_dir)
    _plot_ess_threshold_curves(selection_with_ess, out_dir)
    _plot_oracle_best_actual(trials_long, out_dir)
    _plot_selected_policy_reward(df, out_dir)

    robust = _robustness_ranking(df)
    robust.to_csv(out_dir / "robustness_ranking.csv", index=False)

    delta = _pivot_delta(df, "policy_rewards")
    sig = (
        delta.groupby(["dataset", "noise_mode", "noise_axis", "ctr", "noise_level", "train_size"])["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    sig["se"] = sig["std"] / np.sqrt(sig["count"].clip(lower=1))
    sig["ci_low"] = sig["mean"] - 1.96 * sig["se"]
    sig["ci_high"] = sig["mean"] + 1.96 * sig["se"]
    sig["significant_positive"] = sig["ci_low"] > 0
    sig.to_csv(out_dir / "delta_significance.csv", index=False)


if __name__ == "__main__":
    main()
