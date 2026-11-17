#!/usr/bin/env python3
"""Analyze matched linear+mlp capacity cells vs linear reference (fair-π story).

Reads artifacts/full_study/run_nlres_cap_* and optional linear grid tags.
Writes capacity_fair_pi/did_summary.regen.json and prints a markdown table.

If run folders are missing, prints the frozen did_summary.json and exits 0
(so docs-only checkouts still have a usable report path).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # artifacts/full_study
OUT_DIR = Path(__file__).resolve().parent / "capacity_fair_pi"
FROZEN = OUT_DIR / "did_summary.json"
METHODS = ("opc", "nll", "hybrid_seq_f0.5")


def _mean_ci(xs: list[float]) -> tuple[float, float, float, int]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = sum(xs) / n
    if n < 2:
        return m, float("nan"), float("nan"), n
    s2 = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(s2 / n)
    t = 2.262 if n == 10 else 1.96
    return m, m - t * se, m + t * se, n


def _load_tag(tag: str) -> dict[int, dict[str, float]]:
    d = ROOT / f"run_{tag}"
    out: dict[int, dict[str, float]] = {}
    if not d.is_dir():
        return out
    for seed_dir in d.glob("dataset=*"):
        m = re.search(r"seed=(\d+)", seed_dir.name)
        if not m:
            continue
        seed = int(m.group(1))
        row: dict[str, float] = {}
        for meth in METHODS:
            rp = seed_dir / f"{meth}_runs_long.csv"
            if not rp.exists():
                continue
            with rp.open() as f:
                rr = list(csv.DictReader(f))
            if not rr:
                continue
            row[meth] = float(rr[0]["policy_rewards_greedy"])
        if row:
            out[seed] = row
    return out


def _did(
    a: dict[int, dict[str, float]],
    b: dict[int, dict[str, float]],
    left: str,
    right: str,
) -> list[float]:
    seeds = sorted(set(a) & set(b))
    diffs = []
    for s in seeds:
        if left not in a[s] or right not in a[s] or left not in b[s] or right not in b[s]:
            continue
        d0 = a[s][left] - a[s][right]
        d1 = b[s][left] - b[s][right]
        diffs.append(100.0 * (d1 - d0))
    return diffs


def _record(family: str, capacity: str, contrast: str, diffs: list[float]) -> dict:
    m, lo, hi, n = _mean_ci(diffs)
    return {
        "family": family,
        "capacity": capacity,
        "contrast": contrast,
        "mean": None if n == 0 else round(m, 4),
        "ci_lo": None if n < 2 else round(lo, 4),
        "ci_hi": None if n < 2 else round(hi, 4),
        "n": n,
        "n_pos": sum(1 for x in diffs if x > 0),
    }


def analyze() -> dict:
    did_h: list[dict] = []
    did_opc: list[dict] = []
    missing: list[str] = []

    for short, fam in (("rstar", "rstar"), ("st15", "st15")):
        for hid, cap in ((64, "mlp64"), (256, "mlp256")):
            t0 = f"nlres_cap_{short}_g0_mlp{hid}"
            a = _load_tag(t0)
            if not a:
                missing.append(t0)
            for g in ("0.5", "2"):
                t1 = f"nlres_cap_{short}_g{g}_mlp{hid}"
                b = _load_tag(t1)
                if not b:
                    missing.append(t1)
                    continue
                did_h.append(_record(fam, cap, f"0→{g}", _did(a, b, "hybrid_seq_f0.5", "nll")))
                did_opc.append(_record(fam, cap, f"0→{g}", _did(a, b, "opc", "nll")))

    # Linear references (best-effort)
    for fam, kind in (("rstar", "rstar"), ("st15", "softthresh")):
        t0 = f"nlres_grid_ml_{kind}_g0"
        a = _load_tag(t0)
        if not a:
            missing.append(t0)
            continue
        for g in ("0.5", "2"):
            t1 = f"nlres_grid_ml_{kind}_g{g}"
            b = _load_tag(t1)
            if not b:
                missing.append(t1)
                continue
            did_h.append(_record(fam, "linear", f"0→{g}", _did(a, b, "hybrid_seq_f0.5", "nll")))

    return {
        "did_h_minus_nll": did_h,
        "did_opc_minus_nll": did_opc,
        "missing_tags": sorted(set(missing)),
        "n_cells_with_data": sum(1 for r in did_h if r["n"] > 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="Write did_summary.regen.json")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = analyze()

    if result["n_cells_with_data"] == 0:
        print("No run_nlres_cap_* folders under", ROOT)
        if FROZEN.exists():
            print("Frozen summary:", FROZEN)
            frozen = json.loads(FROZEN.read_text())
            print(json.dumps(frozen.get("did_h_minus_nll", []), indent=2)[:2000])
            print("(using frozen did_summary.json; regenerate on a host with run artifacts)")
            return 0
        print("ERROR: no run data and no frozen summary")
        return 1

    print(f"cells_with_data={result['n_cells_with_data']} missing={len(result['missing_tags'])}")
    print("\n# H−NLL DiD")
    print("| family | capacity | contrast | DiD | 95% CI | n | +signs |")
    print("|---|---|---|---:|---|---:|---:|")
    for r in result["did_h_minus_nll"]:
        if r["n"] == 0:
            continue
        print(
            f"| {r['family']} | {r['capacity']} | {r['contrast']} | "
            f"{r['mean']:+.2f} | [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | "
            f"{r['n']} | {r['n_pos']}/{r['n']} |"
        )

    if args.write:
        out = {
            "title": "Fair reasonable π — regenerated capacity DiD",
            "root": str(ROOT),
            **result,
        }
        path = OUT_DIR / "did_summary.regen.json"
        path.write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
