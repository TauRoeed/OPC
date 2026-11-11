#!/usr/bin/env python3
"""Write full-grid inventory JSON for the PDF builder."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("artifacts/full_study")
LOG = Path("artifacts/full_study/misspec_experiment/logs")
OUT = Path("artifacts/full_study/misspec_experiment/hybrid_by_cell/study_inventory.json")

METHODS = [
    "opc",
    "nll",
    "hybrid_seq_f0.25",
    "hybrid_seq_f0.5",
    "hybrid_seq_f0.75",
    "hybrid_seq_adapt",
]
KIND_LABEL = {
    "rstar": "R*",
    "softand": "SoftAND*",
    "softthresh": "SoftThreshτ1.5",
}
DSKEY = {
    "ml": ("ml", ""),
    "anime": ("anime", ""),
    "myket": ("myket", ""),
    "kuairand_full": ("kuairand", "-full"),
    "kuairec_full": ("kuairec", "-full"),
    "kuairand_ml": ("kuairand", "-mlsize"),
    "kuairec_ml": ("kuairec", "-mlsize"),
}
TARGET_N = 40


def parse_queue():
    qpath = LOG / "full_hybrid_grid_queue.txt"
    rows = []
    for ln in qpath.read_text().splitlines():
        if not ln.strip():
            continue
        tag, ds, emb, kind, spec, g = ln.split("|")
        body = tag[len("nlres_grid_") :]
        left, g_str = body.rsplit("_g", 1)
        dskey = None
        for k in ("softthresh", "softand", "rstar"):
            suf = "_" + k
            if left.endswith(suf):
                dskey = left[: -len(suf)]
                kind = k
                break
        if dskey not in DSKEY:
            continue
        dataset, suffix = DSKEY[dskey]
        misspec = KIND_LABEL[kind] + suffix
        rows.append(
            dict(
                tag=tag,
                dskey=dskey,
                dataset=dataset,
                misspec=misspec,
                kind=kind,
                gamma=float(g_str),
                emb=emb,
                source="full_hybrid_grid",
                target_n=TARGET_N,
                methods=list(METHODS),
            )
        )
    return rows


def method_seed_counts(tag: str) -> dict[str, int]:
    d = ROOT / f"run_{tag}"
    by = {m: 0 for m in METHODS}
    if not d.is_dir():
        return by
    for p in d.rglob("*_runs_long.csv"):
        for m in METHODS:
            if p.name.startswith(m + "_runs"):
                by[m] += 1
                break
    return by


def status_for(by: dict[str, int], target_n: int) -> str:
    vals = list(by.values())
    if sum(vals) == 0:
        return "did_not_run"
    if all(v >= target_n for v in vals):
        return "complete"
    return "partial"


def main():
    planned = parse_queue()
    cells = []
    for row in planned:
        by = method_seed_counts(row["tag"])
        st = status_for(by, row["target_n"])
        cells.append(
            {
                **row,
                "method_n": by,
                "n_min": int(min(by.values()) if by else 0),
                "n_max": int(max(by.values()) if by else 0),
                "status": st,
            }
        )
    # Also note non-grid locked tags present in analyze static set (from REPORT / levels)
    summary = {
        "target_n_grid": TARGET_N,
        "n_planned_grid": len(cells),
        "n_complete": sum(1 for c in cells if c["status"] == "complete"),
        "n_partial": sum(1 for c in cells if c["status"] == "partial"),
        "n_did_not_run": sum(1 for c in cells if c["status"] == "did_not_run"),
        "cells": cells,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(
        f"wrote {OUT}: complete={summary['n_complete']} "
        f"partial={summary['n_partial']} did_not_run={summary['n_did_not_run']} "
        f"/ {summary['n_planned_grid']}"
    )


if __name__ == "__main__":
    main()
