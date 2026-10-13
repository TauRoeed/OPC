"""The structured-shift world scan (docs/structured_scenario_shift_study.md §13.1): world statistics only, no method is
trained. Per dataset, seed and shift × response level: the click calibration (reference and best-item CTR, κ, c), the
pathology rule against the homogeneous world, the shift's agreement statistics and the response factors' spread. A
world that fails to calibrate is a row with its error.

    python scripts/ss_world_scan.py --datasets ml kuairand anime --seeds 100 101 200 201 \
        --out artifacts/full_study/structured_scenario_shift/world_scan/world_scan_amended.csv
    python scripts/ss_world_scan.py --gated --shifts moderate --seeds 100 101 --out .../world_scan_gated.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.representation_bias import WorldConfig  # noqa: E402
from utils.seeding import enable_determinism, pin_cpu_threads, seed_everything  # noqa: E402
from utils.structured_shift import RESPONSE_LEVELS, SHIFT_LEVELS, build_structured_world, structured_label  # noqa: E402,E501


def scan_row(x, a, ds: str, seed: int, shift: str, response: str, gated: bool) -> dict:
    row = {"dataset": ds, "seed": seed, "shift": shift, "response": response, "gated": gated}
    seed_everything(seed)
    try:
        d = build_structured_world(x, a, structured_label(shift, response, gated), seed=seed,
                                   config=WorldConfig(target_ctr=0.05))
    except Exception as e:  # noqa: BLE001 (the scan records an uncalibratable world as a row)
        return {**row, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    w = d["world"]
    cs = w["click_sanity"]
    row.update({"best_gt95": cs["share_best_q_gt_0.95"], "best_gt95_hom": cs["share_best_q_gt_0.95_homogeneous"],
                "excess": cs["share_best_q_gt_0.95"] - cs["share_best_q_gt_0.95_homogeneous"],
                "logger_lt001": cs["share_logger_q_lt_0.001"], "pathological": cs["pathological"],
                "best_ctr": w["best_item_ctr"], "best_ctr_hom": w["best_item_ctr_homogeneous"],
                "ref_ctr": w["reference_ctr"], "logger_ctr": w["logging_ctr"], "best_q99": w["best_item_q"]["q99"],
                "corr": w["shift_corr"], "top1": w["top1_agreement"], "top10": w["top10_overlap"],
                "gamma": w["gamma"], "gain": w["available_gain"], "sd_log_alpha": w["log_alpha"]["sd"],
                "sd_beta": w["beta_stats"]["sd"], "kappa": w["kappa"], "c": w["click_intercept"]})
    if gated and w.get("gate_stats"):
        row.update({f"gate_{k}": v for k, v in w["gate_stats"].items()})
    return row


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", default=["ml", "kuairand", "anime"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 200, 201])
    ap.add_argument("--shifts", nargs="+", default=list(SHIFT_LEVELS))
    ap.add_argument("--responses", nargs="+", default=list(RESPONSE_LEVELS))
    ap.add_argument("--gated", action="store_true")
    ap.add_argument("--emb-dir", default="BPR/embeddings")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    enable_determinism(True)
    pin_cpu_threads(4)
    rows = []
    for ds in args.datasets:
        x = np.load(Path(args.emb_dir) / f"{ds}_user_factors.npy")
        a = np.load(Path(args.emb_dir) / f"{ds}_item_factors.npy")
        for seed in args.seeds:
            for shift in args.shifts:
                for response in args.responses:
                    rows.append(scan_row(x, a, ds, seed, shift, response, args.gated))
                    print(rows[-1], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
