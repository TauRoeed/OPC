"""Summaries of the revalidation's Phase 3 reruns (docs/simulator_fix_opc_revalidation_20261004.md).

Builds, from the fixed-simulator runs, the same tables the old report was built from, so that every old table has a
new counterpart paired by world:
  stage2/     analyze_recoverability stage2 (fractions, paired contrasts, diagnostics, learned rows, the robustness arm)
  followup/   the gap decomposition with the unchanged Stage 1 oracle (and the validated bounds as a sensitivity)
  stage3_lgs_<share>/   the logging-support sweep at 25k (share 0.8 = the Stage 2 rerun's 25k cells)
  reward_model/   learned rows of the external-q̂ and misspecified-q̂ reruns and of their old counterparts
  decomposition/  learned rows of the old Stage 2 configuration on the corrected logs (OPC and DM-only, ml/kuairand)
  old/            the old (buggy-log) reward-model tests and Su slice, rebuilt from their run folders
  m5/             the OPC side of the CausE M5 comparison (read only, for the Phase 5 assessment)

Usage: python -m training.revalidation_phase3 --out artifacts/full_study/opc_revalidation_20261004/summaries
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from training.analyze_recoverability import learned_recovery, load_learned, load_oracle, main as recoverability

RUNS = Path("artifacts/full_study")
ORACLE = RUNS / "run_oracle_repair_20260927"
ORACLE_STAGE3 = {"0.6": RUNS / "run_oracle_repair_stage3_lgs_0_6", "0.95": RUNS / "run_oracle_repair_stage3_lgs_0_95"}
VALIDATION = RUNS / "run_oracle_validation_20260927"
STAGE2 = ["run_reval_stage2_base_mlkr", "run_reval_stage2_base_anime", "run_reval_stage2_opc_mlkr", "run_reval_stage2_opc_anime"]
ROBUST = {"shrink100": ["run_reval_stage2_opc_shrink100_mlkr", "run_reval_stage2_opc_shrink100_anime"]}
STAGE3 = {"0.6": ["run_reval_stage3_base_lgs_0_6", "run_reval_stage3_opc_lgs_0_6"],
          "0.95": ["run_reval_stage3_base_lgs_0_95", "run_reval_stage3_opc_lgs_0_95"]}
REWARD_MODEL = {"external": ["run_reval_budget_external_base", "run_reval_budget_external_opc"],
                "concat": ["run_reval_qhat_concat_base", "run_reval_qhat_concat_opc"]}
# the old Stage 2 configuration (old search space) on the corrected logs: separates the simulator effect from the retuning
OLDSPACE = ["run_reval_stage2_oldspace_opc_mlkr", "run_reval_stage2_oldspace_dm_mlkr"]
# the old (buggy-log) counterparts that are not in summaries_20260927: the reward-model tests of the older pipeline
# (run folder, condition-name pattern; the old report's Table 6) and the Su robustness slice
OLD_REWARD_MODEL = {"external": ("run_logger_explore", r"__lgs=0\.8$"),
                    "budget_fair": ("run_logger_explore", r"__lgs=0\.8__qhat=train__cf=5$"),
                    "interaction": ("run_logger_explore_budget", r"__lgs=0\.8__qhat=train__cf=5__val=20000$"),
                    "concat": ("run_qhat_concat", r"__lgs=0\.8__qhat=train__cf=5__val=20000$")}
OLD_SU = "run_stage2_su_shrink_100"
# the OPC side of the CausE M5 comparison (fixed simulator, the pre-revalidation OPC configuration; Phase 5 only reads it)
M5_OPC = "run_cause_dev_25k_opc_20261004"


def learned_rows_matching(run_dir, oracle_root, pattern: str) -> pd.DataFrame:
    """``learned_recovery`` rows of the condition folders of ``run_dir`` whose name matches the regular expression
    ``pattern`` (the old reward-model runs mix several settings in one run folder)."""
    import re
    import tempfile

    run_dir = Path(run_dir)
    with tempfile.TemporaryDirectory() as tmp:
        sub = Path(tmp) / run_dir.name
        sub.mkdir()
        for cond in run_dir.glob("dataset=*"):
            if re.search(pattern, cond.name):
                (sub / cond.name).symlink_to(cond.resolve())
        rows = load_learned(sub)
    if rows.empty:
        raise FileNotFoundError(f"no condition of {run_dir} matches {pattern!r}")
    return learned_recovery(rows, load_oracle(oracle_root))


def _existing(names) -> list[str]:
    return [str(RUNS / n) for n in names if (RUNS / n).exists()]


def _merged_dir(names, tmp: Path) -> Path:
    """One folder whose condition sub-folders are symlinks into several runs (``--robust`` takes one folder)."""
    tmp.mkdir(parents=True, exist_ok=True)
    for n in names:
        for cond in (RUNS / n).glob("dataset=*"):
            link = tmp / cond.name
            if not link.exists():
                link.symlink_to(cond.resolve())
    return tmp


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        robust = [f"--robust={lab}={_merged_dir(names, tmp / ('robust_' + lab))}" for lab, names in ROBUST.items()
                  if _existing(names)]
        recoverability(["stage2", str(ORACLE), "--runs", *_existing(STAGE2), *robust, "--out", str(out / "stage2")])
        recoverability(["followup", str(ORACLE), "--runs", *_existing(STAGE2), "--candidates", str(VALIDATION),
                        "--out", str(out / "followup")])
        # Stage 3: 0.6 / 0.95 from their reruns; 0.8 = the Stage 2 rerun's 25k cells of the single-type high biases
        for share, names in STAGE3.items():
            if _existing(names):
                recoverability(["stage2", str(ORACLE_STAGE3[share]), "--runs", *_existing(names),
                                "--out", str(out / f"stage3_lgs_{share.replace('.', '_')}")])
        rows = pd.read_csv(out / "stage2" / "stage2_learned_rows.csv")
        singles = {"w-high.g-none.v-none", "w-none.g-high.v-none", "w-none.g-none.v-high"}
        s08 = rows[(rows["train_size"] == 25000) & rows["bias"].isin(singles) & rows["dataset"].isin(["ml", "kuairand"])
                   & rows["method"].isin(["opc", "dm", "no_propensity", "tempered_logger"])]
        (out / "stage3_lgs_0_8").mkdir(exist_ok=True)
        s08.to_csv(out / "stage3_lgs_0_8" / "stage2_learned_rows.csv", index=False)
        # reward-model tests: learned rows of the new runs
        oracle = load_oracle(ORACLE)
        rm = out / "reward_model"
        rm.mkdir(exist_ok=True)
        for setting, names in REWARD_MODEL.items():
            if _existing(names):
                learned_recovery(load_learned(*_existing(names)), oracle).to_csv(rm / f"learned_rows_{setting}.csv", index=False)
        # the old counterparts (buggy logs), rebuilt with the current loader, so that the report needs no run folder
        (out / "old").mkdir(exist_ok=True)
        for setting, (run, pattern) in OLD_REWARD_MODEL.items():
            if (RUNS / run).exists():
                learned_rows_matching(RUNS / run, ORACLE, pattern).to_csv(out / "old" / f"reward_model_{setting}.csv",
                                                                         index=False)
        if (RUNS / OLD_SU).exists():
            learned_recovery(load_learned(RUNS / OLD_SU), oracle).to_csv(out / "old" / "su_shrink100.csv", index=False)
        if (RUNS / M5_OPC).exists():
            (out / "m5").mkdir(exist_ok=True)
            learned_recovery(load_learned(RUNS / M5_OPC), oracle).to_csv(out / "m5" / "learned_rows_m5_opc_side.csv",
                                                                         index=False)
        if _existing(OLDSPACE):
            (out / "decomposition").mkdir(exist_ok=True)
            learned_recovery(load_learned(*_existing(OLDSPACE)), oracle).to_csv(
                out / "decomposition" / "learned_rows_oldspace.csv", index=False)
    print(f"wrote the Phase 3 summaries to {out}")


if __name__ == "__main__":
    main()
