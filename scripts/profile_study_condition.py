#!/usr/bin/env python3
"""Profile one mid-size study condition (OPC arm) and print top hotspots."""

from __future__ import annotations

import argparse
import cProfile
import pstats
from io import StringIO
from pathlib import Path

from training.run_full_study import _run_condition


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb-dir", type=Path, default=Path("BPR/embeddings"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/oom_smoke/profile_mid"))
    p.add_argument("--train-size", type=int, default=100_000)
    p.add_argument("--val-size", type=int, default=50_000)
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--require-cuda", action="store_true", default=True)
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    run_dir = out / "condition"
    run_dir.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    profiler.enable()
    _run_condition(
        dataset_name="ml",
        emb_dir=args.emb_dir,
        noise_mode="kmeans_templates",
        noise_axis="context",
        noise_level="high",
        noise_component="cluster",
        ctr=0.05,
        seed=int(args.seed),
        train_sizes=[int(args.train_size)],
        n_trials=int(args.n_trials),
        batch_size=2048,
        val_size=int(args.val_size),
        val_frac=0.15,
        val_min=5000,
        val_max=None,
        policy_reward_mode="exact",
        policy_reward_mc_sim=8,
        policy_temperature=1.0,
        run_dir=run_dir,
        slim=True,
        policy_loss_types=("sndr",),
        search_use_log_trick=True,
        shared_regression_size=25_000,
        qhat_user_chunk=5000,
        qhat_action_chunk=5000,
        require_cuda=bool(args.require_cuda),
        optuna_batch_sizes=None,
        methods=("opc",),
        logging_uniform_mix=0.0,
        optuna_selection="ci_low",
        reward_model="regression",
    )
    profiler.disable()

    stats_path = out / "profile.pstats"
    profiler.dump_stats(str(stats_path))

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(40)
    report = stream.getvalue()
    (out / "profile_top40.txt").write_text(report, encoding="utf-8")

    # Also by tottime for self-time hotspots
    stream2 = StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.sort_stats("tottime")
    stats2.print_stats(25)
    self_report = stream2.getvalue()
    (out / "profile_tottime25.txt").write_text(self_report, encoding="utf-8")

    print(report)
    print("--- tottime top 25 ---")
    print(self_report)
    print(f"wrote {stats_path}")
    print(f"wrote {out / 'profile_top40.txt'}")
    print(f"wrote {out / 'profile_tottime25.txt'}")


if __name__ == "__main__":
    main()
