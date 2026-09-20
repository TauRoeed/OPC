#!/usr/bin/env python3
"""Estimate wall time for one OPC study condition from train_size × n_trials."""

from __future__ import annotations

import argparse
import json

from training.trainer_trials import (
    batch_schedule,
    estimate_condition_runtime_s,
    format_runtime_estimate,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-size", type=int, required=True)
    p.add_argument("--n-trials", type=int, required=True)
    p.add_argument("--val-size", type=int, default=None)
    p.add_argument("--shared-regression-size", type=int, default=50_000)
    p.add_argument(
        "--methods",
        type=int,
        default=1,
        help="1=OPC only, 2=OPC+no-prop (roughly doubles).",
    )
    p.add_argument("--slim", action="store_true", default=True)
    p.add_argument("--no-slim", action="store_false", dest="slim")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    default_b, choices = batch_schedule(int(args.train_size))
    est = estimate_condition_runtime_s(
        int(args.train_size),
        int(args.n_trials),
        val_size=args.val_size,
        shared_regression_size=int(args.shared_regression_size),
        n_methods=int(args.methods),
        slim=bool(args.slim),
    )
    if args.json:
        print(
            json.dumps(
                {
                    **est,
                    "batch_default": default_b,
                    "batch_choices": choices,
                    "human": format_runtime_estimate(est),
                },
                indent=2,
            )
        )
    else:
        print(format_runtime_estimate(est))
        print(f"batch_schedule: default={default_b} choices={choices}")


if __name__ == "__main__":
    main()
