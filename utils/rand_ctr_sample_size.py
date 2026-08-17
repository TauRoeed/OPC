"""Sample size for estimating rand_CTR (mean reward under uniform exposure)."""

from __future__ import annotations

import argparse
import math


def n_for_rand_ctr(
    *,
    eps: float = 0.01,
    alpha: float = 0.05,
    rho_hat: float | None = None,
    n_users: int | None = None,
    per_user: float = 10.0,
) -> dict[str, float | int]:
    """
    Wald sample size for a Bernoulli mean, plus optional coverage floor.

    If rho_hat is None, use worst-case variance rho=0.5.
    """
    if not (0.0 < eps < 1.0):
        raise ValueError("eps must be in (0, 1)")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    # Approx normal quantile for two-sided 1-alpha (common values).
    z_table = {0.10: 1.64485, 0.05: 1.95996, 0.01: 2.57583}
    z = z_table.get(round(alpha, 2))
    if z is None:
        # Crude fallback via erfinv-free approximation is avoided; require known alpha.
        raise ValueError("alpha must be one of {0.10, 0.05, 0.01}")

    if rho_hat is None:
        var = 0.25
        rho_used = 0.5
    else:
        rho_used = float(rho_hat)
        if not (0.0 <= rho_used <= 1.0):
            raise ValueError("rho_hat must be in [0, 1]")
        var = rho_used * (1.0 - rho_used)

    n_precision = math.ceil((z * z * var) / (eps * eps))
    n_coverage = 0
    if n_users is not None:
        n_coverage = math.ceil(float(per_user) * int(n_users))
    n_rec = max(n_precision, n_coverage)
    return {
        "eps": float(eps),
        "alpha": float(alpha),
        "z": float(z),
        "rho_used": float(rho_used),
        "n_precision": int(n_precision),
        "n_coverage": int(n_coverage),
        "n_recommended": int(n_rec),
        "n_users": int(n_users) if n_users is not None else 0,
        "per_user": float(per_user),
    }


def density_regime(rho_hat: float) -> str:
    if rho_hat < 0.03:
        return "SparseReward"
    if rho_hat < 0.10:
        return "ModerateReward"
    return "DenseReward"


def main():
    p = argparse.ArgumentParser(
        description="Recommend n random exposures to estimate rand_CTR."
    )
    p.add_argument("--eps", type=float, default=0.01, help="Absolute error target.")
    p.add_argument("--alpha", type=float, default=0.05, help="One of 0.10, 0.05, 0.01.")
    p.add_argument(
        "--rho-hat",
        type=float,
        default=None,
        help="Optional pilot estimate of rand_CTR (else use 0.5 worst-case).",
    )
    p.add_argument("--n-users", type=int, default=None)
    p.add_argument("--n-items", type=int, default=None, help="Informational only.")
    p.add_argument(
        "--per-user",
        type=float,
        default=10.0,
        help="Coverage floor: per_user * n_users.",
    )
    args = p.parse_args()

    out = n_for_rand_ctr(
        eps=args.eps,
        alpha=args.alpha,
        rho_hat=args.rho_hat,
        n_users=args.n_users,
        per_user=args.per_user,
    )
    print("rand_CTR sample size")
    for k, v in out.items():
        print(f"  {k}: {v}")
    if args.n_items is not None and args.n_users is not None:
        matrix = int(args.n_users) * int(args.n_items)
        frac = out["n_recommended"] / matrix if matrix else float("nan")
        print(f"  matrix_|U|x|I|: {matrix}")
        print(f"  frac_of_matrix: {frac:.6f}")
    if args.rho_hat is not None:
        print(f"  density_regime: {density_regime(float(args.rho_hat))}")
    else:
        print("  density_regime: (provide --rho-hat after a pilot)")


if __name__ == "__main__":
    main()
