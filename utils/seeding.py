"""Single-seed reproducibility.

One experiment seed (``--seeds`` in the study runners) drives every RNG. Sub-seeds
are derived from it with a stable hash of labels, so a condition's results do not
depend on run order, worker process, or which other methods/conditions ran first.
"""

from __future__ import annotations

import os
import random
import zlib

import numpy as np
import torch

# cuBLAS reads this when its handle is created; set it before any CUDA matmul.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_DETERMINISTIC = False


def derive_seed(seed: int, *labels) -> int:
    """Stable 32-bit sub-seed for ``(seed, *labels)``; identical in every process."""
    key = zlib.crc32("|".join(str(x) for x in labels).encode("utf-8"))
    return int(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, key]).generate_state(1)[0])


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy's global RNG and torch (CPU and all CUDA devices)."""
    seed = int(seed) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def enable_determinism(enabled: bool = True) -> None:
    """Use deterministic torch/cuDNN/cuBLAS kernels so GPU runs repeat exactly."""
    global _DETERMINISTIC
    _DETERMINISTIC = bool(enabled)
    torch.use_deterministic_algorithms(_DETERMINISTIC, warn_only=True)
    torch.backends.cudnn.deterministic = _DETERMINISTIC
    if _DETERMINISTIC:
        torch.backends.cudnn.benchmark = False


def deterministic_enabled() -> bool:
    return _DETERMINISTIC


# CPU thread count changes floating-point summation order in BLAS/torch, which training
# then amplifies; a fixed count keeps serial, parallel and multi-machine runs identical.
DEFAULT_CPU_THREADS = 4
_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def pin_cpu_threads(n_threads: int = DEFAULT_CPU_THREADS) -> int:
    """Pin numpy/BLAS (via threadpoolctl) and torch intra-op threads; children inherit via env."""
    from threadpoolctl import threadpool_limits

    n = max(1, int(n_threads))
    for key in _THREAD_ENV_VARS:
        os.environ[key] = str(n)
    threadpool_limits(limits=n)
    torch.set_num_threads(n)
    return n


def optuna_sampler(seed: int, *labels):
    """Seeded TPE sampler (Optuna's default sampler, made reproducible)."""
    import optuna

    return optuna.samplers.TPESampler(seed=derive_seed(seed, "optuna", *labels))
