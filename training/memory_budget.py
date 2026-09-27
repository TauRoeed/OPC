"""Memory-aware worker limits for the parallel runners.

Each worker runs one condition. Its peak memory is its training steps, which hold several
(batch × catalog) fp32 matrices at once (logits, softmax, q_hat rows, their products and
gradients), plus the dense (users × catalog) q_hat matrices it keeps on the device, plus
the allocator's reserve and the process's own context. We estimate that peak per condition
from the largest batch the Optuna search can pick and the condition's reward-model budget,
then cap how many workers run concurrently so they fit in free GPU memory (or in RAM on a
CPU-only machine): per device, floor(SAFETY_FRACTION × free / peak), at least one and at
most --max-workers. Scheduling only: results are unchanged because every condition seeds
itself.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from training.trainer_trials import _qhat_materialize_limit_bytes, batch_schedule

BYTES_F32 = 4
GIB = 1024**3
# Live (batch × catalog) fp32 matrices per training step; calibrated on measured peaks
# (ml 10M train, batch 327,680: ~20.5 GB; msd 1M train, batch 16,384: ~16.7 GB).
PEAK_BATCH_MATRICES = 6
# Margin on the live tensors (training step + dense q_hat copies):
#   ALLOCATOR_RESERVE multiplies them: PyTorch's caching allocator keeps freed blocks for reuse
#     (Optuna varies the batch size per trial, which fragments the pool), and each train size's
#     q_hat lookup is built while the previous size's two are still referenced (the per-size
#     rebinding in trainer_trials), so a third dense copy exists transiently;
#   CONTEXT_BYTES is per process: CUDA context and kernels, embeddings, sampler blocks.
# Calibration (RTX 6000 Ada, 48 GB, WSL2, 2026-09-27): anime (73,417 users × 10,803 items, train
# 100k, batch 8,192, 5-fold cross-fitting) holds 2.0 GiB of training-step matrices and two dense
# q_hat copies of 3.0 GiB each, 7.9 GiB live; it was observed at ≈ 12.5 GB per worker (4 workers
# demanded ~50 GB and spilled into shared memory): 1.4 × 7.9 + 1.5 ≈ 12.5 GiB.
ALLOCATOR_RESERVE = 1.4
CONTEXT_BYTES = int(1.5 * GIB)
# Share of the queried free memory the planner assigns. The query is a snapshot at launch, and
# under Windows / WSL2 the driver does not raise an out-of-memory error when dedicated memory runs
# out: it spills into shared system memory, silently and much slower (all workers on the device
# slow down). With the calibration above, ~47 GiB free admits 2 anime workers: 3 would leave under
# 10 GB of headroom, and 4 spilled.
SAFETY_FRACTION = 0.75


def catalog_shape(emb_dir: str | Path, dataset: str) -> tuple[int, int]:
    """(n_users, n_actions) from the embedding files, without loading them."""
    emb_dir = Path(emb_dir)
    n_users = np.load(emb_dir / f"{dataset}_user_factors.npy", mmap_mode="r").shape[0]
    n_actions = np.load(emb_dir / f"{dataset}_item_factors.npy", mmap_mode="r").shape[0]
    return int(n_users), int(n_actions)


def max_train_batch(cfg: dict) -> int:
    """Largest training batch the condition can use (Optuna choices and --batch-size)."""
    sizes = [int(x) for x in cfg.get("optuna_batch_sizes") or []]
    for ts in cfg.get("train_sizes") or []:
        default, choices = batch_schedule(int(ts))
        sizes += [default, *choices]
    if cfg.get("batch_size"):
        sizes.append(int(cfg["batch_size"]))
    return max(sizes) if sizes else 1024


def dense_qhat_copies(cfg: dict, shape: tuple[int, int]) -> int:
    """Dense (users × catalog) fp32 q_hat matrices a worker keeps on its device (trainer_trials):
    the shared lookup's copy, plus, with cross-fitting (``crossfit_folds``, which needs train-mode
    reward data), the out-of-fold matrix built next to it. Zero when q_hat is larger than the
    dense-materialize limit (the same test as fit_shared_regression_bundle): it then stays in
    linear form and its rows are built per batch (counted in the training step). The fold models
    are never materialized."""
    n_users, n_actions = shape
    if n_users * n_actions * BYTES_F32 > _qhat_materialize_limit_bytes():
        return 0
    return 2 if int(cfg.get("crossfit_folds", 0) or 0) > 0 else 1  # the worker's own default: no cross-fitting


def estimate_worker_peak_bytes(cfg: dict, shape: tuple[int, int]) -> int:
    n_users, n_actions = shape
    train_step = PEAK_BATCH_MATRICES * max_train_batch(cfg) * n_actions * BYTES_F32
    dense_qhat = dense_qhat_copies(cfg, shape) * n_users * n_actions * BYTES_F32
    return int(ALLOCATOR_RESERVE * (train_step + dense_qhat) + CONTEXT_BYTES)


def _cuda_free_bytes_worker(num_gpus: int, queue) -> None:
    import torch

    n = min(int(num_gpus), torch.cuda.device_count())  # --num-gpus may exceed real devices
    queue.put([int(torch.cuda.mem_get_info(i)[0]) for i in range(n)])


def _host_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def device_capacities(num_gpus: int) -> tuple[str, list[int] | None]:
    """("gpu", free bytes per GPU) or ("cpu", [available RAM]); capacity None if unknown.

    GPU memory is queried in a short-lived child process so the parent does not keep
    a CUDA context (and its memory) on every device.
    """
    import torch

    if torch.cuda.is_available() and num_gpus > 0:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target=_cuda_free_bytes_worker, args=(num_gpus, queue))
        try:
            proc.start()
            free = queue.get(timeout=120)
        except Exception as exc:  # query failed: fall back to --max-workers (no cap)
            print(f"[memory] WARNING: could not query free GPU memory ({exc!r}); no cap", flush=True)
            free = None
        finally:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
        return "gpu", free
    host = _host_available_bytes()
    return "cpu", None if host is None else [host]


def _device_of_worker(k: int, n_devices: int, n_slots: int) -> int:
    slot = k % n_slots
    return slot if slot < n_devices else 0


def workers_per_device(workers: int, n_devices: int, n_slots: int | None = None) -> list[int]:
    """How many of ``workers`` land on each device."""
    n_slots = max(1, int(n_slots or n_devices))
    load = [0] * n_devices
    for k in range(int(workers)):
        load[_device_of_worker(k, n_devices, n_slots)] += 1
    return load


def workers_that_fit(peak_bytes: int, capacities: list[int], n_slots: int | None = None) -> int:
    """Most workers whose round-robin placement fits every device: each device holds at most
    floor(SAFETY_FRACTION × free / peak) of them. Never below one.

    Worker k gets GPU slot ``k % n_slots``; slots beyond the real devices fall back to
    device 0 (as ``OPC_WORKER_GPU`` does), so that device hosts the extra workers.
    """
    n_slots = max(1, int(n_slots or len(capacities)))
    budget = [int(c * SAFETY_FRACTION) for c in capacities]
    load = [0] * len(capacities)
    workers = 0
    while workers < 100_000:
        dev = _device_of_worker(workers, len(capacities), n_slots)
        if (load[dev] + 1) * peak_bytes > budget[dev]:
            break
        load[dev] += 1
        workers += 1
    return max(1, workers)


def _catalog_shape_of(cfg: dict) -> tuple[int, int]:
    return catalog_shape(cfg["emb_dir"], cfg["dataset_name"])


def plan_worker_groups(
    run_configs: list[dict],
    *,
    max_workers: int,
    capacities: list[int] | None,
    n_slots: int | None = None,
    shape_fn=None,
) -> list[tuple[int, list[dict]]]:
    """Group configs by how many copies fit; returns [(workers, configs)], widest first."""
    if capacities is None:
        return [(max_workers, list(run_configs))] if run_configs else []
    shape_fn = shape_fn or _catalog_shape_of
    groups: dict[int, list[dict]] = defaultdict(list)
    for cfg in run_configs:
        peak = estimate_worker_peak_bytes(cfg, shape_fn(cfg))
        fit = workers_that_fit(peak, capacities, n_slots)
        groups[min(int(max_workers), fit)].append(cfg)
    # Never start more workers than a group has conditions.
    return [(min(w, len(g)), g) for w, g in sorted(groups.items(), key=lambda kv: -kv[0])]


def describe_plan(plan, kind: str, capacities: list[int] | None, max_workers: int, *,
                  n_slots: int | None = None, shape_fn=None) -> str:
    """The plan per device (free and usable memory) and per dataset (estimated peak per worker,
    how many fit on each device, the workers chosen)."""
    if capacities is None:
        return f"[memory] capacity unknown on {kind}; using --max-workers {max_workers}"
    shape_fn = shape_fn or _catalog_shape_of
    devices = ", ".join(f"{kind} {i}: {c / GIB:.1f} free, {c * SAFETY_FRACTION / GIB:.1f} usable"
                        for i, c in enumerate(capacities))
    lines = [f"[memory] {devices} (GiB; safety fraction {SAFETY_FRACTION}); requested --max-workers {max_workers}"]
    for workers, cfgs in plan:
        by_dataset: dict[str, list[dict]] = defaultdict(list)
        for cfg in cfgs:
            by_dataset[str(cfg.get("dataset_name", "?"))].append(cfg)
        placed = workers_per_device(workers, len(capacities), n_slots)
        for name, group in by_dataset.items():
            shapes = [shape_fn(c) for c in group]
            peak = max(estimate_worker_peak_bytes(c, s) for c, s in zip(group, shapes))
            copies = max(dense_qhat_copies(c, s) for c, s in zip(group, shapes))
            fit = [int(c * SAFETY_FRACTION // peak) for c in capacities]
            lines.append(
                f"[memory]   {name} ({shapes[0][0]:,} × {shapes[0][1]:,}; {copies} dense q_hat cop{'y' if copies == 1 else 'ies'}): "
                f"est. peak {peak / GIB:.1f} GiB/worker; fit per device {fit}; {len(group)} condition(s) -> "
                f"{workers} concurrent worker(s) (per device {placed})")
    return "\n".join(lines)
