"""Memory-aware worker limits for the parallel runners.

Each worker runs one condition. Its peak memory is dominated by training steps, which
hold several (batch × catalog) fp32 matrices at once (logits, softmax, q_hat rows,
their products and gradients). We estimate that peak per condition from the largest
batch the Optuna search can pick, then cap how many workers run concurrently so they
fit in free GPU memory (or in RAM on a CPU-only machine). Scheduling only: results are
unchanged because every condition seeds itself.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from training.trainer_trials import _qhat_materialize_limit_bytes, batch_schedule

BYTES_F32 = 4
# Live (batch × catalog) fp32 matrices per training step; calibrated on measured peaks
# (ml 10M train, batch 327,680: ~20.5 GB; msd 1M train, batch 16,384: ~16.7 GB).
PEAK_BATCH_MATRICES = 6
FIXED_OVERHEAD_BYTES = int(1.5 * 1024**3)  # CUDA context, embeddings, caches, sampler blocks
SAFETY_FRACTION = 0.9


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


def estimate_worker_peak_bytes(cfg: dict, shape: tuple[int, int]) -> int:
    n_users, n_actions = shape
    batch = max_train_batch(cfg)
    train_step = PEAK_BATCH_MATRICES * batch * n_actions * BYTES_F32
    dense_qhat = n_users * n_actions * BYTES_F32
    if dense_qhat > _qhat_materialize_limit_bytes():
        dense_qhat = 0  # lazy q_hat: rows are built per batch (counted in train_step)
    return int(train_step + dense_qhat + FIXED_OVERHEAD_BYTES)


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


def workers_that_fit(peak_bytes: int, capacities: list[int], n_slots: int | None = None) -> int:
    """Most workers whose round-robin placement fits every device.

    Worker k gets GPU slot ``k % n_slots``; slots beyond the real devices fall back to
    device 0 (as ``OPC_WORKER_GPU`` does), so that device hosts the extra workers.
    """
    n_slots = max(1, int(n_slots or len(capacities)))
    budget = [int(c * SAFETY_FRACTION) for c in capacities]
    load = [0] * len(capacities)
    workers = 0
    while workers < 100_000:
        slot = workers % n_slots
        dev = slot if slot < len(capacities) else 0
        if (load[dev] + 1) * peak_bytes > budget[dev]:
            break
        load[dev] += 1
        workers += 1
    return max(1, workers)


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
    shape_fn = shape_fn or (lambda cfg: catalog_shape(cfg["emb_dir"], cfg["dataset_name"]))
    groups: dict[int, list[dict]] = defaultdict(list)
    for cfg in run_configs:
        peak = estimate_worker_peak_bytes(cfg, shape_fn(cfg))
        fit = workers_that_fit(peak, capacities, n_slots)
        groups[min(int(max_workers), fit)].append(cfg)
    # Never start more workers than a group has conditions.
    return [(min(w, len(g)), g) for w, g in sorted(groups.items(), key=lambda kv: -kv[0])]


def describe_plan(plan, kind: str, capacities: list[int] | None, max_workers: int) -> str:
    if capacities is None:
        return f"[memory] capacity unknown on {kind}; using --max-workers {max_workers}"
    cap = ", ".join(f"{c / 1024**3:.1f}" for c in capacities)
    lines = [f"[memory] free {kind} memory (GB): {cap}; requested --max-workers {max_workers}"]
    for workers, cfgs in plan:
        lines.append(f"  {len(cfgs)} condition(s) -> {workers} concurrent worker(s)")
    return "\n".join(lines)
