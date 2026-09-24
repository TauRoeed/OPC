"""Memory-aware worker planning (scheduling only; no GPU needed)."""

from training.memory_budget import (
    estimate_worker_peak_bytes,
    max_train_batch,
    plan_worker_groups,
    workers_that_fit,
)

GB = 1024**3
ML_SHAPE = (6_040, 3_533)
MSD_SHAPE = (1_019_318, 42_053)


def test_max_batch_follows_optuna_schedule():
    assert max_train_batch({"train_sizes": [5_000]}) == 2048
    assert max_train_batch({"train_sizes": [5_000, 10_000_000]}) == 327_680
    assert max_train_batch({"train_sizes": [5_000], "optuna_batch_sizes": [65_536]}) == 65_536


def test_estimates_cover_measured_peaks():
    # Measured on an RTX 6000 Ada: ml 10M train ~20.5 GB, msd 1M train ~16.7 GB.
    assert estimate_worker_peak_bytes({"train_sizes": [10_000_000]}, ML_SHAPE) > 20.5 * GB
    assert estimate_worker_peak_bytes({"train_sizes": [1_000_000]}, MSD_SHAPE) > 16.7 * GB
    small = estimate_worker_peak_bytes({"train_sizes": [5_000]}, ML_SHAPE)
    assert small < 2 * GB


def test_workers_fit_round_robin_over_devices():
    assert workers_that_fit(10 * GB, [48 * GB]) == 4          # floor(0.9 * 48 / 10)
    assert workers_that_fit(10 * GB, [48 * GB, 24 * GB]) == 5  # round-robin: 3 on the 48 GB, 2 on the 24 GB
    assert workers_that_fit(100 * GB, [48 * GB]) == 1         # never below one worker


def test_plan_groups_by_size_and_respects_max_workers():
    small = [{"run_key": f"small{i}", "train_sizes": [5_000]} for i in range(40)]
    big = [{"run_key": "big", "train_sizes": [10_000_000]}]
    cfgs = small[:20] + big + small[20:]
    plan = plan_worker_groups(cfgs, max_workers=30, capacities=[48 * GB], shape_fn=lambda c: ML_SHAPE)
    small_fit = workers_that_fit(estimate_worker_peak_bytes(small[0], ML_SHAPE), [48 * GB])
    assert 1 < small_fit < 30  # memory, not --max-workers, is the binding limit here
    assert [w for w, _ in plan] == [small_fit, 1]
    assert [c["run_key"] for c in plan[0][1]] == [c["run_key"] for c in small]
    assert [c["run_key"] for c in plan[1][1]] == ["big"]
    plan = plan_worker_groups(cfgs, max_workers=3, capacities=[48 * GB], shape_fn=lambda c: ML_SHAPE)
    assert [w for w, _ in plan] == [3, 1]  # --max-workers stays the upper bound
    plan = plan_worker_groups(small[:2], max_workers=30, capacities=[48 * GB], shape_fn=lambda c: ML_SHAPE)
    assert [w for w, _ in plan] == [2]  # never more workers than conditions


def test_unknown_capacity_keeps_max_workers():
    cfgs = [{"run_key": "a", "train_sizes": [5_000]}]
    assert plan_worker_groups(cfgs, max_workers=7, capacities=None) == [(7, cfgs)]


def test_extra_gpu_slots_fall_back_to_device_zero():
    # --num-gpus 2 on a 1-GPU machine: slot 1 maps to device 0, so it holds all workers.
    assert workers_that_fit(10 * GB, [48 * GB], n_slots=2) == 4
    # Two real devices, three slots: device 0 gets slots 0 and 2.
    assert workers_that_fit(10 * GB, [48 * GB, 48 * GB], n_slots=3) == 6
