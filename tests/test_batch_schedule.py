"""Unit tests for train_size → batch schedule and runtime estimate."""

from training.trainer_trials import (
    batch_schedule,
    estimate_condition_runtime_s,
    format_runtime_estimate,
)


def test_batch_schedule_table():
    assert batch_schedule(10_000) == (1024, [512, 1024, 2048])
    assert batch_schedule(25_000) == (1024, [512, 1024, 2048])
    assert batch_schedule(100_000) == (4096, [2048, 4096, 8192])
    assert batch_schedule(500_000) == (8192, [4096, 8192, 16384])
    assert batch_schedule(2_000_000) == (16384, [8192, 16384, 32768])
    assert batch_schedule(2_000_001) == (32768, [16384, 32768, 65536])


def test_batch_schedule_monotone_default():
    sizes = [1_000, 50_000, 200_000, 1_000_000, 5_000_000]
    defaults = [batch_schedule(n)[0] for n in sizes]
    assert defaults == sorted(defaults)


def test_runtime_estimate_scales():
    a = estimate_condition_runtime_s(100_000, 1)
    b = estimate_condition_runtime_s(100_000, 10)
    c = estimate_condition_runtime_s(1_000_000, 10)
    assert b["total_s"] > a["total_s"]
    assert c["total_s"] > b["total_s"]
    assert abs(b["trials_s"] - 10 * b["per_trial_s"]) < 1e-6
    # Calibrated ballpark: 100k × 1 trial ≈ 1–3 min on profiled GPU
    assert 40.0 < a["total_s"] < 200.0
    assert a["final_s"] == 0.0
    s = format_runtime_estimate(a)
    assert "wall" in s and "train=100000" in s
    assert "final≈" not in s


if __name__ == "__main__":
    test_batch_schedule_table()
    test_batch_schedule_monotone_default()
    test_runtime_estimate_scales()
    print("ok")
