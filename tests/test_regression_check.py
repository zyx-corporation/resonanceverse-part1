from experiments.regression_check import (
    check_efficiency_regression,
    check_instrument_regression,
)


def test_regression_passes_when_within_factor() -> None:
    base = {
        "rows": [
            {
                "seq_len": 32,
                "full_quadratic": {"mean_time_s": 0.01},
                "roi_tiers": {"mean_time_s": 0.02},
            }
        ]
    }
    cur = {
        "rows": [
            {
                "seq_len": 32,
                "full_quadratic": {"mean_time_s": 0.05},
                "roi_tiers": {"mean_time_s": 0.1},
            }
        ]
    }
    assert check_efficiency_regression(base, cur, max_slowdown=10.0) == []


def test_regression_fails_when_too_slow() -> None:
    base = {
        "rows": [
            {
                "seq_len": 32,
                "full_quadratic": {"mean_time_s": 0.01},
                "roi_tiers": {"mean_time_s": 0.02},
            }
        ]
    }
    cur = {
        "rows": [
            {
                "seq_len": 32,
                "full_quadratic": {"mean_time_s": 1.0},
                "roi_tiers": {"mean_time_s": 0.02},
            }
        ]
    }
    errs = check_efficiency_regression(base, cur, max_slowdown=10.0)
    assert len(errs) == 1
    assert "full_quadratic" in errs[0]


def test_instrument_regression_passes() -> None:
    base = {
        "stages": [
            {"name": "a", "elapsed_s": 0.01},
            {"name": "b", "elapsed_s": 0.02},
        ]
    }
    cur = {
        "stages": [
            {"name": "a", "elapsed_s": 0.05},
            {"name": "b", "elapsed_s": 0.1},
        ]
    }
    assert check_instrument_regression(base, cur, max_slowdown=10.0) == []


def test_instrument_regression_fails_on_slow_stage() -> None:
    base = {"stages": [{"name": "a", "elapsed_s": 0.01}]}
    cur = {"stages": [{"name": "a", "elapsed_s": 1.0}]}
    errs = check_instrument_regression(base, cur, max_slowdown=10.0)
    assert len(errs) == 1
