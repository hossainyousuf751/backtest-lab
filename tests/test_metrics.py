import numpy as np
import pandas as pd
import pytest

from btlab import (
    bootstrap_sharpe_ci, deflated_sharpe, expected_max_sharpe, factor_attribution, max_drawdown,
    probabilistic_sharpe, sharpe,
)


def test_sharpe_of_known_series():
    r = np.tile([0.01, -0.005], 500)
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe(r) == pytest.approx(expected)


def test_max_drawdown():
    assert max_drawdown([0.1, -0.5, 0.2]) == pytest.approx(-0.5)


def test_psr_is_half_at_own_sharpe():
    r = np.random.default_rng(0).normal(0.0005, 0.01, 2000)
    assert probabilistic_sharpe(r, sharpe(r)) == pytest.approx(0.5, abs=1e-6)


def test_expected_max_sharpe_grows_with_trials():
    vals = [expected_max_sharpe(n, 0.3) for n in (2, 10, 100, 1000)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0004, 0.01, 2520)
    trials = rng.normal(0, 0.3, 200)
    assert deflated_sharpe(r, trials) < probabilistic_sharpe(r, 0.0)


def test_factor_attribution_recovers_known_loadings():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2010-01-01", periods=2520)
    f = pd.DataFrame(rng.normal(0, 0.01, (2520, 2)), idx, columns=["mkt", "mom"])
    r = pd.Series(0.0002 + 0.8 * f["mkt"] - 0.3 * f["mom"] + rng.normal(0, 0.002, 2520), idx)
    r.iloc[:100] = 0.0                                  # warm-up period is ignored
    t = factor_attribution(r, f)
    assert t.loc["mkt", "coef"] == pytest.approx(0.8, abs=0.02)
    assert t.loc["mom", "coef"] == pytest.approx(-0.3, abs=0.02)
    assert t.loc["alpha", "coef"] == pytest.approx(0.0504, abs=0.02)   # 2 bps/day annualised
    assert t.loc["alpha", "t_stat"] > 3 and t.attrs["r2"] > 0.9


def test_bootstrap_ci_covers_point_estimate():
    r = np.random.default_rng(2).normal(0.0004, 0.01, 2520)
    lo, hi = bootstrap_sharpe_ci(r, n_boot=500)
    assert lo < sharpe(r) < hi
