"""v0.4 hooks: cost models, annualisation, vol estimators, covariance strategies and risk reports."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from btlab import (
    CovarianceStrategy, CrossSectionalMomentum, FlatBps, SpreadImpactCost, TimeSeriesMomentum, check_lookahead,
    metrics, risk_report, run_backtest, synthetic_dollar_volume, synthetic_market,
)

PRICES = synthetic_market(n_assets=30, n_years=4, momentum_strength=0.15, delist_rate=0.02, seed=3)
WEIGHTS = CrossSectionalMomentum(min_assets=5).weights(PRICES)


def test_flat_bps_reproduces_the_cost_arguments():
    old = run_backtest(PRICES, WEIGHTS, cost_bps=7, slippage_bps=2, borrow_bps=50)
    new = run_backtest(PRICES, WEIGHTS, cost_model=FlatBps(7, 2), borrow_bps=50)
    np.testing.assert_allclose(new.returns, old.returns, rtol=0, atol=1e-15)
    np.testing.assert_allclose(new.costs, old.costs, rtol=0, atol=1e-15)


def test_spread_impact_cost_reproduces_the_square_root_impact_path():
    volume = synthetic_dollar_volume(PRICES, median_adv=2e7, seed=3)
    old = run_backtest(PRICES, WEIGHTS, cost_bps=5, dollar_volume=volume, capital=3e8, impact_coef=0.8)
    model = SpreadImpactCost.from_dollar_volume(PRICES, volume, half_spread_bps=5, impact_coef=0.8, capital=3e8)
    new = run_backtest(PRICES, WEIGHTS, cost_model=model)
    assert old.costs.sum() > 2 * run_backtest(PRICES, WEIGHTS, cost_bps=5).costs.sum()  # impact matters here
    np.testing.assert_allclose(new.costs, old.costs, rtol=1e-12, atol=1e-18)


def test_spread_only_cost_warns_and_equals_flat_bps():
    with pytest.warns(UserWarning, match="half-spread only"):
        model = SpreadImpactCost(half_spread_bps=4)
    a = run_backtest(PRICES, WEIGHTS, cost_model=model).costs
    b = run_backtest(PRICES, WEIGHTS, cost_model=FlatBps(4)).costs
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-18)
    with pytest.raises(ValueError, match="vol is required"):
        SpreadImpactCost(5, adv=1e7)


def test_cost_model_cannot_be_combined_with_cost_arguments():
    with pytest.raises(ValueError, match="cost_model replaces"):
        run_backtest(PRICES, WEIGHTS, cost_bps=10, cost_model=FlatBps(10))


def test_periods_per_year_flows_into_summary():
    weekly = PRICES.iloc[::5]
    res = run_backtest(weekly, CrossSectionalMomentum(lookback=52, skip=4, min_assets=5).weights(weekly),
                       rebalance=4, periods_per_year=52)
    s = res.summary()
    assert s["sharpe"] == pytest.approx(metrics.sharpe(res.returns, 52), rel=1e-15)
    assert s["annual_vol"] == pytest.approx(metrics.annual_vol(res.returns, 52), rel=1e-15)
    assert run_backtest(PRICES, WEIGHTS).summary() == pytest.approx(metrics.summary(
        run_backtest(PRICES, WEIGHTS).returns, run_backtest(PRICES, WEIGHTS).turnover), rel=0)


def test_vol_estimator_default_is_the_rolling_std():
    rolling = TimeSeriesMomentum(vol_estimator=lambda r: r.rolling(63, min_periods=31).std() * np.sqrt(252))
    pd.testing.assert_frame_equal(rolling.weights(PRICES), TimeSeriesMomentum().weights(PRICES))
    ewma = TimeSeriesMomentum(vol_estimator=lambda r: r.ewm(alpha=0.06).std() * np.sqrt(252))
    assert not ewma.weights(PRICES).equals(TimeSeriesMomentum().weights(PRICES))
    assert check_lookahead(ewma, PRICES)[0]


def inverse_variance(cov):
    w = 1 / np.diag(cov)
    return w / w.sum()


def test_covariance_strategy_is_causal_and_holds_between_dates():
    strat = CovarianceStrategy(lambda x: np.cov(x, rowvar=False), inverse_variance, window=126, every=21)
    w = strat.weights(PRICES)
    assert check_lookahead(strat, PRICES)[0]
    changes = np.flatnonzero(w.diff().abs().sum(axis=1).to_numpy() > 1e-12)
    computed = set(range(20, len(PRICES), 21))
    assert all(i in computed or PRICES.iloc[i].isna().any() or PRICES.iloc[i - 1].isna().any() for i in changes)
    assert (w.iloc[:125] == 0).all().all()
    live = w.iloc[125:].sum(axis=1)
    assert live.max() == pytest.approx(1.0) and (live <= 1 + 1e-12).all()
    # the decision made on position 21k-1 is traded the next day by the matching engine schedule
    res = run_backtest(PRICES, w, rebalance=21, execution_lag=1)
    d = res.weights.index[-1]
    i = PRICES.index.get_loc(d)
    assert i % 21 == 0
    np.testing.assert_allclose(res.weights.loc[d].to_numpy(), w.iloc[i - 1].to_numpy(), atol=1e-12)


def test_risk_report_kupiec():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 5000), index=pd.bdate_range("2000-01-03", periods=5000))
    exact = risk_report(r, lambda x: pd.Series(-0.01 * stats.norm.ppf(0.01), x.index), alpha=0.01)
    assert abs(exact["exceptions"] - exact["expected"]) < 3 * np.sqrt(exact["expected"])
    assert exact["kupiec_p"] > 0.01
    too_low = risk_report(r, lambda x: pd.DataFrame({"var": 0.015}, index=x.index), alpha=0.01)
    assert too_low["exceptions"] > 3 * too_low["expected"] and too_low["kupiec_p"] < 1e-6
    default = risk_report(run_backtest(PRICES, WEIGHTS))
    assert default["n"] > 0 and default["var"].notna().sum() == default["n"]
