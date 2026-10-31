import numpy as np
import pandas as pd
import pytest

from btlab import rebalance_dates, run_backtest
from btlab.engine import asset_returns

IDX = pd.bdate_range("2020-01-01", periods=60)


def prices_from_returns(rets: dict) -> pd.DataFrame:
    df = pd.DataFrame(rets, index=IDX)
    return 100 * (1 + df).cumprod()


def test_execution_lag_delays_exposure():
    rng = np.random.default_rng(0)
    p = prices_from_returns({"A": rng.normal(0, 0.01, 60)})
    r = asset_returns(p)["A"]
    w = pd.DataFrame({"A": 1.0}, index=IDX)
    lag1 = run_backtest(p, w, rebalance="D", cost_bps=0, execution_lag=1).returns
    lag0 = run_backtest(p, w, rebalance="D", cost_bps=0, execution_lag=0).returns
    np.testing.assert_allclose(lag1.iloc[:2], 0.0)
    np.testing.assert_allclose(lag1.iloc[2:], r.iloc[2:], atol=1e-12)
    np.testing.assert_allclose(lag0.iloc[1:], r.iloc[1:], atol=1e-12)


def test_costs_equal_turnover_times_bps():
    rng = np.random.default_rng(1)
    p = prices_from_returns({"A": rng.normal(0, 0.01, 60), "B": rng.normal(0, 0.01, 60)})
    w = pd.DataFrame({"A": np.where(np.arange(60) % 2, 1.0, 0.0), "B": 0.0}, index=IDX)
    res = run_backtest(p, w, rebalance="D", cost_bps=10)
    np.testing.assert_allclose(res.costs.sum(), res.turnover.sum() * 10 / 1e4)
    np.testing.assert_allclose(res.gross_returns - res.costs, res.returns)


def test_weights_drift_between_rebalances():
    p = prices_from_returns({"A": np.full(60, 0.01), "B": np.zeros(60)})
    w = pd.DataFrame({"A": 0.5, "B": 0.5}, index=IDX)
    res = run_backtest(p, w, rebalance=20, cost_bps=0, execution_lag=0)
    # after rebalancing back to 50/50, turnover equals the drift away from 50/50
    growth = 1.01 ** 20
    drifted_a = 0.5 * growth / (0.5 * growth + 0.5)
    assert res.turnover.iloc[1] == pytest.approx(2 * (drifted_a - 0.5), rel=1e-9)


def test_delisted_position_takes_delisting_return_then_cash():
    rets = {"A": np.full(60, 0.001), "B": np.full(60, 0.001)}
    p = prices_from_returns(rets)
    p.loc[IDX[30]:, "A"] = np.nan                   # A delists on day 30
    w = pd.DataFrame({"A": 1.0, "B": 0.0}, index=IDX)
    res = run_backtest(p, w, rebalance="M", cost_bps=0, execution_lag=0, delisting_return=-0.3)
    assert res.returns.iloc[30] == pytest.approx(-0.3)
    assert (res.returns.iloc[31:] == 0).all()        # cash, and A cannot be re-bought


def test_square_root_impact():
    rng = np.random.default_rng(3)
    p = prices_from_returns({"A": rng.normal(0, 0.01, 60)})
    w = pd.DataFrame({"A": np.where(np.arange(60) % 2, 1.0, -1.0)}, index=IDX)
    dv = pd.DataFrame({"A": 1e7}, index=IDX)
    kw = dict(rebalance="D", cost_bps=0, dollar_volume=dv)
    res = run_backtest(p, w, capital=1e6, **kw)
    big = run_backtest(p, w, capital=4e6, **kw).costs
    np.testing.assert_allclose(big, 2 * res.costs)                 # cost per unit ~ sqrt(size)
    vol = asset_returns(p)["A"].iloc[:41].std()                     # 63-day window, all data so far
    trade = res.turnover.loc[IDX[40]]                               # ~2: flip from short to long
    assert res.costs.iloc[40] == pytest.approx(trade * vol * np.sqrt(trade * 1e6 / 1e7))
    assert run_backtest(p, w, rebalance="D", cost_bps=0).costs.sum() == 0


def test_borrow_fee_charged_daily_on_short_value_only():
    p = prices_from_returns({"A": np.zeros(60), "B": np.zeros(60)})
    w = pd.DataFrame({"A": -1.0, "B": 1.0}, index=IDX)
    res = run_backtest(p, w, cost_bps=0, execution_lag=0, borrow_bps=pd.Series({"A": 252.0, "B": 999.0}))
    first = res.turnover.index[0]
    np.testing.assert_allclose(res.costs.loc[first:].iloc[1:], 1e-4)  # 252 bps/yr = 1 bp/day on 100% short
    assert res.costs.loc[:first].sum() == 0


def test_cannot_buy_unlisted_assets():
    rng = np.random.default_rng(2)
    p = prices_from_returns({"A": rng.normal(0, 0.01, 60), "B": rng.normal(0, 0.01, 60)})
    p.loc[:IDX[39], "B"] = np.nan                    # B lists on day 40
    w = pd.DataFrame({"A": 0.0, "B": 1.0}, index=IDX)
    res = run_backtest(p, w, rebalance="D", cost_bps=0)
    assert (res.returns.iloc[:41] == 0).all()


def test_weekly_schedule_has_no_extra_rebalance_at_year_end():
    d = rebalance_dates(pd.bdate_range("2024-12-20", "2025-01-15"), "W")
    assert [x.strftime("%m-%d") for x in d] == ["12-20", "12-27", "01-03", "01-10", "01-15"]


def test_monthly_schedule_is_month_ends():
    d = rebalance_dates(pd.bdate_range("2021-01-01", "2021-06-30"), "M")
    assert len(d) == 6 and all(x.is_month_end or (x + pd.offsets.BDay()).month != x.month for x in d)


def test_point_in_time_universe_recipe():
    """Mask the strategy's view, not the engine's prices: index exits are sales, not delistings."""
    p = prices_from_returns({"A": np.full(60, 0.01), "B": np.zeros(60)})
    member = pd.DataFrame({"A": np.arange(60) < 30, "B": True}, index=IDX)
    view = p.where(member)
    w = view.notna().astype(float).div(view.notna().sum(axis=1), axis=0)   # equal-weight members
    res = run_backtest(p, w, rebalance="D", cost_bps=0, execution_lag=0)
    assert res.returns.iloc[30] == pytest.approx(0.005, rel=0.01)   # A's real return on exit day
    assert (res.returns.iloc[31:] == 0).all()                        # then sold: all in B
    wrong = run_backtest(view, w, rebalance="D", cost_bps=0, execution_lag=0, delisting_return=-1)
    assert wrong.returns.iloc[30] < -0.4                             # masked prices = fake delisting
