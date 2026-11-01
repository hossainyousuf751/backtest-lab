"""Vectorised backtest engine with execution lag, drift, costs and delistings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from btlab import metrics
from btlab.costs import CostModel


def rebalance_dates(index: pd.DatetimeIndex, schedule: str | int = "M") -> pd.DatetimeIndex:
    """'D' daily, 'W' last day of each week, 'M' last day of each month, or every n days."""
    if isinstance(schedule, int):
        return index[::schedule]
    if schedule == "D":
        return index
    if schedule not in ("D", "W", "M"):
        raise ValueError(f"rebalance must be 'D', 'W', 'M' or an int, got {schedule!r}")
    if schedule == "W":
        iso = index.isocalendar()                   # ISO year, not calendar year, at Dec/Jan
        key = iso.week.to_numpy() + 100 * iso.year.to_numpy()
    else:
        key = np.asarray(index.month + 100 * index.year)
    last = np.append(key[1:] != key[:-1], True)
    return index[last]


def listed_span(prices: pd.DataFrame) -> pd.DataFrame:
    """True from an asset's first price to its last, including gaps such as trading halts."""
    return prices.ffill().notna() & prices.bfill().notna()


def asset_returns(prices: pd.DataFrame, delisting_return: float = 0.0) -> pd.DataFrame:
    """Simple returns; a delisting day (price present, then NaN forever) gets `delisting_return`.

    Returns across a gap (a halt) are booked on the day trading resumes.
    """
    rets = prices.ffill().where(listed_span(prices)).pct_change(fill_method=None)
    listed_before = prices.shift(1).notna()
    gone = prices.isna() & listed_before & prices[::-1].isna().cummin()[::-1]
    rets = rets.mask(gone, delisting_return)
    return rets.fillna(0.0)


@dataclass
class BacktestResult:
    returns: pd.Series          # net daily returns
    gross_returns: pd.Series
    costs: pd.Series            # cost charged each day (fraction of NAV)
    turnover: pd.Series         # one-way turnover at each rebalance
    weights: pd.DataFrame       # executed weights at each rebalance
    periods_per_year: int = 252  # bars per year, for annualising (252 daily, 52 weekly, ...)

    @property
    def equity(self) -> pd.Series:
        return (1 + self.returns).cumprod()

    def summary(self) -> dict:
        return metrics.summary(self.returns, self.turnover, self.periods_per_year)


def run_backtest(prices: pd.DataFrame, target_weights: pd.DataFrame, rebalance: str | int = "M",
                 cost_bps: float = 5.0, slippage_bps: float = 0.0, execution_lag: int = 1,
                 delisting_return: float = 0.0, dollar_volume: pd.DataFrame | None = None,
                 capital: float = 1e8, impact_coef: float = 1.0,
                 borrow_bps: float | pd.Series = 0.0, cost_model: CostModel | None = None,
                 periods_per_year: int = 252) -> BacktestResult:
    """Simulate trading `target_weights` (fractions of NAV, row t decided at close t).

    * execution_lag=1 (default): weights decided at close t are traded at close
      t+1. execution_lag=0 assumes you can trade at the same close the signal
      used, which is optimistic with close-only data.
    * Between rebalances, positions drift with their returns.
    * Costs: (cost_bps + slippage_bps) per unit of one-way turnover.
    * Market impact (only if `dollar_volume` is given): the square-root law,
      impact_coef * daily_vol * sqrt(trade / ADV) per unit traded, with the
      trade sized from `capital` and ADV the trailing 20-day mean dollar
      volume. This is what makes a strategy's capacity finite.
    * Borrow: short positions pay `borrow_bps` per year on their (drifting)
      value, charged daily. Pass a Series for per-asset fees; hard-to-borrow
      names can cost several hundred bps. Assets missing from it pay nothing.
    * cost_model (optional, see btlab.costs) replaces cost_bps, slippage_bps
      and dollar_volume: it prices each rebalance's trade per asset, e.g.
      SpreadImpactCost with a spread and impact calibrated elsewhere. With a
      cost_model, leave those three at their defaults (anything else raises).
    * periods_per_year: bars per year (252 for daily). It sets the borrow
      accrual per bar and the annualisation in result.summary().
    * Assets that are not listed on a rebalance date cannot be bought; a
      delisted position becomes cash. A position inside a price gap (a halt)
      is held unchanged until prices resume.
    """
    if cost_model is not None and (cost_bps != 5.0 or slippage_bps != 0.0 or dollar_volume is not None):
        raise ValueError("cost_model replaces cost_bps, slippage_bps and dollar_volume; do not pass them too")
    target_weights = target_weights.reindex(index=prices.index, columns=prices.columns)
    rets = asset_returns(prices, delisting_return)
    listed = prices.notna()
    halted = listed_span(prices).to_numpy() & ~listed.to_numpy()
    targets = target_weights.shift(execution_lag)

    dates = prices.index
    rebal = [d for d in rebalance_dates(dates, rebalance) if targets.loc[d].notna().any()]
    R = rets.to_numpy()
    pos = {d: i for i, d in enumerate(dates)}
    gross = np.zeros(len(dates))
    costs = np.zeros(len(dates))
    turnover, executed = {}, {}
    w = np.zeros(prices.shape[1])
    bps = (cost_bps + slippage_bps) / 1e4
    borrow = pd.Series(borrow_bps, index=prices.columns, dtype=float).fillna(0.0).to_numpy() / 1e4 / periods_per_year
    if dollar_volume is not None:
        # ponytail: unknown vol or ADV (first days after listing) means zero impact
        vol = rets.where(listed).rolling(63, min_periods=5).std().fillna(0.0).to_numpy()
        # shift(1): the trade day's own volume is not known when the order is sized
        adv = dollar_volume.reindex_like(prices).rolling(20, min_periods=1).mean().shift(1).to_numpy()

    for j, d in enumerate(rebal):
        i0 = pos[d]
        alive = listed.iloc[i0].to_numpy()
        w = np.where(alive | halted[i0], w, 0.0)            # delisted holdings are now cash
        new = np.where(alive, targets.iloc[i0].fillna(0.0).to_numpy(), 0.0)
        new = np.where(halted[i0], w, new)                  # halted holdings cannot be traded
        trade = np.abs(new - w)
        to = trade.sum()
        turnover[d], executed[d] = to, new
        if cost_model is not None:
            context = {"assets": prices.columns, "returns": rets.iloc[:i0 + 1]}
            costs[i0] += float(np.nansum(cost_model.cost(new - w, d, context)))
        else:
            costs[i0] += to * bps                           # += keeps that day's borrow fee
        if dollar_volume is not None:
            participation = np.divide(trade * capital, adv[i0], out=np.zeros_like(trade),
                                      where=adv[i0] > 0)
            costs[i0] += impact_coef * np.nansum(trade * vol[i0] * np.sqrt(participation))

        i1 = pos[rebal[j + 1]] if j + 1 < len(rebal) else len(dates) - 1
        if i1 > i0:
            growth = np.cumprod(1 + R[i0 + 1:i1 + 1], axis=0)
            nav = 1 + (growth - 1) @ new
            gross[i0 + 1:i1 + 1] = nav / np.concatenate([[1.0], nav[:-1]]) - 1
            if borrow.any():
                start_of_day = new * np.vstack([np.ones_like(new), growth[:-1]]) \
                    / np.concatenate([[1.0], nav[:-1]])[:, None]
                costs[i0 + 1:i1 + 1] += np.clip(-start_of_day, 0, None) @ borrow
            w = new * growth[-1] / nav[-1]
        else:
            w = new

    idx = prices.index
    return BacktestResult(
        returns=pd.Series(gross - costs, idx, name="net"),
        gross_returns=pd.Series(gross, idx, name="gross"),
        costs=pd.Series(costs, idx, name="costs"),
        turnover=pd.Series(turnover, name="turnover", dtype=float),
        weights=pd.DataFrame(executed, index=prices.columns).T,
        periods_per_year=periods_per_year,
    )


def cost_sensitivity(prices: pd.DataFrame, target_weights: pd.DataFrame, bps_grid=(0, 2, 5, 10, 20, 30, 50),
                     **kwargs) -> pd.DataFrame:
    """Net Sharpe ratio across a grid of cost assumptions."""
    rows = []
    for bps in bps_grid:
        res = run_backtest(prices, target_weights, cost_bps=bps, **kwargs)
        rows.append({"cost_bps": bps, "sharpe": metrics.sharpe(res.returns),
                     "annual_return": metrics.annual_return(res.returns)})
    return pd.DataFrame(rows)
