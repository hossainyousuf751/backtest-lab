"""Strategies map prices to target weights using only information up to each date.

Convention: row t of `weights(prices)` is the portfolio decided at the close of
day t. The engine is responsible for execution delay; strategies must never
look past row t (verified by `validation.check_lookahead`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


class Strategy(ABC):
    @abstractmethod
    def weights(self, prices: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class CrossSectionalMomentum(Strategy):
    """Rank assets on return from t-lookback to t-skip; long top, short bottom quantile.

    The classic 12-1 momentum is lookback=252, skip=21: skipping the latest
    month avoids the short-term reversal effect.
    """

    lookback: int = 252
    skip: int = 21
    quantile: float = 0.2
    long_short: bool = True
    min_assets: int = 10

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        logp = np.log(prices)
        score = logp.shift(self.skip) - logp.shift(self.lookback)
        score = score.where(prices.notna())
        n_valid = score.notna().sum(axis=1)
        rank = score.rank(axis=1, pct=True)
        long = (rank > 1 - self.quantile).astype(float)
        w = long.div(long.sum(axis=1).replace(0, np.nan), axis=0)
        if self.long_short:
            short = (rank <= self.quantile).astype(float)
            w = 0.5 * w - 0.5 * short.div(short.sum(axis=1).replace(0, np.nan), axis=0)
        w[n_valid < self.min_assets] = 0.0
        return w.fillna(0.0)


@dataclass(frozen=True)
class TimeSeriesMomentum(Strategy):
    """Each asset long/short by the sign of its own trailing return, scaled to a vol target.

    vol_estimator (optional) replaces the rolling standard deviation: it maps the
    daily returns DataFrame to annualised vol forecasts of the same shape, row t
    using returns up to t only (e.g. a GARCH or EWMA forecast). The default is
    the rolling std over vol_lookback days, times sqrt(252).
    """

    lookback: int = 252
    vol_lookback: int = 63
    target_vol: float = 0.10
    max_weight: float = 0.25
    vol_estimator: Callable[[pd.DataFrame], pd.DataFrame] | None = None

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        logp = np.log(prices)
        signal = np.sign(logp - logp.shift(self.lookback))
        rets = prices.pct_change(fill_method=None)
        if self.vol_estimator is None:
            rolling = rets.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2)
            asset_vol = rolling.std() * np.sqrt(TRADING_DAYS)
        else:
            asset_vol = self.vol_estimator(rets).reindex_like(rets)
        n_valid = signal.notna().sum(axis=1).replace(0, np.nan)
        w = signal * (self.target_vol / asset_vol)
        w = w.div(n_valid, axis=0).clip(-self.max_weight, self.max_weight)
        return w.where(prices.notna()).fillna(0.0)


@dataclass(frozen=True)
class CovarianceStrategy(Strategy):
    """Risk-based allocation: estimate a covariance matrix, then turn it into weights.

    Every `every` trading days, `estimator` gets the trailing `window` daily
    returns of the assets fully observed over that window and listed today, as a
    (window x n) array, and returns an (n x n) covariance; `allocator` maps it to
    n weights (e.g. inverse variance, minimum variance, risk parity, HRP). Weights
    are held between those dates. Plain callables keep this library independent
    of any risk package.

    Weights are computed only on positions every-1, 2*every-1, ... so that
    run_backtest(rebalance=every, execution_lag=1) trades each decision the next
    day. The schedule depends only on the date's position, never on later dates,
    so the strategy passes check_lookahead; a calendar month-end schedule would
    not, because spotting the last trading day of a month needs tomorrow's date.
    """

    estimator: Callable[[np.ndarray], np.ndarray]
    allocator: Callable[[np.ndarray], np.ndarray]
    window: int = 252
    min_history: int | None = None  # days of data before the first allocation; default window
    every: int = 21

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        rets = prices.pct_change(fill_method=None)
        start = max(self.window if self.min_history is None else self.min_history, 2)
        out = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        for i in range(self.every - 1, len(prices), self.every):
            if i + 1 < start:
                continue
            win = rets.iloc[max(1, i + 1 - self.window):i + 1]
            cols = prices.columns[win.notna().all().to_numpy() & prices.iloc[i].notna().to_numpy()]
            out.iloc[i] = 0.0
            if len(cols) == 0:
                continue
            w = np.asarray(self.allocator(np.asarray(self.estimator(win[cols].to_numpy()))), dtype=float)
            out.loc[out.index[i], cols] = w
        return out.ffill().where(prices.notna()).fillna(0.0)
