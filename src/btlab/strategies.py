"""Strategies map prices to target weights using only information up to each date.

Convention: row t of `weights(prices)` is the portfolio decided at the close of
day t. The engine is responsible for execution delay; strategies must never
look past row t (verified by `validation.check_lookahead`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
    """Each asset long/short by the sign of its own trailing return, scaled to a vol target."""

    lookback: int = 252
    vol_lookback: int = 63
    target_vol: float = 0.10
    max_weight: float = 0.25

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        logp = np.log(prices)
        signal = np.sign(logp - logp.shift(self.lookback))
        rets = prices.pct_change(fill_method=None)
        asset_vol = rets.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2).std() * np.sqrt(TRADING_DAYS)
        n_valid = signal.notna().sum(axis=1).replace(0, np.nan)
        w = signal * (self.target_vol / asset_vol)
        w = w.div(n_valid, axis=0).clip(-self.max_weight, self.max_weight)
        return w.where(prices.notna()).fillna(0.0)
