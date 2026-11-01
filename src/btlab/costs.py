"""Pluggable transaction-cost models for run_backtest(cost_model=...).

A cost model maps the trade at a rebalance to a cost per asset, as a fraction
of NAV. The engine sums it and charges it on the trade day, like the flat
cost_bps it replaces.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class CostModel(Protocol):
    def cost(self, trade_weights: np.ndarray, date: pd.Timestamp, context: dict) -> np.ndarray:
        """Cost per asset (fraction of NAV) of trading `trade_weights` (new minus old weights) at `date`.

        context["assets"] is the asset index; context["returns"] holds asset returns up to and including `date`.
        """
        ...


@dataclass(frozen=True)
class FlatBps:
    """A fixed cost per unit traded: exactly run_backtest's cost_bps + slippage_bps."""

    cost_bps: float = 5.0
    slippage_bps: float = 0.0

    def cost(self, trade_weights: np.ndarray, date: pd.Timestamp, context: dict) -> np.ndarray:
        return np.abs(trade_weights) * ((self.cost_bps + self.slippage_bps) / 1e4)


def _at(x, date: pd.Timestamp, assets: pd.Index) -> np.ndarray:
    """A scalar, a per-asset Series, or a date x asset DataFrame, as an array over `assets` at `date`."""
    if isinstance(x, pd.DataFrame):
        x = x.loc[date]
    if isinstance(x, pd.Series):
        return x.reindex(assets).to_numpy(dtype=float)
    return np.full(len(assets), float(x))


@dataclass(frozen=True, eq=False)
class SpreadImpactCost:
    """Half-spread plus square-root market impact, per unit of NAV traded.

        cost_i = |dw_i| * half_spread_bps / 1e4 + impact_coef * |dw_i| * vol_i * sqrt(|dw_i| * capital / adv_i)

    The square-root law (Almgren, Thum, Hauptmann & Li 2005; Toth et al. 2011):
    impact per unit traded grows like daily vol times the square root of the
    trade's share of average daily dollar volume. vol is the daily return
    volatility; adv is in dollars, capital (NAV) in dollars. Each of adv and vol
    can be a scalar, a per-asset Series or a date x asset DataFrame (row d must
    use only data before d). Without adv only the half-spread is charged, with a
    warning; assets with unknown or zero adv pay no impact.
    """

    half_spread_bps: float
    impact_coef: float = 1.0
    adv: float | pd.Series | pd.DataFrame | None = None
    vol: float | pd.Series | pd.DataFrame | None = None
    capital: float = 1e8

    def __post_init__(self) -> None:
        if self.adv is None:
            warnings.warn("SpreadImpactCost without adv charges the half-spread only (no market impact)",
                          stacklevel=3)
        elif self.vol is None:
            raise ValueError("vol is required when adv is given")

    @classmethod
    def from_dollar_volume(cls, prices: pd.DataFrame, dollar_volume: pd.DataFrame, half_spread_bps: float,
                           impact_coef: float = 1.0, capital: float = 1e8,
                           delisting_return: float = 0.0) -> SpreadImpactCost:
        """Estimate vol and ADV exactly as run_backtest(dollar_volume=...) does.

        vol: 63-day standard deviation of listed daily returns; ADV: trailing 20-day mean dollar volume,
        lagged a day so a trade is sized before that day's volume is known.
        """
        from btlab.engine import asset_returns

        vol = asset_returns(prices, delisting_return).where(prices.notna()).rolling(63, min_periods=5).std().fillna(0.0)
        adv = dollar_volume.reindex_like(prices).rolling(20, min_periods=1).mean().shift(1)
        return cls(half_spread_bps, impact_coef, adv, vol, capital)

    def cost(self, trade_weights: np.ndarray, date: pd.Timestamp, context: dict) -> np.ndarray:
        trade = np.abs(trade_weights)
        spread = trade * (self.half_spread_bps / 1e4)
        if self.adv is None:
            return spread
        adv, vol = _at(self.adv, date, context["assets"]), _at(self.vol, date, context["assets"])
        participation = np.divide(trade * self.capital, adv, out=np.zeros_like(trade), where=adv > 0)
        return spread + np.nan_to_num(self.impact_coef * trade * vol * np.sqrt(participation))
