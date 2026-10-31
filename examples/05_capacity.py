"""Study 5: capacity. A real signal, paid for with square-root market impact.

Flat basis-point costs say nothing about size. Impact grows with the square
root of trade size relative to daily volume, so every strategy has a fund size
at which its edge is traded away. Trading less often buys capacity.
"""

import pandas as pd
from _common import DOCS

from btlab import (
    CrossSectionalMomentum, annual_return, run_backtest, sharpe, synthetic_dollar_volume,
    synthetic_market,
)
from btlab.plotting import plot_capacity

prices = synthetic_market(n_assets=100, n_years=12, momentum_strength=0.15, seed=21)
volume = synthetic_dollar_volume(prices, median_adv=2e7, seed=21)
weights = CrossSectionalMomentum().weights(prices)
CAPITAL = [1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9]   # beyond ~$1bn trades exceed daily volume

curves = {}
for label, rebalance in (("monthly rebalance", "M"), ("quarterly rebalance", 63)):
    rows = []
    for cap in CAPITAL:
        res = run_backtest(prices, weights, rebalance=rebalance, cost_bps=5,
                           dollar_volume=volume, capital=cap)
        rows.append({"capital": cap, "sharpe": sharpe(res.returns),
                     "annual_return": annual_return(res.returns),
                     "cost_per_year": res.costs.sum() / 12})
    curves[label] = pd.DataFrame(rows)
    print(f"{label} (5 bps + square-root impact, median ADV $20m):")
    print(curves[label].to_string(index=False, formatters={
        "capital": "${:,.0f}".format, "sharpe": "{:.2f}".format,
        "annual_return": "{:.1%}".format, "cost_per_year": "{:.2%}".format}))
    print()

plot_capacity(curves, DOCS / "capacity.png", "Capacity: net Sharpe as the fund grows")
