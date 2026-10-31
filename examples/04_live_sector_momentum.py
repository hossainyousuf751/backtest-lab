"""Study 4: cross-sectional momentum across US sector ETFs (requires yfinance).

Sector ETFs avoid survivorship bias: none of the SPDR sector funds has closed,
and late launches (XLRE 2015, XLC 2018) enter the universe only once listed.
With 11 assets, use the top/bottom 3 sectors (quantile ~0.27).
"""

from _common import DOCS, fmt

from btlab import (
    CrossSectionalMomentum, check_lookahead, cost_sensitivity, factor_attribution, run_backtest,
)
from btlab.data import load_french_factors, load_yfinance
from btlab.plotting import plot_equity

SECTORS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
prices = load_yfinance(SECTORS + ["SPY"], start="2000-01-01")
spy = prices.pop("SPY")

strategy = CrossSectionalMomentum(quantile=0.27, min_assets=8, long_short=True)
print(f"Look-ahead check: passed={check_lookahead(strategy, prices)[0]}")
res = run_backtest(prices, strategy.weights(prices), cost_bps=3)
print(f"Sector momentum (long-short, 3 bps):\n  {fmt(res.summary())}\n")
print(cost_sensitivity(prices, strategy.weights(prices)).round(3).to_string(index=False))

# Is it alpha, or a sector-level echo of known factors? Long-short is self-financing,
# so its return is already an excess return.
factors = load_french_factors(start="2000-01-01").drop(columns="RF")
table = factor_attribution(res.returns, factors)
print(f"\nFama-French + momentum attribution (Newey-West t-stats), R2 {table.attrs['r2']:.2f}:")
print(table.round(3).to_string())

plot_equity({"sector momentum L/S": res.returns,
             "SPY buy & hold": spy.pct_change().fillna(0).loc[res.returns.index]},
            DOCS / "live_sector_momentum.png", "Sector ETF momentum vs. SPY")
