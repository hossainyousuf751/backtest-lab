"""Study 2: survivorship bias.

Testing only on firms that still exist today silently removes every firm that
failed along the way. The same strategy looks better for no legitimate reason.
"""

import pandas as pd
from _common import DOCS, fmt

from btlab import CrossSectionalMomentum, run_backtest, synthetic_market
from btlab.plotting import plot_equity

prices = synthetic_market(n_assets=150, n_years=12, delist_rate=0.04, late_listing_frac=0.2, seed=5)
survivors = prices.loc[:, prices.iloc[-1].notna()]
print(f"Universe: {prices.shape[1]} firms ever listed, {survivors.shape[1]} survive to the end\n")


def equal_weight(p: pd.DataFrame) -> pd.DataFrame:
    listed = p.notna().astype(float)
    return listed.div(listed.sum(axis=1), axis=0)


results = {}
for label, p in (("point-in-time universe", prices), ("survivors only (biased)", survivors)):
    ew = run_backtest(p, equal_weight(p), cost_bps=2)
    lo = run_backtest(p, CrossSectionalMomentum(long_short=False).weights(p))
    results[label] = ew.returns
    print(f"Equal-weight, {label}:\n  {fmt({k: v for k, v in ew.summary().items() if k in ('annual_return', 'sharpe', 'max_drawdown')})}")
    print(f"Long-only momentum, {label}:\n  {fmt({k: v for k, v in lo.summary().items() if k in ('annual_return', 'sharpe', 'max_drawdown')})}\n")

plot_equity(results, DOCS / "survivorship.png",
            "Same equal-weight portfolio, two universes: survivorship bias")
