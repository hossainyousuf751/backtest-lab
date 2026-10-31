"""Study 3: a market with a real (planted) momentum effect.

Here the full methodology should agree the signal is genuine, and the
remaining questions are economic: how much survives costs and turnover?
"""

from _common import DOCS, MOMENTUM_GRID, fmt

from btlab import (
    CrossSectionalMomentum, check_lookahead, cost_sensitivity, deflated_sharpe, effective_trials,
    factor_attribution, probability_of_backtest_overfitting, run_backtest, sharpe,
    synthetic_market, walk_forward,
)
from btlab.plotting import plot_cost_sensitivity, plot_equity, plot_walk_forward

prices = synthetic_market(n_assets=100, n_years=12, momentum_strength=0.15, seed=21)
strategy = CrossSectionalMomentum()
print(f"Look-ahead check: passed={check_lookahead(strategy, prices)[0]}")

weights = strategy.weights(prices)
base = run_backtest(prices, weights, cost_bps=5)
print(f"12-1 momentum, 5 bps costs:\n  {fmt(base.summary())}")
for fee in (50, 300):  # general collateral vs hard to borrow, bps/yr on the short leg
    r = run_backtest(prices, weights, cost_bps=5, borrow_bps=fee).returns
    print(f"  with {fee} bps/yr borrow fee: Sharpe {sharpe(r):.2f}")
print()

wf = walk_forward(prices, CrossSectionalMomentum, MOMENTUM_GRID, cost_bps=5)
best = wf.trial_returns[wf.best_in_sample]
n_eff = effective_trials(wf.trial_returns)
print(f"Deflated Sharpe of best of {len(MOMENTUM_GRID)} configs: "
      f"{deflated_sharpe(best, wf.trial_sharpes):.3f} (N = {n_eff} effective: "
      f"{deflated_sharpe(best, wf.trial_sharpes, n_trials=n_eff):.3f})")
print(f"Probability of Backtest Overfitting: {probability_of_backtest_overfitting(wf.trial_returns)[0]:.2f}")
print(f"Walk-forward out-of-sample Sharpe: {sharpe(wf.oos_returns):.2f}\n")

# Attribution: is the return alpha, or market exposure? The equal-weight index of all
# listed assets stands in for the market factor.
market = prices.pct_change(fill_method=None).mean(axis=1).fillna(0.0).to_frame("market")
long_only = run_backtest(prices, CrossSectionalMomentum(long_short=False).weights(prices))
for label, r in (("long-short", base.returns), ("long-only", long_only.returns)):
    t = factor_attribution(r, market)
    print(f"{label}: alpha {t.loc['alpha', 'coef']:.1%}/yr (t={t.loc['alpha', 't_stat']:.1f}), "
          f"market beta {t.loc['market', 'coef']:.2f}, R2 {t.attrs['r2']:.2f}")
print()

costs = cost_sensitivity(prices, weights)
print(costs.round(3).to_string(index=False))

gross = run_backtest(prices, weights, cost_bps=0).returns
plot_equity({"gross": gross, "net of 5 bps": base.returns, "net of 30 bps":
             run_backtest(prices, weights, cost_bps=30).returns},
            DOCS / "planted_equity.png", "Momentum on a market with a genuine signal")
plot_walk_forward(wf.folds, DOCS / "planted_walkforward.png",
                  "Walk-forward with a genuine signal: performance persists out of sample")
plot_cost_sensitivity(costs, DOCS / "planted_costs.png", "How much edge survives trading costs")
