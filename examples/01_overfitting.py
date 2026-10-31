"""Study 1: search 45 momentum configurations on a market with NO signal.

The best configuration looks like a discovery. The deflated Sharpe ratio and
walk-forward testing both expose it as the expected maximum of noise.
"""

import numpy as np
from _common import DOCS, MOMENTUM_GRID

from btlab import (
    CrossSectionalMomentum, deflated_sharpe, effective_trials, expected_max_sharpe,
    probabilistic_sharpe, probability_of_backtest_overfitting, run_backtest, sharpe,
    synthetic_market, walk_forward,
)
from btlab.plotting import plot_trial_sharpes, plot_walk_forward

# Part A: one strategy, 30 independent noise markets
noise_srs = np.array([
    sharpe(run_backtest(p, CrossSectionalMomentum().weights(p)).returns)
    for p in (synthetic_market(n_assets=100, n_years=12, seed=s) for s in range(30))
])
print("12-1 momentum on 30 independent pure-noise markets:")
print(f"  mean Sharpe {noise_srs.mean():+.2f}, std {noise_srs.std(ddof=1):.2f}, "
      f"range [{noise_srs.min():+.2f}, {noise_srs.max():+.2f}]")
print(f"  theory: std of a 12-year Sharpe estimate ~ 1/sqrt(11) = {1 / np.sqrt(11):.2f}\n")

# Part B: search 45 configurations on one noise market
# seed 6 is the "luckiest" of 12 noise markets scanned: the situation a researcher
# who stops at the first good-looking result ends up in.
prices = synthetic_market(n_assets=100, n_years=12, seed=6)
wf = walk_forward(prices, CrossSectionalMomentum, MOMENTUM_GRID, train_days=756, test_days=252)
srs = wf.trial_sharpes
best = wf.best_in_sample
best_returns = wf.trial_returns[best]
e_max = expected_max_sharpe(len(srs), srs.std(ddof=1))

print(f"Searched {len(srs)} configurations on one pure-noise market:")
print(f"  best full-sample Sharpe: {srs[best]:.2f} with {MOMENTUM_GRID[best]}")
print(f"  expected best of {len(srs)} with zero skill: {e_max:.2f}")
print(f"  PSR vs 0 (ignores the search): {probabilistic_sharpe(best_returns):.3f}")
print(f"  Deflated Sharpe (accounts for it): {deflated_sharpe(best_returns, srs):.3f}")
n_eff = effective_trials(wf.trial_returns)
print(f"  effective number of trials (clusters with corr > 0.5): {n_eff}")
print(f"  Deflated Sharpe with N = {n_eff}: {deflated_sharpe(best_returns, srs, n_trials=n_eff):.3f}")
pbo, _ = probability_of_backtest_overfitting(wf.trial_returns)
print(f"  Probability of Backtest Overfitting (CSCV, 16 blocks): {pbo:.2f}")
print(f"  walk-forward out-of-sample Sharpe: {sharpe(wf.oos_returns):.2f}")
print(f"  mean in-sample Sharpe of selected config: {wf.folds['is_sharpe'].mean():.2f}")

plot_trial_sharpes(srs, e_max, DOCS / "overfitting_trials.png",
                   "45 momentum configurations on pure noise")
plot_walk_forward(wf.folds, DOCS / "overfitting_walkforward.png",
                  "Walk-forward on noise: in-sample winners do not persist")

# Part C: the only cure for a lucky sample is genuinely new data
chosen = CrossSectionalMomentum(**MOMENTUM_GRID[best])
fresh = np.array([
    sharpe(run_backtest(p, chosen.weights(p)).returns)
    for p in (synthetic_market(n_assets=100, n_years=12, seed=s) for s in range(100, 120))
])
print(f"\nChosen config on 20 fresh noise markets: mean Sharpe {fresh.mean():+.2f} "
      f"(std {fresh.std(ddof=1):.2f})")
