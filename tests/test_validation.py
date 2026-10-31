from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from btlab import CrossSectionalMomentum, Strategy, TimeSeriesMomentum, check_lookahead, sharpe, run_backtest, synthetic_market, walk_forward
from btlab.validation import effective_trials, probability_of_backtest_overfitting

PRICES = synthetic_market(n_assets=40, n_years=4, seed=3)


@dataclass(frozen=True)
class PeekingStrategy(Strategy):
    """Deliberately cheats: buys whatever goes up tomorrow."""

    def weights(self, prices):
        tomorrow = prices.pct_change(fill_method=None).shift(-1)
        return (tomorrow > 0).astype(float).div(prices.notna().sum(axis=1), axis=0)


def test_causal_strategies_pass_lookahead_check():
    assert check_lookahead(CrossSectionalMomentum(), PRICES)[0]
    assert check_lookahead(TimeSeriesMomentum(), PRICES)[0]


def test_peeking_strategy_is_caught():
    passed, diff = check_lookahead(PeekingStrategy(), PRICES)
    assert not passed and diff > 0


def test_engine_cannot_save_a_strategy_that_peeks():
    """A one-day peek survives zero execution lag: only the truncation test catches it."""
    w = PeekingStrategy().weights(PRICES)
    lag0 = run_backtest(PRICES, w, rebalance="D", execution_lag=0, cost_bps=0).returns
    lag1 = run_backtest(PRICES, w, rebalance="D", execution_lag=1, cost_bps=0).returns
    assert sharpe(lag0) > 10          # absurd result that should never be believed
    assert abs(sharpe(lag1)) < 1.5    # a one-day delay happens to neutralise a one-day peek


def test_walk_forward_folds_are_out_of_sample():
    grid = [{"lookback": lb, "skip": 21} for lb in (63, 126, 252)]
    wf = walk_forward(PRICES, CrossSectionalMomentum, grid, train_days=252, test_days=126)
    assert (wf.folds["test_start"] > wf.folds["train_start"]).all()
    assert wf.oos_returns.index.is_unique and wf.oos_returns.index.is_monotonic_increasing
    assert wf.oos_returns.index[0] == wf.folds["test_start"].iloc[0]


def test_walk_forward_trades_the_chosen_config_and_pays_to_enter():
    grid = [{"lookback": 126, "skip": 21}]
    wf = walk_forward(PRICES, CrossSectionalMomentum, grid, train_days=252, test_days=126, cost_bps=10)
    trial, oos = wf.trial_returns[0].loc[wf.oos_returns.index], wf.oos_returns
    entry = oos.ne(0).idxmax()                      # first rebalance inside the first test window
    assert oos.loc[entry] == pytest.approx(-0.001)  # from cash: 10 bps on 100% gross exposure...
    np.testing.assert_allclose(oos.loc[entry:].iloc[1:], trial.loc[entry:].iloc[1:])  # ...then identical


def test_noise_momentum_is_not_significant_on_average():
    srs = [sharpe(run_backtest(p, CrossSectionalMomentum().weights(p)).returns)
           for p in (synthetic_market(n_assets=60, n_years=8, seed=s) for s in range(8))]
    assert abs(np.mean(srs)) < 0.3


def test_planted_momentum_is_detected():
    p = synthetic_market(n_assets=100, n_years=10, momentum_strength=0.25, seed=4)
    assert sharpe(run_backtest(p, CrossSectionalMomentum().weights(p)).returns) > 0.8


def _noise_trials(n_trials=40, n_days=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 0.01, (n_days, n_trials)), index=pd.bdate_range("2010-01-01", periods=n_days))


def test_pbo_is_about_half_on_noise():
    pbos = [probability_of_backtest_overfitting(_noise_trials(seed=s), n_splits=10)[0] for s in range(5)]
    assert 0.3 < np.mean(pbos) < 0.7


def test_pbo_is_near_zero_with_a_genuine_edge():
    trials = _noise_trials()
    trials[0] += 0.0015                     # annualised Sharpe ~ 2.4
    pbo, logits = probability_of_backtest_overfitting(trials, n_splits=10)
    assert pbo < 0.05 and len(logits) == 252  # C(10, 5) splits


def test_effective_trials_counts_clusters_of_correlated_configs():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.01, (1500, 3))
    cols = {f"{k}{i}": base[:, k] + rng.normal(0, 0.003, 1500) for k in range(3) for i in range(10)}
    assert effective_trials(pd.DataFrame(cols)) == 3
    assert effective_trials(_noise_trials(n_trials=12)) == 12


@dataclass(frozen=True)
class SignPeekingStrategy(Strategy):
    """Cheats via shift(-1) and keeps the NaN it leaves on the last row."""

    def weights(self, prices):
        tomorrow = prices.pct_change(fill_method=None).shift(-1)
        return np.sign(tomorrow).div(prices.notna().sum(axis=1), axis=0)


def test_peek_that_leaves_nan_is_caught():
    passed, diff = check_lookahead(SignPeekingStrategy(), PRICES)
    assert not passed and diff == np.inf


def test_short_inputs_fail_clearly():
    with pytest.raises(ValueError, match="no test window"):
        walk_forward(PRICES, CrossSectionalMomentum, [{}], train_days=5000)
    assert effective_trials(_noise_trials(n_trials=1)) == 1


@dataclass(frozen=True)
class HoldOne(Strategy):
    target: str = "A"

    def weights(self, prices):
        return pd.DataFrame({c: float(c == self.target) for c in prices.columns}, index=prices.index)


def test_walk_forward_never_selects_a_config_that_only_shines_later():
    """B loses steadily, then surges after day 900: best in hindsight, never best in time."""
    rng = np.random.default_rng(7)
    n = 4 * 252
    a = 0.0005 + rng.normal(0, 0.005, n)
    b = np.where(np.arange(n) < 900, -0.0005, 0.01) + rng.normal(0, 0.005, n)
    prices = 100 * pd.DataFrame({"A": 1 + a, "B": 1 + b}, index=pd.bdate_range("2015-01-01", periods=n)).cumprod()
    grid = [{"target": "A"}, {"target": "B"}]
    wf = walk_forward(prices, HoldOne, grid, train_days=504, test_days=252, cost_bps=0)
    assert wf.best_in_sample == 1                        # the trap is real: B wins with hindsight
    assert len(wf.folds) == 2 and (wf.folds["chosen"] == 0).all()   # but is never chosen in time
