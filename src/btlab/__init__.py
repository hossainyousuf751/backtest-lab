"""btlab: honest backtesting, with tools to detect look-ahead, survivorship and overfitting."""

from btlab.data import synthetic_dollar_volume, synthetic_market
from btlab.engine import BacktestResult, cost_sensitivity, rebalance_dates, run_backtest
from btlab.metrics import (
    annual_return, annual_vol, bootstrap_sharpe_ci, deflated_sharpe, expected_max_sharpe,
    factor_attribution, max_drawdown, probabilistic_sharpe, sharpe, sortino, summary,
)
from btlab.strategies import CrossSectionalMomentum, Strategy, TimeSeriesMomentum
from btlab.validation import (
    WalkForwardResult, check_lookahead, effective_trials, probability_of_backtest_overfitting,
    walk_forward,
)

__all__ = [
    "BacktestResult", "CrossSectionalMomentum", "Strategy", "TimeSeriesMomentum",
    "WalkForwardResult", "annual_return", "annual_vol", "bootstrap_sharpe_ci", "check_lookahead",
    "cost_sensitivity", "deflated_sharpe", "effective_trials", "expected_max_sharpe",
    "factor_attribution", "max_drawdown", "probabilistic_sharpe",
    "probability_of_backtest_overfitting", "rebalance_dates", "run_backtest", "sharpe", "sortino",
    "summary", "synthetic_dollar_volume", "synthetic_market", "walk_forward",
]
