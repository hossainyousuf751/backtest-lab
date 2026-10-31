"""Guards against the two most common backtest failures: look-ahead and overfitting."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from btlab import metrics
from btlab.engine import run_backtest
from btlab.strategies import Strategy


def check_lookahead(strategy: Strategy, prices: pd.DataFrame, n_cutoffs: int = 5,
                    seed: int | None = 0, tol: float = 1e-10) -> tuple[bool, float]:
    """Truncation test: weights up to date c must not change when data after c is removed.

    Returns (passed, max_abs_difference). Any difference means the strategy
    used information from the future.
    """
    rng = np.random.default_rng(seed)
    full = strategy.weights(prices)
    worst = 0.0
    lo = len(prices) // 4
    for c in sorted(rng.integers(lo, len(prices) - 1, n_cutoffs)):
        trunc = strategy.weights(prices.iloc[:c + 1])
        diff = (full.iloc[:c + 1] - trunc).abs().to_numpy()
        worst = max(worst, float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0)
    return worst <= tol, worst


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    folds: pd.DataFrame
    trial_returns: pd.DataFrame       # every configuration's full-history returns
    params: list[dict]

    @property
    def trial_sharpes(self) -> pd.Series:
        return self.trial_returns.apply(metrics.sharpe)

    @property
    def best_in_sample(self) -> int:
        return int(self.trial_sharpes.idxmax())


def walk_forward(prices: pd.DataFrame, strategy_cls: type[Strategy], param_grid: list[dict],
                 train_days: int = 756, test_days: int = 252, expanding: bool = True,
                 select=metrics.sharpe, **backtest_kwargs) -> WalkForwardResult:
    """Choose parameters on each training window, then trade them on the next test window.

    Every configuration is backtested once over the full history (strategies
    are causal, so this is equivalent to re-running up to each date) to score
    the training windows. The out-of-sample series is then one backtest of
    the stitched weights: the config chosen at the close of each training
    window's last day is traded from the next rebalance on, so entering the
    first fold and switching configs later pay trading costs like any other
    trade. It is the honest estimate of what the selection procedure would
    have delivered.
    """
    trials = {}
    for i, p in enumerate(param_grid):
        w = strategy_cls(**p).weights(prices)
        trials[i] = run_backtest(prices, w, **backtest_kwargs).returns
    R = pd.DataFrame(trials)

    start = R.index.get_loc(_trim_warmup(R).index[0])
    folds, chosen_weights = [], {}
    stitched = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    train_end = start + train_days
    while train_end < len(R):
        test_end = min(train_end + test_days, len(R))
        train_start = start if expanding else train_end - train_days
        scores = R.iloc[train_start:train_end].apply(select)
        best = int(scores.idxmax())
        if best not in chosen_weights:
            chosen_weights[best] = strategy_cls(**param_grid[best]).weights(prices)
        stitched.iloc[train_end - 1:] = chosen_weights[best].iloc[train_end - 1:]  # later folds overwrite
        folds.append({
            "train_start": R.index[train_start], "test_start": R.index[train_end],
            "test_end": R.index[test_end - 1], "chosen": best, "params": param_grid[best],
            "is_sharpe": scores[best],
        })
        train_end = test_end

    folds = pd.DataFrame(folds)
    oos = run_backtest(prices, stitched, **backtest_kwargs).returns.loc[folds["test_start"].iloc[0]:]
    folds["oos_sharpe"] = [metrics.sharpe(oos.loc[a:b]) for a, b in zip(folds["test_start"], folds["test_end"])]
    return WalkForwardResult(oos, folds, R, param_grid)


def _trim_warmup(trial_returns: pd.DataFrame) -> pd.DataFrame:
    """Drop the leading rows before every configuration has started trading."""
    return trial_returns.loc[trial_returns.ne(0).idxmax().max():]


def probability_of_backtest_overfitting(trial_returns: pd.DataFrame, n_splits: int = 16
                                        ) -> tuple[float, np.ndarray]:
    """Combinatorially symmetric cross-validation (Bailey, Borwein, Lopez de Prado & Zhu, 2017).

    The history is cut into `n_splits` blocks. For every way of choosing half
    of them as in-sample, the best in-sample configuration is located in the
    out-of-sample ranking. PBO is the fraction of splits in which it lands in
    the bottom half. Pure noise gives PBO ~ 0.5; a genuine edge gives ~ 0.

    Returns (pbo, logits), where logit > 0 means the in-sample winner
    beat the out-of-sample median.
    """
    if n_splits % 2:
        raise ValueError("n_splits must be even")
    R = _trim_warmup(trial_returns).to_numpy()
    n_trials = R.shape[1]
    blocks = np.array_split(R, n_splits)
    s1 = np.array([b.sum(axis=0) for b in blocks])        # per-block sums: Sharpe of any
    s2 = np.array([(b**2).sum(axis=0) for b in blocks])   # union of blocks follows exactly
    n = np.array([len(b) for b in blocks], dtype=float)

    is_mask = np.array([np.isin(np.arange(n_splits), c)
                        for c in combinations(range(n_splits), n_splits // 2)], dtype=float)

    def _sr(mask):
        cnt = (mask @ n)[:, None]
        mean = (mask @ s1) / cnt
        var = ((mask @ s2) - cnt * mean**2) / (cnt - 1)
        return mean / np.sqrt(np.maximum(var, 1e-300))

    is_sr, oos_sr = _sr(is_mask), _sr(1 - is_mask)
    best = is_sr.argmax(axis=1)
    oos_best = oos_sr[np.arange(len(best)), best][:, None]
    rank = (oos_sr < oos_best).sum(axis=1) + 0.5 * ((oos_sr == oos_best).sum(axis=1) + 1)
    omega = rank / (n_trials + 1)
    logits = np.log(omega / (1 - omega))
    return float(np.mean(logits <= 0)), logits


def effective_trials(trial_returns: pd.DataFrame, min_corr: float = 0.5) -> int:
    """Number of clusters of configurations whose returns correlate above `min_corr`.

    45 momentum variants are not 45 independent experiments. Passing this as
    `n_trials` to `deflated_sharpe` gives a less conservative, more accurate
    multiple-testing correction. Average-linkage hierarchical clustering on
    the distance sqrt((1 - corr) / 2).
    """
    corr = _trim_warmup(trial_returns).corr().fillna(0.0).to_numpy()
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
    np.fill_diagonal(dist, 0.0)
    tree = linkage(squareform(dist, checks=False), "average")
    return int(fcluster(tree, np.sqrt(0.5 * (1 - min_corr)), "distance").max())
