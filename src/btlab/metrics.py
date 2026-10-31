"""Performance statistics, including multiple-testing-aware Sharpe ratio inference.

References: Bailey & Lopez de Prado (2012) "The Sharpe Ratio Efficient
Frontier" (PSR); Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PERIODS = 252
EULER_GAMMA = 0.5772156649015329


def _clean(r) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return r[np.isfinite(r)]


def _trim_leading_zeros(r) -> np.ndarray:
    r = _clean(r)
    nz = np.flatnonzero(r)
    return r[nz[0]:] if len(nz) else r[:0]


def sharpe(r, periods: int = PERIODS) -> float:
    r = _trim_leading_zeros(r)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    # a constant series has sd 0 up to rounding error: treat that as undefined, not huge
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 and sd > 1e-9 * abs(r.mean()) else float("nan")


def sortino(r, periods: int = PERIODS) -> float:
    r = _trim_leading_zeros(r)
    if not len(r):
        return float("nan")
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    return float(r.mean() / downside * np.sqrt(periods)) if downside > 0 else float("nan")


def annual_return(r, periods: int = PERIODS) -> float:
    r = _trim_leading_zeros(r)
    return float(np.prod(1 + r) ** (periods / len(r)) - 1) if len(r) else float("nan")


def annual_vol(r, periods: int = PERIODS) -> float:
    r = _trim_leading_zeros(r)
    return float(r.std(ddof=1) * np.sqrt(periods)) if len(r) > 1 else float("nan")


def max_drawdown(r) -> float:
    equity = np.concatenate([[1.0], np.cumprod(1 + _clean(r))])   # the peak starts at 1, not day 1
    return float(np.min(equity / np.maximum.accumulate(equity) - 1))


def probabilistic_sharpe(r, sr_benchmark: float = 0.0, periods: int = PERIODS) -> float:
    """P(true Sharpe > sr_benchmark), adjusting for sample length, skew and kurtosis.

    Both Sharpe ratios are annualised at the interface and converted to
    per-period internally, as the formula requires.
    """
    r = _trim_leading_zeros(r)
    n = len(r)
    if np.isnan(sharpe(r)):                                 # empty or constant: undefined
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    sr_b = sr_benchmark / np.sqrt(periods)
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - sr_b) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Expected maximum Sharpe among n_trials skill-less strategies (annualised in, out)."""
    if n_trials < 2:
        return 0.0
    return float(sharpe_std * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
                               + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * np.e))))


def deflated_sharpe(r, trial_sharpes, periods: int = PERIODS, n_trials: int | None = None) -> float:
    """PSR against the Sharpe you'd expect from the best of N skill-less trials.

    `trial_sharpes` are the annualised Sharpe ratios of every configuration
    tried, including the one being evaluated. Correlated trials overstate N,
    so by default (N = number of trials) this is conservative when
    configurations are similar; pass `n_trials=validation.effective_trials(...)`
    to count clusters of correlated configurations instead.
    """
    trial_sharpes = _clean(trial_sharpes)
    n = len(trial_sharpes) if n_trials is None else n_trials
    sr0 = expected_max_sharpe(n, trial_sharpes.std(ddof=1))
    return probabilistic_sharpe(r, sr0, periods)


def bootstrap_sharpe_ci(r, n_boot: int = 2000, mean_block: int = 20, alpha: float = 0.05,
                        seed: int | None = 0, periods: int = PERIODS) -> tuple[float, float]:
    """Stationary-bootstrap (Politis & Romano) confidence interval for the Sharpe ratio.

    Random-length blocks preserve the autocorrelation and volatility
    clustering that an i.i.d. bootstrap would destroy.
    """
    r = _trim_leading_zeros(r)
    n = len(r)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, (n_boot, n))
    new_block = rng.random((n_boot, n)) < 1 / mean_block
    new_block[:, 0] = True
    block_id = np.cumsum(new_block, axis=1) - 1
    block_start = np.take_along_axis(starts, block_id, axis=1)
    offset = np.arange(n)[None, :] - np.maximum.accumulate(np.where(new_block, np.arange(n), 0), axis=1)
    sample = r[(block_start + offset) % n]
    srs = sample.mean(axis=1) / sample.std(axis=1, ddof=1) * np.sqrt(periods)
    return float(np.quantile(srs, alpha / 2)), float(np.quantile(srs, 1 - alpha / 2))


def factor_attribution(r: pd.Series, factors: pd.DataFrame, periods: int = PERIODS,
                       lags: int | None = None) -> pd.DataFrame:
    """Regress returns on factor returns: how much is alpha, how much is beta?

    OLS with Newey-West (HAC) standard errors, since daily strategy returns
    are autocorrelated and heteroskedastic. `r` should be an excess return
    (subtract the risk-free rate unless the strategy is self-financing, e.g.
    long-short). Returns a table of coef and t_stat for alpha (annualised)
    and each factor loading; R-squared and the number of observations are in
    `.attrs["r2"]` and `.attrs["n_obs"]`.
    """
    r = pd.Series(r)
    r = r.iloc[int(r.ne(0).to_numpy().argmax()):]                    # skip the warm-up
    df = pd.concat([r.rename("_r"), factors], axis=1, join="inner").dropna()
    y = df.pop("_r").to_numpy()
    X = np.column_stack([np.ones(len(df)), df.to_numpy()])
    n = len(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef

    lags = int(4 * (n / 100) ** (2 / 9)) if lags is None else lags
    xe = X * resid[:, None]
    S = xe.T @ xe
    for lag in range(1, lags + 1):
        g = xe[lag:].T @ xe[:-lag]
        S += (1 - lag / (lags + 1)) * (g + g.T)
    bread = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(bread @ S @ bread))

    out = pd.DataFrame({"coef": coef, "t_stat": coef / se}, index=["alpha"] + list(df.columns))
    out.loc["alpha", "coef"] *= periods
    out.attrs["r2"] = float(1 - resid.var() / y.var())
    out.attrs["n_obs"] = n
    return out


def summary(r, turnover: pd.Series | None = None, periods: int = PERIODS) -> dict:
    out = {
        "annual_return": annual_return(r, periods),
        "annual_vol": annual_vol(r, periods),
        "sharpe": sharpe(r, periods),
        "sortino": sortino(r, periods),
        "max_drawdown": max_drawdown(r),
        "psr_vs_0": probabilistic_sharpe(r, 0.0, periods),
        "sharpe_ci95": bootstrap_sharpe_ci(r, periods=periods),
    }
    traded = turnover[turnover > 0] if turnover is not None else []
    if len(traded) and isinstance(r, pd.Series):
        years = (r.index[-1] - traded.index[0]).days / 365.25   # from the first trade, not the warm-up
        out["annual_turnover"] = float(traded.sum() / years) if years > 0 else float("nan")
    return out
