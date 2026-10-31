# Changelog

All notable changes. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [semantic versioning](https://semver.org/).

## Unreleased

## 0.3.1 - 2026-11-01

Release-process changes only: the library code is identical to 0.3.0.

### Changed

- Releases are automatic: merging a version bump to `main` creates the tag and GitHub release
  (notes from this changelog, package files attached) and uploads to PyPI.
- The documented release steps work in PowerShell as well as bash.

## 0.3.0 - 2026-11-01

### Added

- `publish` workflow: a published GitHub release uploads the package to PyPI (trusted publishing).
- `CONTRIBUTING.md`, `CLAUDE.md`, PR template and this changelog.
- `factor_attribution` reports the number of observations in `.attrs["n_obs"]`.
- CI tests Python 3.13 and the oldest supported numpy/pandas/scipy.
- Test that `walk_forward` never selects a configuration whose edge only appears after the
  training window (fails if selection is scored on test data).

### Changed

- CI uses `actions/checkout@v7` and `actions/setup-python@v7` (Node 24).
- `rebalance_dates` / `run_backtest` reject unknown schedules (e.g. `"Q"`) instead of
  silently rebalancing monthly.
- `walk_forward` raises a clear `ValueError` when the history is too short for one test window.
- License metadata uses the SPDX form (`license = "MIT"`); requires setuptools>=77 to build.
- README chart links are absolute so they also render on PyPI.

### Fixed

- `check_lookahead` passed strategies whose peek leaves a NaN in the truncated run
  (e.g. `sign(returns.shift(-1))`); a NaN-vs-number mismatch now fails the check.
- `max_drawdown` ignored a loss on the first day (the peak now starts at 1).
- `summary` crashed on a strategy that never trades; Sharpe-family metrics return NaN
  on empty or constant series instead of crashing, warning or returning ~1e16.
- `annual_turnover` counted the warm-up period; Study 3 turnover is ~6.4x, not ~5.9x.
- A gap in prices (e.g. a trading halt) lost the return across it; the position is now held
  through the gap and the return booked when trading resumes.
- Market impact used the trade day's own volume in its ADV estimate.
- `factor_attribution` skipped one extra warm-up row per NaN in the return series.
- `effective_trials` crashed with a single configuration.

## 0.2.0 - 2026-10-31

### Added

- Probability of Backtest Overfitting via CSCV (`probability_of_backtest_overfitting`).
- Effective number of trials by correlation clustering (`effective_trials`);
  `deflated_sharpe(..., n_trials=)`.
- Square-root market impact in `run_backtest` (`dollar_volume`, `capital`, `impact_coef`);
  `synthetic_dollar_volume`; `load_yfinance(field=)`.
- Short borrow fees (`borrow_bps`, scalar or per asset).
- Factor attribution with Newey-West t-stats (`factor_attribution`); `load_french_factors`.
- Study 5 (capacity), attribution in Studies 3 and 4, point-in-time universe recipe.

### Changed

- `walk_forward` out-of-sample returns now come from one backtest of the stitched weights, so
  entry and configuration switches pay costs. Out-of-sample Sharpe in the README studies fell
  (0.87 → 0.82, 1.14 → 1.01).

### Fixed

- Weekly rebalancing added a spurious rebalance at ISO year boundaries.
- Trading costs on a rebalance day overwrote that day's borrow fee.

## 0.1.0

### Added

- Engine, momentum strategies, look-ahead truncation test, walk-forward, PSR/DSR,
  stationary-bootstrap CIs, four studies.
