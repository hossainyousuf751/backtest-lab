# Changelog

All notable changes. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [semantic versioning](https://semver.org/).

## Unreleased

### Added
- `CONTRIBUTING.md`, `CLAUDE.md`, PR template and this changelog.

### Changed
- CI uses `actions/checkout@v7` and `actions/setup-python@v7` (Node 24).

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
