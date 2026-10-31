# backtest-lab

![tests](https://github.com/hossainyousuf751/backtest-lab/actions/workflows/ci.yml/badge.svg)

A backtesting framework built around one question: **should you believe this backtest?**

Most backtests that look good are wrong for one of a handful of reasons: look-ahead bias,
survivorship bias, ignored trading costs and capacity, returns that are really just factor
exposure, or picking the best of many configurations and reporting it as if it were the only one. This project builds the tools to detect each of
these, and validates them on synthetic markets where the true answer is known.

```bash
pip install -e ".[dev]"
pytest
cd examples && python 01_overfitting.py && python 02_survivorship.py && python 03_planted_signal.py && python 05_capacity.py
pip install -e "..[data]" && python 04_live_sector_momentum.py   # needs internet: Yahoo Finance + Ken French
```

## Design

| Module | What it does |
|---|---|
| `strategies.py` | Strategies return target weights; row *t* is decided at the close of day *t* |
| `engine.py` | Execution lag, drift between rebalances, turnover-based costs, square-root market impact, delistings, point-in-time universe |
| `validation.py` | Truncation test for look-ahead bias; walk-forward selection; Probability of Backtest Overfitting (CSCV); effective number of trials |
| `metrics.py` | Sharpe, Sortino, drawdown, Probabilistic and Deflated Sharpe ratios, block-bootstrap CIs, factor attribution with Newey-West t-stats |
| `data.py` | Synthetic markets with controllable signal, listings, delistings and traded volume; Yahoo Finance and Ken French factor loaders |

**Timing is explicit.** With close-only data, weights decided at the close of day *t* are
traded at the close of *t+1* by default (`execution_lag=1`). Positions then drift with
returns until the next rebalance, so turnover and costs are measured on what the portfolio
actually holds, not on idealised constant weights.

**Look-ahead is tested, not assumed.** `check_lookahead` recomputes a strategy's weights
on data truncated at random dates and requires them to match the full-history weights.
The test suite includes a deliberately cheating strategy: the engine reports a Sharpe
above 10 for it with zero execution lag, which is why the truncation test, not the engine,
is the line of defence.

**Walk-forward is a real backtest.** The out-of-sample series trades the stitched weights of
each fold's chosen configuration, so entering the market and switching configurations pay costs.

**Point-in-time universes need no special engine support.** Mask what the *strategy* sees, not
the prices the *engine* trades on:

```python
weights = strategy.weights(prices.where(index_membership))   # rank only current members
result = run_backtest(prices, weights)                       # exits sold at real prices
```

Masking the engine's prices instead would turn every index deletion into a fake delisting.

## Study 1: overfitting and luck

On 30 independent markets with **no signal**, 12-1 momentum has a mean Sharpe of +0.01
but a standard deviation of 0.33, matching the theoretical 1/√years. Twelve years of noise
routinely produces Sharpe ratios of ±0.6.

On the luckiest of 12 noise markets (deliberately chosen, as a researcher who stops at the
first good result would), a search over 45 momentum configurations finds:

| Measure | Value | Verdict |
|---|---|---|
| Best full-sample Sharpe | 0.86 | looks like a discovery |
| Probabilistic Sharpe vs 0 | 0.997 | "significant" (ignores the search) |
| **Deflated Sharpe** | **0.92** | not significant at 95% once 45 trials are accounted for |
| Deflated Sharpe, N = 2 effective trials | 0.99 | **fooled**: the 45 configs are really ~2 independent bets |
| Probability of Backtest Overfitting | 0.34 | **fooled**: the in-sample winner usually stays above median |
| Walk-forward out-of-sample Sharpe | 0.82 | **fooled**: the whole sample is lucky, not just the selection |
| Chosen config on 20 fresh noise markets | +0.00 | the truth |

![Trial Sharpes](https://raw.githubusercontent.com/hossainyousuf751/backtest-lab/main/docs/overfitting_trials.png)

The lesson is that every in-sample defence (DSR, PBO, walk-forward) protects against
*selection* bias but not against a lucky *sample*. Only independent data (other markets,
periods, or asset classes) does that.

The two refinements cut in opposite directions. Counting all 45 near-duplicate configurations as
independent trials makes the Deflated Sharpe too strict; clustering them (`effective_trials`)
is more accurate, but it removes the margin that rejected this lucky sample. The cluster count also
depends on the correlation threshold: here it is 1, 2, 10 and 35 at ρ > 0.3, 0.5, 0.7 and 0.9.
The threshold-free eigenvalue participation ratio is about 3, which supports the 0.5 default.

## Study 2: survivorship bias

150 firms are listed over 12 years; 57 fail, after a decline, with a −50% final return.
Testing only on the 93 survivors:

| Strategy | Point-in-time universe | Survivors only |
|---|---|---|
| Equal-weight: annual return | 9.3% | **12.8%** |
| Equal-weight: Sharpe | 0.67 | 0.88 |
| Long-only momentum: annual return | 16.1% | 15.6% |

The bias is large for strategies that would have held the failures, and nearly absent for
momentum, which sells declining firms before they delist. The engine handles this correctly:
assets cannot be bought before listing, and delisted holdings become cash.

![Survivorship](https://raw.githubusercontent.com/hossainyousuf751/backtest-lab/main/docs/survivorship.png)

## Study 3: a genuine signal, and what costs leave of it

With a planted persistent-return effect, every check agrees the signal is real: the
look-ahead test passes, the Deflated Sharpe is 0.996 (1.000 with effective N = 2), and walk-forward
out-of-sample Sharpe, after the costs of switching configuration, is 1.01 against 1.06 in the full sample.

PBO is 0.54, no better than on noise, and that is the correct answer to a narrower question.
PBO asks whether picking the in-sample winner *among these configurations* beats picking one at
random. All 45 variants capture the same signal, so choosing among them adds nothing. A high PBO
says "don't trust the tuning", not "there is no edge".

Factor attribution against the market (equal-weight index) separates the edge from exposure:

| | Alpha | t-stat | Market beta | R² |
|---|---|---|---|---|
| Long-short momentum | 6.1%/yr | 3.3 | 0.01 | 0.00 |
| Long-only momentum | 6.8%/yr | 3.0 | 1.03 | 0.82 |

The long-only version earns 16%/yr in raw terms, but most of that is beta.

The remaining question is economic. At ~6.4× annual turnover, net Sharpe falls from 1.11
gross to 0.79 at 30 bps and 0.58 at 50 bps. Borrowing the short leg costs more: a 50 bps/yr
general-collateral fee takes Sharpe from 1.06 to 1.01, and a 300 bps/yr hard-to-borrow fee takes it to 0.80.

![Cost sensitivity](https://raw.githubusercontent.com/hossainyousuf751/backtest-lab/main/docs/planted_costs.png)

## Study 4: real data

`04_live_sector_momentum.py` runs long-short momentum across the 11 SPDR sector ETFs since
2000. Sector ETFs are a deliberately survivorship-free universe; late launches (XLRE, XLC)
enter only once listed. For single stocks, a point-in-time constituent list is required.

Over 2000 to October 2026, the result at 3 bps is a Sharpe of 0.12 (95% CI −0.23 to 0.47), and it
is gone by 20 bps. Regressed on the Fama-French factors plus momentum:

| | Coef | Newey-West t |
|---|---|---|
| Alpha | −0.5%/yr | −0.4 |
| Market | 0.02 | 2.0 |
| Size (SMB) | −0.02 | −1.3 |
| Value (HML) | −0.08 | −5.5 |
| **Momentum** | **0.40** | **39.6** |

R² is 0.58. Sector momentum is not a new strategy: it is a weak, diluted copy of the stock-level
momentum factor, which you could hold directly.

![Sector momentum](https://raw.githubusercontent.com/hossainyousuf751/backtest-lab/main/docs/live_sector_momentum.png)

## Study 5: capacity

Flat basis-point costs ignore size. `run_backtest(..., dollar_volume=, capital=)` adds
square-root market impact, charging `σ_daily × √(trade / ADV)` per unit traded. On the Study 3
market, with a median stock trading $20m/day:

| Capital | Net Sharpe, monthly | Net Sharpe, quarterly | Cost/yr, monthly |
|---|---|---|---|
| $1m | 0.96 | 0.94 | 0.8% |
| $10m | 0.76 | 0.82 | 1.9% |
| $100m | 0.13 | 0.43 | 5.3% |
| $300m | −0.49 | 0.06 | 8.9% |
| $1bn | −1.41 | −0.52 | 16.1% |

The edge that survives 30 bps of flat costs is gone by about $150m. Rebalancing quarterly
gives up a little at small size but roughly doubles capacity. The square-root law is
calibrated for trades below about 10–20% of daily volume, so the far right of the curve
(where trades exceed ADV) is indicative only.

![Capacity](https://raw.githubusercontent.com/hossainyousuf751/backtest-lab/main/docs/capacity.png)

## Roadmap

- [x] **M1** Engine, strategies, look-ahead test, walk-forward, PSR/DSR, bootstrap CIs, four studies
- [x] **M2** Probability of Backtest Overfitting via combinatorially symmetric cross-validation
- [x] **M3** Effective number of trials (cluster correlated configurations) for a sharper DSR
- [x] **M4** Realistic costs: spread plus square-root market impact as a function of trade size and volume
- [x] **M5** Factor attribution: regress returns on market, size, value and momentum factors to separate alpha from beta
- [x] **M6** Walk-forward pays to switch configuration; borrow fees on shorts; point-in-time universe recipe

A real-data single-stock study remains out of scope: free sources such as Yahoo Finance drop
delisted tickers, so it needs a survivorship-free database (e.g. CRSP or Norgate) plus
historical index membership.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the project's invariants, roles, workflow and
testing rules, and [CHANGELOG.md](CHANGELOG.md) for what changed between versions.

## References

- Bailey, D. & López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk* 15(2).
- Bailey, D. & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management* 40(5).
- Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance* 20(4).
- Politis, D. & Romano, J. (1994). The Stationary Bootstrap. *JASA* 89(428).
- Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and Selling Losers. *Journal of Finance* 48(1).
- López de Prado, M. & Lewis, M. (2019). Detection of False Investment Strategies Using Unsupervised Learning Methods. *Quantitative Finance* 19(9).
- Almgren, R., Thum, C., Hauptmann, E. & Li, H. (2005). Direct Estimation of Equity Market Impact. *Risk* 18(7).
- Newey, W. & West, K. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica* 55(3).
- Fama, E. & French, K. (1993). Common Risk Factors in the Returns on Stocks and Bonds. *Journal of Financial Economics* 33(1).
- Carhart, M. (1997). On Persistence in Mutual Fund Performance. *Journal of Finance* 52(1).
