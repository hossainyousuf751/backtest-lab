"""Pin the README headline numbers to the offline examples (same markets, seeds and settings)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btlab import (
    CrossSectionalMomentum, annual_return, cost_sensitivity, deflated_sharpe, effective_trials,
    probabilistic_sharpe, probability_of_backtest_overfitting, run_backtest, sharpe,
    synthetic_dollar_volume, synthetic_market, walk_forward,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from _common import MOMENTUM_GRID  # noqa: E402


def approx(x):
    return pytest.approx(x, rel=1e-12)


def equal_weight(p: pd.DataFrame) -> pd.DataFrame:
    listed = p.notna().astype(float)
    return listed.div(listed.sum(axis=1), axis=0)


def test_study2_survivorship():
    p = synthetic_market(n_assets=150, n_years=12, delist_rate=0.04, late_listing_frac=0.2, seed=5)
    want = {"pit": ((0.09275159098134589, 0.6672911554000381), (0.1612048957892651, 0.975010746985403)),
            "surv": ((0.12777134181852468, 0.8770110000831877), (0.15616589541407788, 0.9315362624874359))}
    for label, q in (("pit", p), ("surv", p.loc[:, p.iloc[-1].notna()])):
        ew = run_backtest(q, equal_weight(q), cost_bps=2).returns
        lo = run_backtest(q, CrossSectionalMomentum(long_short=False).weights(q)).returns
        assert (annual_return(ew), sharpe(ew)) == approx(want[label][0])
        assert (annual_return(lo), sharpe(lo)) == approx(want[label][1])


def test_study3_planted_signal_and_study5_capacity():
    p = synthetic_market(n_assets=100, n_years=12, momentum_strength=0.15, seed=21)
    w = CrossSectionalMomentum().weights(p)
    base = run_backtest(p, w, cost_bps=5)
    assert (sharpe(base.returns), annual_return(base.returns), base.turnover.sum()) == approx(
        (1.0569624846827514, 0.06290842463601831, 68.07207801720337))
    borrow = [sharpe(run_backtest(p, w, cost_bps=5, borrow_bps=f).returns) for f in (50, 300)]
    assert borrow == approx([1.014837374866543, 0.804212269266613])
    assert list(cost_sensitivity(p, w)["sharpe"]) == approx(
        [1.1093705569641976, 1.0883169765622018, 1.0569624846827514, 1.0045202768770523, 0.8990753629579024,
         0.7931303437370779, 0.5809273500793127])

    vol = synthetic_dollar_volume(p, median_adv=2e7, seed=21)
    for rebalance, want in (("M", [0.9643619238967769, 0.130824838704816, -1.4058797506596943]),
                            (63, [0.9383047947932504, 0.4346273189129186, -0.5150818109920978])):
        got = [sharpe(run_backtest(p, w, rebalance=rebalance, cost_bps=5, dollar_volume=vol, capital=c).returns)
               for c in (1e6, 1e8, 1e9)]
        assert got == approx(want)


@pytest.mark.slow
def test_study1_overfitting_and_study3_walk_forward():
    srs = np.array([sharpe(run_backtest(p, CrossSectionalMomentum().weights(p)).returns)
                    for p in (synthetic_market(n_assets=100, n_years=12, seed=s) for s in range(30))])
    assert (srs.mean(), srs.std(ddof=1)) == approx((0.014964030326057387, 0.3319227062972986))

    wf = walk_forward(synthetic_market(n_assets=100, n_years=12, seed=6), CrossSectionalMomentum, MOMENTUM_GRID,
                      train_days=756, test_days=252)
    s, best = wf.trial_sharpes, wf.trial_returns[wf.best_in_sample]
    got = (s[wf.best_in_sample], probabilistic_sharpe(best), deflated_sharpe(best, s),
           probability_of_backtest_overfitting(wf.trial_returns)[0], sharpe(wf.oos_returns),
           effective_trials(wf.trial_returns))
    assert got == approx((0.8575397287118948, 0.9971587834152211, 0.9198087669115991, 0.3439005439005439,
                          0.8191936239531838, 2))

    p3 = synthetic_market(n_assets=100, n_years=12, momentum_strength=0.15, seed=21)
    wf3 = walk_forward(p3, CrossSectionalMomentum, MOMENTUM_GRID, cost_bps=5)
    b3 = wf3.trial_returns[wf3.best_in_sample]
    got = (deflated_sharpe(b3, wf3.trial_sharpes), probability_of_backtest_overfitting(wf3.trial_returns)[0],
           sharpe(wf3.oos_returns))
    assert got == approx((0.995930140088623, 0.544055944055944, 1.0112705367526902))
