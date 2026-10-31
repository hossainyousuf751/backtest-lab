# CLAUDE.md

Agent entry point. The rules live in [CONTRIBUTING.md](CONTRIBUTING.md); read it first,
especially section 1 (invariants) and section 7 (agent rules).

## Commands

```bash
pip install -e ".[dev]"          # add ,data for examples/04 (needs internet)
pytest -q                        # deterministic, no network, ~5 s
cd examples && python 01_overfitting.py   # each study regenerates its README numbers and docs/ charts
```

## Map

- `src/btlab/strategies.py`: prices → target weights (row t decided at close t)
- `src/btlab/engine.py`: `run_backtest`: lag, drift, costs, impact, borrow, delistings
- `src/btlab/validation.py`: `check_lookahead`, `walk_forward`, PBO, `effective_trials`
- `src/btlab/metrics.py`: Sharpe family, PSR/DSR, bootstrap CI, `factor_attribution`
- `src/btlab/data.py`: synthetic markets and volume; Yahoo and Ken French loaders
- `examples/0N_*.py`: one script per README study

## Things that are easy to get wrong here

- Any rolling statistic used at date t must not include t+1. Prove it with `check_lookahead`
  or a truncation comparison.
- Costs on a day accumulate (`+=`): rebalance costs, impact and borrow can all land on one day.
- `sharpe` and friends trim leading zeros (warm-up), so slices that start in cash are not penalised.
- Changing the engine moves README numbers: re-run the examples and update the README in the
  same change.
