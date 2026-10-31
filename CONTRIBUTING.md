# Contributing to backtest-lab

This file is the rulebook. Everyone working on the project, human or AI agent, follows it so
that changes land in one consistent direction. If a rule here conflicts with a habit, the rule
wins; if a rule is wrong, change this file in its own pull request first.

## 1. What the project protects

`btlab` exists to answer one question: *should you believe this backtest?* Every change is
judged against these invariants. A change that breaks one is a bug, however good the numbers look.

1. **No look-ahead.** Row *t* of a strategy's weights uses data up to the close of *t* only.
   Every strategy must pass `check_lookahead`. Any rolling window (volatility, ADV, scores) must
   end at or before the decision date.
2. **Explicit timing.** Weights decided at close *t* trade at close *t + execution_lag*. Costs
   are charged on the day of the trade; returns accrue from the next day.
3. **Costs are never free by accident.** Anything that trades (rebalances, walk-forward
   switches, entry from cash) pays costs. A cost you deliberately skip is documented in the
   docstring and README.
4. **Point-in-time data.** Assets cannot be bought before they list. Delisted holdings take the
   delisting return and become cash. Strategies see only the universe that existed at the time.
5. **Results are reproducible.** Every number in the README comes from a script in `examples/`
   with a fixed seed. If a code change moves a number, the README is updated in the same PR.
6. **Honest reporting.** Results that make a method look bad (e.g. PBO being fooled by a lucky
   sample) are kept, not tuned away.

## 2. Roles and responsibilities

| Role | Who | Responsible for |
|---|---|---|
| **Maintainer** | @hossainyousuf751 | Roadmap and priorities, merging to `main`, releases and version numbers, final say on this file |
| **Contributor** | anyone opening a PR | One focused change, its tests, README/CHANGELOG updates, green CI before asking for review |
| **Reviewer** | maintainer or a delegate | Checking the invariants in section 1, that tests would fail if the logic broke, that README numbers match a fresh run |
| **AI agent** | Claude Code or similar | Same as a contributor, plus the agent rules in section 7. Never pushes to `main`, never merges, never tags releases |

Module responsibilities (what each module guarantees to the others):

| Module | Owns | Must not |
|---|---|---|
| `strategies.py` | Mapping prices to target weights, causally | Know about costs, lag, or rebalancing |
| `engine.py` | Execution: lag, drift, costs, impact, borrow, delistings | Make investment decisions |
| `validation.py` | Look-ahead test, walk-forward, PBO, effective trials | Change engine semantics (it calls `run_backtest`) |
| `metrics.py` | Statistics on return series | Depend on the engine or strategies |
| `data.py` | Synthetic and external data loaders | Clean data in ways that add survivorship bias |
| `plotting.py` | Charts only | Compute anything a test should check |

## 3. Workflow

1. **Plan.** Non-trivial work starts as a GitHub issue or a roadmap line in the README, so two
   people never build the same thing. Assign yourself before starting.
2. **Branch** from up-to-date `main`: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `ci/<topic>`.
3. **One change per PR.** A bug fix does not also refactor; a new feature does not also restyle.
4. **Commits** use a short imperative subject (`Fix weekly rebalance at ISO year boundary`).
   Explain *why* in the body when it is not obvious.
5. **Before pushing**, run locally:
   ```bash
   pytest -q
   cd examples && python 01_overfitting.py && python 02_survivorship.py \
     && python 03_planted_signal.py && python 05_capacity.py
   ```
6. **Open a PR** using the template; CI must be green on Python 3.10–3.12.
7. **Merge** is squash-merge by the maintainer. Delete the branch afterwards.
8. `main` is always releasable. Nobody pushes to it directly.

## 4. Code rules

- **Match the surrounding code.** Vectorised numpy/pandas, dataclasses for results, short
  docstrings that state the *assumption* being made, not a restatement of the code.
- **Dependencies:** runtime stays `numpy`, `scipy`, `pandas`. Anything else goes in an optional
  extra (`plot`, `data`, `dev`) and is imported inside the function that needs it.
- **No speculative abstraction.** No base class with one implementation, no config option for a
  value nobody changes. Add flexibility when the second use case arrives.
- **Deliberate shortcuts** are marked `# ponytail: <limit>, <upgrade path>` so they can be found
  with `grep -rn "ponytail:" src`.
- **Public API** is what `btlab/__init__.py` exports. Adding to it is a minor version bump;
  changing or removing from it is a breaking change and must be called out in the CHANGELOG.
- **Units at the interface:** Sharpe ratios are annualised, costs are in bps, weights are
  fractions of NAV, returns are simple (not log) daily returns.

## 5. Testing rules

- Every bug fix comes with a test that fails before the fix.
- Every new piece of logic gets **one test with a known answer**: a synthetic case where the
  correct output can be computed by hand or is fixed by theory (e.g. PBO ≈ 0.5 on noise, a
  recovered regression coefficient, a cost equal to turnover × bps).
- Tests are deterministic: fixed seeds, no network. Code needing the internet
  (`load_yfinance`, `load_french_factors`) is exercised by `examples/04`, not by `pytest`.
- Never weaken a test to get to green: no loosened tolerances, skipped tests or deleted asserts
  without a written reason in the PR.

## 6. Documentation and release rules

- README study numbers are regenerated, not hand-edited, and updated in the same PR as the code
  that moved them. Charts in `docs/` are regenerated by the same scripts.
- `CHANGELOG.md` gets an entry under `Unreleased` in every PR that changes behaviour.
- Versions follow semantic versioning in `pyproject.toml`. Release = maintainer moves
  `Unreleased` to a version heading, bumps the version, tags `vX.Y.Z` on `main`.
- New methods cite their source paper in the docstring and in the README references.

## 7. Rules for AI agents

- Read this file and `CLAUDE.md` before making changes.
- Work on a branch; propose changes as a PR. Do not push to `main`, merge, tag, change repository
  settings or visibility unless the maintainer asks for that specific action.
- Verify before claiming: run the tests and the affected example scripts, and quote the actual
  output. Report failures as failures.
- Check that external facts (library versions, API behaviour, paper citations) are current
  instead of relying on memory.
- Keep diffs minimal and on-topic; list anything deliberately skipped in the PR description.
