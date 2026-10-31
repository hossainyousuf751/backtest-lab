## What and why

<!-- One focused change. Link the issue or roadmap item. -->

## Checklist ([CONTRIBUTING.md](../CONTRIBUTING.md))

- [ ] `pytest -q` passes locally
- [ ] Bug fix: a test that failed before the fix; new logic: a test with a known answer
- [ ] No look-ahead: new strategies pass `check_lookahead`; rolling windows end at the decision date
- [ ] Anything that trades pays costs, or the skipped cost is documented
- [ ] Affected `examples/` scripts re-run; README numbers and `docs/` charts updated if they moved
- [ ] `CHANGELOG.md` entry under `Unreleased` if behaviour changed
- [ ] No new runtime dependency (optional extras only)

## Deliberately skipped

<!-- Anything left out on purpose, and when it should be added. -->
