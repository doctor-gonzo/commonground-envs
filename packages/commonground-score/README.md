# commonground-score

Pure-Python scoring utilities for Common Ground deliberation evaluations.

## API

`commonground_score` exposes:

- `prop_test(successes, trials)`
- `two_prop_test(s_in, s_out, p_in, p_out)`
- `comment_stats(votes)`
- `rating_to_vote(value)`
- `vote_accuracy(predictions, held_out)`
- `brier_score(predictions, held_out)`

`rating_to_vote(value)` implements the canonical 0-10 rating conversion used for
dataset parity:

```text
signed = clamp((2 * (value - 5)) / 10, -1, 1)
vote = sign(signed)
```

An exact rating of `5` maps to neutral/pass (`0`); values below `5` map to
disagree (`-1`), and values above `5` map to agree (`1`).

## Parity Fixtures

The test suite includes a parity fixture loader for
`tests/fixtures/parity_*.json`. Fixture files should use:

```json
{"function":"prop_test","cases":[{"args":[2,4],"expected":0.4472135954999579}]}
```

When no parity fixtures are present, the parity test skips cleanly.
