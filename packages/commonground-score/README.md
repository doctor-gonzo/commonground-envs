# commonground-score

Pure-Python scoring utilities for Common Ground deliberation evaluations.

## API

`commonground_score` exposes:

- `prop_test(successes, trials)`
- `two_prop_test(s_in, s_out, p_in, p_out)`
- `comment_stats(votes)`
- `vote_entropy(votes)`
- `cluster_separation(votes)`
- `rating_to_vote(value)`
- `vote_accuracy(predictions, held_out)`
- `brier_score(predictions, held_out)`

`vote_entropy` normalizes agree/disagree/pass entropy to `[0, 1]`.
`cluster_separation` returns the fraction of faction-vote pairs taking different
stances. Together they provide the deterministic panel-disagreement math used
by `commonground-elicit`; missing votes are excluded from both calculations.

`brier_score` accepts bare point predictions (`1`, `-1`, or `0`) and
probability mappings keyed by `agree`, `disagree`, and `pass` or their numeric
equivalents. Valid non-negative finite mappings are normalized before scoring.
Invalid or non-normalizable mappings score as the uniform distribution
(`1/3`, `1/3`, `1/3`) to represent no information. Version 0.2 divides the
three-class squared-error sum by two, so the returned score is bounded to
`[0,1]`; callers requiring the unnormalized convention must multiply by two.

`rating_to_vote(value)` implements the canonical 0-10 rating conversion used for
dataset parity:

```text
signed = (2 * (value - 5)) / 10
```

Values outside the inclusive `0`-`10` range, plus non-finite values, map to
neutral/pass (`0`). In-range values below `5` return a negative score, values
above `5` return a positive score, and exactly `5` returns `0`.

## Parity Fixtures

The test suite includes a parity fixture loader for
`tests/fixtures/parity_*.json`. The harness requires fixtures for
`prop_test`, `two_prop_test`, `comment_stats`, and `rating_to_vote`. Fixture
files should use:

```json
{"function":"prop_test","cases":[{"args":[2,4],"expected":0.4472135954999579}]}
```

JSON cannot encode NaN, so NaN arguments are represented as the exact string
`"NaN"` and decoded by the harness before invocation.
