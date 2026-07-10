# commonground-predict

`commonground-predict` is a deterministic Verifiers `SingleTurnEnv` for
predicting held-out votes in deliberation snapshots. It has no judge model: the
rubric compares strict JSON predictions against masked ground truth.

## Dataset Card

Each source line is a JSON session snapshot:

- `session_id`: snapshot identifier.
- `statements`: ordered objects with `index` and `text`.
- `participants`: pseudonymous participant IDs.
- `votes`: participant-by-statement matrix with `1` agree, `-1` disagree, `0`
  pass/unsure, and `null` for not seen or masked.
- `masked_cells`: `[participant_idx, statement_idx]` cells hidden from the
  prompt.
- `held_out`: true labels for masked cells keyed as `"<p>,<s>"`.
- `clusters`: planted participant cluster IDs or exported cluster objects.
- `meta`: includes `k_anonymity`, `source`, and `synthetic`.

Bundled splits:

- `data/eval_synthetic.jsonl`: 20 seeded synthetic snapshots with coherent
  planted clusters plus noise (`meta.synthetic: true`).
- `data/eval_ce_demo.jsonl`: one real
  [Context Engine](https://github.com/AgalmicSoftware/context-engine)
  demo-corpus snapshot with `meta.source: "ce-demo"` and
  `meta.synthetic: false`. It was exported via the CE snapshot exporter with
  redaction and k-anonymity tests, and remains demonstration-corpus data.
  This demo split and raw CE exporter output are unmasked and require
  `masked_vote_count` when loaded.

Real exported splits should use plaintext/public content only, pseudonymized
participants, and a k-anonymity floor of `k=5`.

## Methodology Attribution

This environment's statistics follow open-source Polis math conventions. Polis
methodology is attributed to the Computational Democracy Project and the Polis
open-source community.

## Configuration

Environment variable:

- `COMMONGROUND_DATA_PATH`: optional path to a JSONL file of session snapshots.
  If unset, the bundled synthetic eval split is used. Set it to
  `data/eval_ce_demo.jsonl` to run against the bundled CE demo split.

Difficulty knobs passed to `load_environment(**kwargs)`:

- `masked_vote_count`: deterministically remasks each snapshot to this many
  held-out votes. Non-positive values mask no votes; values larger than the
  known-vote pool mask every known vote. If omitted, each snapshot uses the
  masks already present in the data; entirely unmasked data is rejected.
- `min_cluster_count`: filters snapshots to those with at least this many
  participant clusters.

## Prompt And Response

The prompt renders statements, a compact visible vote matrix, and the list of
masked cells. Models must return strict JSON:

```json
{"predictions":{"<participant_idx>,<statement_idx>":1}}
```

Values must be `1`, `-1`, or `0`. The parser tolerates fenced JSON and extracts
the first JSON object from the completion.

## Scoring

The rubric uses:

- `vote_accuracy`, weight `1.0`: exact-match fraction over held-out cells.
- `brier`, weight `0.0`: logged multiclass Brier metric only.

Baseline numbers are TBD for both bundled splits.

## Evaluation

After installing dependencies with uv, run a local Verifiers evaluation with:

```bash
uv run vf-eval commonground-predict
```

This repository proves wiring with pytest; it does not require API keys for the
test suite.
