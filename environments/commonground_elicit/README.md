# commonground-elicit

`commonground-elicit` is a deterministic Verifiers `SingleTurnEnv` for finding
planted ambiguities, contradictions, and gaps in small sets of fictional policy
documents. Its v0 scenarios are synthetic, generated offline from committed
templates, and carry explicit provenance.

The find task accepts strict JSON:

```json
{"findings":[{"doc_id":"policy","quote":"ambiguous passage","type":"ambiguity"}]}
```

The reward is one-to-one finding F1 against the planted answer key. A candidate
must match the document, finding type, and normalized quote at the configured
overlap threshold. Extra findings therefore reduce precision.

Difficulty arguments are `docs_count`, `docs_length`, `planted_density`, and
`distractor_density`. Generation, loading, and scoring do not use the network,
wall clock, or a judge model.
