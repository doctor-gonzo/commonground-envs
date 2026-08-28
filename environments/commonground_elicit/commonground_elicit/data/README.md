# commonground-elicit dataset card

## Scope and provenance

Both bundled splits are fictional and synthetic. Every row records a seed,
template ID/set, generator family, fixed generation date, and generation mode.
The default generator is offline and calls no model or network service.

## Planting and visible structure

Each scenario contains three policy issues: an ambiguity, a cross-document
contradiction, and an uncovered case. The hidden key records exact evidence,
issue type, a decision-focused yes/no diagnosis, decision terms, decision
value, faction stances, and paired evidence for contradictions.

Prompt-visible document and faction IDs are opaque. Titles, styles, document
order, sentence order, faction order, faction count, and issue-specific stance
patterns vary by row. Public faction summaries describe row-specific decision
tendencies so the varied stances remain inferable; hidden numeric priors and
planted keys are not rendered.

## Split separation

| Split | Rows | Template set | Variants per template | Generator family |
| --- | ---: | --- | ---: | --- |
| `train_synthetic.jsonl` | 100 | 4 training templates | 25 | `train-rotating-layout-v2` |
| `eval_synthetic_heldout.jsonl` | 100 | 20 held-out templates | 5 | `heldout-opaque-layout-v2` |

Generation blocks duplicate prompt-only or answer fingerprints within either
split, exact prompt/answer overlap across splits, shared generator-family
labels, and the legacy held-out document IDs. Structural signatures separately
capture layout, counts, stance patterns, and evidence relationships. The
committed files regenerate byte-for-byte.

## Find and Ask targets

Find requires localized primary evidence, a single type, a semantic diagnosis,
and paired evidence for contradictions. Ask requires exactly two questions for
the three visible candidate issues, plus complete faction stance predictions.
Its reward is normalized by the top-two attainable utility on that row.

## Model-free comparators

| Comparator class | Task | Comparator | mean reward |
| --- | --- | --- | ---: |
| Prompt-observable | find | Random visible spans | 0.030 |
| Prompt-observable | find | Flag vague-sounding spans | 0.140 |
| Prompt-observable | find | Legacy 0.2 document-ID/position codebook | 0.000 |
| Prompt-observable | elicit-ask | Template clarity questions | 0.000 |
| Prompt-observable | elicit-ask | Randomly targeted questions | 0.000 |
| Component oracle | elicit-ask | Exact top-K issues + random stances | 0.655 |
| Component oracle | elicit-ask | Exact top-K issues + visible-summary stances | 1.000 |

These are deterministic comparators, not model evaluations. Component oracles
read the hidden issue selection but use random or prompt-visible stances; they
diagnose rubric components and are not model-achievable floors.

The answer keys are public. This corpus is for reproducibility, open training,
and experimental evaluation—not a contamination-resistant leaderboard.
