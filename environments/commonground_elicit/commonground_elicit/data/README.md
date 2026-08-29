# commonground-elicit dataset card

## Scope and provenance

Both bundled splits are fictional and synthetic. Every row records a seed,
template ID/set, template/layout profile, fixed generation date, and generation mode.
The default generator is offline and calls no model or network service.

## Planting and visible structure

Each scenario contains three policy issues: an ambiguity, a cross-document
contradiction, and an uncovered case. The hidden key records exact evidence,
issue type, a decision-focused yes/no diagnosis, decision terms, decision
value, faction stances, and paired evidence for contradictions.

Prompt-visible document and faction IDs are opaque. Titles, styles, document
order, sentence order, faction order, faction count, and issue-specific stance
patterns vary by row. Public faction summaries describe indirect policy
principles; they do not repeat issue decision terms or exact yes/no/pass
stances. Hidden numeric priors and planted keys are not rendered.

## Split separation

| Split | Rows | Template set | Variants per template | Template/layout profile |
| --- | ---: | --- | ---: | --- |
| `train_synthetic.jsonl` | 100 | 4 training templates | 25 | `train-template-layout-profile-v3` |
| `eval_synthetic_heldout.jsonl` | 100 | 20 held-out templates | 5 | `heldout-template-layout-profile-v3` |

Generation distinguishes instance, canonical-prompt, and policy-issue
fingerprints and blocks cross-split overlap. It also enforces token-Jaccard and
word-ngram TF-IDF nearest-neighbor thresholds, profile-label separation, and
the absence of legacy held-out document IDs. Structural signatures separately
capture layout, counts, stance patterns, and evidence relationships. Both
profiles share the same core generator. The committed files regenerate
byte-for-byte.

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
| Prompt-observable | elicit-ask | Removed 0.3 summary/stance codebook | 0.000 |
| Component oracle | elicit-ask | Exact top-K issues + random stances | 0.659 |

These are deterministic comparators, not model evaluations. The removed
codebook parser scores 0.787 on the historical 0.3 public corpus. The component
oracle reads hidden issue selection and is not a model-achievable floor.

The answer keys are public. This corpus is for reproducibility, open training,
and experimental evaluation—not a contamination-resistant leaderboard.
