# commonground-elicit dataset card

## Scope and provenance

Both bundled splits are fictional and synthetic. Every row records a seed,
template ID/set, template/layout profile, fixed generation date, and generation
mode. The offline generator calls no model or network service.

## Planting and visible structure

Each scenario contains an ambiguity, a cross-document contradiction, and an
uncovered case. The hidden key records exact primary evidence, issue type, a
yes/no authoring question, the yes-side polarity, value trade-off, composed
faction stances, decision value, and explicitly authored second evidence for a
contradiction.

Prompt-visible document and faction IDs are opaque. Titles, styles, document
order, sentence order, faction order, faction count, and stance patterns vary
by row. Public faction summaries render general values—access, adaptability,
continuity, oversight, and safety—without consulting planted issues or answer
labels. The generator derives issue stances compositionally from those values
and explicit alternatives; it does not render a phrase selected by hidden
issue type or stance.

Every contradiction template authors a stable second document and anchor.
Generation remaps document IDs while preserving that relationship. Validation
rejects a related passage that is absent, same-document, another planted
anchor, or a distractor.

## Split separation

| Split | Rows | Template set | Variants per template | Template/layout profile |
| --- | ---: | --- | ---: | --- |
| `train_synthetic.jsonl` | 100 | 4 training templates | 25 | `train-template-layout-profile-v4` |
| `eval_synthetic_heldout.jsonl` | 100 | 20 held-out templates | 5 | `heldout-template-layout-profile-v4` |

Generation distinguishes instance, canonical-prompt, and policy-issue
fingerprints and blocks cross-split overlap. It also enforces token-Jaccard and
word-ngram TF-IDF nearest-neighbor thresholds, profile-label separation, and
the absence of legacy held-out document IDs. Structural signatures separately
capture layout, counts, value/stance patterns, and evidence relationships.
Both profiles share one core generator. The committed files regenerate
byte-for-byte.

## Find and Ask targets

Find requires localized primary evidence, a single issue type, a well-formed
yes/no diagnosis, and authored paired evidence for contradictions. Ask requires
exactly two of three visible candidate issues plus type, yes-side polarity,
relationship evidence, and complete faction stance predictions. Free-form
question prose is not compared with hidden canonical tokens. Ask reward is
normalized by the top-two attainable utility on that row.

## Model-free comparators

| Comparator class | Task | Comparator | mean reward |
| --- | --- | --- | ---: |
| Prompt-observable | find | Random visible spans | 0.080 |
| Prompt-observable | find | Flag vague-sounding spans | 0.195 |
| Prompt-observable | find | Legacy 0.2 document-ID/position codebook | 0.000 |
| Prompt-observable | elicit-ask | Template clarity questions | 0.078 |
| Prompt-observable | elicit-ask | Randomly targeted questions | 0.069 |
| Prompt-observable | elicit-ask | Removed 0.3 summary/stance codebook | 0.000 |
| Component oracle | elicit-ask | Exact issues + removed 0.4 principle-table parser | 0.000 |
| Source-aware prompt-only | elicit-ask | Public template detector + removed 0.4 principle-table parser | 0.000 |
| Component oracle | elicit-ask | Exact top-K issues + random stances | 0.670 |
| Component oracle | elicit-ask | Exact top-K issues + exact stances (ceiling) | 1.000 |

The removed 0.4 phrase decoder reaches 1.0 on its frozen historical fixture
with either exact issues or the public template detector, and 0.0 on this
corpus. Component oracles read hidden issue selection and are not
prompt-observable floors.

The answer keys are public. This corpus is for reproducibility, open training,
and experimental evaluation—not a contamination-resistant leaderboard.
