# Common Ground 0.5.0 evaluation plan

Status: released public artifacts, complete exact-artifact study, and 12
version-pinned public evaluation records. The formal source tag/release remains
an operator-owned finalization step.

## Release target

| Distribution | Candidate version | Role |
| --- | --- | --- |
| `commonground-score` | 0.5.0 | Probability scoring and Brier-skill helpers |
| `commonground-scenarios` | 0.5.0 | Scenario schema 3, generation, and validation |
| `commonground-predict` | 0.5.0 | Probabilistic masked-vote prediction |
| `commonground-elicit` | 0.5.0 | Grounded finding and clarification selection |

Version 0.5 is an immutable correction to Elicit 0.4.1. It replaces inferred
contradiction relationships with authored references, replaces issue/stance
phrases with issue-independent faction values, makes shaped Find reward
precision-sensitive, and removes hidden canonical vocabulary from Ask scoring.
Because answer keys and the visible construct change, every 0.4.1 Elicit model
score is historical and must not be carried forward.

Predict's task and corpus remain compatible in concept, but its package is
versioned with the release family and its comparator report changes. New 0.5
artifacts should therefore receive fresh installation and smoke evidence even
if a complete Predict model rerun is treated as confirmatory rather than a
correction.

## Release-blocking integrity evidence

Before publication, retain machine-readable evidence that:

1. every contradiction in all 24 template families resolves to its explicitly
   authored opposing document and quote;
2. no contradiction relationship duplicates another planted anchor or a
   distractor, including all 200 generated Elicit rows;
3. all target stances recompute from general faction values, issue trade-offs,
   and explicit question polarity;
4. changing an issue's trade-offs cannot change public faction summaries, and
   reversing question polarity reverses agree/disagree labels consistently;
5. the removed 0.4 principle-table decoder scores zero on the 0.5 corpus and
   one on its historical fixture;
6. adding false positives or overlapping hedges strictly lowers shaped Find
   reward relative to a concise exact answer;
7. synonym-rich valid questions receive the same score when their structured
   evidence, type, polarity, relationship, and stance fields are identical;
8. generated splits are byte-identical after regeneration and pass exact,
   semantic, token-Jaccard, and TF-IDF separation gates; and
9. clean wheels and source archives install all four exact versions together,
   include required legal/provenance files, and load all three task modes.

## Model-free evidence

Publish Predict probability-native comparators with probability reward,
normalized Brier, vote accuracy, and Brier skill against uniform. At minimum:

- uniform, train-split empirical, and explicitly transductive visible priors;
- per-statement visible class frequencies;
- one-hot nearest-neighbor votes;
- neighbor vote-frequency distributions;
- smoothed distance-weighted k-NN;
- a train-split text-only probabilistic baseline; and
- clearly labeled held-out-label and generator diagnostics.

Publish Elicit floors for random/vague spans, removed 0.2/0.3/0.4 shortcut
parsers, a public-template prompt detector paired with the removed 0.4 decoder,
exact issues with random stances, and the exact structured ceiling. Component
oracles must remain labeled as non-prompt-observable.

## Private artifact study

The completed study pinned both 0.5 Hub versions and ran 100 tasks with five
rollouts for Predict, Elicit Find, and Elicit Ask across four model families,
including an open-weight trainable model. Eleven runs contain no recovery or
error history. One Qwen Ask rollout contains a provider 502 caused by a
malformed upstream finish reason followed by a successful internal retry. It
is retained and explicitly disclosed because the same provider failure
reproduced in two replacement attempts, including at concurrency one; no run
was selected based on score.

For Elicit, rerunning both modes is mandatory because the corpus, public faction
descriptions, answer polarity, and scoring diagnostics changed. Report:

- strict reward, rollout standard deviation, and template-hierarchical 95%
  bootstrap intervals;
- paired model differences using shared template/variant resamples;
- Find localization, type, diagnosis, and relationship diagnostics; and
- Ask format, grounding, and stance diagnostics.

For Predict, report task-clustered intervals, all probability-native
comparators, and Brier skill. Avoid describing raw `1 - Brier` without the
uniform reference, because uniform already earns two thirds under the
three-class normalized convention.

## Publication and claim boundary

The 0.5 evaluation report records source commit, wheel/sdist hashes, Hub
version IDs/content hashes, exact configs, run IDs, and all 12 public
evaluation IDs. Create an annotated `v0.5.0` tag and formal source release only
after the recorded documentation commit is final; these are operator-owned
remote actions.

Supported claims remain limited to experimental synthetic
semantic-conditioned matrix completion and structured policy-issue
localization/clarification selection. Do not claim real organizational
information gain, human-preference validity, contamination-resistant
leaderboard performance, or demonstrated RL training utility without a
multi-seed intervention and a fresh independently implemented private
generator.
