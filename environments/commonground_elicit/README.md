# commonground-elicit

`commonground-elicit` 0.5.0 is an unreleased deterministic Verifiers candidate
for two structured tasks over fictional stakeholder-policy scenarios:

- `find`: localize and diagnose planted ambiguities, contradictions, and gaps.
- `elicit-ask`: select the two highest-value clarification targets from three
  planted issues and predict faction stances on each yes/no question.

The current public 0.4.1 artifact remains available for reproducibility but is
superseded for benchmark use. Five `regional-archives-access` rows selected the
wrong second contradiction passage, and public faction summaries used a finite
phrase table keyed by issue type and exact target stance. Version 0.5 replaces
both mechanisms and requires a fresh Elicit study before promotion.

This environment is associated with [Context Engine](https://contextengine.sh),
whose organizational workflow can surface unresolved decisions and collect
stakeholder responses. A future governed exporter could allow consenting
individuals and groups to retain, license, or sell derived preference data.
Version 0.5 is entirely synthetic and includes no such exporter.

## Data and separation

| Split | Rows | Templates × variants | Template/layout profile |
| --- | ---: | --- | --- |
| `train` | 100 | 4 × 25 | `train-template-layout-profile-v4` |
| `eval` | 100 | 20 × 5 | `heldout-template-layout-profile-v4` |

Every row contains one ambiguity, contradiction, and gap plus three to five
factions. Each faction has a general value vector spanning access,
adaptability, continuity, oversight, and safety. Its public summary renders
those values once without consulting any planted issue or target stance. Each
issue separately defines a value trade-off and whether “yes” selects the
primary rule or an alternative. The hidden stance vector is recomputed from
that composition, so reversing question polarity also reverses agree/disagree.

Every contradiction template authors a stable opposing document and exact
quote. Generation remaps opaque IDs but never infers the relationship from
lexical overlap. Validation rejects a related passage that is absent, in the
same document, another planted anchor, or a distractor. Corpus-wide tests cover
all 24 template families, including the corrected regional-archives phone-
photograph rule.

Generation separately audits exact instances, canonical visible propositions,
and policy-issue semantics after removing opaque identity/layout fields. It
blocks cross-split overlap for all relevant hashes and enforces bounded token-
Jaccard and word unigram/bigram TF-IDF neighbors. These deterministic checks
are not an embedding proof of conceptual independence. Both profiles still
share one core synthetic generator.

## Find contract

Return one diagnosis per suspected issue:

```json
{
  "findings": [{
    "doc_id": "doc-a1b2c3d4",
    "quote": "The copied primary passage.",
    "type": "contradiction",
    "diagnosis": "Should the practical exception override the primary rule?",
    "related_evidence": {
      "doc_id": "doc-e5f6a7b8",
      "quote": "The copied conflicting rule."
    }
  }]
}
```

`diagnosis` must be a well-formed yes/no question. It is not compared with a
hidden canonical vocabulary: exact public evidence, issue type, and
relationship carry deterministic semantic grounding. Contradictions require a
contiguous second passage from another document; ambiguity and gap findings
require `related_evidence: null`. Duplicate normalized spans invalidate the
response, so repeating one anchor under all types cannot hedge.

End-to-end matching requires the correct document and type, at least 90%
contiguous anchor coverage, at least 80% evidence-token precision, valid
diagnosis form, and the correct relationship. The environment reports strict
`finding_f1` plus localization, type, diagnosis, and relationship diagnostics.
`question_utility` remains a weight-zero companion signal.

## Ask contract

The default row has three candidate issues and requires exactly two questions.
Each item must identify the issue structurally and declare what “yes” means:

```json
{
  "questions": [{
    "doc_id": "doc-a1b2c3d4",
    "quote": "The copied issue passage.",
    "type": "contradiction",
    "question": "Should the practical exception override the primary rule?",
    "yes_choice": "alternative",
    "related_evidence": {
      "doc_id": "doc-e5f6a7b8",
      "quote": "The copied conflicting rule."
    },
    "target_stances": {
      "group-1a2b3c": "agree",
      "group-4d5e6f": "disagree"
    }
  }]
}
```

Question prose must have yes/no form but is not compared with hidden authored
tokens or aliases. Exact visible grounding, type, `yes_choice`, relationship,
and complete faction stances determine semantic credit. If a model reverses
the question orientation and inverts agree/disagree consistently, it receives
the same stance credit. Separate metrics report question format validity,
grounding recall, and stance accuracy.

Utility combines structured issue grounding, stance accuracy, panel
disagreement, and the share of factions whose composed position is non-pass.
There is no policy-keyword value table. Global assignment prevents duplicate
claims, and normalization by the best attainable two-item sum makes the exact
top-two response score 1.0. This is clarification-target selection—not a
measurement of real-world information gain. Three candidates versus K=2
remains a deliberately small ranking problem.

## Model-free comparators

Exact results on the bundled 100-row 0.5 candidate evaluation split:

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

The historical 0.4 decoder is tested against a frozen old-format fixture, where
both the exact-issue component and public-template prompt detector reach 1.0,
and against the 0.5 corpus, where both reach 0.0. Component oracles read hidden
issue selection and are not prompt-observable floors.

## Usage

```bash
uv run validate commonground-elicit --runtime.type subprocess --rich false
uv run validate commonground-elicit --taskset.task-mode elicit-ask \
  --runtime.type subprocess --rich false
uv run eval commonground-elicit -m MODEL --no-push
uv run eval commonground-elicit --env.taskset.task-mode elicit-ask \
  -m MODEL --no-push
```

Difficulty controls are `docs_count`, `docs_length`, `planted_density`,
`distractor_density`, `panel_polarization`, and `question_count`. Parsing is
bounded and malformed output fails closed. Public planted keys support open
training but not contamination-resistant comparison; use a fresh private
generator family for consequential evaluation.

Find additionally accepts `reward_mode="shaped"` for training. It averages
localization, type, diagnosis, and relation F1 scores on `[0,1]`, charging every
unmatched candidate at every stage. Adding a false positive or overlapping
hedge strictly lowers reward relative to a concise exact answer. The default is
`reward_mode="strict"`; reported evaluations must use it. Neither objective has
demonstrated learning value until multi-seed training and fresh-family transfer
are run.
