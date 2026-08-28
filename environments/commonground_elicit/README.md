# commonground-elicit

`commonground-elicit` 0.3.0 is a deterministic Verifiers environment for two
structured tasks over fictional stakeholder-policy scenarios:

- `find`: localize and diagnose planted ambiguities, contradictions, and gaps.
- `elicit-ask`: select the two highest-value clarification targets from three
  planted issues and predict faction stances on each yes/no question.

Version 0.3.0 replaces the 0.2.x corpus and response contract. The prior
held-out split exposed stable document IDs, issue positions, faction IDs, and
stance patterns. The new generator uses opaque IDs, neutral randomized titles,
varied document and sentence order, varied faction count/order, and
issue-specific stance patterns. The exact legacy ID/position codebook now
scores 0.000.

This environment is associated with [Context Engine](https://contextengine.sh),
whose organizational workflow can surface unresolved decisions and collect
stakeholder responses. A future governed exporter could allow consenting
individuals and groups to retain, license, or sell derived preference data.
Version 0.3.0 is entirely synthetic and includes no such exporter.

## Data and separation

| Split | Rows | Templates × variants | Generator family |
| --- | ---: | --- | --- |
| `train` | 100 | 4 × 25 | `train-rotating-layout-v2` |
| `eval` | 100 | 20 × 5 | `heldout-opaque-layout-v2` |

Every row contains three planted issue types and three to five factions. Public
faction summaries include row-specific decision tendencies, so varied stance
targets remain inferable rather than becoming arbitrary hidden labels.
Generation enforces unique prompt-only and answer-key fingerprints within each
split, no exact prompt/answer overlap across splits, and disjoint generator
families. Structural signatures cover issue locations, document/sentence
counts, faction counts, stance patterns, evidence relationships, and generator
family. These checks catch exact structural reuse; they are not a substitute
for embedding-neighbor or low-complexity-classifier audits.

## Find contract

Return one diagnosis per suspected issue:

```json
{
  "findings": [{
    "doc_id": "doc-a1b2c3d4",
    "quote": "The copied primary passage.",
    "type": "contradiction",
    "diagnosis": "Should the emergency exception override prior approval?",
    "related_evidence": {
      "doc_id": "doc-e5f6a7b8",
      "quote": "The copied conflicting rule."
    }
  }]
}
```

`diagnosis` must be a yes/no question that identifies at least half of the
hidden decision terms. A contradiction must cite a contiguous related passage
from a different document; ambiguity and gap findings require
`related_evidence: null`. A gap's diagnosis must state the missing decision,
not merely label a passage vague. Duplicate normalized spans make the response
invalid, so repeating one anchor under all three types cannot hedge.

End-to-end matching requires the correct document and type, at least 90%
contiguous anchor coverage, at least 80% evidence-token precision, and the
structured diagnosis/relationship fields. The environment reports:

- `finding_f1` as the primary reward;
- `finding_localization_recall` as an evidence-location diagnostic;
- `finding_type_accuracy` as a type-classification diagnostic;
- `question_utility` as a companion issue/question signal.

## Ask contract

The default row has three candidate issues and requires exactly two questions,
so selection is real but modest. Each response item copies the source
document/quote, asks a yes/no question expressing at least two latent decision
terms, and predicts `agree`, `disagree`, or `pass` for every visible faction.

```json
{
  "questions": [{
    "doc_id": "doc-a1b2c3d4",
    "quote": "The copied issue passage.",
    "question": "Should an emergency exception override prior approval?",
    "target_stances": {"group-1a2b3c": "agree", "group-4d5e6f": "disagree"}
  }]
}
```

Utility combines exact visible grounding, semantic decision identification,
faction-stance accuracy, panel disagreement, and an issue decision-value
weight. Global assignment prevents duplicate claims. The raw numerator is
divided by the sum of the best two attainable issue utilities for that row, so
an exact top-two response scores 1.0 under every polarization setting. Replacing
a selected issue with a lower-value issue scores strictly less.

This deterministic metric does not judge elegance, conversational usefulness,
or real-world information gain. Three candidates versus K=2 is a first
selection benchmark, not the suggested harder 8–12-candidate design.

## Model-free comparators

Exact results on the bundled 100-row evaluation split:

| Comparator class | Task | Comparator | mean reward |
| --- | --- | --- | ---: |
| Prompt-observable | find | Random visible spans | 0.030 |
| Prompt-observable | find | Flag vague-sounding spans | 0.140 |
| Prompt-observable | find | Legacy 0.2 document-ID/position codebook | 0.000 |
| Prompt-observable | elicit-ask | Template clarity questions | 0.000 |
| Prompt-observable | elicit-ask | Randomly targeted questions | 0.000 |
| Component oracle | elicit-ask | Exact top-K issues + random stances | 0.655 |
| Component oracle | elicit-ask | Exact top-K issues + visible-summary stances | 1.000 |

The zero Ask rows are weak floors that rarely clear the structured grounding
gate; they do not establish model competence. The component oracles read the
hidden top-K issue targets and therefore are not prompt-observable baselines.
Their 0.655-to-1.000 gap isolates the stance component and verifies that the
visible summaries contain enough information for exact stance recovery.
Stronger prompt-only issue detectors remain useful follow-up work.

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
