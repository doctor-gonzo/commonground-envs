# commonground-elicit

`commonground-elicit` 0.6.0 is an unreleased deterministic Verifiers candidate
for two structured tasks over fictional stakeholder-policy scenarios:

- `find`: localize and diagnose planted ambiguities, contradictions, and gaps.
- `elicit-ask`: select the single highest-value clarification target from
  three planted issues and predict faction stances on its yes/no question.

Version 0.5.0 remains the current public artifact and its evaluation records
remain immutable historical evidence. Version 0.6 changes the visible corpus,
response contract, issue ranking, and diagnostics; it requires a fresh
exact-artifact study before publication and does not inherit 0.5 scores.

This environment is associated with [Context Engine](https://contextengine.sh),
whose organizational workflow can surface unresolved decisions and collect
stakeholder responses. A future governed exporter could allow consenting
individuals and groups to retain, license, or sell derived preference data.
Version 0.6 is entirely synthetic and includes no such exporter.

## Data and separation

| Split | Rows | Templates × variants | Template/layout profile |
| --- | ---: | --- | --- |
| `train` | 100 | 4 × 25 | `train-template-layout-profile-v6` |
| `eval` | 100 | 20 × 5 | `heldout-template-layout-profile-v6` |

Every row contains one ambiguity, contradiction, and gap plus three to five
factions. Each faction has a general value vector spanning access,
adaptability, continuity, oversight, and safety. Its prompt-visible summary prints the
exact signed values used by the synthetic generator and then renders those
values as issue-independent prose. Each issue separately defines a value
trade-off and whether “yes” selects the primary rule or an alternative. The
hidden stance vector is recomputed from that composition, so reversing question
polarity also reverses agree/disagree.

The 18 helper-built held-out families have five documents each: one primary
document per issue, a separate authored contradiction-related document, and a
plant-free operations ledger. Each document receives an independently sampled
number of seeded, compositionally generated type-neutral administrative
distractors before sentence order, title, and style are randomized. There is no
fixed distractor sentence pool or equal sentence-count layout. Neutral
sentences independently use one or two clauses from the same procedural
vocabulary as authored spans, balancing sentence lengths without a fixed
clause-count marker. The two independently authored held-out families retain
their own layouts. A source-aware predicate-exclusion floor
therefore has no exclusive marker. Audits report planted and distractor rates
for that vocabulary plus residual title, style, position, sentence-count, and
length signal instead of assuming randomization removed it.

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

Return one diagnosis per suspected issue. The primary root may contain just the
`findings` array. It may also include exactly one companion `questions` entry,
as illustrated below:

```json
{
  "findings": [{
    "doc_id": "doc-a1b2c3d4",
    "quote": "Only the records desk may release a file.",
    "type": "contradiction",
    "diagnosis": "Should the duty archivist release files for urgent requests (yes selects the alternative outcome: duty archivist release urgent requests)?",
    "decision": {
      "actor": "duty archivist",
      "action": "release file",
      "condition": "urgent requests",
      "anchor_outcome": "only records desk release file",
      "alternative_outcome": "duty archivist release urgent requests"
    },
    "related_evidence": {
      "doc_id": "doc-e5f6a7b8",
      "quote": "The duty archivist may release urgent requests."
    }
  }],
  "questions": [{
    "doc_id": "doc-a1b2c3d4",
    "quote": "Only the records desk may release a file.",
    "type": "contradiction",
    "question": "Should the duty archivist release files for urgent requests (yes selects the alternative outcome: duty archivist release urgent requests)?",
    "decision": {
      "actor": "duty archivist",
      "action": "release file",
      "condition": "urgent requests",
      "anchor_outcome": "only records desk release file",
      "alternative_outcome": "duty archivist release urgent requests"
    },
    "yes_choice": "alternative",
    "related_evidence": {
      "doc_id": "doc-e5f6a7b8",
      "quote": "The duty archivist may release urgent requests."
    },
    "target_stances": {
      "group-1a2b3c": "agree",
      "group-4d5e6f": "disagree"
    }
  }]
}
```

`diagnosis` must be a well-formed yes/no question. Its five-field `decision`
object must name an actor, action, condition, primary-rule outcome, and
alternative outcome. It must recover, through normalized non-verbatim concept
matching, one accepted concept in each slot of the prompt-hidden,
candidate-bundled, hand-authored reference. The actor, action, and condition cannot move
between roles, while the two outcomes stay separately oriented and remain
anchored to the cited source passages where applicable. Candidate diagnosis
prose is not evidence for its own frame.
The prose must express that same decision. It need not equal one canonical
sentence, but its structured meaning is checked against those authored
reference concepts. Contradictions require a contiguous second
passage from another document; ambiguity and gap findings require
`related_evidence: null`.
The optional `questions` array follows the Ask contract below and contributes
only the logged weight-zero `question_utility` metric. Omitting it, using the
wrong cardinality, or making it malformed sets that companion metric to zero
without erasing otherwise valid strict or shaped Find reward.
Unknown issue types and duplicate normalized spans invalidate the complete
response, so a type typo cannot be partially scored and repeating one anchor
under all types cannot hedge.

End-to-end matching requires the correct document and type, at least 90%
contiguous anchor coverage, at least 80% evidence-token precision, valid
decision grounding and diagnosis form, and the correct relationship. The
environment reports strict
`finding_f1` plus localization, type, diagnosis, and relationship diagnostics.
`question_utility` remains a weight-zero companion signal.

## Ask contract

The default row has three candidate issues and requires exactly one question.
The prompt renders one unordered profile per candidate. Each profile contains
the exact canonical five-field decision and its exact signed alternative-side
trade-off weights. The prompt also defines the faction-preference composition,
the `+/-0.25` pass band, yes-side orientation, `decision_value`, disagreement,
and utility ranking formulas. It does not reveal evidence locations, issue
types, relationships, accepted alias sets, stored stance labels,
`decision_value`, or precomputed utility. Ask is therefore an honest selection
and grounding task over public candidate profiles, while Find remains the task
that must discover and localize issues without those profiles.
The item must identify the issue structurally and declare what “yes” means:

```json
{
  "questions": [{
    "doc_id": "doc-a1b2c3d4",
    "quote": "Only the records desk may release a file.",
    "type": "contradiction",
    "question": "Should the duty archivist release files for urgent requests (yes selects the alternative outcome: duty archivist release urgent requests)?",
    "decision": {
      "actor": "duty archivist",
      "action": "release file",
      "condition": "urgent requests",
      "anchor_outcome": "only records desk release file",
      "alternative_outcome": "duty archivist release urgent requests"
    },
    "yes_choice": "alternative",
    "related_evidence": {
      "doc_id": "doc-e5f6a7b8",
      "quote": "The duty archivist may release urgent requests."
    },
    "target_stances": {
      "group-1a2b3c": "agree",
      "group-4d5e6f": "disagree"
    }
  }]
}
```

Question prose must have yes/no form. It is not compared by verbatim equality
with a canonical question or alias. The response may copy the prompt-visible
canonical decision profile or use a source-grounded paraphrase accepted by the
hidden, role-specific alias set. Exact primary and contradiction evidence
anchors the issue and applicable outcomes; repeating a profile or decision
field in the response does not establish the required evidence match. The
scorer rejects tiny keyword answers,
unrelated evidence-word dumps, and unsupported semantic clauses appended to
either the question stem or its orientation tail. The prose must name the same
core decision and express the outcome selected by `yes_choice`; an
unchanged-prose polarity flip fails. The
stance map must contain exactly the visible faction
IDs. Separate metrics report question format validity,
`question_top1_selection_accuracy`, semantic grounding recall, end-to-end grounded stance recall,
exact-evidence match recall, and stance accuracy over the same deterministic
exact-evidence assignment. The coverage metric makes conditional accuracy
interpretable when no passage matches.

Utility combines structured issue grounding, stance accuracy, panel
disagreement, and a continuous value derived from the strength and spread of
faction preferences on that issue's prompt-visible value trade-off. There is no
policy-keyword or answer-count value table. Global assignment prevents
duplicate claims, and normalization by the best attainable single-item utility
makes the exact top-one response score 1.0. The release gate requires no
top-one boundary ties, a normalized top-one margin of at least `0.10`, an exact
uniform-random selection expectation below `0.90`, and a public-profile
composition oracle at `1.0` reward and selection accuracy. The runner-up oracle
must score below `1.0` with zero top-one accuracy. This is
clarification-target selection—not a measurement of
real-world information gain. Selecting one of three remains a deliberately
small ranking problem.

## Model-free comparators

The candidate suite separates three evidence classes:

- **Prompt-observable baselines** use only the rendered task, including random
  and vague-span strategies, removed-marker checks, layout/position probes,
  legacy shortcut parsers, and simple Ask strategies.
- **Component oracles and selection diagnostics** read specified hidden
  components to isolate localization, ranking, decision, or stance difficulty;
  they are not prompt-observable floors.
- **Structural attacks** include exact-nonlocalization component-oracle
  rescoring of the retired-marker, layout/length/position, longest-sentence,
  and sector-team locators. Combined title/length/position and helper-document
  role/relationship classifiers exclude the complete test template family and
  report macro recall (balanced accuracy), observed class shares, and explicit
  chance references. Gold-filled locator rows and corpus-trained classifiers
  are diagnostics, not deployable prompt-only baselines.
- **Source-aware replay** uses the candidate source's generator, templates,
  seed, and answer construction to reconstruct the exact key. It is expected to attain
  the deterministic ceiling and is labeled a memorization ceiling—not a model
  baseline, a meaningful benchmark floor, or evidence of generalization.

These attacks were developed against the candidate source. After source and
scorer freeze, an independently constructed attack holdout is required; a
failed holdout reopens the candidate and cannot be reused as final evidence
after tuning.

Exact model-free results on the regenerated 100-row candidate evaluation split:

| Comparator class | Task | Comparator | mean reward |
| --- | --- | --- | ---: |
| Prompt-observable | find | Random visible spans | 0.000 |
| Prompt-observable | find | Flag vague-sounding spans | 0.000 |
| Prompt-observable | find | Removed fixed filler/type marker | 0.000 |
| Prompt-observable diagnostic | find | Removed fixed marker localization recall | 0.000 |
| Prompt-observable diagnostic | find | Removed fixed marker conditional type accuracy | 0.000 |
| Localization component oracle | find | Removed fixed marker locator + exact non-localization components | 0.000 |
| Prompt-observable | find | Document length rank + fixed sentence position | 0.000 |
| Prompt-observable diagnostic | find | Layout/position/length localization recall | 0.200 |
| Prompt-observable diagnostic | find | Layout/position/length conditional type accuracy | 0.130 |
| Localization component oracle | find | Layout/position/length locator + exact non-localization components | 0.200 |
| Prompt-observable | find | Select the three longest visible sentences | 0.000 |
| Prompt-observable diagnostic | find | Longest-sentence localization recall | 0.087 |
| Prompt-observable diagnostic | find | Longest-sentence conditional type accuracy | 0.023 |
| Localization component oracle | find | Longest-sentence locator + exact non-localization components | 0.087 |
| Source-aware diagnostic | find | Exclude current shared procedural predicates | 0.000 |
| Source-aware diagnostic | find | Shared-predicate exclusion localization recall | 0.000 |
| Source-aware diagnostic | find | Shared-predicate exclusion conditional type accuracy | 0.000 |
| Source-aware diagnostic | find | Select every sector-team marker | 0.000 |
| Source-aware diagnostic | find | Sector-team marker localization recall | 0.000 |
| Source-aware diagnostic | find | Sector-team marker localization F1 | 0.000 |
| Localization component oracle | find | Sector-team locator + exact non-localization components | 0.000 |
| Source-aware memorization ceiling | find | Regenerate exact answer key from candidate template and seed | 1.000 |
| Prompt-observable | find | Legacy 0.2 document-ID/position codebook | 0.000 |
| Prompt-observable | elicit-ask | Template clarity questions | 0.000 |
| Prompt-observable | elicit-ask | Randomly targeted questions | 0.000 |
| Prompt-observable | elicit-ask | Removed 0.3 summary/stance codebook | 0.000 |
| Component oracle | elicit-ask | Exact issues + removed 0.4 principle-table parser | 0.000 |
| Source-aware prompt-only | elicit-ask | Candidate template detector + removed 0.4 principle-table parser | 0.000 |
| Selection diagnostic | elicit-ask | Exact uniform-random issue expectation + exact components | 0.557 |
| Selection diagnostic | elicit-ask | Exact uniform-random top-1 selection accuracy | 0.333 |
| Selection diagnostic | elicit-ask | Exact runner-up issue + exact per-issue components | 0.398 |
| Selection diagnostic | elicit-ask | Runner-up top-1 selection accuracy | 0.000 |
| Component oracle | elicit-ask | Exact grounding + public-profile stance/rank composition | 1.000 |
| Component oracle diagnostic | elicit-ask | Public-profile composition top-1 selection accuracy | 1.000 |
| Component oracle | elicit-ask | Exact top-1 issue + random stances | 0.655 |
| Component oracle | elicit-ask | Exact top-1 issue + exact stances (component ceiling) | 1.000 |
| Component oracle diagnostic | elicit-ask | Exact top-1 selection accuracy | 1.000 |
| Source-aware memorization ceiling | elicit-ask | Regenerate exact top-1 answer from candidate template and seed | 1.000 |
| Corpus diagnostic | audit | Rows containing a removed fixed type marker | 0.000 |
| Corpus diagnostic | audit | Planted anchors carrying shared procedural predicates | 1.000 |
| Corpus diagnostic | audit | Distractors carrying shared procedural predicates | 1.000 |
| Corpus diagnostic | audit | Absolute planted/distractor shared-predicate rate gap | 0.000 |
| Corpus diagnostic | audit | Planted anchors carrying the sector-team marker | 0.000 |
| Corpus diagnostic | audit | Distractors carrying the sector-team marker | 0.054 |
| Corpus diagnostic | audit | Absolute planted/distractor sector-team rate gap | 0.054 |
| Corpus diagnostic | audit | Minimum planted issue-class proportion | 0.333 |
| Corpus diagnostic | audit | Maximum planted issue-class proportion | 0.333 |
| Corpus diagnostic | audit | Issue-type balanced-accuracy chance reference | 0.333 |
| Cross-template structural diagnostic | audit | Combined title/length/anchor-position LOTO balanced accuracy | 0.403 |
| Corpus diagnostic | audit | Minimum helper document-role class proportion | 0.200 |
| Corpus diagnostic | audit | Maximum helper document-role class proportion | 0.200 |
| Corpus diagnostic | audit | Helper document-role balanced-accuracy chance reference | 0.200 |
| Cross-template structural diagnostic | audit | Helper document-role LOTO balanced accuracy | 0.240 |
| Corpus diagnostic | audit | Helper contradiction-related document prevalence | 0.200 |
| Corpus diagnostic | audit | Minimum related-document binary class proportion | 0.200 |
| Corpus diagnostic | audit | Maximum related-document binary class proportion | 0.800 |
| Corpus diagnostic | audit | Related-document balanced-accuracy chance reference | 0.500 |
| Cross-template structural diagnostic | audit | Related-document LOTO balanced accuracy | 0.524 |
| Corpus diagnostic | audit | Top-1 boundary tie rate | 0.000 |
| Corpus diagnostic | audit | Mean top-1 boundary utility gap | 0.133 |
| Corpus diagnostic | audit | Minimum top-1 boundary utility gap | 0.023 |
| Corpus diagnostic | audit | Mean normalized top-1 utility margin | 0.602 |
| Corpus diagnostic | audit | Minimum normalized top-1 utility margin | 0.144 |
| Corpus diagnostic | audit | Title-root leave-one-template-out issue-label accuracy | 0.267 |
| Corpus diagnostic | audit | Style leave-one-template-out issue-label accuracy | 0.397 |
| Corpus diagnostic | audit | Anchor-position leave-one-template-out issue-label accuracy | 0.363 |
| Corpus diagnostic | audit | Sentence-count leave-one-template-out issue-label accuracy | 0.430 |
| Corpus diagnostic | audit | Document-length leave-one-template-out issue-label accuracy | 0.383 |

The top-one boundary tie rate is `0.000`; its mean/minimum raw utility gaps are
`0.133`/`0.023`, and its mean/minimum normalized margins are `0.602`/`0.144`.
The exact uniform-random selector has `0.557` expected reward and `0.333`
top-one accuracy; the exact runner-up has `0.398` reward and `0.000` top-one
accuracy. Composition from the public candidate profiles reaches `1.000` on
both measures.

Leave-one-template-out issue-label accuracy is `0.267` from title root,
`0.397` from style, `0.363` from anchor position, `0.430` from sentence count,
and `0.383` from document length. The combined title/length/position classifier
has `0.403` balanced accuracy against a `0.333` reference. Helper document-role
classification has `0.240` against `0.200`; related-document classification has
`0.524` against `0.500`. All remain below their preregistered release gates,
but none proves that all structural signal is absent.

The source-aware sector-team attack has `0.000` localization recall and F1;
that phrase appears in `0.000` of planted anchors and `0.054` of distractors.
The genuine prompt-only three-longest-sentences probe scores `0.000` strict F1,
localizes `0.087` of planted anchors, and has `0.023` conditional type accuracy.
Supplying exact type, diagnosis, and relationship only for correctly located
spans yields component-oracle F1 of `0.000`, `0.200`, `0.087`, and `0.000` for
the removed-marker, layout/position/length, longest-sentence, and sector-team
locators. Seeded neutral sentences use the same procedural-component vocabulary
as planted sentences. This is a bounded shortcut audit, not proof that every
prompt-observable signal is absent.
Reproduce these values with `python scripts/compute_elicit_floors.py`; exact
wheel/source-archive reproduction remains release-blocking. An independently
implemented private generator is still required for consequential transfer
evidence.

No 0.6 model scores or training-utility evidence are claimed yet. The
historical
[0.5.0 evaluation report](https://github.com/doctor-gonzo/commonground-envs/blob/master/docs/evaluation-report-0.5.0.md)
applies only to the immutable 0.5 artifacts. Follow the
[0.6.0 evaluation plan](https://github.com/doctor-gonzo/commonground-envs/blob/master/docs/evaluation-plan-0.6.0.md)
before publishing or promoting this candidate.

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
`distractor_density`, `panel_polarization`, and `question_count`. Actor-support
passages are required candidate evidence, not optional distractors, so
`distractor_density` changes only true neutral noise. Bounded `docs_length`
views drop optional neutral spans before required actor support; if a view still
cannot preserve both an issue anchor and every accepted actor alias, it removes
that issue and its visible anchor rather than emitting an unobservable answer
key. Parsing is bounded and malformed output fails closed. Public planted keys
support open training but not contamination-resistant comparison; use a fresh
private generator family for consequential evaluation.

Find additionally accepts `reward_mode="shaped"` for training. It averages
localization, type, diagnosis, and relation F1 scores on `[0,1]`, charging every
unmatched candidate at every stage. Adding a false positive or overlapping
hedge strictly lowers reward relative to a concise exact answer. The default is
`reward_mode="strict"`; reported evaluations must use it. Neither objective has
demonstrated learning value until multi-seed training and fresh-family transfer
are run.
