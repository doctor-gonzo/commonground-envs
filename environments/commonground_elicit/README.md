# commonground-elicit

`commonground-elicit` 0.6.0 is an unpublished deterministic Verifiers
candidate for two related tasks over synthetic organizational documents:

- `find`: locate and classify material ambiguities, contradictions, and gaps.
- `elicit-ask`: choose the most consequential clarification candidate and
  predict each visible faction's yes/no/pass stance.

Every bundled row and answer is synthetic and public. This is an open training
environment, not a private test set or a measure of real organizational value.

## Data model

Each scenario contains public documents, public faction value vectors, and
three authored issues: one ambiguity, one contradiction, and one gap. Each
issue has one source of truth:

- an exact primary `doc_id` and authored passage;
- an issue `type`;
- a five-slot decision frame: `actor`, `action`, `condition`,
  `anchor_outcome`, and `alternative_outcome`;
- a canonical yes/no presentation question and explicit `yes_choice`;
- general value trade-off weights used to compose faction stances;
- for contradictions only, an explicitly authored second `doc_id` and passage.

The generator never rediscovers the opposing contradiction passage through
lexical overlap. Faction prose is rendered from issue-independent values, not
from an `(issue type, stance) -> phrase` table. Reversing the yes-side
alternative reverses agree/disagree labels and preserves pass.

The checked-in train and eval files contain 100 scenarios each. Their template
IDs are disjoint. Release generation rejects:

- exact instance overlap;
- duplicate policy-semantic identities, including across splits;
- word unigram/bigram TF-IDF cross-split similarity above the declared bound.

These checks reduce known synthetic leakage. They do not make the shared core
generator an independent transfer distribution.

## Find contract

The default task is `find`. Return strict JSON:

```json
{
  "findings": [
    {
      "doc_id": "<exact opaque id>",
      "quote": "<grounding passage>",
      "type": "ambiguity|contradiction|gap",
      "diagnosis": "<yes/no presentation question>",
      "decision": {
        "actor": "<decision maker>",
        "action": "<decision action>",
        "condition": "<scope or trigger>",
        "anchor_outcome": "<outcome preserving the primary rule>",
        "alternative_outcome": "<clarification, fallback, or conflicting rule>"
      },
      "related_evidence": null
    }
  ]
}
```

For a contradiction, `related_evidence` must instead be an object containing
the exact opposing `doc_id` and its passage.

Strict Find reward is one-to-one finding F1. A match requires:

1. correct document and sufficient contiguous grounding;
2. correct issue type;
3. each decision slot to equal one explicitly authored slot alias after only
   Unicode compatibility, case, and whitespace normalization;
4. valid yes/no form for `diagnosis` (its wording is otherwise unscored);
5. the authored second passage for a contradiction and `null` otherwise.

The complete response must start with `{` and end with `}` without Markdown or
prose wrappers. A diagnosis must start with one of the validator's listed
yes/no auxiliaries (for example `Should`, `Can`, or `May`) and end with one
question mark.

The optional `reward_mode="shaped"` is the mean of four cumulative stage F1
scores: localization, type, diagnosis, and final relation. The final relation
stage is the strict finding F1. Every stage charges unmatched candidates as
false positives, so an exact answer plus spam scores below the exact concise
answer.

## Ask contract

Load Ask with `task="elicit-ask"`. The prompt publishes an unordered profile
for each candidate: the complete canonical decision frame and its signed value
trade-off weights. It also publishes the faction values with round-trip
precision, pass threshold, stance composition rule, and ranking formula. It
does not reveal evidence locations, issue types, relationships, stance labels,
decision values, or final utilities.

Return strict JSON:

```json
{
  "questions": [
    {
      "doc_id": "<exact opaque id>",
      "quote": "<complete authored passage>",
      "type": "ambiguity|contradiction|gap",
      "question": "<yes/no presentation question>",
      "decision": {
        "actor": "<copy from selected public profile>",
        "action": "<copy from selected public profile>",
        "condition": "<copy from selected public profile>",
        "anchor_outcome": "<copy from selected public profile>",
        "alternative_outcome": "<copy from selected public profile>"
      },
      "yes_choice": "anchor|alternative",
      "related_evidence": null,
      "target_stances": {"<exact faction id>": "agree|disagree|pass"}
    }
  ]
}
```

Ask's primary reward combines:

- selecting high-utility candidates under the public ranking rule;
- exact primary and contradiction evidence;
- exact copying of the selected public decision profile;
- exact faction-key coverage and stance accuracy in the declared orientation.

The free-form `question` is only required to have yes/no form. Hidden canonical
wording and Find aliases cannot affect Ask reward. Opaque IDs and schema keys
are exact. Ask evidence and decision text normalize only Unicode compatibility,
case, and whitespace; fragments, padding, punctuation edits, or moved fields do
not match.
The complete response must start with `{` and end with `}` without Markdown or
prose wrappers. Questions use the same auxiliary-first surface rule as Find,
and supporting passages must be copied without ellipses or omitted words.

Logged diagnostics separate schema/format validity, top-one selection,
grounding, and stance accuracy. They are deterministic synthetic conformance
metrics, not human judgments of question usefulness.

## Minimal baseline suite

Run:

```bash
uv run python scripts/compute_elicit_floors.py
```

The active baseline tool intentionally contains only:

- random and longest-visible-sentence Find probes;
- an exact Find ceiling;
- uniform-candidate and runner-up Ask references;
- an exact Ask answer ceiling;
- top-one tie/margin and issue-class-balance diagnostics.

Retired codebook attacks and experimental structural classifiers live only in
historical reports. They are not parallel release gates.

## Usage

```python
from commonground_elicit import load_taskset

find = load_taskset(split="eval", task="find")
ask = load_taskset(split="eval", task="elicit-ask")

assert len(find.load()) == 100
assert len(ask.load()) == 100
```

Legacy callers may use `load_environment` with the same task arguments.

## Scientific limits

- One synthetic generator and public answers permit memorization.
- Ask is candidate selection and composition, not open-ended information gain.
- Generator clause ordering leaves a source-aware first-clause shortcut for
  localizing candidate issue sentences. It does not provide the remaining
  scored finding fields, but limits what localization success demonstrates.
- The same public trade-off vector wins Ask in 90 of 100 evaluation rows, so a
  source-aware selector can recognize that vector instead of demonstrating
  general utility computation.
- Decision-slot aliases are authored references, not general semantic judges.
- Low or high model reward does not establish useful real-world questions.
- Training utility and transfer require multiple seeds and a fresh,
  independently implemented private generator.
- No live-human collection or export pipeline is included.
