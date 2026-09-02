# Methodology and benchmark scope

Common Ground 0.6 is a synthetic benchmark family with two constructs. Predict
tests calibrated masked-vote prediction. Elicit tests grounded policy-issue
identification and deterministic clarification-candidate selection. Neither
construct is evidence about actual human preferences or organizational value.

## Predict

Each snapshot contains policy statements, a partially visible
participant-by-statement vote matrix, and masked cells. Votes use three classes:
agree, disagree, and pass.

The response must contain exactly one probability distribution for every
masked cell. Missing, extra, duplicate-normalized, malformed, non-finite, or
out-of-range values invalidate the response.

For one-hot target vector `y` and predicted probability vector `p`, normalized
multiclass Brier loss is:

```text
Brier = 1/2 * sum_c (p_c - y_c)^2
reward = 1 - Brier
```

This is the only primary reward. Argmax accuracy, Brier, and Brier skill against
declared references are diagnostics. The fixed argmax tie order is agree, pass,
then disagree.

The active probability-native references are uniform, empirical class prior,
per-statement visible frequency, and smoothed neighbor frequency. Prompt views
support full, matrix-only, text-only, and deterministic shuffled-text
ablations. These are enough to test whether text contributes beyond matrix
completion without introducing a general modeling framework.

## Elicit scenario model

Each synthetic scenario has public documents, issue-independent faction value
vectors, and exactly one ambiguity, contradiction, and gap. Every issue authors:

- one primary passage;
- one five-slot decision frame;
- one canonical yes/no presentation question;
- one explicit yes-side orientation;
- one value trade-off vector;
- for contradictions, one explicit opposing passage.

Faction alternative preferences are the normalized dot product between public
faction values, rendered with round-trip precision, and issue trade-off
weights. Values at or above `+0.25` agree with the alternative, values at or
below `-0.25` disagree, and values between those thresholds pass.
`yes_choice="anchor"` swaps agree and disagree; pass is unchanged.

Faction summaries describe only general values. They are generated before and
independently of issue stance labels. Contradiction relationships come directly
from authored template references, not lexical search.

## Elicit Find

Strict Find scoring uses one-to-one F1. Candidate and gold findings match only
when source grounding, issue type, the five decision slots, and required
contradiction evidence match. Diagnosis prose must have yes/no form but carries
no lexical answer key.

Find permits only explicitly authored aliases in the same decision slot. Human
text is normalized with Unicode NFKC, case folding, and whitespace collapse.
Opaque IDs and JSON keys are exact.

The optional shaped reward is the mean of four cumulative stage F1 scores:
localization, type, diagnosis, and final relation. The final relation stage is
the strict finding F1. Every stage counts unmatched candidates as false
positives, so exact-plus-spam scores below a concise exact answer.

## Elicit Ask

Ask presents three unordered public decision profiles and requests the top
candidate by a declared utility:

```text
utility = decision_value * disagreement
```

`decision_value` derives from the spread and magnitude of faction preferences;
`disagreement` combines normalized three-label entropy with the fraction of
faction pairs holding different labels. The exact formula and inputs are in the
prompt.

Ask reward requires the selected issue's exact source evidence, exact public
decision profile, explicit orientation, exact faction-key set, and stance
predictions. The free-form question is presentation checked only for yes/no
form. The benchmark does not claim open-ended question generation or actual
information gain.

## Split controls

Train and eval use disjoint templates but one shared generator. Release checks
retain three focused identities:

1. exact instance fingerprint, including the full scenario;
2. policy-semantic identity, excluding opaque IDs and neutral layout;
3. maximum cross-split word unigram/bigram TF-IDF similarity.

These target exact duplicates, identity-only variants, and lexical near
neighbors. They do not prove independent domain generalization.

## Evidence and uncertainty

The local release gate stores the source commit, exact verified artifacts, and
separate hashes for critical source inputs and retained evidence, plus
concise model-free summaries. It does not store pre-generated bootstrap
schedules or an attempt ledger.

If a later exact-artifact model study is run:

- freeze source, artifacts, prompts, models, sampling, seeds, and task IDs
  before inference;
- retain every attempted rollout and provider error;
- average repeated rollouts within a task;
- resample Predict tasks directly;
- resample Elicit base templates, then variants within templates;
- report component diagnostics and paired intervals without selecting runs by
  score.

Failed historical studies remain historical evidence outside the ordinary
release path.

## Scope limits

- The generator, taxonomy, and answers are public.
- Synthetic template separation is weaker than an independently implemented
  private generator.
- Model-free ceilings demonstrate scorer reachability, not model capability.
- Generator clause ordering leaves a source-aware first-clause shortcut for
  locating candidate issue sentences. It does not supply the scored type,
  decision frame, or relationship, but limits what localization demonstrates.
- The same public trade-off vector identifies the winning Ask candidate in 90
  of 100 evaluation rows, so fixed-corpus selection need not demonstrate
  general utility computation.
- Low Elicit scores may reflect task difficulty, formatting, or construct
  mismatch; they do not prove useful RL dynamics.
- Training value needs multiple seeds and fresh private transfer evaluation.
- Human-data use needs separate collection, consent, privacy, and redistribution
  governance; none is implemented by these environments.
