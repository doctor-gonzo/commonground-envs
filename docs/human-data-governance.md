# Human-data governance

## Current release boundary

Every dataset bundled with the environments is synthetic, including the public
0.5.0 release and the 0.6.0 candidate. The open packages contain their
evaluation answers and are intended for reproducible development and training,
not contamination-resistant leaderboards. No file may be described or
published as a human-data split merely because it passes automated validation.

## Required export contract

Human snapshots must be plaintext Context Engine exports and must satisfy the
shared commonground-human-snapshot-v2 contract enforced by
commonground_scenarios.validate_human_snapshot:

- participant identifiers are positional pseudonyms p000, p001, and so on;
- every cluster partitions participants exactly once and contains at least the
  declared k-anonymity value, which must be at least five;
- statements, vote dimensions, vote values, cluster membership, and summary
  counts are internally consistent;
- cluster center is the empty list, while comment extremity and divisiveness
  are null. These are exporter-derived PCA values that this Apache-licensed
  validator cannot independently verify, so human-v2 rejects rather than
  trusts them;
- snapshots contain at most 10,000 participants, 1,000 statements, 1,000,000
  vote cells, and 100 clusters, bounding validation work before publication;
- evaluation masks and held-out labels are absent;
- source is context-engine-session and synthetic is false;
- consent_scope is public-benchmark;
- redistribution_rights_approved is true;
- schema_version is exactly commonground-human-snapshot-v2, exporter_version is
  a semantic version, and source_commit is a pinned lowercase 40-character Git
  commit;
- privacy_review.attested is true, reviewed_at is a valid date, and the
  canonical direct-identifiers, free-text, and participant-pseudonyms checks
  are recorded.

The intake command rebuilds each accepted row from an allowlist and writes a
versioned manifest containing the output SHA-256, input hashes, source lines,
counts, provenance, rights, and privacy-review attestations:

    uv run python scripts/ingest_snapshots.py /path/to/exports/*.jsonl \
      --output /approved/data/eval_real.jsonl \
      --manifest /approved/data/eval_real.manifest.json

By default any rejected row prevents publication. The --skip-invalid option is
for operator investigation; it does not waive the review requirements below.

## Human review is mandatory

The identifier scanner detects several high-confidence shapes, including email,
EVM and ENS addresses, phone numbers, IP addresses, and honorific-prefixed
names. It cannot establish anonymity, consent, legal authority, or the absence
of indirect identification. Before publication, an accountable reviewer must:

1. verify the consent record and redistribution authority outside this
   repository;
2. inspect every free-text statement for direct and indirect identifiers;
3. confirm that positional pseudonyms cannot be joined back to a public source;
4. verify the fixed schema version, pinned exporter version, and source commit,
   and remove PCA-derived centers/extremity/divisiveness as required by
   human-v2;
5. confirm the declared k-anonymity policy is appropriate for the population;
6. record the approval and compare the final file hash with the generated
   manifest.

Operators must keep consent records and any identity mapping outside the public
repository and release archive. If any check is uncertain, do not publish the
snapshot.

## Elicit human-feedback socket

The scenario package delegates human_feedback validation to the same strict
snapshot validator. A scenario carrying human feedback therefore has the same
contract as predict intake. This socket is reserved for future reviewed data;
the bundled Elicit corpora do not use it.
