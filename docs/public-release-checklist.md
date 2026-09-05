# Public release checklist

This checklist describes Common Ground publication. Running local verification
does not authorize a Git push, PyPI upload, Hub push, visibility change, or
public evaluation upload.

## 1. Freeze and verify source

- [ ] All four distributions declare the same intended version.
- [ ] `uv.lock` and exact dependency manifests match.
- [ ] The worktree is clean and the intended source commit is recorded.
- [ ] `bash scripts/local_smoke.sh` prints `LOCAL RELEASE VERIFICATION PASS`.
- [ ] Retain its evidence directory: source commit, concise floor summaries,
      critical-input hashes, and fresh artifact report.

The command regenerates and byte-compares Elicit splits, runs the complete
tests/static gates, builds fresh artifacts, installs exact manifests, verifies
legal files, and loads packaged tasksets. It performs no hosted inference or
publication.

## 2. Review the candidate

- [ ] Predict exact-key/probability, proper-reward, and high-mask parser tests
      pass.
- [ ] Elicit explicit-relationship, issue-independent faction prose, polarity,
      round-trip public values, false-positive, and opaque-ID regressions pass.
- [ ] Exact instance, policy-semantic, and TF-IDF split checks pass.
- [ ] Model-free summaries have reachable exact ceilings and no top-one ties.
- [ ] READMEs, dataset card, methodology, changelog, and version strings agree.
- [ ] Hub-bundled READMEs name the intended version, link any cited evidence to
      its exact artifact version, and contain no stale pre-release status text.
- [ ] No secrets, live human data, private study fixtures, or generated local
      outputs are included.

## 3. Optional exact-artifact model evidence

This section is required when publishing new model-result claims alongside the
environment. It is not required to verify or publish the environment contract.

- [ ] Use only the frozen artifacts.
- [ ] Preserve all traces and provider failures.
- [ ] Complete the predeclared task/model/rollout matrix.
- [ ] Produce task-bootstrap Predict and template-hierarchical Elicit summaries.
- [ ] Do not replace or select runs by score.
- [ ] Update the versioned evaluation report with hashes, IDs, failures, and
      limitations.

If source, corpus, prompt, scorer, dependency, or artifact content changes,
stop and create a new candidate.

## 4. Publish shared packages

Publish `commonground-score` and `commonground-scenarios` first. Verify the
exact versions from a fresh Python 3.12 environment and the public PyPI index
before pushing environments that depend on them.

Never paste tokens into commands, logs, or repository files. Use hidden input
or a configured trusted publisher.

## 5. Push private Hub candidates

- [ ] Push exact environment versions as `PRIVATE`, without auto-bumping.
- [ ] Wait for each Hub build action to report success.
- [ ] Pull the exact versions to a fresh directory.
- [ ] Verify bundled legal files and dependency pins.
- [ ] Install/load all tasksets from the pulled artifacts.
- [ ] Compare Hub content hashes with the release report.

## 6. Public transition

- [ ] Change visibility only after private verification passes.
- [ ] Verify anonymous status, pull, install, and taskset loading.
- [ ] Upload only exact-artifact evaluation records.
- [ ] Verify public evaluation pages and record their IDs.
- [ ] Tag the exact source commit and create a formal release.

Keep older versions available when they are needed to reproduce cited evidence.
Mark a flawed version superseded rather than rewriting or silently deleting its
historical record.

## 7. Claims review

The release may be described as a reproducible synthetic benchmark with proper
Predict scoring and grounded Elicit contracts. It must not be described as
real-human preference inference, real information gain, independent-domain
generalization, or proven RL utility without separate evidence. Disclose that
one public trade-off vector identifies the Ask winner in 90 of 100 evaluation
rows.
