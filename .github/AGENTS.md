# CI maintenance — agent rules

Applies when you change anything under `.github/` or the scripts the workflows
call (`scripts/gate.sh`, `scripts/gate-receipt.py`, `scripts/ci-impact.py`,
`scripts/performance-build.py`, `scripts/validate-paired-results.py`,
`scripts/smoke-receipt.py`, `scripts/validate-doc-coverage.py`,
`scripts/check-doc-validator-negative-control.sh`,
`scripts/check-gate-wall-clock.py`). Read [`CI.md`](CI.md) first: it
describes every workflow, check name and local command. Repository-wide rules
are in the root [`AGENTS.md`](../AGENTS.md) and
[`CONTRIBUTING.md`](../CONTRIBUTING.md).

An `AGENTS.md` file governs its own directory. This one is reached from the root
instructions by a link; it is not loaded automatically when you start at the
repository root or edit `scripts/`.

## Shared standard (context-graph-ai organisation OSS CI policy)

These rules are shared by the organisation's open-source repositories —
contextdb, context-graph and Vigil.

- Keep four separate paths: PR checks, extended integration, performance
  measurement and publication preparation. Do not merge them, and do not let a
  scheduled or manual result stand in for a check on the PR's commit.
- Workflows call repository-owned entry points. Add or change commands in
  `scripts/gate.sh` and the scripts it calls, then call that stage from YAML; do
  not write a second copy of a command inline in a workflow.
- One stable aggregate (`PR checks`) fails closed: any failed, cancelled,
  skipped, missing or unexpected job fails it. When you add, rename or remove a
  job in `ci.yml`, update the aggregate's `needs:` and its expected job set
  together.
- Unknown impact fails toward more coverage. A path the classifier does not
  recognise, or a diff it cannot compute, must report pending qualification;
  documentation-only routing never removes an original PR check.
- Fork safety: PR, extended public and performance workflows keep
  `permissions: contents: read`, use public inputs only and receive no secrets.
  Never use `pull_request_target`, and never use a `workflow_run` job to build or
  run untrusted code with secrets or write permissions. Pin every third-party
  action to a full commit SHA.
- Correctness is not speed. Ordinary test and job durations are reported, not
  gates; timeouts are runaway protection. Benchmarks are descriptive; only a
  controlled native run judged against a quoted product requirement can be
  acceptance evidence. Never run benchmarks on shared hosted runners or report
  a hosted result as a measurement or qualification.
- Report exact identity: receipts name the source SHA, the locked ContextDB SHA,
  the toolchain and the executed commands. Missing evidence is `PENDING` or a
  failure, never an inferred pass.
- No authority by implementation: a CI change does not change branch-protection
  required checks, publish anything, create tags or alter secrets. Those are
  owner actions.

## context-graph obligations

- Linux qualification is required: `source`, `lint`, `docs`, `test`, `features`,
  complete `controls` and the stable `PR checks` aggregate remain fail-closed.
  macOS runs the same lint, default, feature and complete controls commands in
  `macos-advisory.yml`, with all five physical jobs and six logical receipts.
  Its aggregate reports failures honestly and never blocks Linux or delivery.
  Preserve model proofs, test-estate accounting, documentation validation and
  every feature shape. See [macOS follow-up](CI.md#macos-qualification-follow-up).
- The CI guard tests that read the workflows (for example
  `crates/context-graph/tests/ci_hardening.rs`,
  `crates/cg-catalog/tests/acceptance/ci_hardening.rs`,
  `crates/context-graph/tests/acceptance/cg_main_dep_audit.rs` and
  `scripts/tests/test_ci_qualification.py`) are adapted honestly when a
  workflow changes: they keep asserting what the workflow now guarantees, with a
  mutation control that fails when the guarantee is broken. Never delete a guard
  or weaken an assertion elsewhere to make a change pass.
- Secrets are read only by the trusted jobs of `extended-integration.yml`, which
  run only on a manual dispatch with `trusted: true` in the protected
  `trusted-integration` environment and only when the requested source SHA
  equals the dispatched revision. Do not add a secret to any other job.
- Performance measurement is manual.
  [`docs/performance/README.md`](../docs/performance/README.md) is the only
  home for performance commands and the publishing rules; elsewhere, link to it
  rather than copying commands. `performance.yml` stays dormant: no
  `workflow_dispatch`, `schedule`, `pull_request` or matching push trigger, no
  checkout, no secret, no self-hosted runner and no step that can succeed. A
  dedicated, isolated self-hosted runner requires explicit repository-owner authorization.
- Recorded source defects are fixed in source, not hidden by narrowing CI.

## When you change CI

- Update [`CI.md`](CI.md) in the same change whenever a workflow name,
  trigger, job or check name, local command, prerequisite or classifier rule
  changes, and keep [`CONTRIBUTING.md`](../CONTRIBUTING.md) consistent.
- Validate statically before claiming anything: YAML parse, `actionlint` when
  available, `python3 scripts/tests/test_ci_qualification.py`,
  `python3 scripts/tests/test_ci_impact.py`, and the Rust guard tests above.
- A hosted run proves only the commit it ran on. Do not describe CI as passing,
  fast or qualified without that run's evidence.
- Keep this directory free of private machine paths, internal references and run
  histories; it must make sense to an external contributor.
