# context-graph CI

How this repository's continuous integration is organised, what each workflow
proves, and how to run the same thing locally. The workflow files under
[`workflows/`](workflows/) and [`../scripts/gate.sh`](../scripts/gate.sh) are
authoritative; this page explains them and must change when they change.
Agents maintaining CI also read [`AGENTS.md`](AGENTS.md).

The CI rules here follow the shared policy of the context-graph-ai
organisation's open-source repositories (contextdb, context-graph and Vigil):
separate paths for PR checks, extended integration, performance measurement and
publication, one fail-closed aggregate for PR checks, read-only correctness jobs, and
manual performance measurement.

## Workflows

Every workflow defaults to `permissions: contents: read`; the development-binaries caller grants write permissions to the development
build publisher alone. Build jobs also receive actions: read for manual admission. No workflow uses
`pull_request_target`. Third-party actions are pinned to full commit SHAs. Correctness
workflows call stages of [`scripts/gate.sh`](../scripts/gate.sh), so a
contributor runs the same commands locally.

| Path | Workflow | Trigger | What a result means |
|---|---|---|---|
| PR checks | [`ci.yml`](workflows/ci.yml) (`Linux CI`) | pushes to `dev` and every pull request; manual dispatch with exact `cg_sha` and `contextdb_sha` | Correctness of this exact commit. The aggregate job **`PR checks`** passes only if every required Linux, source and documentation job succeeded. |
| macOS advisory | [`macos-advisory.yml`](workflows/macos-advisory.yml) (`macOS CI (advisory)`) | pushes to `dev`, every pull request and exact-SHA dispatch | Full macOS coverage with a failing aggregate on failure; nonblocking for Linux and delivery. |
| Extended integration | [`extended-integration.yml`](workflows/extended-integration.yml) (`extended integration`) | callable; Mondays 03:23 UTC; manual dispatch with exact SHAs | Public Nomic model and event-search discovery on Linux and macOS (aggregate `extended-required`). A manual dispatch with `trusted: true` additionally runs the protected trusted jobs (aggregate `trusted-qualification`). |
| Performance measurement | [`performance.yml`](workflows/performance.yml) (`Performance measurement (manual guide)`) | Never — dormant placeholder | **Manual today.** The workflow cannot start: its only trigger is a push filter that ignores every branch and every tag, and manual dispatch is commented out. It checks out nothing, runs no benchmark and never reports a status. The manual journey is [`docs/performance/README.md`](../docs/performance/README.md). |
| Development downloads | [`build.yml`](workflows/build.yml) (`Build development binaries`) | called after PR checks for each push to dev; maintainer dispatch on dev | Packages and smokes exact-source binaries, then publishes the rolling development prerelease. |
| Publication preparation | [`publication.yml`](workflows/publication.yml) (`publication preparation`) | manual dispatch with exact SHAs | Locked optimized CLI build and local package inspection (aggregate `publication-receipt`). It publishes nothing, creates no tag and performs no crates.io action. |

Automatic pushes run only on `dev`, preserving development publication. Candidate
branches receive both platform workflows through pull requests or exact-SHA manual
dispatch, without duplicate push runs. Display-name changes apply to future runs;
existing runs retain their recorded names. The sync compatibility job was renamed on
both platforms from `Feature compatibility — sync, CLI and doctests` to
`Feature compatibility — sync and CLI` (`ci.yml` and `macos-advisory.yml`) because its
feature doctests moved to the event-search acceptance job. `PR checks` and
`macOS advisory results` are unchanged.

## PR checks (`ci.yml`)

Every job first validates the requested ContextDB SHA, checks out the selected
commit and runs `bash scripts/gate.sh source`, which prints and verifies the
checked Context Graph SHA and the ContextDB revision locked in `Cargo.lock`.
Each test bundle uploads a sanitized receipt `pr-<job>-<os>-<shard>`; other jobs use `pr-<job>-<os>`. Superseded pull-request
runs are cancelled; push and dispatch runs are not.

| Check name | Runners | Local command | What it proves |
|---|---|---|---|
| `source` | `ubuntu-latest` | `bash scripts/gate.sh source` | The checked commit and the locked ContextDB revision are exact 40-character SHAs and match the selection. It also records `qualification.json` from [`ci-impact.py`](../scripts/ci-impact.py) (see below). |
| `lint` | `ubuntu-latest` | `bash scripts/gate.sh lint` | `cargo deny`, the expanded-library and dependency-tree scans that keep EmbeddingGemma out of the default library, `cargo fmt --check` and `clippy -D warnings` for the workspace. |
| `docs` | `ubuntu-latest` | `bash scripts/gate.sh docs` | Every behaviour sentence in the published pages names a live test ([`validate-doc-coverage.py`](../scripts/validate-doc-coverage.py)), and the validator still rejects a dead reference and an unbound claim ([`check-doc-validator-negative-control.sh`](../scripts/check-doc-validator-negative-control.sh)). |
| `test` | `ubuntu-latest` | `bash scripts/gate.sh public-model`, then `bash scripts/gate.sh default` | Offline real Nomic model proofs, `cargo test --workspace --no-run`, explicit doctests, three disjoint nextest bundles, the deterministic test-estate audit and the combined JUnit duration report. |
| `features` | `ubuntu-latest` | `python3 scripts/ci-test-shards.py bundle features --tests-only` | The complete `event-search-test-support` runtime suite and, in the acceptance bundle, the complete feature doctests, with all model proofs and disjoint receipt accounting. |
| `Feature compatibility — default and minimal libraries (ubuntu-latest)` | `ubuntu-latest` | `bash scripts/gate.sh feature-qualification-core` | Locked library builds with default features and no default features. |
| `Feature compatibility — sync and CLI (ubuntu-latest)` | `ubuntu-latest` | `bash scripts/gate.sh feature-qualification-sync` | Locked `fabric-sync` library and CLI builds. |
| `CI controls` | `ubuntu-latest` | `python3 scripts/tests/test_ci_qualification.py` | Complete CI qualification and release safety controls, once per platform. |
| **`PR checks`** | `ubuntu-latest` | — | Stable aggregate. Fails unless exactly `source`, `lint`, `docs`, `test`, `features`, `controls`, `feature-qualification-core` and `feature-qualification-sync` report `success`; a failed, cancelled, skipped, missing or unexpected job fails it. The complete Linux default and feature inventories must also verify before this check succeeds. |

`windows-latest` is excluded deliberately: the ContextDB sync substrate requires
Unix descriptor-relative filesystem operations.

Test and feature jobs restore the shared Nomic model cache with `actions/cache`
and run `bash scripts/gate.sh model-seed` on a cold miss before any test.

Default and event-search support execution each use the same three bundles on both
platforms: acceptance, imports and CLI (`import-and-cli`), and integration and remaining
cases (`integration-and-rest`). The last bundle includes newly discovered binaries.
`scripts/ci-test-shards.json` defines the exact binary allocation. Set `GATE_SHARD` to
one of these names with the `default` or `features` stage to reproduce a bundle.
Without that variable, local stages retain complete workspace execution.

Each bundle records nextest's discovered test inventory, including ignored status, its
commands and JUnit results. `PR checks` downloads these existing receipt artifacts and
runs `bash scripts/gate.sh shard-union-linux-isolated`: it requires all six successful Linux bundle
receipts at the selected source and locked dependency revisions, compares the broad
workspace inventories, rejects missing, extra, duplicate, failed or overlapping executed
cases, and verifies exactly every nonignored test ran once per platform and feature
variant. Ignored cases remain excluded; the existing ignored measurement audit runs
explicitly against each merged default JUnit only after the complete union verifies.
Default compile-only examples and default doctests run in the integration
bundle. Feature compatibility builds run in the two independent required jobs; feature doctests run in the event-search support acceptance bundle. Default acceptance compiles only `context-graph --test acceptance`; the other
bundles still compile the workspace. Feature builds retain workspace-wide feature
resolution, so filtering alone does not reduce their compilation closure.

CI alone sets development and test optimization to level 1 and debug info to 0, with
debug assertions and overflow checks enabled. Local Cargo profile defaults stay as
specified in `Cargo.toml`. Linux jobs reclaim preinstalled SDKs with sudo and print
root disk headroom before and after linking. Disposable `target/tests` scratch
is removed before compiler cache post hooks, including on failure; `target/tests/trybuild`
stays so the compile-fail dependency build is cached with the rest of `target`. The nested
Cargo runs that tests start keep their own build directories under `target`
(`tests/trybuild`, `no-default-features-check` and `expand-check`), so they are cached
with it and never rerun, under a test's Cargo environment, the build scripts the
surrounding build reuses.

These changes require hosted evidence to establish end-to-end completion under
30 minutes; queue time, cold model downloads, repeated workspace compilation and the
aggregate audit build remain part of that measurement.

### Impact classification

[`ci-impact.py`](../scripts/ci-impact.py) classifies the changed paths and
records the result in the `source` job's `qualification.json` and step summary.
Markdown under `docs/` and `skills/` and the named root documentation and licence
files are unaffected; every other path, an unknown path and an unreadable or
empty diff report `qualification: pending maintainer native/trusted run`. That
label never removes or skips a PR check, is not a failure and is not a pass. For
performance it means a manual measurement may be wanted; see
[`docs/performance/README.md`](../docs/performance/README.md).

## Extended integration (`extended-integration.yml`)

Local: `bash scripts/gate.sh extended-public` for the public jobs, and
`bash scripts/gate.sh extended-trusted` for the trusted jobs. The public jobs
take no secrets. The trusted jobs (`trusted-inputs`, `answer-quality`,
`llm-byok`, `deterministic-llm`, `capture`) run only on a manual dispatch with
`trusted: true`, in the protected `trusted-integration` environment, and only
when the requested source SHA equals the dispatched workflow revision. Missing
keys, sessions, gold answers, configuration or acceptance receipts return
`PENDING` with a nonzero exit, never a pass.

## Performance measurement (`performance.yml`)

**Manual today.** Every performance command and the publishing rules are in
[`docs/performance/README.md`](../docs/performance/README.md), the single home
for them. `performance.yml` is a dormant placeholder that cannot run: its only
active trigger is `push` with `branches-ignore: ['**']` and
`tags-ignore: ['**']`, and `workflow_dispatch` is a comment. Its one step exits
nonzero, so even a wrongly enabled run cannot report a green result. It is not a
required check. Enabling a dedicated, isolated self-hosted runner requires
repository-owner authorization, runner admission, resource isolation, security
controls and real measurement steps.
Nothing hosted measures or qualifies performance.

## Development downloads (`build.yml`)

The build jobs check out the exact qualifying commit with credentials disabled,
read the pinned Rust toolchain and assert native runner architecture. Each builds
one archive with the binary, licence and README excerpt, plus a checksum and JSON
receipt. The unpacked archive supplies the binary for version, public help and a
synthetic import/find/show/forget smoke with the same public Nomic model cache as
CI. No hosted model credentials are supplied.

The `development-binaries` job in ci.yml needs only `required` and runs only
for a push to dev in this repository; it is outside the `PR checks` aggregate.
It grants actions: read and contents/id-token/attestations: write to the called
workflow. Correctness jobs and the aggregate stay read-only. The automatic
path verifies the caller identity and uses the same source SHA in the current
CI run, without waiting for that run to complete. Automatic delivery works
whichever branch is the default branch once ci.yml is on dev.

Only the publisher receives `contents: write`, `id-token: write` and
`attestations: write`. It checks no source out and executes only inline trusted
steps: bind the workflow revision to the built SHA and both receipts, validate
the exact six assets, attest them, then replace the
single `dev-latest` prerelease. Publication and attestation require github.sha
to equal the built source SHA. The publisher selects GH_REPO explicitly, uploads
all six write-once run-named copies first, then reads back every uploaded asset as
data and verifies its bytes against the validated local assets. Existing run names
fail closed. Only after the complete group for both targets passes read-back does
it replace stable files per platform in archive, checksum, receipt order. Release
title and notes listing the run-named files change last. Pruning follows success,
retaining the newest three complete publications plus the group matching the
current stable receipt; incomplete and unknown groups are preserved. First
publication stays a hidden draft until both complete sets exist. Authenticated
release listing covers all pages and includes drafts: no match bootstraps, one
match is reused by its ID, and multiple matches refuse publication. The tag is
created once at the default branch head; its commit is the bootstrap point, not
the current binary source. The receipt names the exact source commit, workflow
run and attempt. Run-named assets use
`cg-<triple>-<sha12>-r<run_id>a<run_attempt>` with archive, checksum and JSON
suffixes. If stable verification fails during an interrupted update, use the
newest complete run-named group described in the download guide.

Publisher concurrency uses `group: build-publish`, `cancel-in-progress: false`
and `queue: max`. GitHub documents that queue as holding at most 100 pending runs,
cancelling arrivals beyond that bound, and does not guarantee queue ordering. A dev-tip comparison at publisher
start and immediately before every stable asset mutation skips an obsolete source
commit successfully. Metadata changes last. The pinned local actionlint currently
rejects only this new queue key; workflow guards enforce its exact placement and
value. GitHub acceptance of job-level queue remains pending the first hosted run.

For a manual retry, dispatch with `--ref dev`; the dispatch form uses the
workflow file from the selected ref (dev). Admission verifies a completed push
CI run of ci.yml at exactly github.sha in this repository and a successful
`PR checks` job from its latest attempt. A run whose overall conclusion failed
only because `development-binaries` failed qualifies for retry; every correctness
job must have succeeded, with exactly thirteen displayed checks: source, lint, docs, controls, six Rust runtime bundles, both feature compatibility jobs and PR checks. Missing or failed PR checks, any other failing job,
and a source different from the current dev tip refuse admission.

This is a development build, not a stable release. The first development build
is published by the first run of the Build development binaries workflow.
[Download instructions and provenance verification](../docs/download.md) are
canonical there. Hosted publication and cross-platform execution still require
that first hosted run; local checks alone do not establish asset availability.
Stable release preparation and manual performance measurement stay separate.

## Publication preparation (`publication.yml`)

Local: `bash scripts/gate.sh publication-check`. It builds the locked release
CLI and lists the local package contents. Stable releasing, tagging and crates.io publishing remain
owner actions outside this preparation workflow.

## Running the same checks locally

```sh
GATE_BASE=<base-commit-sha> bash scripts/gate.sh pr
```

With nextest available, the default stage keeps Cargo's default workspace build closure, including compile-only examples, through `cargo test --workspace --no-run`. The unfiltered nextest run executes the libtest suite, and explicit doctests execute rustdoc cases once. Without nextest, the no-argument compatibility gate retains full `cargo test --workspace` execution and explicit doctests; PR stages still require nextest.

`pr` runs the `source`, `lint`, `docs`, `public-model`, `default` and `features`
stages in that order, the same commands the hosted jobs run. Each stage writes a
sanitized JSON receipt under `$GATE_RECEIPT_DIR` (default
`$RUNNER_TEMP/job-receipt`), including the executing Bash path and version.
Command wrappers record the exact arguments they execute through a dedicated
descriptor, closed for child processes; ordinary stderr cannot supply command
evidence. This capture also supports macOS system Bash 3.2.
The `pr` stage needs `cargo-nextest`; the
no-argument `bash scripts/gate.sh` keeps the full contributor gate. Neither the
workflows nor the gate set `CARGO_BUILD_JOBS`, `RUST_TEST_THREADS` or
`NEXTEST_TEST_THREADS`, so CI builds and tests with all of the runner's cores; a
caller's value passes through and each receipt records it, or `unset`. The
concurrent-writes reindex test is serialised under both runners: an exclusive
lock over its binary under `cargo test`; under nextest, the exact case reserves
the entire actual test pool with `threads-required = "num-test-threads"`. This
excludes sibling cases in that invocation, not unrelated processes. Other cases
keep normal parallelism.
[`CONTRIBUTING.md`](../CONTRIBUTING.md) lists the plain Cargo commands and
the prerequisites.

## Reading results

- **Failed `PR checks`**: its error line names the job inventory; open the
  failed job. Every job's receipt artifact is uploaded even on failure.
- **Skipped or cancelled job**: fails `PR checks`. A skip is never a pass.
- **`qualification: pending maintainer native/trusted run`** in the `source`
  summary: the change touches more than documentation. It is not a failure, not
  a pass and not a required gate.
- **`PENDING`** from a trusted or performance stage: a required input was
  missing. It exits nonzero and is never a pass.
- **`Performance measurement (manual guide)`** never appears as a run; the
  workflow is dormant.
- Timeouts are runaway protection, not latency limits. Ordinary test and job
  durations are reported, never a merge blocker.

Which checks GitHub requires before merging is a repository setting, not
something this directory controls. `PR checks` is designed to be the single
required check.

## Security boundaries

PR, extended public and performance workflows run with `contents: read`, use
only public inputs and receive no repository secrets, so fork pull requests run
the same PR checks. Secrets (`OPENROUTER_API_KEY`, `CG_SMOKE_HUB_TICKET`) are
read only by the trusted jobs of `extended-integration.yml`, which run only on a
trusted manual dispatch in the protected `trusted-integration` environment.
Caches hold rebuildable model and compiler output only; a cache miss must still
pass.

CI qualification and release safety controls run once on Linux and once on macOS,
in their respective workflows, on the same selected source commit. The Linux required and macOS advisory
`CI controls` jobs execute the complete `scripts/tests/test_ci_qualification.py`
module, including all development download controls. Locally, use Python 3.11 or
newer with `PyYAML==6.0.2` installed and run
`python3 scripts/tests/test_ci_qualification.py`. The entrypoint runs every collected
test once, each in its own Python process, several at a time, and fails unless each one
reports exactly one passing test; naming a test (for example
`python3 scripts/tests/test_ci_qualification.py StageEvidence`) runs only that selection
in one process. These controls use synthetic
compiler and model inputs; Cargo features do not change their inputs. Rust workflow
guards verify both workflows and their qualification roles without recursively executing the Python suite.
The jobs need Python and Git, with no Rust compilation or model cache setup, and
upload the same source identity receipts as the other correctness checks.

Historical wall-clock baseline comparison runs only with the complete default JUnit,
after the unfiltered local run or verified shard union. This permits removal of a
retired binary while still rejecting a live binary without a recorded ceiling.
The compatibility gate without nextest skips JUnit-dependent comparisons.

Each platform runs the same Rust jobs, one per logical shard receipt. Each variant runs
acceptance, integration and remaining cases, and imports and CLI in three independent
jobs: six runtime jobs plus two independent compatibility build jobs, eight Rust jobs per
platform and sixteen in total. Each runtime job seeds and proves the model once and keeps
its own Cargo target. Jobs that compile the same selection share one compiler cache entry
per platform (their cache keys omit the job identity, `add-job-id-key: false`) (default acceptance compiles the narrow selection; the other default bundles
compile the workspace; every feature bundle compiles the workspace with event-search test
support). Only pushes to `dev` save compiler caches, and a `dev` job that fails still saves
its compiled dependencies; pull requests restore the `dev` entries, which GitHub serves to
every branch targeting `dev`, and save nothing, so pull-request entries cannot evict them.
Compiler cache keys name the pinned toolchain in their prefix (`v1-rust-1.94.1`) instead of
hashing every Rust release the runner image preinstalls, so a runner image update does not
miss the `dev` cache. `python3 scripts/ci-test-shards.py bundle default` (or `features`) uses
`GATE_PHYSICAL_BUNDLE` to select the displayed bundle. Each logical stage retains its
own `GATE_SHARD`, receipt directory and existing artifact identity. A failed stage
stops the bundle, preserves its exit status and uploads available evidence; later
stages have no execution receipt. Complete successful hosted runs are still required
to establish queue-inclusive completion under 30 minutes.

CI model seeding uses a core-owned narrow example and a private catalog broad example,
sharing the CLI-owned root resolver and
production `Catalog::prefetch_defaults`. `GATE_CI_MODEL_MODE` requires `CI=true` and accepts
`catalog-narrow`, `catalog-workspace`, or `catalog-features`; unknown values fail closed.
The default acceptance jobs keep the narrow core proof selection; broad default jobs use
the workspace and feature jobs use the workspace with event-search test support. Cold
seeding still runs the document proof, and cache hits still run all three offline proofs.
Local model seeding retains `cg model fetch --defaults`. Ordinary CLI builds and tests remain required.

Broad CI seed modes build the catalog example with `cargo build --locked --workspace
--example ci_catalog_model_seed --message-format=json-render-diagnostics`; feature mode adds
`--features context-graph-cli/event-search-test-support`. The launcher obtains Cargo's
configured target directory from locked metadata and executes exactly one approved
example artifact after a successful build, rejecting malformed or ambiguous output.
The stage receipt records the actual build arguments, executable path and SHA-256.
Narrow mode builds `cargo build --locked -p context-graph --example ci_model_seed
--message-format=json-render-diagnostics`. Its source includes the unchanged private
catalog helper. Both helpers are excluded from their packages. Compiler seed caches use family v3; the
OS/source-bound Nomic model cache key is unchanged. Timing improvements require native
unit-reuse evidence and complete cold/warm hosted measurements.

## macOS qualification follow-up

Linux is the current required qualification platform. The separate `macOS CI (advisory)`
workflow runs the Linux correctness jobs on `macos-latest` with identical steps: lint,
complete Python controls, three default bundles, three feature bundles and the two
compatibility build jobs, ten physical jobs that produce six logical Rust receipts. Its `macOS advisory results` aggregate fails on failed, cancelled,
skipped, missing or unexpected prerequisites, verifies only macOS receipts and runs
macOS default accounting through the isolated audit. It never enters `PR checks` needs or manual delivery admission.
The legacy local `shard-union` stage still requires twelve receipts; explicit Linux
and macOS stages each require exactly six, with no fallback to available platforms.

Qualification remains pending: seven macOS test failures remain unresolved.
The pinned ContextDB macOS default locator now canonicalizes the fixed system
temporary root `/tmp` and uses a private per-effective-UID directory with the full
socket key. Focused upstream engine coverage consists of five integration cases
actually run on the pinned revision and three unchanged library cases carried
from an older revision, not eight newly executed cases. The only reader
authentication proof is from a separate earlier ContextDB candidate, not the
pinned revision; prior reader authentication results do not prove the pinned
default locator or full Context Graph qualification. Full Context Graph macOS
consumer qualification, including the integrated default owner/reader locator,
remains pending. These are open follow-up items, not passing evidence.
Completion requires identifying and fixing each failure in source, verifying
locator behavior, running all ten macOS jobs with all six valid receipts on the exact
integrated source, and reviewing the resulting evidence before changing qualification.
No issue number or hosted green result is claimed.

Measure required latency from event admission through `PR checks`, including queues,
setup and union/accounting. No under-thirty-minute result is established here.
Report advisory latency separately. Development binary delivery after required success
can extend the dev-push workflow; required-check latency does not establish full dev-push
workflow latency. Manual retry requires exactly eleven successful displayed Linux/source/docs
checks from `ci.yml`; advisory runs cannot authorize delivery.

The compatibility build jobs are unconditional and independent of the six runtime
jobs on each platform. `feature-qualification-core` runs the original default and minimal library
builds. `feature-qualification-sync` runs the original fabric-sync library and CLI
builds. The feature doctests run in the event-search support acceptance job, before its
tests, where they reuse the workspace libraries that job already compiles with
event-search test support instead of compiling that closure a second time.
`feature-qualification` reproduces both build groups and then the feature doctests.
Each compile job records its selected source, actual commands, compiler identities and
job result without a shard or nextest inventory. The existing local `features`, sharded
`features`, `pr` and `extended-public` stages retain all builds, tests and doctests in
their original order. Only the explicit singleton dispatcher option
`bundle features --tests-only` selects the internal `features-tests` stage, on either
platform; default and combined bundles reject that option.

After all six Linux shard receipts verify, `shard-union-linux-isolated` copies the
complete merged default JUnit, exports `CG_GATE_JUNIT`, compares the historical baseline,
runs the original ignored Rust audit, and runs the independent Python checker. The Rust
audit compiles the same `crates/context-graph/tests/test_estate_audit.rs` through
`scripts/gate-wall-audit/Cargo.toml`, an independent edition-2024 workspace with only
the exact root-locked TOML dependency closure. Its ephemeral target is
`$RUNNER_TEMP/gate-wall-audit-target`. No engine compilation or executable transport is
needed for this aggregate. The macOS aggregate runs
`shard-union-macos-isolated`, the same audit over the six macOS receipts. Existing local
`shard-union`, `shard-union-linux`, `shard-union-macos` and default wall-audit routes keep
their original behavior. Actual cold compilation, model downloads, queues and total hosted
completion remain unproved until measured on the final source.
