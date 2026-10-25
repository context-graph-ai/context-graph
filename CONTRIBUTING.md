# Contributing

## Build from source

Source installation is the fallback to [development downloads](docs/download.md). Have Git and rustup available.

```sh
git clone --branch dev https://github.com/context-graph-ai/context-graph.git
cd context-graph
cargo install --locked --path crates/cg-cli
cg model fetch --defaults
```

Follow the [first import and search](README.md#first-import-and-search) example after installation.

## How to contribute

Run `scripts/gate.sh pr` before submitting a change. This is the contributor path matching every required PR stage; install cargo-nextest first. Use Git and the Rust toolchain selected by `rust-toolchain.toml`.

```sh
GATE_BASE=<base-commit-sha> bash scripts/gate.sh pr
```

Build individual targets from the repository root:

```sh
cargo build --workspace
cargo install --locked --path crates/cg-cli
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --workspace --doc --no-fail-fast
```

The no-argument `scripts/gate.sh` retains the full contributor compatibility path.

### Review documentation

Read [AGENTS.md](AGENTS.md) and the owning crate's guide before changing behavior. Every behavioral sentence in the README, crate READMEs, documentation and skills must carry an inline `<!-- enforced by: test_name -->` naming a live default-tier test that exercises the claim. Bind to the assertion, not a similar test title. Pure explanation needs no tag. Update help and public contracts together.

```sh
python3 scripts/validate-doc-coverage.py --all
bash scripts/check-doc-validator-negative-control.sh
bash scripts/generate-agent-bundle.sh
```

The generated `llms-full.txt` is ignored; do not commit it. Use synthetic fixtures and sanitized output. Never include private history, tokens or credentials in issues or PRs. Fork PR checks must not depend on repository secrets.

## Contributor reference

### Gate checks

| Step | What it checks |
|---|---|
| Formatting | Rust formatting matches rustfmt |
| Clippy | Workspace and all-target warnings are errors |
| Workspace tests | Behavioral proofs and deterministic test-estate audits |
| Doctests | Compiles documentation examples |
| JUnit duration report | Reports binary, measured seconds and ceiling for overruns |
| Historical baseline comparison | Rejects increases to recorded baseline values |
| Documentation coverage | Resolves claim tags and rejects unbound claims in changed pages; audits all pages without a usable base |
| Negative control | Proves the validator rejects a dead reference and an unbound claim |

Without cargo-nextest, the script runs workspace tests but skips the two checks requiring JUnit. Ordinary test duration is reported, never a merge blocker. Missing measurements or missing baseline entries still fail. Sleep/raw-clock call-site counts may only decrease; per-crate default-tier test inventory must match the recorded baseline exactly. Update inventory in the same change that adds or deliberately removes tests. Do not assert elapsed time or use sleep as synchronization.

### Continuous integration

The `Linux CI` and `macOS CI (advisory)` workflows run on pushes to `dev`, every pull request and exact-SHA manual dispatch. Candidate branches use pull requests or manual dispatch for validation. Required PR test and feature jobs run on `ubuntu-latest`; the separate `macOS CI (advisory)` workflow retains the same coverage on `macos-latest`.
`windows-latest` is excluded deliberately because the ContextDB sync substrate requires Unix descriptor-relative filesystem operations.
The workflows install the repository's Rust 1.94.1 toolchain explicitly, use immutable action pins, and verify the selected Context Graph head and locked ContextDB revision. Base, head and event SHAs are recorded separately. Fork PR jobs use public inputs and no provider secrets.

CI sets `CG_MODEL_CACHE_ROOT` at runner temporary space and uses `actions/cache` for the shared Nomic artifact cache. Without that override, the cache falls back to `~/.cache/cg/models`, with `$XDG_CACHE_HOME` replacing `~/.cache` when set.
Key the cache on the embedder source file rather than on `Cargo.lock`:
`hashFiles('workspace/context-graph/crates/context-graph/src/embedder/nomic.rs')`.
A cold miss fetches the default model; the document, query and event-path Nomic proofs then run unconditionally against that root.

### Verification paths

[`.github/CI.md`](.github/CI.md) describes every workflow, check and local command; agents changing CI also follow [`.github/AGENTS.md`](.github/AGENTS.md). Performance measurement is manual; [`docs/performance/README.md`](docs/performance/README.md) is its only guide.

| Path | Workflow and local entry point | Result |
|---|---|---|
| PR checks | `ci.yml`; `GATE_BASE=<base> bash scripts/gate.sh pr` | Every PR, including docs-only changes, requires source, lint, docs, Linux default tests, Linux features and complete Linux CI controls; `PR checks` fails on any failed, cancelled, skipped, missing or unexpected prerequisite |
| Extended public integration | `extended-integration.yml`; `bash scripts/gate.sh extended-public` | Callable, scheduled and manual public Nomic and event-search discovery; the same public coverage remains required inline in PR jobs |
| Extended trusted integration | Manual protected environment in `extended-integration.yml`; `bash scripts/gate.sh extended-trusted` | Separate exact-tuple answer quality and the three named live smokes; unavailable keys, sessions, gold, configuration, adjudication or acceptance receipt return PENDING with nonzero exit |
| Performance measurement | Manual; dormant `performance.yml` never runs; [performance guide](docs/performance/README.md) | Manual-first: build the exact baseline and candidate on one controlled host, keep raw evidence private, publish only a reviewed summary; nothing hosted measures performance |
| Publication preparation | Manual `publication.yml`; `bash scripts/gate.sh publication-check` | Locked optimized CLI build and local package inspection; no publishing, tags or crates.io action |

The PR source job always records `qualification.json` and a step summary bound to the checked source and locked dependency SHA. `scripts/ci-impact.py` contains path-map version 1: Markdown under `docs/` and `skills/` and the named root documentation/license files are unaffected; runtime, build, scripts, tests and CI paths require performance and trusted integration qualification. Unknown paths, unreadable or empty diffs and initial parentless roots require both classes. All Linux checks remain required. The same macOS commands and full controls run in the separate, nonblocking `macOS CI (advisory)` workflow; failures remain visible. See the [macOS qualification follow-up](.github/CI.md#macos-qualification-follow-up). Affected changes report `qualification: pending maintainer native/trusted run`; public check success does not imply native or trusted qualification. Fork PRs use no secrets and never run these privileged workloads. For performance, the pending label means a manual measurement may be wanted; see the [performance guide](docs/performance/README.md).

A maintainer can explicitly supply `--qualification-receipt <path>` when running `python3 scripts/ci-impact.py --base <sha> --output <path>`. Receipt schema 1 requires the exact `cg_sha` and `contextdb_sha`, `result: "qualified"`, the ordered `qualified_classes` array (`performance`, `trusted-integration`) and an `evidence_sha256` digest for each class. Mismatched or incomplete receipts fail and retain the pending status. Scheduled results are never discovered or accepted implicitly.

Hosted jobs invoke these same stage functions, with the model cache restored and seeded before default or feature tests. Each stage writes a sanitized JSON receipt in `$GATE_RECEIPT_DIR` (default `$RUNNER_TEMP/job-receipt`) containing commands, tool versions, source/dependency revisions, test counts, failed names and executed feature inventories. Cargo preserves the default workspace compilation closure with `cargo test --workspace --no-run`, including compile-only examples; nextest executes the complete default libtest suite, and explicit doctests execute rustdoc cases.

Neither hosted CI nor the gate sets `CARGO_BUILD_JOBS`, `RUST_TEST_THREADS` or `NEXTEST_TEST_THREADS`, so builds and tests use all of the runner's cores. A value you export is passed through unchanged, and every stage receipt records each of the three as given, or `unset`. `reindex::reindex_concurrent_writes_during_reindex_window_land_in_target_space_deterministically` is serialised under both runners: under `cargo test` it takes an exclusive lock that every other test in its binary waits on, and under nextest its exact-case override in `.config/nextest.toml` reserves the entire actual test pool with `threads-required = "num-test-threads"`. This excludes sibling cases in that invocation, not unrelated processes; other cases keep normal parallelism.

The reusable local stages are `source`, `lint`, `docs`, `default`, `features` and `public-model`. With no argument, `scripts/gate.sh` retains the full contributor gate and its compatibility fallback when nextest is unavailable. The PR stage requires nextest and uses `cargo test --workspace --no-run`, the unfiltered nextest gate run and explicit doctests. Without nextest, the no-argument compatibility gate retains `cargo test --workspace` execution and explicit doctests. Standalone library builds cover default features, no default features and `fabric-sync`; default CLI and the complete `event-search-test-support` suite run separately from default tests.

Ordinary suite duration is report-only. Missing JUnit measurements or baseline rows fail, as do increases to recorded historical baseline values. Product work, memory, recall and latency limits remain enforced, and job/test timeouts still terminate runaway execution. Default test jobs dispatch the shards. After their receipts are merged and the complete test unions are verified, `PR checks` invokes the measurement arm for Linux; the macOS advisory aggregate invokes it for macOS:

```yaml
      - name: Run default ${{ matrix.bundle }} tests
        env:
          CG_MODEL_CACHE_ROOT: ${{ runner.temp }}/cg-model-cache
          MODEL_CACHE_HIT: 'true'
          GATE_MODEL_READY: '1'
          TESTED_SHA: ${{ github.event.pull_request.head.sha || inputs.cg_sha || github.sha }}
          DOCUMENTATION_BASE_SHA: ${{ github.event.pull_request.base.sha || github.event.before }}
        run: python3 scripts/ci-test-shards.py bundle default
```

```yaml
      - name: Verify complete default and feature unions before wall reports
        env:
          TESTED_SHA: ${{ github.event.pull_request.head.sha || inputs.cg_sha || github.sha }}
        run: bash scripts/gate.sh shard-union-linux-isolated
```

Manual workflows require lowercase 40-character source/dependency SHAs. Trusted jobs additionally require the source SHA to equal the dispatched workflow revision and use the protected `trusted-integration` environment; configure its approval rules before supplying credentials. Callable public integration accepts no secrets and cannot select the trusted path. Scheduled discovery on another revision does not qualify an affected PR. PR supersession cancels only PR-number runs; manual qualification receipts are retained.

For local exact-tuple qualification, set `TESTED_SHA` and `REQUESTED_CONTEXTDB_SHA` to the source and locked dependency revisions. Trusted qualification requires `APPROVED_SESSION_LIST` (comma-separated absolute JSONL paths), `APPROVED_GOLD_DIR`, a disposable configured work root, model ID, input identity receipt and completed product acceptance receipt. The gate builds and runs the existing cg-bench driver, adjudicator and smoke scripts. Smoke stages build the selected CLI tuple or verify `CG_SMOKE_BUILD_RECEIPT` against an explicitly supplied `CG_BIN`; each script receives that path explicitly and writes a small result receipt. Capture also requires `CONTEXTDB_BUILD_RECEIPT` binding its installed dependency binary to the locked source SHA and digest. Installed build receipts require `build_exit_code: 0`, `binary_sha256`, and the exact `cg_sha`/`contextdb_sha` tuple (or `source_sha` for the dependency). A receipt must bind the source/dependency tuple, corpus/gold identities, contract, output hashes, every gray pair, no-gray cases, sample and observations. A successful driver exit does not establish answer quality. Hosted disposable runners do not provision private data or acceptance decisions automatically; their missing-input result is PENDING.

Performance is measured manually on a controlled native host, never on hosted runners. The baseline/candidate build, the paired `scripts/gate.sh performance-manifest` and `scripts/gate.sh performance` stages, the required inputs and the publishing rules are documented only in the [performance guide](docs/performance/README.md). See also the [benchmark notes](crates/context-graph/benches/README.md).

### Live smokes

These contributor scripts use hosted models and can spend money. They exercise additional implementation paths beyond the presented commands. Read each script's required environment first; choose disposable data roots and provide keys locally, never in a PR.

The capture smoke requires an explicit JSONL input file before any other check and accepts installed release binaries:

```sh
CG_SMOKE_REAL_JSONL=/absolute/path/to/session.jsonl CG_SMOKE_SKIP_BUILD=1 CG=/absolute/path/to/cg CONTEXTDB=/absolute/path/to/contextdb bash scripts/cg-capture-live-smoke.sh
```

Supply `OPENROUTER_API_KEY` through your environment. Authenticated sync qualification additionally requires an isolated hub ticket and tenant; without them the capture smoke does not establish release readiness. The other scripts are `scripts/llm-byok-live-smoke.sh` (hosted and local compatible endpoints) and `scripts/cg-deterministic-llm-live-smoke.sh` (hosted extraction and an installed provider CLI). Consult their usage blocks for endpoint, credential-variable and binary settings before invoking them.

Hosted default tests and event-search support tests each run three disjoint bundles on
both platforms: `acceptance`, `import-and-cli` and `integration-and-rest`. To run one
locally, use `GATE_SHARD=<bundle> bash scripts/gate.sh default` (or `features`). The
unfiltered local stage remains available with `GATE_SHARD` unset. `PR checks` verifies
the complete Linux nonignored test union from every bundle's discovery inventory and JUnit;
the macOS advisory aggregate verifies the same macOS union. Each then runs the explicit ignored measurement audit against the merged default reports.
Missing, extra, duplicate or failed cases cannot produce a successful aggregate.

CI uses optimization level 1 and debug info level 0 for development and test profiles,
retaining debug assertions and overflow checks. Local Cargo defaults are unchanged.
See [CI details](.github/CI.md#pr-checks-ciyml) for build scope, disk cleanup and the
remaining hosted timing qualification.

CI qualification and release safety controls run once on Linux and once on macOS,
in their respective workflows, on the same selected source commit. The Linux required
and macOS advisory `CI controls` jobs execute the complete `scripts/tests/test_ci_qualification.py`
module, including all development download controls. Locally, use Python 3.11 or
newer with `PyYAML==6.0.2` installed and run
`python3 scripts/tests/test_ci_qualification.py`. The entrypoint runs every collected
test once, each in its own Python process, several at a time, and fails unless each one
reports exactly one passing test; naming a test (for example
`python3 scripts/tests/test_ci_qualification.py StageEvidence`) runs only that selection
in one process. These controls use synthetic
compiler and model inputs; Cargo features do not change their inputs. Rust workflow
guards verify both qualification roles without recursively executing the Python suite.
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
per platform (default acceptance compiles the narrow selection; the other default bundles
compile the workspace; every feature bundle compiles the workspace with event-search test
support), and a job that fails still saves its compiled dependencies, so a later run does
not start cold because an earlier one failed. `python3 scripts/ci-test-shards.py bundle default` (or `features`) uses
`GATE_PHYSICAL_BUNDLE` to select the displayed bundle. Each logical stage retains its
own `GATE_SHARD`, receipt directory and existing artifact identity. A failed stage
stops the bundle, preserves its exit status and uploads available evidence; later
stages have no execution receipt. Complete successful hosted runs are still required
to establish queue-inclusive completion under 30 minutes.

CI model seeding uses the private catalog example with the CLI-owned root resolver and
production `Catalog::prefetch_defaults`. `GATE_CI_MODEL_MODE` requires `CI=true` and accepts
`catalog-narrow`, `catalog-workspace`, or `catalog-features`; unknown values fail closed.
The default acceptance jobs keep the narrow core proof selection; broad default jobs use
the workspace and feature jobs use the workspace with event-search test support. Cold
seeding still runs the document proof, and cache hits still run all three offline proofs.
Local model seeding retains `cg model fetch --defaults`. Ordinary CLI builds and tests remain required.

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
