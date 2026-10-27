# Performance measurement

<!-- concept -->
Use this guide for manual consumer measurements of context-graph. Keep raw
evidence private and submit only a reviewed public summary.

The [`performance.yml`](../../.github/workflows/performance.yml) placeholder has
no reachable trigger, checkout, secret or successful step. <!-- enforced by: `ci_runs_cargo_deny_and_cargo_expand_symbol_check` -->

## Choose the scope

<!-- concept -->
Choose a consumer path: event-history import, projection, search, graph
maintenance, store operations or embedding. Consult the benchmark targets in
[`crates/context-graph/Cargo.toml`](../../crates/context-graph/Cargo.toml) and the
[benchmark notes](../../crates/context-graph/benches/README.md).

<!-- concept -->
For engine storage, vector search, sync or server throughput, use the
[ContextDB repository](https://github.com/context-graph-ai/contextdb).
Do not use a consumer result as engine evidence, or the reverse.

<!-- concept -->
Choose broader coverage for shared code, dependencies, `Cargo.lock` or build
settings. Label a single-target measurement with its exact scope. Treat all
benchmark numbers as descriptive; cite a product requirement, statistic and
scope for any separate acceptance judgment. Do not infer acceptance from a
successful benchmark exit.

## Build the baseline and candidate

<!-- concept -->
Select a committed baseline and candidate, each in its own clean checkout on
one controlled native host. Use the same toolchain, build environment and
resource limits. Keep other heavy workloads off the host during measurement.
Do not measure performance on shared hosted runners.

For an ordinary benchmark target, run from each checkout:

```sh
cargo bench -p context-graph --bench <target>
```

<!-- concept -->
Compare the two checkouts' estimate files explicitly; do not substitute a
checkout's cached Criterion comparison for the selected baseline.

For paired event-history measurement, build each checkout with the current
[`performance-build.py`](../../scripts/performance-build.py):

```sh
CG_EVENT_BENCH_CONSUMER_SHA=<selected-cg-sha> CG_EVENT_BENCH_ENGINE_SHA=<locked-contextdb-sha> \
  python3 /absolute/path/to/current-tooling/scripts/performance-build.py /absolute/path/to/build-receipt.json
```

The builder rejects unrecorded Cargo configuration, compiler-wrapper settings,
flag overrides and profile overrides before compilation. <!-- enforced by: `ci_runs_cargo_deny_and_cargo_expand_symbol_check` -->

<!-- concept -->
Copy each built binary out of its checkout before building the other side.
Retain both build receipts. Consult the builder source for its receipt fields
and the [`gate.sh`](../../scripts/gate.sh) `performance-build` entry point.

## Run the pair

From the current tooling checkout, supply:

<!-- concept -->
- Four exact revisions: `BASELINE_CG_SHA`, `CANDIDATE_CG_SHA`,
  `BASELINE_CONTEXTDB_SHA`, `CANDIDATE_CONTEXTDB_SHA`.
- An environment receipt at `PERFORMANCE_ENVIRONMENT_RECEIPT`: describe the
  hardware, filesystem, resource limits, fixture, model, profile and isolation.
- A maintainer acceptance receipt at `PERFORMANCE_ACCEPTANCE_RECEIPT`.
- Corpus and separately sealed fixture roots: `CG_EVENT_BENCH_CORPUS_JSONL_DIR`,
  `BASELINE_FIXTURE_DIR`, `CANDIDATE_FIXTURE_DIR`.
- Workload sizes: `CG_EVENT_BENCH_TOTALS`, `CG_EVENT_BENCH_CONTEXTS`,
  `CG_EVENT_BENCH_RECALL_QUERIES`.
- A fresh `PERFORMANCE_RESULT_DIR`.
- Both binaries and build receipts: `BASELINE_BENCH_BINARY`,
  `CANDIDATE_BENCH_BINARY`, `BASELINE_BUILD_RECEIPT`, `CANDIDATE_BUILD_RECEIPT`.

Run these stages in order:

```sh
bash scripts/gate.sh performance-manifest
bash scripts/gate.sh performance
```

Without `BASELINE_CG_SHA`, `performance-manifest` exits with status 2 before
running contributor commands. <!-- enforced by: `ci_runs_cargo_deny_and_cargo_expand_symbol_check` -->

The paired-result validator rejects mismatched compiler/build identities,
substituted source inputs, missing selectors, failed processes, incomplete
workloads and unequal fixture identities, including corpus, model and held-out
queries. <!-- enforced by: `ci_runs_cargo_deny_and_cargo_expand_symbol_check` -->

Complete paired records can bind different binaries to equal measured inputs. <!-- enforced by: `ci_runs_cargo_deny_and_cargo_expand_symbol_check` -->

<!-- concept -->
Inspect [`validate-paired-results.py`](../../scripts/validate-paired-results.py)
and its tests in [`test_ci_qualification.py`](../../scripts/tests/test_ci_qualification.py)
for the exact validation fields. Make a separate human acceptance judgment
against the cited product requirement.

## Keep raw evidence private

<!-- concept -->
Keep raw results, receipts and logs outside this repository. Do not upload host
identities, absolute paths, resource details, user names, credentials or private
corpus content. Do not add automated import or publication of measurement data.

## Publish a reviewed summary

<!-- concept -->
1. Copy [`results/TEMPLATE.md`](results/TEMPLATE.md) to
   `docs/performance/results/<date>-<target>-<short-candidate-sha>.md`.
2. Fill in reviewed public fields only. Put descriptive numbers in the
   descriptive table without a verdict. For an acceptance row, cite the product
   requirement and exact scope. Use `PASS`, `FAIL`, `NOT RUN` or `NOT EVIDENCE`;
   mark missing proof `NOT RUN` and incompatible or uncertain provenance
   `NOT EVIDENCE`.
3. Add the target, hardware label, date and measured candidate SHA to the
   [results list](results/README.md).
4. Open a normal pull request. Do not tag a release, change branch protection
   or add a required check as part of publishing a summary.

<!-- concept -->
Require explicit repository-owner authorization before enabling any dedicated
runner, dispatch trigger or automated measurement. Review runner admission,
resource isolation and security controls separately from result publication.
