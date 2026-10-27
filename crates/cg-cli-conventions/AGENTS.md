# cg-cli-conventions

Purpose: Shared finding, severity, diagnostic registry and report conventions.

Owns: Diagnostic check definitions, free/spending selection and shared report shapes.
Must not own: Store behavior, credential persistence, provider calls or command parsing.

## Seams

CLI surfaces consume this registry; registered check callbacks supply findings.

## Where a change lives

`src/lib.rs` for registry, report construction and condition mappings.

## Invariants and guarding tests

Ownership boundaries above are review-only architectural constraints unless a guard below covers them.

Doctor must not run spending checks; selftest explicitly selects spending checks.

Guards: `crates/cg-cli/tests/cli_acceptance.rs` guards `cg_doctor_runs_only_free_checks_names_drift_and_spends_nothing` and `cg_doctor_and_selftest_runners_mount_from_shared_conventions_layer`.

Fast check: `cargo test -p cg-cli-conventions`. Full gate: `scripts/gate.sh`.
