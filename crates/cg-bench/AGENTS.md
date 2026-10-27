# cg-bench

Purpose: Correctness measurement with gold answers and a frozen scoring method.

Owns: Gold format, matching thresholds, scoring, production capture driver and reports.
Must not own: Shipped operator commands, ontology behavior or model connection policy.

## Seams

Consumes the core capture pipeline and a supplied similarity implementation.

## Where a change lives

`src/gold.rs` for format; `src/scoring.rs` for matching; `src/driver.rs` for capture; `src/report.rs` for output.

## Invariants and guarding tests

Ownership boundaries above are review-only architectural constraints unless a guard below covers them.

Thresholds and gray-zone treatment are measurement contracts.
Review-only: changes require a separately documented benchmark contract.

Guards: `tests/bench_contract.rs`: `scoring_matcher_applies_frozen_thresholds`, `scoring_routes_gray_zone_out_instead_of_deciding`; `tests/driver_contract.rs`: `bench_driver_exercises_production_capture_path`.

Fast check: `cargo test -p cg-bench`. Full gate: `scripts/gate.sh`.
