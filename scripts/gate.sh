#!/usr/bin/env bash
#
# The gate every change passes before a commit, a release, or a "done" claim.
#
# Runs formatting, lints, the workspace tests, the doc tests, and then the checks that keep
# the test estate honest: the wall-clock arm, which refuses a test binary that got slower than
# the frozen baseline allows, the documentation-claim check, which refuses a documented behavior
# claim with no test behind it, and the negative control that proves that check still detects a
# dead reference and an unbound claim rather than having gone quietly blind.
#
# The wall-clock arm compares seconds someone else already measured, so it needs a JUnit file
# from this run's own tests: the test step below writes one and hands it over in CG_GATE_JUNIT.
# Without cargo-nextest installed there is no JUnit to write, and the run says so and skips the
# two checks that need one; install cargo-nextest to get the full gate.
#
# GATE_BASE names the ref the documentation and baseline-direction checks compare against
# (default origin/dev).
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

baseline="crates/context-graph/tests/fixtures/test_estate_baseline.toml"
gate_base="${GATE_BASE:-origin/dev}"
target_dir="${CARGO_TARGET_DIR:-$repository_root/target}"

cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings

junit=""
if command -v cargo-nextest > /dev/null 2>&1; then
    # The `gate` profile is checked in at .config/nextest.toml, so this run and the CI workflow
    # produce the measurement the same way.
    cargo nextest run --workspace --no-fail-fast --profile gate
    junit="$target_dir/nextest/gate/junit.xml"
else
    echo "cargo-nextest is not installed: running cargo test --workspace instead, which writes"
    echo "no JUnit, so the gate wall-clock checks below are skipped this run."
    cargo test --workspace
fi

cargo test --workspace --doc --no-fail-fast

if [[ -n "$junit" ]]; then
    export CG_GATE_JUNIT="$junit"
    cargo test -p context-graph --test test_estate_audit gate_wall_seconds_never_exceed_the_frozen_baseline -- --ignored --exact
    python3 scripts/check-gate-wall-clock.py --junit "$CG_GATE_JUNIT" --baseline "$baseline"
fi

python3 scripts/check-gate-wall-clock.py --base "$gate_base"
python3 scripts/validate-doc-coverage.py --base "$gate_base"
bash scripts/check-doc-validator-negative-control.sh
