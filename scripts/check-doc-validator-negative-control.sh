#!/usr/bin/env bash
#
# Prove the documentation-claim validator still detects what it publishes it detects.
#
# `scripts/validate-doc-coverage.py` is a gate step whose ordinary result is a clean pass, so a
# detector that silently stops finding anything looks exactly like a healthy repository. This
# builds a throwaway repository carrying one dead `enforced by` reference and one unbound claim
# paragraph, runs the validator over it, and requires it to FAIL naming both. The Rust twin of the
# same control is `the_documentation_binding_checks_report_a_dead_reference_and_an_unbound_claim`
# in `crates/context-graph/tests/doc_claim_bindings.rs`.
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture="$(mktemp -d "${TMPDIR:-/tmp}/cg-doc-validator-control-XXXXXX")"
trap 'rm -rf "$fixture"' EXIT

mkdir -p "$fixture/crates/fixture_crate/tests"
printf '# Fixture repository\n' > "$fixture/README.md"
cat > "$fixture/crates/fixture_crate/tests/behaviour.rs" <<'FIXTURE_TEST'
#[test]
fn the_store_refuses_a_second_writer() {}
FIXTURE_TEST
cat > "$fixture/crates/fixture_crate/README.md" <<'FIXTURE_README'
# Fixture crate

Opening the store a second time while another writer holds it is refused with a typed error
naming the process that holds it.
<!-- enforced by: `the_store_refuses_a_second_writer` -->

Closing the store flushes every pending write before the handle is released, so a reader that
opens afterwards never observes a partial commit.

Reopening the store after a crash replays the write-ahead log before answering any read.
<!-- enforced by: `the_store_replays_the_write_ahead_log_after_a_crash` -->
FIXTURE_README

report="$fixture/report.txt"
status=0
python3 "$repository_root/scripts/validate-doc-coverage.py" \
    --repo "$fixture" \
    --changed-files crates/fixture_crate/README.md > "$report" 2>&1 || status=$?

if [[ "$status" -ne 1 ]]; then
    echo "the documentation validator must exit 1 on a repository carrying a dead reference and" >&2
    echo "an unbound claim; it exited $status. Report:" >&2
    cat "$report" >&2
    exit 1
fi
if ! grep -q "the_store_replays_the_write_ahead_log_after_a_crash" "$report"; then
    echo "the validator must name the dead reference it found. Report:" >&2
    cat "$report" >&2
    exit 1
fi
if ! grep -q "unbound claim(s) in a file this change touched" "$report"; then
    echo "the validator must report the unbound claim paragraph it found. Report:" >&2
    cat "$report" >&2
    exit 1
fi
echo "PASS: the documentation validator reports a dead reference and an unbound claim."
