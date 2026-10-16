#!/usr/bin/env bash
# Shared local entry points for the four CI paths. No argument retains the full contributor gate.
set -euo pipefail
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"
export RUNNER_TEMP="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
target_dir="${CARGO_TARGET_DIR:-$repository_root/target}"
stage="${1:-full}"
export TESTED_SHA="${TESTED_SHA:-$(git rev-parse HEAD)}"
export HEAD_SHA="${HEAD_SHA:-$TESTED_SHA}"
export BASE_SHA="${BASE_SHA:-}"
export EVENT_SHA="${EVENT_SHA:-}"
export REQUESTED_CONTEXTDB_SHA="${REQUESTED_CONTEXTDB_SHA:-}"
export DOCUMENTATION_BASE_SHA="${DOCUMENTATION_BASE_SHA:-${GATE_BASE:-}}"
export GATE_RECEIPT_DIR="${GATE_RECEIPT_DIR:-$RUNNER_TEMP/job-receipt}"
export CG_MODEL_CACHE_ROOT="${CG_MODEL_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/cg/models}"
# Commands are captured separately; receipts never copy arbitrary command output or environment.
receipt_stage=""
finish_receipt() {
    local result=$?
    trap - EXIT
    if [[ -n "$receipt_stage" ]]; then
        local receipt_result=0
        python3 scripts/gate-receipt.py "$receipt_stage" "$result" "$receipt_trace" "$receipt_output" || receipt_result=$?
        unlink "$receipt_trace"
        unlink "$receipt_output"
        if [[ "$receipt_result" != 0 ]]; then exit "$receipt_result"; fi
    fi
    exit "$result"
}
trace_stage() {
    # Log the exact arguments passed to each receipted tool, independently of stderr.
    # These functions live only in the pipeline subshell, including on Bash 3.2.
    receipt_command() {
        printf '+ ' >&9
        printf '%q ' "$@" >&9
        printf '\n' >&9
        # Children cannot write their output into the command channel.
        command "$@" 9>&-
    }
    cargo() { receipt_command cargo "$@"; }
    rustup() { receipt_command rustup "$@"; }
    python3() { receipt_command python3 "$@"; }
    bash() { receipt_command bash "$@"; }
    "$1"
}
run_stage() {
    receipt_stage="$1"
    if [[ "$receipt_stage" == default || "$receipt_stage" == features || "$receipt_stage" == features-tests ]]; then
        if [[ -e "$repository_root/target/nextest/gate/junit.xml" ]]; then unlink "$repository_root/target/nextest/gate/junit.xml"; fi
    fi
    mkdir -p "$GATE_RECEIPT_DIR"
    receipt_trace="$(mktemp "$GATE_RECEIPT_DIR/.commands.XXXXXX")"
    receipt_output="$(mktemp "$GATE_RECEIPT_DIR/.output.XXXXXX")"
    exec 9>"$receipt_trace"
    export GATE_BASH_PATH="$BASH" GATE_BASH_VERSION="$BASH_VERSION"
    printf 'Gate shell: %s (%s)\n' "$GATE_BASH_PATH" "$GATE_BASH_VERSION"
    trap finish_receipt EXIT
    trace_stage "$2" | tee "$receipt_output"
    python3 scripts/gate-receipt.py "$receipt_stage" 0 "$receipt_trace" "$receipt_output"
    trap - EXIT
    receipt_stage=""
    exec 9>&-
    unlink "$receipt_trace"
    unlink "$receipt_output"
}
pending() {
    printf 'PENDING: %s\n' "$*" >&2
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        printf 'PENDING: %s\n' "$*" >> "$GITHUB_STEP_SUMMARY"
    fi
    exit 2
}
require_value() {
    [[ -n "${!1:-}" ]] || pending "missing $1"
}
require_file() {
    require_value "$1"
    [[ -s "${!1}" ]] || pending "missing or empty file: $1"
}
source() {
    context_graph_sha="$(git rev-parse HEAD)"
    contextdb_sha="$(grep -o -m 1 -E 'contextdb[?]rev=[0-9a-f]{40}' Cargo.lock)"
    contextdb_sha="${contextdb_sha##*=}"
    if [[ ! "$context_graph_sha" =~ ^[0-9a-f]{40}$ ]] || [[ ! "$contextdb_sha" =~ ^[0-9a-f]{40}$ ]]; then
      echo "Context Graph checkout and locked ContextDB revision must be immutable 40-character commit SHAs" >&2
      exit 1
    fi
    if [[ "$context_graph_sha" != "$TESTED_SHA" ]]; then
      echo "checked-out Context Graph SHA does not match selected tested SHA" >&2
      exit 1
    fi
    if [[ -n "$REQUESTED_CONTEXTDB_SHA" ]] && [[ "$contextdb_sha" != "$REQUESTED_CONTEXTDB_SHA" ]]; then
      echo "locked ContextDB SHA does not match contextdb_sha" >&2
      exit 1
    fi
    printf 'Context Graph SHA: %s\n' "$context_graph_sha"
    printf 'ContextDB SHA: %s\n' "$contextdb_sha"
    printf 'Base SHA: %s\nHead SHA: %s\nEvent SHA: %s\n' "$BASE_SHA" "$HEAD_SHA" "$EVENT_SHA"
}

lint_audit() {
    require_file_without_match() {
      local pattern="$1"
      local file="$2"
      local grep_status=0
      grep -q -E "$pattern" "$file" || grep_status=$?
      case "$grep_status" in
        0)
          echo "forbidden dependency marker found in $file" >&2
          return 1
          ;;
        1) return 0 ;;
        *)
          echo "dependency-marker scan failed for $file" >&2
          return "$grep_status"
          ;;
      esac
    }
    cargo deny --config deny.toml check
    cargo expand --lib -p context-graph | tee "$RUNNER_TEMP/context-graph-expanded.rs"
    test -s "$RUNNER_TEMP/context-graph-expanded.rs"
    require_file_without_match 'embeddinggemma|candle_embeddinggemma' "$RUNNER_TEMP/context-graph-expanded.rs"
    cargo tree -p context-graph -e normal,build | tee "$RUNNER_TEMP/context-graph-tree.txt"
    test -s "$RUNNER_TEMP/context-graph-tree.txt"
    require_file_without_match '(^|[[:space:]])candle-embeddinggemma([[:space:]]|$)' "$RUNNER_TEMP/context-graph-tree.txt"
}
lint_format() {
    rustup run 1.94.1 cargo fmt --all --check
}
lint_clippy() {
    rustup run 1.94.1 cargo clippy --workspace --all-targets -- -D warnings
}
lint() {
    lint_audit
    lint_format
    lint_clippy
}

docs() {
    # Documentation and baseline comparison
    base="${DOCUMENTATION_BASE_SHA:-}"
    if [[ -z "$base" ]] || [[ "$base" =~ ^0+$ ]] || ! git rev-parse --verify --quiet "HEAD^" > /dev/null || ! git rev-parse --verify --quiet "${base}^{commit}" > /dev/null; then
      echo "initial state, no comparison: auditing all documentation"
      python3 scripts/validate-doc-coverage.py --all
    else
      printf 'comparing the touched doc files against %s\n' "$base"
      python3 scripts/validate-doc-coverage.py --base "$base"
    fi
    bash scripts/check-doc-validator-negative-control.sh
}

default_tests() {
    if command -v cargo-nextest > /dev/null 2>&1; then
        # Keep Cargo's complete default build closure, including compile-only examples.
        # nextest executes libtest cases; default_docs executes rustdoc cases once.
        rustup run 1.94.1 cargo test --workspace --no-run
    else
        # Preserve execution in the existing full-gate compatibility fallback.
        rustup run 1.94.1 cargo test --workspace
    fi
}
default_docs() {
    rustup run 1.94.1 cargo test --workspace --doc --no-fail-fast
}
default_baseline() {
    base="${DOCUMENTATION_BASE_SHA:-}"
    if [[ -z "$base" ]] || [[ "$base" =~ ^0+$ ]] || ! git rev-parse --verify --quiet "HEAD^" > /dev/null || ! git rev-parse --verify --quiet "${base}^{commit}" > /dev/null; then
      echo "initial state, no comparison: gate wall-clock baseline"
    else
      printf 'comparing the gate wall-clock baseline against %s\n' "$base"
      python3 scripts/check-gate-wall-clock.py --base "$base"
    fi
}
default_nextest() {
    rustup run 1.94.1 cargo nextest run --workspace --no-fail-fast --profile gate
}
default_wall() {
    export CG_GATE_JUNIT="$repository_root/target/nextest/gate/junit.xml"
    default_baseline
    rustup run 1.94.1 cargo test -p context-graph --test test_estate_audit gate_wall_seconds_never_exceed_the_frozen_baseline -- --ignored --exact
    python3 scripts/check-gate-wall-clock.py --junit "$CG_GATE_JUNIT"
}
default_wall_isolated() {
    export CG_GATE_JUNIT="$repository_root/target/nextest/gate/junit.xml"
    default_baseline
    rustup run 1.94.1 cargo test --locked --manifest-path scripts/gate-wall-audit/Cargo.toml --target-dir "$RUNNER_TEMP/gate-wall-audit-target" --test test_estate_audit gate_wall_seconds_never_exceed_the_frozen_baseline -- --ignored --exact
    python3 scripts/check-gate-wall-clock.py --junit "$CG_GATE_JUNIT"
}
default() {
    if [[ -n "${GATE_SHARD:-}" ]]; then
        sharded_default
        return
    fi
    default_tests
    default_docs
    if command -v cargo-nextest > /dev/null 2>&1; then
        default_nextest
        default_wall
    else
        if [[ "$stage" != full ]]; then
            pending 'cargo-nextest is required for this stage'
        fi
        export GATE_ALLOW_NO_NEXTEST=1
        echo 'cargo-nextest is not installed: JUnit checks skipped in the compatibility full gate'
    fi
}

feature_builds_core() {
    cargo build --locked -p context-graph --lib
    cargo build --locked -p context-graph --no-default-features --lib
}
feature_builds_sync() {
    cargo build --locked -p context-graph --features fabric-sync --lib
    cargo build --locked -p context-graph-cli
}
feature_builds() {
    feature_builds_core
    feature_builds_sync
}
feature_docs() {
    cargo test --workspace --doc --no-fail-fast --features context-graph-cli/event-search-test-support
}
feature_qualification() {
    feature_builds
    feature_docs
}
feature_qualification_sync() {
    feature_builds_sync
    feature_docs
}
features_tests() {
    [[ "${RUNNER_OS:-}" == Linux ]] || pending 'feature tests stage requires Linux'
    case "${GATE_SHARD:-}" in
        acceptance|integration-and-rest|import-and-cli) ;;
        *) pending 'feature tests stage requires an exact shard' ;;
    esac
    python3 scripts/ci-test-shards.py run features "$GATE_SHARD"
}

features() {
    if [[ -n "${GATE_SHARD:-}" ]]; then
        sharded_features
        return
    fi
    feature_builds
    cargo nextest run --workspace --no-fail-fast --profile gate --features context-graph-cli/event-search-test-support
    feature_docs
}

sharded_default() {
    if [[ "$GATE_SHARD" == integration-and-rest ]]; then
        default_tests
        default_docs
    fi
    python3 scripts/ci-test-shards.py run default "$GATE_SHARD"
}
sharded_features() {
    if [[ "$GATE_SHARD" == integration-and-rest ]]; then
        feature_builds
        feature_docs
    fi
    python3 scripts/ci-test-shards.py run features "$GATE_SHARD"
}

shard_union() {
    local systems=(Linux macOS)
    case "$stage" in
        shard-union-linux|shard-union-linux-isolated) systems=(Linux) ;;
        shard-union-macos) systems=(macOS) ;;
        shard-union) ;;
        *) pending 'unknown shard union stage' ;;
    esac
    if [[ "$stage" == shard-union ]]; then
        python3 scripts/ci-test-shards.py verify "$RUNNER_TEMP/shard-receipts" --sha "$TESTED_SHA"
    else
        python3 scripts/ci-test-shards.py verify "$RUNNER_TEMP/shard-receipts" --sha "$TESTED_SHA" --system "${systems[0]}"
    fi
    mkdir -p "$repository_root/target/nextest/gate"
    for system in "${systems[@]}"; do
        cp "$RUNNER_TEMP/shard-receipts/default-$system.xml" "$repository_root/target/nextest/gate/junit.xml"
        if [[ "$stage" == shard-union-linux-isolated ]]; then
            default_wall_isolated
        else
            default_wall
        fi
    done
}

model_selection() {
    model_cargo_selection=(-p context-graph)
    case "${GATE_CI_MODEL_MODE:-}" in
        "") ;;
        catalog-narrow|catalog-workspace|catalog-features)
            [[ "${CI:-}" == true ]] || pending 'CI model mode requires CI=true'
            case "$GATE_CI_MODEL_MODE" in
                catalog-workspace) model_cargo_selection=(--workspace) ;;
                catalog-features) model_cargo_selection=(--workspace --features context-graph-cli/event-search-test-support) ;;
            esac
            ;;
        *) pending 'unknown CI model mode' ;;
    esac
}
model_seed() {
    model_selection
    if [[ -n "${GATE_CI_MODEL_MODE:-}" ]]; then
        python3 scripts/ci-model-seed.py
    else
        rustup run 1.94.1 cargo run -p context-graph-cli --bin cg -- model fetch --defaults
    fi
    rustup run 1.94.1 cargo test "${model_cargo_selection[@]}" --test nomic_document_preparer_adds_exactly_one_system_task_prefix
}
model_proofs() {
    model_selection
    rustup run 1.94.1 cargo test "${model_cargo_selection[@]}" \
      --test nomic_document_preparer_adds_exactly_one_system_task_prefix \
      --test nomic_query_preparer_adds_exactly_one_system_task_prefix \
      --test event_search_uses_role_preparation_while_digest_record_and_evidence_paths_remain_byte_for_byte_unchanged
}

public_model() {
    if [[ "${MODEL_CACHE_HIT:-false}" != true ]]; then
        model_seed
    fi
    model_proofs
}

trusted_inputs() {
    for key in APPROVED_SESSION_LIST APPROVED_GOLD_DIR DISPOSABLE_WORK_ROOT VARIANT_LABEL CONFIGURED_MODEL_ID REPORT_PATH GRAY_PATH ADJUDICATED_PATH OPENROUTER_API_KEY; do
        require_value "$key"
    done
    [[ -d "$APPROVED_GOLD_DIR" ]] || pending 'approved gold directory is unavailable'
    IFS=',' read -r -a sessions <<< "$APPROVED_SESSION_LIST"
    [[ "${#sessions[@]}" -gt 0 ]] || pending 'approved session list is empty'
    for session in "${sessions[@]}"; do
        [[ -s "$session" ]] || pending 'approved session is unavailable'
        stem="$(basename "$session" .jsonl)"
        [[ -s "$APPROVED_GOLD_DIR/${stem:0:8}.yaml" ]] || pending 'session gold is unavailable'
    done
    [[ -s "$DISPOSABLE_WORK_ROOT/cg/capture.toml" ]] || pending 'approved hosted model configuration is unavailable'
    require_file QUALITY_INPUT_RECEIPT
    # Acceptance is supplied by the product's existing contract; a runner exit alone is insufficient.
    python3 - "$QUALITY_INPUT_RECEIPT" <<'PY'
import json, os, subprocess, sys
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
for path in sys.argv[1:]:
    receipt = json.load(open(path))
    if receipt.get('cg_sha') != head or receipt.get('contextdb_sha') != os.environ.get('REQUESTED_CONTEXTDB_SHA'):
        raise SystemExit('PENDING: quality receipt tuple mismatch')
    if not receipt.get('contract') or not receipt.get('corpus_identity') or not receipt.get('gold_identity'):
        raise SystemExit('PENDING: quality receipt lacks approved input identity or contract')
PY
}
answer_quality() {
    trusted_inputs
    cargo build --locked --release -p context-graph-cli --bin cg -p cg-bench --bin cg-bench
    target/release/cg-bench --cg "$PWD/target/release/cg" \
      --sessions "$APPROVED_SESSION_LIST" --gold-dir "$APPROVED_GOLD_DIR" \
      --work-root "$DISPOSABLE_WORK_ROOT" --label "$VARIANT_LABEL" \
      --model-name "$CONFIGURED_MODEL_ID" --report "$REPORT_PATH" --gray-out "$GRAY_PATH"
    [[ -f "$GRAY_PATH" ]] || pending 'gray-pair result is unavailable'
    python3 - "$GRAY_PATH" <<'PY_GRAY'
import pathlib, sys
for row in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if row.strip() and len(row.split('\t')) < 3:
        raise SystemExit('PENDING: malformed gray-pair row requires review')
PY_GRAY
    if [[ -s "$GRAY_PATH" ]]; then
        python3 scripts/cg-bench-adjudicate.py "$GRAY_PATH" "$ADJUDICATED_PATH"
    else
        # The existing adjudicator cannot sample an empty list. Preserve the explicit no-gray result.
        : > "$ADJUDICATED_PATH"
    fi
    require_file REPORT_PATH
    [[ -f "$ADJUDICATED_PATH" ]] || pending 'adjudication result is unavailable'
    require_file QUALITY_ACCEPTANCE_RECEIPT
    python3 - "$QUALITY_ACCEPTANCE_RECEIPT" "$REPORT_PATH" "$ADJUDICATED_PATH" <<'PY'
import hashlib, json, os, subprocess, sys
receipt = json.load(open(sys.argv[1]))
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if receipt.get('cg_sha') != head or receipt.get('contextdb_sha') != os.environ.get('REQUESTED_CONTEXTDB_SHA'):
    raise SystemExit('PENDING: acceptance source tuple mismatch')
inputs = json.load(open(os.environ['QUALITY_INPUT_RECEIPT']))
for key in ['contract', 'corpus_identity', 'gold_identity']:
    if receipt.get(key) != inputs[key]:
        raise SystemExit('PENDING: acceptance input identity mismatch')
for key, path in [('report_sha256', sys.argv[2]), ('adjudicated_sha256', sys.argv[3])]:
    if receipt.get(key) != hashlib.sha256(open(path, 'rb').read()).hexdigest():
        raise SystemExit('PENDING: acceptance receipt does not bind this result')
for key in ['all_gray_pairs_reviewed', 'no_gray_case_accounted', 'sample_complete', 'observations_accounted', 'accepted']:
    if receipt.get(key) is not True:
        raise SystemExit('PENDING: incomplete product acceptance: ' + key)
PY
}
smoke_binary() {
    mkdir -p "$GATE_RECEIPT_DIR"
    if [[ -n "${CG_BIN:-}" ]]; then
        require_file CG_SMOKE_BUILD_RECEIPT
        [[ -x "$CG_BIN" ]] || pending 'selected smoke binary is unavailable'
        python3 scripts/smoke-receipt.py verify "$CG_BIN" "$CG_SMOKE_BUILD_RECEIPT"
    else
        git diff --quiet HEAD -- || pending 'selected smoke build requires an unchanged checkout'
        cargo build --locked --release -p context-graph-cli --bin cg
        export CG_BIN="$target_dir/release/cg"
        export CG_SMOKE_BUILD_RECEIPT="$GATE_RECEIPT_DIR/cg-build.json"
        python3 scripts/smoke-receipt.py build "$CG_BIN" "$CG_SMOKE_BUILD_RECEIPT"
    fi
    export CG_BIN
}
smoke_run() {
    local smoke="$1"
    local result=0
    # Explicit paths bind every script, including capture's CG setting.
    CG_BIN="$CG_BIN" CG="$CG_BIN" bash "scripts/$smoke-live-smoke.sh" || result=$?
    python3 scripts/smoke-receipt.py result "$CG_BIN" "$GATE_RECEIPT_DIR/$smoke.json" "$smoke" "$result"
    return "$result"
}
llm_byok() {
    for key in CG_LLM_SMOKE_HOSTED_ENDPOINT CG_LLM_SMOKE_LOCAL_ENDPOINT CG_LLM_SMOKE_HOSTED_API_KEY_ENV; do
        require_value "$key"
    done
    require_value "$CG_LLM_SMOKE_HOSTED_API_KEY_ENV"
    smoke_binary
    smoke_run llm-byok
}
deterministic_llm() {
    require_value OPENROUTER_API_KEY
    command -v claude > /dev/null 2>&1 || pending 'installed provider CLI is unavailable'
    smoke_binary
    smoke_run cg-deterministic-llm
}
capture() {
    require_value OPENROUTER_API_KEY
    require_file CG_SMOKE_REAL_JSONL
    require_value CG_SMOKE_HUB_TICKET
    require_value CG_SMOKE_TENANT
    require_file CONTEXTDB
    require_file CONTEXTDB_BUILD_RECEIPT
    python3 scripts/smoke-receipt.py dependency "$CONTEXTDB" "$CONTEXTDB_BUILD_RECEIPT"
    smoke_binary
    export CG_SMOKE_SKIP_BUILD=1
    smoke_run cg-capture
}
performance_inputs() {
    for key in BASELINE_CG_SHA CANDIDATE_CG_SHA BASELINE_CONTEXTDB_SHA CANDIDATE_CONTEXTDB_SHA; do
        require_value "$key"
        [[ "${!key}" =~ ^[0-9a-f]{40}$ ]] || pending "invalid exact SHA: $key"
    done
    for key in PERFORMANCE_ENVIRONMENT_RECEIPT PERFORMANCE_ACCEPTANCE_RECEIPT; do
        require_file "$key"
    done
    for key in CG_EVENT_BENCH_CORPUS_JSONL_DIR BASELINE_FIXTURE_DIR CANDIDATE_FIXTURE_DIR; do
        require_value "$key"
        [[ -d "${!key}" ]] || pending "missing sealed fixture: $key"
    done
    for key in CG_EVENT_BENCH_TOTALS CG_EVENT_BENCH_CONTEXTS CG_EVENT_BENCH_RECALL_QUERIES PERFORMANCE_RESULT_DIR; do
        require_value "$key"
    done
    python3 - "$PERFORMANCE_ENVIRONMENT_RECEIPT" <<'PY'
import json, os, sys
receipt = json.load(open(sys.argv[1]))
for key in ['hardware', 'filesystem', 'resource_limits', 'fixture_identity', 'model_identity', 'profile']:
    if not receipt.get(key):
        raise SystemExit('PENDING: environment receipt lacks ' + key)
if receipt.get('isolated') is not True:
    raise SystemExit('PENDING: controlled performance environment is unavailable')
for key in ['baseline_cg_sha', 'candidate_cg_sha', 'baseline_contextdb_sha', 'candidate_contextdb_sha']:
    if receipt.get(key) != os.environ[key.upper()]:
        raise SystemExit('PENDING: environment tuple mismatch: ' + key)
PY
}
performance_build() {
    require_value CG_EVENT_BENCH_ENGINE_SHA
    require_value CG_EVENT_BENCH_CONSUMER_SHA
    [[ "$CG_EVENT_BENCH_CONSUMER_SHA" == "$(git rev-parse HEAD)" ]] || pending 'benchmark source tuple mismatch'
    REQUESTED_CONTEXTDB_SHA="$CG_EVENT_BENCH_ENGINE_SHA" GITHUB_EVENT_NAME=workflow_dispatch source
    require_value PERFORMANCE_BUILD_RECEIPT
    python3 scripts/performance-build.py "$PERFORMANCE_BUILD_RECEIPT"
}
performance() {
    performance_inputs
    require_file BASELINE_BENCH_BINARY
    require_file CANDIDATE_BENCH_BINARY
    require_file BASELINE_BUILD_RECEIPT
    require_file CANDIDATE_BUILD_RECEIPT
    python3 scripts/validate-paired-results.py --fixtures
    mkdir -p "$PERFORMANCE_RESULT_DIR"
    export CG_EVENT_BENCH_ACCEPTANCE=1
    # Paired observations are serialized in this process on this machine, with separate fresh destinations.
    for variant in baseline candidate; do
        binary_key="${variant^^}_BENCH_BINARY"
        fixture_key="${variant^^}_FIXTURE_DIR"
        export CG_EVENT_BENCH_FIXTURE_DIR="${!fixture_key}"
        export CG_EVENT_BENCH_FRESH_GRAPH_SOURCE="$CG_EVENT_BENCH_FIXTURE_DIR/P10000-K1-sessions"
        export CG_EVENT_BENCH_EVIDENCE="$PERFORMANCE_RESULT_DIR/$variant.jsonl"
        [[ ! -e "$CG_EVENT_BENCH_EVIDENCE" ]] || pending 'result destination must be fresh'
        "${!binary_key}" --bench event_find_recall_at_10_against_exact_on_the_real_model_fixture
        export CG_EVENT_BENCH_FRESH_GRAPH_DIR="$PERFORMANCE_RESULT_DIR/$variant-fresh-graph"
        [[ ! -e "$CG_EVENT_BENCH_FRESH_GRAPH_DIR" ]] || pending 'fresh graph destination already exists'
        "${!binary_key}" --bench event_fresh_graph_build
        [[ -s "$CG_EVENT_BENCH_EVIDENCE" ]] || pending 'benchmark emitted no evidence'
        build_key="${variant^^}_BUILD_RECEIPT"
        python3 - "${!build_key}" "$CG_EVENT_BENCH_EVIDENCE" <<'PY_IDENTITY'
import json, pathlib, sys
identity = json.loads(pathlib.Path(sys.argv[1]).read_text())
with pathlib.Path(sys.argv[2]).open('a') as output:
    output.write(json.dumps({'build_identity': identity}, sort_keys=True) + '\n')
PY_IDENTITY
        python3 - "${!binary_key}" "$PERFORMANCE_RESULT_DIR/$variant-process.json" <<'PY_PROCESS'
import hashlib, json, pathlib, sys
binary = hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()
pathlib.Path(sys.argv[2]).write_text(json.dumps({'binary_sha256': binary, 'selectors': ['event_find_recall_at_10_against_exact_on_the_real_model_fixture', 'event_fresh_graph_build'], 'exit_codes': [0, 0]}, sort_keys=True) + '\n')
PY_PROCESS
    done
    python3 scripts/validate-paired-results.py "$PERFORMANCE_ACCEPTANCE_RECEIPT" "$PERFORMANCE_RESULT_DIR"

}
publication_check() {
    source
    cargo build --locked --release -p context-graph-cli
    cargo package --locked -p context-graph-cli --list
    cargo metadata --locked --no-deps --format-version 1 > "$RUNNER_TEMP/package-metadata.json"
    python3 - "$RUNNER_TEMP/package-metadata.json" "$target_dir/release/cg" <<'PY'
import json, pathlib, sys
packages = json.load(open(sys.argv[1]))['packages']
package = next(p for p in packages if p['name'] == 'context-graph-cli')
if package['publish'] != [] or not pathlib.Path(sys.argv[2]).is_file():
    raise SystemExit('local package inspection failed')
print('local package inspected; no publication performed')
PY
}
case "$stage" in
    shard-union|shard-union-linux|shard-union-macos|shard-union-linux-isolated) run_stage "$stage" shard_union ;;
    full)
        cargo fmt --all --check
        cargo clippy --workspace --all-targets -- -D warnings
        default
        docs
        ;;
    source|lint|docs) run_stage "$stage" "$stage" ;;
    feature-qualification) run_stage "$stage" feature_qualification ;;
    feature-qualification-core) run_stage "$stage" feature_builds_core ;;
    feature-qualification-sync) run_stage "$stage" feature_qualification_sync ;;
    features-tests)
        if [[ "${GATE_MODEL_READY:-}" != 1 ]]; then run_stage public-model public_model; fi
        run_stage "$stage" features_tests ;;
    default|features)
        if [[ "${GATE_MODEL_READY:-}" != 1 ]]; then run_stage public-model public_model; fi
        run_stage "$stage" "$stage" ;;
    model-seed) run_stage model-seed model_seed ;;
    public-model) run_stage public-model public_model ;;
    pr) run_stage source source; run_stage lint lint; run_stage docs docs; run_stage public-model public_model; run_stage default default; run_stage features features ;;
    extended-public) run_stage source source; run_stage public-model public_model; run_stage features features ;;
    extended-trusted) source; trusted_inputs; answer_quality; llm_byok; deterministic_llm; capture ;;
    trusted-inputs) source; trusted_inputs ;;
    answer-quality) source; answer_quality ;;
    llm-byok) source; llm_byok ;;
    deterministic-llm) source; deterministic_llm ;;
    capture) source; capture ;;
    performance-manifest) performance_inputs ;;
    performance-build) performance_build ;;
    performance) performance ;;
    publication-check) publication_check ;;
    *) echo "unknown gate stage: $stage" >&2; exit 1 ;;
esac
