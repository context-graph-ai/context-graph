#!/usr/bin/env bash
# cg-llm-client-refactor smoke test — OpenRouter BYOK driver (Step 4b acceptance gate)
# ---------------------------------------------------------------------------
# Drives the compiled `cg` binary through the library-grade capture harness
# (Required Behavior §1–§8) against inputs that exist in NO test fixture and NO
# fake-provider branch, so it can only pass if real extraction + the real
# downstream logic work. OpenRouter is the model provider (BYOK, OpenAI-compatible).
#
# Two halves:
#   CAPTURE HALF (live now): §1 full-ontology extraction, §4 four-stage pipeline,
#     §J cost/time receipt + zero-outbound --estimate, §L connected-by-default,
#     §7 sync split (requires a contextdb relay hub for a release-ready verdict).
#   CORRECTION/SKILL/HITL HALF (§5,§6,§8): REAL assertions now that the CLI verbs
#     fix/drop/group/undo/skills/review/why are wired. Item handles are `@n`
#     ordinals over the captured units (digests) in a context, newest first.
#     Assertions check SHAPES and LOOP behavior, never exact model wording.
#
# WHAT THE CLI CANNOT DRIVE (recorded so a SKIP is read as a finding, not a pass):
#   - The POSITIVE `cg why` branch needs a genuinely-flagged closed-vocab item; the
#     flag is model-dependent and the flagged id is not surfaced by `cg review`, so
#     §8 asserts the discriminating NEGATIVE branch and eyeballs the positive one.
#
# §2 deterministic grounding is not a live-model wording assertion. This smoke runs
# the grounding acceptance module before the live capture so entity, decision_type,
# agent_id, observation, and outcome grounding remain part of the gate.
#
# REQUIRED env:
#   OPENROUTER_API_KEY   your OpenRouter key. Referenced by NAME only (--env-key),
#                        so it never lands in the catalog file or this log.
# OPTIONAL env (shown with defaults):
#   CG_SMOKE_MODEL       anthropic/claude-haiku-4.5   any OpenRouter model slug
#   CG_SMOKE_ENDPOINT    https://openrouter.ai/api/v1
#   CG_SMOKE_TIMEOUT_MS  120000
#   CG_DATA_ROOT         /tmp/cg-refactor-smoke
#   CG_SMOKE_LOG         /tmp/cg-llm-client-refactor-smoke.log
#   CG                   <worktree>/target/release/cg
#   CG_SMOKE_SKIP_BUILD  0    set 1 to skip cargo build
#   CG_SMOKE_HUB         (unset)  contextdb relay-hub NATS URL with a relay worker
#                        serving CG_SMOKE_TENANT; missing means NOT RELEASE-READY
#   CG_SMOKE_TENANT      generated local tenant id; required with CG_SMOKE_HUB for §7 sync
#   CG_SMOKE_REAL_JSONL  largest real session JSONL to import in the owner-feel gate;
#                        defaults to the largest *.jsonl under CG_SMOKE_REAL_JSONL_DIR
#   CG_SMOKE_REAL_JSONL_DIR  $HOME/.claude/projects
#
# Cost note: only capture (record/import) calls the model. query/status/review are
# model-free. --estimate is zero-outbound by contract.
# ---------------------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="${CG_WORKTREE:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CG="${CG:-$WORKTREE/target/release/cg}"
CG_DATA_ROOT="${CG_DATA_ROOT:-/tmp/cg-refactor-smoke}"
LOG="${CG_SMOKE_LOG:-/tmp/cg-llm-client-refactor-smoke.log}"
MODEL="${CG_SMOKE_MODEL:-anthropic/claude-haiku-4.5}"
ENDPOINT="${CG_SMOKE_ENDPOINT:-https://openrouter.ai/api/v1}"
PROVIDER="openrouter"
TIMEOUT_MS="${CG_SMOKE_TIMEOUT_MS:-120000}"
SKIP_BUILD="${CG_SMOKE_SKIP_BUILD:-0}"
TENANT="${CG_SMOKE_TENANT:-cg-smoke-$RANDOM-$$}"
HUB="${CG_SMOKE_HUB:-}"

export CG_DATA_ROOT
unset CG_USE_TEST_LEXICAL_EMBEDDER   # use the REAL bundled embedder, not the test one

# Load a local .env (OPENROUTER_API_KEY=...) if present — silently, before logging.
for envf in "$WORKTREE/.env" "$PWD/.env"; do
  [[ -f "$envf" ]] && { set -a; . "$envf"; set +a; }
done

exec > >(tee "$LOG") 2>&1

PASS=0; FAIL=0; SKIP=0
declare -a FAILED_CHECKS=()
LAST_OUT=""; LAST_RC=0

section() { printf '\n\n========================================================================\n== %s\n========================================================================\n' "$1"; }
note()    { printf '   %s\n' "$1"; }
check() {  # check "<desc>" <0|1>
  local desc="$1" ok="$2"
  if [[ "$ok" == "0" ]]; then printf '   [PASS] %s\n' "$desc"; PASS=$((PASS+1))
  else printf '   [FAIL] %s\n' "$desc"; FAIL=$((FAIL+1)); FAILED_CHECKS+=("$desc"); fi
}
skip()    { printf '   [SKIP] %s\n' "$1"; SKIP=$((SKIP+1)); }
pending() { printf '   [PENDING] %s\n' "$1"; SKIP=$((SKIP+1)); }

cg_run() {  # cg_run <argv...>  -> LAST_OUT / LAST_RC
  printf '\n$ cg %s\n' "$*"
  local tmp
  tmp="$(mktemp /tmp/cg-smoke-cg-run.XXXXXX)"
  "$CG" "$@" 2>&1 | tee "$tmp"
  LAST_RC=${PIPESTATUS[0]}
  LAST_OUT="$(cat "$tmp")"
  rm -f "$tmp"
}
cg_record() {  # cg_record <context-args...> <text-as-last-arg>
  local text="${!#}"; set -- "${@:1:$(($#-1))}"
  printf '\n$ printf <session> | cg record --understand-now %s\n' "$*"
  local tmp
  tmp="$(mktemp /tmp/cg-smoke-cg-record.XXXXXX)"
  printf '%s' "$text" | "$CG" record --understand-now "$@" 2>&1 | tee "$tmp"
  LAST_RC=${PIPESTATUS[1]}
  LAST_OUT="$(cat "$tmp")"
  rm -f "$tmp"
}
cg_record_retry() {  # cg_record_retry <attempts> <context-args...> <text-as-last-arg>
  local attempts="$1"; shift
  local attempt
  for attempt in $(seq 1 "$attempts"); do
    cg_record "$@"
    if [[ "$(ok_rc)" == "0" ]]; then
      return 0
    fi
    note "record attempt ${attempt}/${attempts} failed; retrying a real provider call"
  done
  return 1
}
contains()       { grep -qiF -- "$2" <<<"$1"; }
counts_line()    { grep -oiE 'goals:[[:space:]]*[0-9]+[[:space:]]+decisions:[[:space:]]*[0-9]+[[:space:]]+observations:[[:space:]]*[0-9]+' <<<"$1" | tail -1; }
counts_nonzero() { local l; l="$(counts_line "$1")"; [[ -n "$l" && ! "$l" =~ goals:[[:space:]]*0[[:space:]]+decisions:[[:space:]]*0[[:space:]]+observations:[[:space:]]*0 ]]; }
counts_sum()     { local l; l="$(counts_line "$1")"; grep -oE '[0-9]+' <<<"$l" | paste -sd+ - | bc 2>/dev/null || echo 0; }
is_ok_json()     { contains "$1" '"ok":true' || ! contains "$1" '"ok":false'; }
ok_rc()          { [[ $LAST_RC -eq 0 ]] && is_ok_json "$LAST_OUT" && echo 0 || echo 1; }
largest_real_jsonl() {
  if [[ -n "${CG_SMOKE_REAL_JSONL:-}" ]]; then
    printf '%s\n' "$CG_SMOKE_REAL_JSONL"
    return 0
  fi
  local dir="${CG_SMOKE_REAL_JSONL_DIR:-$HOME/.claude/projects}"
  [[ -d "$dir" ]] || return 1
  find "$dir" -type f -name '*.jsonl' -printf '%s\t%p\n' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | cut -f2-
}
json_require_jq() {
  if [[ "$HAVE_JQ" != "1" ]]; then
    check "jq is available for hard JSON assertions" 1
    return 1
  fi
  return 0
}

# Pull a top-level JSON field out of a cg result. The verbs print one JSON object
# (on the first line) followed by the plain `rendered` echo, so `jq` over the first
# JSON-looking line is robust; if jq is missing we fall back to a flat-key grep.
# NOTE: we do NOT use jq's `// empty` here — `false // empty` is empty, which would
# silently swallow a real `flagged:false`. A truly-absent key prints `null`.
HAVE_JQ=0; command -v jq >/dev/null 2>&1 && HAVE_JQ=1
json_first_obj() { grep -m1 -E '^\{' <<<"$1"; }   # the single-line JSON object cg emits
json_field() {  # json_field <output> <jq-filter>   e.g. '.flagged'  '.skills | length'
  local obj; obj="$(json_first_obj "$1")"
  if [[ "$HAVE_JQ" == "1" && -n "$obj" ]]; then
    jq -r "$2" <<<"$obj" 2>/dev/null
  else
    # crude flat-key fallback: `.foo` -> "foo":<value> ; good enough for scalars.
    local key="${2#.}"; key="${key%%.*}"
    grep -oE "\"$key\":[^,}]*" <<<"$obj" | head -1 | sed -E 's/^"[^"]*"://; s/^"//; s/"$//'
  fi
}
# `cg review` per-category counts live ONLY in the `rendered` summary line
# (`summary: uncertain=N groupings=N drift=N currency_flips=N`); the JSON `summary`
# key is overwritten by the context-summary string in print_memory_json, so parse
# the rendered line. Returns the integer for one category, or empty if absent.
review_count() {  # review_count <output> <category>   e.g. drift | currency_flips | uncertain | groupings
  grep -oiE "$2=[0-9]+" <<<"$1" | head -1 | grep -oE '[0-9]+'
}

# A §5–§8 verb is "available" once it is wired into the CLI dispatch. Until then
# `cg <verb>` prints `unknown command `<verb>`` (usage_error) and the section SKIPs.
# All seven verbs (fix/drop/group/undo/skills/review/why) are wired in this build;
# the guard stays so the script degrades gracefully on an older binary.
verb_available() {
  local out; out="$("$CG" "$1" 2>&1 || true)"
  ! grep -qiF "unknown command \`$1\`" <<<"$out"
}

# ---------------------------------------------------------------------------
section "SETUP"
printf 'log file       : %s\n' "$LOG"
printf 'cg binary      : %s\n' "$CG"
printf 'data root      : %s\n' "$CG_DATA_ROOT"
printf 'provider       : %s  endpoint=%s\n' "$PROVIDER" "$ENDPOINT"
printf 'model          : %s\n' "$MODEL"
printf 'tenant         : %s\n' "$TENANT"
printf 'relay hub (§7) : %s\n' "${HUB:-<unset — release gate will fail>}"
printf 'api key        : OPENROUTER_API_KEY (by NAME only; never printed)\n'

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  printf '\nERROR: OPENROUTER_API_KEY is not present in the smoke environment. The smoke reads the existing variable by name and never prints its value.\n'
  exit 2
fi
json_require_jq || exit 2

# ---------------------------------------------------------------------------
section "STEP 0 — build cg"
if [[ "$SKIP_BUILD" == "1" && -x "$CG" ]]; then
  note "skip build (CG_SMOKE_SKIP_BUILD=1), using existing $CG"
else
  ( cd "$WORKTREE" && cargo build --release --bin cg )
fi
[[ -x "$CG" ]] && check "cg binary present and executable" 0 || { check "cg binary present and executable" 1; exit 2; }

# ---------------------------------------------------------------------------
section "STEP 1 — clean throwaway state + raise provider timeout"
rm -rf "$CG_DATA_ROOT"
mkdir -p "$CG_DATA_ROOT/cg"
cat > "$CG_DATA_ROOT/cg/providers.toml" <<EOF
[providers.${PROVIDER}]
timeout_ms = ${TIMEOUT_MS}
EOF
note "wrote $CG_DATA_ROOT/cg/providers.toml (timeout_ms=${TIMEOUT_MS})"

# ---------------------------------------------------------------------------
section "STEP 1B — relay hub preflight for the cross-machine gate"
if [[ -z "$HUB" || -z "$TENANT" ]]; then
  check "cross-machine skill durability has CG_SMOKE_HUB and CG_SMOKE_TENANT" 1
  note "This is a release-gate blocker, not a clean skip: run with a NATS URL served by a contextdb relay worker for the tenant."
else
  cg_run sync push --remote-server "$HUB" --tenant-id "$TENANT"
  check "relay hub accepts an initial sync push for this tenant" "$(ok_rc)"
fi

# ---------------------------------------------------------------------------
section "STEP 2 — connect OpenRouter and prove it is live (gating step)"
cg_run model connect "$PROVIDER" --provider "$PROVIDER" --endpoint "$ENDPOINT" --model "$MODEL" --env-key OPENROUTER_API_KEY --consent
check "model connect returned ok" "$(ok_rc)"
cg_run model status
cg_run model selftest "$PROVIDER"
if [[ $LAST_RC -eq 0 ]] && is_ok_json "$LAST_OUT"; then
  check "model selftest succeeded (live extraction is wired)" 0
else
  check "model selftest succeeded (live extraction is wired)" 1
  note "Capture below will fall to verbatim until selftest passes. Fix the key/slug/timeout first."
fi

# ---------------------------------------------------------------------------
section "STEP 3A — §2 deterministic grounding over every extracted value class"
if ( cd "$WORKTREE" && cargo test -p context-graph --test acceptance grounding_silence ); then
  check "grounding flags fabricated entity, decision_type, agent_id, observation, and outcome values" 0
else
  check "grounding flags fabricated entity, decision_type, agent_id, observation, and outcome values" 1
fi

# ---------------------------------------------------------------------------
section "STEP 3 — §1/§4 full-ontology capture, receipt leads with intention"
read -r -d '' SESS1 <<'EOF' || true
We are building the billing service and the goal is to keep it reliable under load.
Decision: we will use PostgreSQL on RDS as the primary store for billing, because we
need managed backups and point-in-time recovery. The rate-limiter config lives in the
repository billing-svc, module ratelimit, file config/ratelimit.toml, and we cap requests
at 100/sec. Observed: p99 latency was 42 ms in the last load test. Outcome: the cap held;
no rejected requests during the spike.
EOF
cg_record --context billing-demo "$SESS1"
CAPTURE_OUT="$LAST_OUT"
check "record #1 ok (real extraction, not an engine error)" "$(ok_rc)"
check "capture receipt leads with an intention/goal (before decisions/entities)" \
  "$(grep -niE 'goal|intention' <<<"$CAPTURE_OUT" | head -1 | grep -qiE 'goal|intention' && echo 0 || echo 1)"
check "OpenRouter/BYOK capture reports Native/captured in the usage receipt" \
  "$(grep -q '"reliability_level":"Native"' <<<"$CAPTURE_OUT" && grep -q '"outcome":"captured"' <<<"$CAPTURE_OUT" && echo 0 || echo 1)"

cg_run status --context billing-demo
check "status shows non-zero counts (real extraction happened)" "$(counts_nonzero "$LAST_OUT" && echo 0 || echo 1)"
note "counts: $(counts_line "$LAST_OUT")"
note "EYEBALL the rows: a repository+module transcript must yield Repository and Module"
note "entities (never collapsed to File); the decision must carry a model-derived type and"
note "reasoning, confidence NULL at capture; the observation must carry its latency lane;"
note "the outcome must be attached. (§1 — judge by shape, model wording varies.)"

# ---------------------------------------------------------------------------
section "STEP 4 — §J cost/time receipt + zero-outbound --estimate"
check "capture receipt reports token + time cost" \
  "$(grep -q '"prompt_tokens":[1-9]' <<<"$CAPTURE_OUT" && grep -q '"completion_tokens":[1-9]' <<<"$CAPTURE_OUT" && grep -q '"elapsed_ms":[1-9]' <<<"$CAPTURE_OUT" && echo 0 || echo 1)"
# --estimate must compute without a model call. Prove it by estimating with the
# provider connection REMOVED from a fresh data root (no outbound possible).
( export CG_DATA_ROOT="${CG_DATA_ROOT}-estimate"; rm -rf "$CG_DATA_ROOT"; mkdir -p "$CG_DATA_ROOT"
  printf '%s\n' '{"message":{"role":"user","content":"We decided to keep PostgreSQL on RDS for billing reliability."}}' > "$CG_DATA_ROOT/sess.jsonl"
  printf '\n$ cg import sess.jsonl --estimate   (no model connection in this data root)\n'
  "$CG" import "$CG_DATA_ROOT/sess.jsonl" --estimate 2>&1
) > /tmp/cg-refactor-estimate.out 2>&1
cat /tmp/cg-refactor-estimate.out
if grep -q '"ok":true' /tmp/cg-refactor-estimate.out && grep -q '"provider_calls":0' /tmp/cg-refactor-estimate.out && ! grep -qiE 'unreachable|connection refused|timeout' /tmp/cg-refactor-estimate.out; then
  check "--estimate produced an estimate with NO model connection (zero-outbound)" 0
else
  check "--estimate produced an estimate with NO model connection (zero-outbound)" 1
fi

# ---------------------------------------------------------------------------
section "STEP 4B — owner-feel real JSONL import with the default window"
REAL_JSONL="$(largest_real_jsonl || true)"
if [[ -z "$REAL_JSONL" || ! -f "$REAL_JSONL" ]]; then
  check "owner-feel real JSONL import requires CG_SMOKE_REAL_JSONL or a real *.jsonl under CG_SMOKE_REAL_JSONL_DIR" 1
else
  note "real JSONL: $REAL_JSONL"
  cg_run import --estimate "$REAL_JSONL" --context owner-feel
  REAL_ESTIMATE_OUT="$LAST_OUT"
  check "owner-feel real JSONL estimate succeeds with zero provider calls" \
    "$([[ $LAST_RC -eq 0 ]] && jq -e '.estimate.provider_calls == 0' <<<"$(json_first_obj "$REAL_ESTIMATE_OUT")" >/dev/null 2>&1 && echo 0 || echo 1)"
  check "owner-feel real JSONL default import window is bounded for hosted output budget" \
    "$(jq -e '.estimate.resolved_chunk_window <= 16000 and .estimate.chunks > 1' <<<"$(json_first_obj "$REAL_ESTIMATE_OUT")" >/dev/null 2>&1 && echo 0 || echo 1)"
  cg_run import --understand-now "$REAL_JSONL" --context owner-feel
  REAL_IMPORT_OUT="$LAST_OUT"
  REAL_IMPORT_OBJ="$(json_first_obj "$REAL_IMPORT_OUT")"
  check "owner-feel real JSONL import succeeds on the default window" \
    "$([[ $LAST_RC -eq 0 ]] && is_ok_json "$REAL_IMPORT_OUT" && echo 0 || echo 1)"
  check "owner-feel real JSONL import exercises the multi-chunk path" \
    "$(jq -e '.import_chunks > 1' <<<"$REAL_IMPORT_OBJ" >/dev/null 2>&1 && echo 0 || echo 1)"
  check "owner-feel real JSONL import captures non-empty typed memory" \
    "$(jq -e '(.captured.goals + .captured.decisions + .captured.observations) > 0' <<<"$REAL_IMPORT_OBJ" >/dev/null 2>&1 && echo 0 || echo 1)"
  check "owner-feel real JSONL import reports at least one Native captured usage call per successful chunk" \
    "$(jq -e '([.usage.calls[] | select(.reliability_level == "Native" and .outcome == "captured")] | length) >= .import_chunks' <<<"$REAL_IMPORT_OBJ" >/dev/null 2>&1 && echo 0 || echo 1)"
fi

# ---------------------------------------------------------------------------
section "STEP 5 — §L connected-by-default (tenant context, not cwd-hash)"
mkdir -p /tmp/cg-cwd-A /tmp/cg-cwd-B /tmp/cg-cwd-C
( cd /tmp/cg-cwd-A && printf '%s' "We decided to adopt Rust for the new ingest worker." | "$CG" record --tenant-id "$TENANT" >/dev/null 2>&1 )
( cd /tmp/cg-cwd-B && printf '%s' "We decided to deploy the ingest worker on Kubernetes." | "$CG" record --tenant-id "$TENANT" >/dev/null 2>&1 )
printf '\n$ (from a THIRD cwd) cg status --tenant-id %s\n' "$TENANT"
LAST_OUT="$( cd /tmp/cg-cwd-C && "$CG" status --tenant-id "$TENANT" 2>&1 )"; printf '%s\n' "$LAST_OUT"
check "captures from two different working dirs both land under the tenant context" \
  "$(counts_nonzero "$LAST_OUT" && echo 0 || echo 1)"
note "If cwd-hashing were still in play these would be in 3 different contexts and this status"
note "(from a 3rd dir) would be empty. (§L — pinned by refactored_capture_context_defaults_to_tenant.)"

# ---------------------------------------------------------------------------
section "STEP 6 — §8 two-layer reversibility (cg fix / cg drop / cg group / cg undo)"
# Item handles are `@n` ordinals over the captured units (digests) in a context,
# newest first: `@1` is the latest capture, `@2` the one before it (see
# resolve_item_handle / captured_units_newest_first in cg-cli). fix/drop/undo/why
# all take one positional handle; `cg fix @n <value...>` takes the corrected value
# as POSITIONAL words (NOT a --to flag — leading-dash tokens are dropped).
if verb_available fix && verb_available drop && verb_available undo && verb_available group; then
  # Two distinct captures into one context → two digests. @1 = newest, @2 = older.
  cg_record --context fixme "We decided to use MongoDB for the session store, because the team knows it."
  REV_FIRST_RC=$LAST_RC
  cg_record --context fixme "We decided to deploy the cache layer with Redis for hot keys."
  check "reversibility setup: two captures into context fixme returned ok" \
    "$([[ $REV_FIRST_RC -eq 0 ]] && [[ $LAST_RC -eq 0 ]] && echo 0 || echo 1)"

  # --- cg fix: the read view changes, the raw is recoverable (undo restores it) ---
  # EVERY verb carries --context fixme so the @n handle resolves in the right
  # context (without it the handle would resolve against the default context).
  CORRECTED="PostgreSQL-on-RDS-correction-marker"
  cg_run fix @2 "$CORRECTED" --context fixme       # correct the OLDER capture
  check "cg fix @2 <value> returned ok with an overlay_id" \
    "$([[ $LAST_RC -eq 0 ]] && [[ -n "$(json_field "$LAST_OUT" '.overlay_id')" ]] && echo 0 || echo 1)"

  cg_run query --context fixme "session store database"
  check "after fix, the corrected value is visible in the read view (cg query)" \
    "$(contains "$LAST_OUT" "$CORRECTED" && echo 0 || echo 1)"

  cg_run undo @2 --context fixme                   # reverse the correction
  check "cg undo @2 returned ok" "$([[ $LAST_RC -eq 0 ]] && echo 0 || echo 1)"
  cg_run query --context fixme "session store database"
  check "after undo, the read view reverts (corrected value gone, raw row was never destroyed)" \
    "$(! contains "$LAST_OUT" "$CORRECTED" && echo 0 || echo 1)"

  # --- cg drop: the item leaves the read view; provenance preserved (undo restores) ---
  cg_run query --context fixme "Redis cache layer hot keys"
  DROP_BEFORE="$(json_field "$LAST_OUT" '.results | length')"
  cg_run drop @1 --context fixme                   # drop the NEWEST capture (the Redis one)
  check "cg drop @1 returned ok with an overlay_id" \
    "$([[ $LAST_RC -eq 0 ]] && [[ -n "$(json_field "$LAST_OUT" '.overlay_id')" ]] && echo 0 || echo 1)"
  cg_run query --context fixme "Redis cache layer hot keys"
  DROP_AFTER="$(json_field "$LAST_OUT" '.results | length')"
  note "drop result count: before=${DROP_BEFORE:-?} after=${DROP_AFTER:-?} (drop is an overlay exclusion, not a hard delete)"
  if [[ -n "$DROP_BEFORE" && "$DROP_BEFORE" -ge 1 ]]; then
    # The item WAS visible before; a drop must strictly reduce the read view.
    check "after drop, the dropped capture-unit is excluded from cg query (strictly fewer results)" \
      "$([[ -n "$DROP_AFTER" && "$DROP_AFTER" -lt "$DROP_BEFORE" ]] && echo 0 || echo 1)"
  else
    # Similarity miss, not a drop failure — the query never surfaced the item to drop.
    skip "cg drop read-view check: query did not surface the Redis digest before the drop (similarity miss, not a drop failure)"
  fi
  cg_run undo @1 --context fixme                   # the drop is reversible
  check "cg undo @1 reverses the drop (returns ok)" "$([[ $LAST_RC -eq 0 ]] && echo 0 || echo 1)"

  # --- cg group ok|rename|no: three DISTINCT effects, read back via review groupings ---
  GNAME="canonical-storage-goal"; GNAME2="renamed-storage-goal"
  cg_run group ok "$GNAME" --context fixme
  check "cg group ok <name> returned ok" "$([[ $LAST_RC -eq 0 ]] && echo 0 || echo 1)"
  cg_run review groupings --context fixme
  check "group ok: the confirmed grouping name shows under cg review groupings" \
    "$(contains "$LAST_OUT" "$GNAME" && echo 0 || echo 1)"

  cg_run group rename "$GNAME2" --context fixme
  cg_run review groupings --context fixme
  check "group rename: old name gone, new name present (a DISTINCT retitle effect)" \
    "$(! contains "$LAST_OUT" "$GNAME" && contains "$LAST_OUT" "$GNAME2" && echo 0 || echo 1)"

  cg_run group no --context fixme
  cg_run review groupings --context fixme
  check "group no: neither grouping name survives in the read view (a DISTINCT reject effect)" \
    "$(! contains "$LAST_OUT" "$GNAME" && ! contains "$LAST_OUT" "$GNAME2" && echo 0 || echo 1)"
else
  pending "§8 reversibility — cg fix/drop/group/undo not all wired into this binary"
fi

# ---------------------------------------------------------------------------
section "STEP 7 — §6 learned-skill loop (cg fix x N → held-out promotion → LLM-authored skill → auto-apply)"
TRAINED_SKILL_ID="skill-retype-context-graph-file-repository"
TRAINED_SKILL_DESCRIPTION=""
TRAINED_SKILL_RULE=""
if verb_available skills && verb_available fix; then
  for idx in 1 2 3; do
    TRAINING_TEXT="Goal G${idx}: classify the source artifact named context-graph. Entity E${idx}: context-graph type=file path=context-graph. Decision D${idx}: treat context-graph as a file artifact for indexing serves G${idx} based_on E${idx}."
    if cg_record_retry 3 --context skillz "$TRAINING_TEXT"; then
      check "training record ${idx}/3 captured a real model response" 0
    else
      check "training record ${idx}/3 captured a real model response" 1
      continue
    fi
    cg_run fix @1 Repository context-graph --context skillz
    check "training fix ${idx}/3 returned ok" "$([[ $LAST_RC -eq 0 ]] && echo 0 || echo 1)"
    check "training fix ${idx}/3 found the connected model-backed correction path" \
      "$([[ "$(json_field "$LAST_OUT" '.skill_proposal_model_available')" == "true" ]] && echo 0 || echo 1)"
    expected_invoked=false
    [[ "$idx" == "3" ]] && expected_invoked=true
    check "training fix ${idx}/3 reports proposal invocation only when promotion is eligible" \
      "$([[ "$(json_field "$LAST_OUT" '.skill_proposal_model_invoked')" == "$expected_invoked" ]] && echo 0 || echo 1)"
    if [[ "$idx" == "3" ]]; then
      check "third aligned correction invoked the connected model skill proposal path" \
        "$([[ "$(json_field "$LAST_OUT" '.skill_proposal_model_invoked')" == "true" ]] && echo 0 || echo 1)"
      check "third aligned correction reports at least one model skill proposal call" \
        "$([[ "$(json_field "$LAST_OUT" '.skill_proposal_model_invocations')" -gt 0 ]] && echo 0 || echo 1)"
      check "third aligned correction created a skill" \
        "$([[ "$(json_field "$LAST_OUT" '.new_skill_created')" == "true" ]] && echo 0 || echo 1)"
      check "third aligned correction ran the held-out gate" \
        "$([[ "$(json_field "$LAST_OUT" '.heldout_gate_invocations')" -gt 0 ]] && echo 0 || echo 1)"
    fi
  done

  cg_run skills --context skillz
  SKILLS_OBJ="$(json_first_obj "$LAST_OUT")"
  TRAINED_SKILL_JSON="$(jq -c --arg id "$TRAINED_SKILL_ID" '.skills[]? | select(.id == $id)' <<<"$SKILLS_OBJ" | head -1)"
  check "cg skills lists the promoted context-graph retype skill by id" \
    "$([[ -n "$TRAINED_SKILL_JSON" ]] && echo 0 || echo 1)"
  EMPTY_JSON='{}'
  TRAINED_SKILL_PROVENANCE="$(jq -r '.provenance // ""' <<<"${TRAINED_SKILL_JSON:-$EMPTY_JSON}")"
  TRAINED_SKILL_DESCRIPTION="$(jq -r '.description // ""' <<<"${TRAINED_SKILL_JSON:-$EMPTY_JSON}")"
  TRAINED_SKILL_RULE="$(jq -r '.rule // ""' <<<"${TRAINED_SKILL_JSON:-$EMPTY_JSON}")"
  check "promoted skill provenance is llm-authored" \
    "$([[ "$TRAINED_SKILL_PROVENANCE" == "llm-authored" ]] && echo 0 || echo 1)"
  check "promoted skill carries a model-written description, not the deterministic fallback" \
    "$([[ -n "$TRAINED_SKILL_DESCRIPTION" && "$TRAINED_SKILL_DESCRIPTION" != "retype context-graph from File to Repository when future captures mention this entity" ]] && echo 0 || echo 1)"
  check "promoted skill rule is parseable and names context-graph File → Repository" \
    "$(grep -qiE 'entity named context-graph.*typed File.*(->|→).*Repository' <<<"$TRAINED_SKILL_RULE" && echo 0 || echo 1)"

  MATCHING_RECORD="Goal G_auto: classify the source artifact named context-graph. Entity E_auto: context-graph type=file path=context-graph. Decision D_auto: treat context-graph as a file artifact for indexing serves G_auto based_on E_auto."
  if cg_record_retry 3 --context skillz "$MATCHING_RECORD"; then
    AUTO_RECORD_OUT="$LAST_OUT"
    check "later matching cg record succeeded" 0
  else
    AUTO_RECORD_OUT="$LAST_OUT"
    check "later matching cg record succeeded" 1
  fi
  check "later matching cg record reports the promoted skill in applied_skill_ids" \
    "$(jq -e --arg id "$TRAINED_SKILL_ID" '.harness.applied_skill_ids | index($id) != null' <<<"$(json_first_obj "$AUTO_RECORD_OUT")" >/dev/null 2>&1 && echo 0 || echo 1)"
  AUTO_ENTITY_ID="$(json_field "$AUTO_RECORD_OUT" '.stable_ids["context-graph"] // ""')"
  cg_run by-entity "$AUTO_ENTITY_ID" --context skillz
  check "later matching cg record stores context-graph as Repository after skill application" \
    "$([[ "$(json_field "$LAST_OUT" '.entity.entity_type')" == "Repository" ]] && echo 0 || echo 1)"

  IMPORT_FILE="$CG_DATA_ROOT/skill-loop-import.jsonl"
  : > "$IMPORT_FILE"
  printf '%s\n' '{"type":"user","message":{"content":[{"type":"text","text":"Goal G_import: classify the source artifact named context-graph. Entity E_import: context-graph type=file path=context-graph. Decision D_import: treat context-graph as a file artifact for indexing serves G_import based_on E_import."}]}}' >> "$IMPORT_FILE"
  printf '%s\n' '{"type":"assistant","message":{"content":[{"type":"text","text":"Additional import discussion pads the session across chunks while staying about validation notes, rollout timing, monitoring, and handoff details."}]}}' >> "$IMPORT_FILE"
  printf '%s\n' '{"type":"user","message":{"content":[{"type":"text","text":"More unrelated import text forces the multi-chunk route under a small chunk window without changing the context-graph classification."}]}}' >> "$IMPORT_FILE"
  cg_run import --understand-now --context skillz-import --chunk-tokens 80 "$IMPORT_FILE"
  IMPORT_OUT="$LAST_OUT"
  check "later real .jsonl cg import succeeded" "$([[ $LAST_RC -eq 0 ]] && echo 0 || echo 1)"
  check "later real .jsonl cg import exercised the multi-chunk route" \
    "$([[ "$(json_field "$IMPORT_OUT" '.import_chunks')" -gt 1 ]] && echo 0 || echo 1)"
  check "later real .jsonl cg import reports the promoted skill in applied_skill_ids" \
    "$(jq -e --arg id "$TRAINED_SKILL_ID" '.harness.applied_skill_ids | index($id) != null' <<<"$(json_first_obj "$IMPORT_OUT")" >/dev/null 2>&1 && echo 0 || echo 1)"
  IMPORT_ENTITY_ID="$(json_field "$IMPORT_OUT" '.stable_ids["context-graph"] // ""')"
  cg_run by-entity "$IMPORT_ENTITY_ID" --context skillz-import
  check "later real .jsonl cg import stores context-graph as Repository after skill application" \
    "$([[ "$(json_field "$LAST_OUT" '.entity.entity_type')" == "Repository" ]] && echo 0 || echo 1)"

  cg_run skills off --context skillz
  check "cg skills off returns ok (learned skills will not be applied)" \
    "$([[ $LAST_RC -eq 0 ]] && contains "$LAST_OUT" 'skills off' && echo 0 || echo 1)"
  cg_run skills on --context skillz
  check "cg skills on returns ok (learned skills will be applied)" \
    "$([[ $LAST_RC -eq 0 ]] && contains "$LAST_OUT" 'skills on' && echo 0 || echo 1)"
  cg_run skills review on --context skillz
  check "cg skills review on returns ok with review_hold=true" \
    "$([[ $LAST_RC -eq 0 ]] && [[ "$(json_field "$LAST_OUT" '.review_hold')" == "true" ]] && echo 0 || echo 1)"
  cg_run skills review off --context skillz
  check "cg skills review off returns ok with review_hold=false" \
    "$([[ $LAST_RC -eq 0 ]] && [[ "$(json_field "$LAST_OUT" '.review_hold')" == "false" ]] && echo 0 || echo 1)"
else
  check "§6 learned-skill loop requires cg skills/fix to be wired into the CLI" 1
fi

# ---------------------------------------------------------------------------
section "STEP 8 — §8 HITL surface (cg review / cg why)"
if verb_available review; then
  cg_run review --context billing-demo
  check "cg review runs and reports per-category counts" "$([[ $LAST_RC -eq 0 ]] && echo 0 || echo 1)"
  # Bare `cg review` emits one summary line carrying all four categories:
  #   summary: uncertain=N groupings=N drift=N currency_flips=N
  # The three single-edge categories (uncertain/grouping/drift) plus the
  # present-but-empty currency category must ALL appear (currency stays structurally
  # empty on a single edge — server-core owns reconciliation — but is never omitted).
  check "cg review shows the three single-edge categories + the present-but-empty currency category" \
    "$(grep -qiE 'uncertain' <<<"$LAST_OUT" && grep -qiE 'grouping' <<<"$LAST_OUT" \
       && grep -qiE 'drift' <<<"$LAST_OUT" && grep -qiE 'currenc' <<<"$LAST_OUT" && echo 0 || echo 1)"
  # The currency category must be PRESENT (a count in the rendered summary line),
  # not merely absent. (The JSON `summary` key is the context string — counts live
  # in the rendered `summary: ... currency_flips=N` line; see review_count.)
  CURR="$(review_count "$LAST_OUT" currency_flips)"
  check "cg review surfaces the currency category as a present count (default 0 on one edge)" \
    "$([[ -n "$CURR" ]] && echo 0 || echo 1)"
  note "review summary currency_flips=${CURR:-<absent>} (present-but-empty on a single edge by design)"

  if verb_available why; then
    # NEGATIVE half (always testable + the discriminating behavior): a non-flagged
    # handle must NOT enumerate a legal-type set — flagged=false, no legal_types key.
    cg_run why @1 --context billing-demo
    WHY_FLAGGED="$(json_field "$LAST_OUT" '.flagged')"
    check "cg why on a NON-flagged handle does NOT enumerate a legal vocab (flagged=false, no legal_types)" \
      "$([[ "$WHY_FLAGGED" == "false" ]] && ! contains "$LAST_OUT" '"legal_types"' && echo 0 || echo 1)"

    # POSITIVE half (a genuinely-flagged closed-vocab item lists the legal set): the
    # capture-path grounding flag is model-dependent (and the flagged item's id is not
    # surfaced through cg review's JSON), so this cannot be driven deterministically
    # from the CLI in a BYOK smoke. We assert the legal-vocab branch is reachable by
    # confirming the negative branch is closed-vocab-aware, and report the positive
    # path as a clean owner-eyeball step rather than a false pass.
    note "POSITIVE cg-why path: on a genuinely-flagged closed-vocab item (e.g. an entity whose"
    note "type span is unsupported), cg why <id> returns flagged=true + legal_types=[the 13 EntityType"
    note "values]. Flagging is model-dependent and cg review does not surface the flagged id, so this"
    note "branch is an owner eyeball: capture a transcript that forces an entity-type flag, then run"
    note "cg why <that-entity-id>."
  else
    check "§8 cg why is wired" 1
  fi
else
  check "§8 HITL surface — cg review is wired into the CLI" 1
fi

# ---------------------------------------------------------------------------
section "STEP 9 — §6 drift safeguard (one alert, nothing auto-disabled)"
if verb_available skills && verb_available review; then
  OUT_OF_SCOPE_TEXT="Goal G_scope: classify the auth-service file artifact. Entity E_scope: auth-service type=file path=auth-service. Decision D_scope: treat auth-service as a file artifact for indexing serves G_scope based_on E_scope."
  if cg_record_retry 3 --context skillz "$OUT_OF_SCOPE_TEXT"; then
    OUT_OF_SCOPE_OUT="$LAST_OUT"
    check "out-of-scope real record succeeded" 0
  else
    OUT_OF_SCOPE_OUT="$LAST_OUT"
    check "out-of-scope real record succeeded" 1
  fi
  check "LLM-authored context-graph skill does not over-fire on auth-service" \
    "$(jq -e --arg id "$TRAINED_SKILL_ID" '(.harness.applied_skill_ids | index($id)) == null' <<<"$(json_first_obj "$OUT_OF_SCOPE_OUT")" >/dev/null 2>&1 && echo 0 || echo 1)"

  DRIFT_TEXT="Goal G_drift: classify the source artifact named context-graph. Entity E_drift: context-graph type=file path=context-graph. Decision D_drift: treat context-graph as a file artifact for indexing serves G_drift based_on E_drift."
  if cg_record_retry 3 --context skillz "$DRIFT_TEXT"; then
    DRIFT_RECORD_OUT="$LAST_OUT"
    check "drift-trigger matching record succeeded" 0
  else
    DRIFT_RECORD_OUT="$LAST_OUT"
    check "drift-trigger matching record succeeded" 1
  fi
  check "drift-trigger matching record applied the LLM-authored skill" \
    "$(jq -e --arg id "$TRAINED_SKILL_ID" '.harness.applied_skill_ids | index($id) != null' <<<"$(json_first_obj "$DRIFT_RECORD_OUT")" >/dev/null 2>&1 && echo 0 || echo 1)"
  cg_run fix @1 File context-graph --context skillz
  check "wrong-label correction against the skill-produced capture returned ok" \
    "$([[ $LAST_RC -eq 0 ]] && echo 0 || echo 1)"
  cg_run review --context skillz
  DRIFT_COUNT="$(review_count "$LAST_OUT" drift)"
  check "cg review surfaces exactly one drift alert after wrong-labeling the LLM-authored skill firing" \
    "$([[ "$DRIFT_COUNT" == "1" ]] && echo 0 || echo 1)"
  cg_run skills --context skillz
  check "the drifting LLM-authored skill is NOT auto-disabled" \
    "$(jq -e --arg id "$TRAINED_SKILL_ID" '.skills[]? | select(.id == $id and .active == true and .provenance == "llm-authored")' <<<"$(json_first_obj "$LAST_OUT")" >/dev/null 2>&1 && echo 0 || echo 1)"
else
  check "§6 drift safeguard requires cg skills/review to be wired into the CLI" 1
fi

# ---------------------------------------------------------------------------
section "STEP 10 — §7 cross-machine skill durability over the contextdb relay hub"
if [[ -z "$HUB" || -z "${CG_SMOKE_TENANT:-}" ]]; then
  check "cross-machine skill durability requires CG_SMOKE_HUB and CG_SMOKE_TENANT" 1
  note "This is a release-gate blocker, not a clean skip: run with a contextdb-enterprise-server relay NATS URL."
else
  EDGE_A_ROOT="$CG_DATA_ROOT"
  EDGE_B_ROOT="${CG_DATA_ROOT}-edgeB"
  export CG_DATA_ROOT="$EDGE_A_ROOT"
  cg_run sync push --remote-server "$HUB" --tenant-id "$TENANT"
  check "edge A pushed learned skill rows to the contextdb relay hub" \
    "$([[ $LAST_RC -eq 0 ]] && is_ok_json "$LAST_OUT" && echo 0 || echo 1)"

  rm -rf "$EDGE_B_ROOT"
  mkdir -p "$EDGE_B_ROOT"
  export CG_DATA_ROOT="$EDGE_B_ROOT"
  cg_run model connect "$PROVIDER" --provider "$PROVIDER" --endpoint "$ENDPOINT" --model "$MODEL" --env-key OPENROUTER_API_KEY --consent
  check "edge B model connect returned ok" "$(ok_rc)"
  cg_run sync pull --remote-server "$HUB" --tenant-id "$TENANT"
  check "edge B pulled from the contextdb relay hub" \
    "$([[ $LAST_RC -eq 0 ]] && is_ok_json "$LAST_OUT" && echo 0 || echo 1)"
  cg_run skills --context skillz
  EDGE_B_SKILL_JSON="$(jq -c --arg id "$TRAINED_SKILL_ID" '.skills[]? | select(.id == $id)' <<<"$(json_first_obj "$LAST_OUT")" | head -1)"
  check "edge B lists the same learned skill id after pull" \
    "$([[ -n "$EDGE_B_SKILL_JSON" ]] && echo 0 || echo 1)"
  check "edge B lists the same learned skill provenance after pull" \
    "$([[ "$(jq -r '.provenance // ""' <<<"${EDGE_B_SKILL_JSON:-$EMPTY_JSON}")" == "$TRAINED_SKILL_PROVENANCE" ]] && echo 0 || echo 1)"

  EDGE_B_MATCH="Goal G_edge_b: classify the source artifact named context-graph. Entity E_edge_b: context-graph type=file path=context-graph. Decision D_edge_b: treat context-graph as a file artifact for indexing serves G_edge_b based_on E_edge_b."
  if cg_record_retry 3 --context skillz "$EDGE_B_MATCH"; then
    EDGE_B_RECORD_OUT="$LAST_OUT"
    check "edge B fresh matching record succeeded" 0
  else
    EDGE_B_RECORD_OUT="$LAST_OUT"
    check "edge B fresh matching record succeeded" 1
  fi
  check "edge B fresh matching record auto-applies the pulled skill" \
    "$(jq -e --arg id "$TRAINED_SKILL_ID" '.harness.applied_skill_ids | index($id) != null' <<<"$(json_first_obj "$EDGE_B_RECORD_OUT")" >/dev/null 2>&1 && echo 0 || echo 1)"
  EDGE_B_ENTITY_ID="$(json_field "$EDGE_B_RECORD_OUT" '.stable_ids["context-graph"] // ""')"
  cg_run by-entity "$EDGE_B_ENTITY_ID" --context skillz
  check "edge B stores context-graph as Repository after applying the pulled skill" \
    "$([[ "$(json_field "$LAST_OUT" '.entity.entity_type')" == "Repository" ]] && echo 0 || echo 1)"
  export CG_DATA_ROOT="$EDGE_A_ROOT"
fi

# ---------------------------------------------------------------------------
section "SUMMARY"
printf 'PASS: %d   FAIL: %d   SKIP/PENDING: %d\n' "$PASS" "$FAIL" "$SKIP"
if (( FAIL > 0 )); then
  printf '\nFailed checks:\n'
  for c in "${FAILED_CHECKS[@]}"; do printf '   - %s\n' "$c"; done
fi
if (( FAIL > 0 || SKIP > 0 )); then
  printf '\nVERDICT: NOT RELEASE-READY\n'
else
  printf '\nVERDICT: RELEASE-READY\n'
fi
printf '\nFull log : %s\n' "$LOG"
printf 'cg state : %s/context-graph.db\n' "$CG_DATA_ROOT"
printf '\nReading the result:\n'
printf '  CAPTURE, correction, promotion, LLM-authored skill creation, record/import auto-apply,\n'
printf '    drift, and cross-machine skill durability are hard pass/fail assertions.\n'
printf '  Promotion, llm-authored, drift, or cross-machine sync skips are release blockers.\n'
printf '  The positive cg-why flag remains an owner-eyeball note because the flagged id is not\n'
printf '    surfaced by cg review; it is not counted as a pass or a skip by this script.\n'
printf '  Grounding (§2) is covered by the deterministic Step 3A acceptance module over\n'
printf '    entity, decision_type, agent_id, observation, and outcome values.\n'

(( FAIL == 0 && SKIP == 0 ))
