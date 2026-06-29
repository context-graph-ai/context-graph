#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/cg-deterministic-llm-live-smoke.sh

Required by default:
  OPENROUTER_API_KEY must be set. cg reads it through --env and this script
  never prints the key value.

What this smoke verifies:
  1. cg connects to OpenRouter with a live Haiku model.
  2. cg record captures a marker-bearing session through deterministic LLM
     extraction.
  3. The capture receipt reports Native/captured for the OpenRouter call.
  4. cg query can read the captured marker back.
  5. claude -p can produce schema-shaped JSON with a Haiku model. This is a
     direct CLI check, not cg's installed-Claude path; cg currently records
     installed CLI providers as Floor unless that runtime is wired separately.

Optional:
  CG_BIN=target/release/cg
  CG_DETERMINISTIC_SMOKE_ROOT=/tmp/cg-deterministic-llm-live-smoke
  CG_DETERMINISTIC_SMOKE_KEEP=1

  CG_DETERMINISTIC_SMOKE_OPENROUTER_API_KEY_ENV=OPENROUTER_API_KEY
  CG_DETERMINISTIC_SMOKE_OPENROUTER_ENDPOINT=https://openrouter.ai/api/v1
  CG_DETERMINISTIC_SMOKE_OPENROUTER_MODEL=anthropic/claude-haiku-4.5
  CG_DETERMINISTIC_SMOKE_OPENROUTER_CONNECTION=live-openrouter-haiku
  CG_DETERMINISTIC_SMOKE_OPENROUTER_TIMEOUT_MS=45000
  CG_DETERMINISTIC_SMOKE_SKIP_OPENROUTER=1

  CG_DETERMINISTIC_SMOKE_CLAUDE_MODEL=haiku
  CG_DETERMINISTIC_SMOKE_CLAUDE_MAX_BUDGET_USD=0.25
  CG_DETERMINISTIC_SMOKE_ALLOW_NON_HAIKU=1
  CG_DETERMINISTIC_SMOKE_SKIP_CLAUDE_DIRECT=1

  CG_DETERMINISTIC_SMOKE_RUN_CLAUDE_CG=1
  CG_DETERMINISTIC_SMOKE_SKIP_CLAUDE_CG=1

  CG_DETERMINISTIC_SMOKE_USE_PROD_EMBEDDER=1

By default the script uses cg's lexical test embedder so the smoke stays
focused on live LLM extraction instead of model download/cache state. Set
CG_DETERMINISTIC_SMOKE_USE_PROD_EMBEDDER=1 to exercise the production text
embedder too.
USAGE
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

note() {
  printf '\n==> %s\n' "$*"
}

resolve_cg() {
  if [[ -n "${CG_BIN:-}" ]]; then
    [[ -x "$CG_BIN" ]] || die "CG_BIN is not executable: $CG_BIN"
    printf '%s\n' "$CG_BIN"
    return
  fi

  if [[ -x target/release/cg ]]; then
    printf '%s\n' "target/release/cg"
    return
  fi

  if [[ -x target/debug/cg ]]; then
    printf '%s\n' "target/debug/cg"
    return
  fi

  die "could not find cg binary; run cargo build --release or set CG_BIN"
}

json_field() {
  local file="$1"
  local pointer="$2"
  python3 - "$file" "$pointer" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)

value = data
for part in sys.argv[2].strip("/").split("/"):
    if part:
        value = value[part]

if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("null")
else:
    print(value)
PY
}

assert_json_field() {
  local file="$1"
  local pointer="$2"
  local expected="$3"
  local actual
  actual="$(json_field "$file" "$pointer")"
  if [[ "$actual" != "$expected" ]]; then
    die "$file expected $pointer=$expected, got $actual"
  fi
}

assert_model_selftest_passed() {
  local file="$1"
  local connection="$2"
  local provider="$3"

  assert_json_field "$file" "/status" "pass"
  assert_json_field "$file" "/connection" "$connection"
  assert_json_field "$file" "/provider" "$provider"
}

assert_record_receipt() {
  local file="$1"
  local expected_level="$2"
  local expected_outcome="$3"
  local marker="$4"

  python3 - "$file" "$expected_level" "$expected_outcome" "$marker" <<'PY'
import json
import sys

path, expected_level, expected_outcome, marker = sys.argv[1:5]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

captured = data.get("captured") or {}
goals = int(captured.get("goals") or 0)
decisions = int(captured.get("decisions") or 0)
observations = int(captured.get("observations") or 0)
if goals < 1 or decisions < 1 or observations < 1:
    raise SystemExit(
        f"{path} expected at least one goal, decision, and observation; "
        f"got goals={goals}, decisions={decisions}, observations={observations}"
    )

calls = ((data.get("usage") or {}).get("calls")) or []
if not calls:
    raise SystemExit(f"{path} missing usage.calls receipt")

call = calls[0]
level = call.get("reliability_level")
outcome = call.get("outcome")
if level != expected_level:
    raise SystemExit(f"{path} expected reliability_level={expected_level}, got {level}")
if outcome != expected_outcome:
    raise SystemExit(f"{path} expected outcome={expected_outcome}, got {outcome}")

receipt_text = json.dumps(data, sort_keys=True).lower()
for forbidden in [
    "response_format",
    "guided_json",
    "grammar",
    "json-schema",
    "output-schema",
]:
    if forbidden in receipt_text:
        raise SystemExit(f"{path} leaked provider-specific receipt vocabulary: {forbidden}")

if marker not in json.dumps(data, sort_keys=True):
    raise SystemExit(f"{path} did not preserve smoke marker {marker!r}")

total_tokens = (data.get("usage") or {}).get("total_tokens")
print(
    "PASS record receipt: "
    f"goals={goals} decisions={decisions} observations={observations} "
    f"level={level} outcome={outcome} total_tokens={total_tokens}"
)
PY
}

assert_query_contains() {
  local file="$1"
  local needle="$2"

  if ! grep -Fq "$needle" "$file"; then
    printf 'Query output did not contain expected marker %s\n' "$needle" >&2
    printf '%s\n' '--- query output ---' >&2
    sed -n '1,160p' "$file" >&2
    die "captured marker was not readable through cg query"
  fi
}

assert_claude_direct() {
  local file="$1"
  local marker="$2"
  local allow_non_haiku="${CG_DETERMINISTIC_SMOKE_ALLOW_NON_HAIKU:-0}"

  python3 - "$file" "$marker" "$allow_non_haiku" <<'PY'
import json
import sys

path, marker, allow_non_haiku = sys.argv[1:4]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

if data.get("is_error") is True:
    raise SystemExit(f"{path} claude returned is_error=true: {data.get('result')}")

result = data.get("result")
if isinstance(result, str):
    try:
        payload = json.loads(result)
    except json.JSONDecodeError as err:
        raise SystemExit(f"{path} result is not JSON: {err}: {result[:240]}")
elif isinstance(result, dict):
    payload = result
else:
    raise SystemExit(f"{path} result has unsupported type: {type(result).__name__}")

if payload.get("marker") != marker:
    raise SystemExit(f"{path} expected marker {marker!r}, got {payload.get('marker')!r}")
if payload.get("status") != "ok":
    raise SystemExit(f"{path} expected status=ok, got {payload.get('status')!r}")
if payload.get("confidence") != "high":
    raise SystemExit(f"{path} expected confidence=high, got {payload.get('confidence')!r}")

usage = data.get("modelUsage") or data.get("model_usage") or {}
models = sorted(str(key) for key in usage.keys())
if not models:
    raise SystemExit(f"{path} missing modelUsage; cannot verify the actual Claude model")

haiku_used = any("haiku" in model.lower() for model in models)
if not haiku_used and allow_non_haiku != "1":
    raise SystemExit(
        f"{path} did not use a Haiku model. actual modelUsage keys={models}. "
        "Set CG_DETERMINISTIC_SMOKE_CLAUDE_MODEL to a valid local Haiku alias "
        "or CG_DETERMINISTIC_SMOKE_ALLOW_NON_HAIKU=1 to only test schema shape."
    )

print(f"PASS claude -p schema: marker={marker} models={','.join(models)}")
PY
}

write_openrouter_timeout_override() {
  local timeout_ms="${CG_DETERMINISTIC_SMOKE_OPENROUTER_TIMEOUT_MS:-45000}"
  local config_dir="$CG_DATA_ROOT/cg"
  local providers_file="$config_dir/providers.toml"

  [[ "$timeout_ms" =~ ^[0-9]+$ ]] || die "CG_DETERMINISTIC_SMOKE_OPENROUTER_TIMEOUT_MS must be an integer"
  mkdir -p "$config_dir"

  if [[ -e "$providers_file" && "${CG_DETERMINISTIC_SMOKE_ALLOW_PROVIDER_OVERRIDE_OVERWRITE:-0}" != "1" ]]; then
    if ! grep -q '^\[providers\.openrouter\]' "$providers_file" \
      || ! grep -q '^timeout_ms[[:space:]]*=' "$providers_file"; then
      die "$providers_file already exists; set CG_DETERMINISTIC_SMOKE_ALLOW_PROVIDER_OVERRIDE_OVERWRITE=1 to replace it for this isolated smoke root"
    fi
    return
  fi

  {
    printf '[providers.openrouter]\n'
    printf 'timeout_ms = %s\n' "$timeout_ms"
  } >"$providers_file"
  printf 'Using OpenRouter provider timeout override: %sms\n' "$timeout_ms"
}

run_openrouter_smoke() {
  if [[ "${CG_DETERMINISTIC_SMOKE_SKIP_OPENROUTER:-0}" == "1" ]]; then
    note "Skipping OpenRouter cg record smoke"
    return
  fi

  local key_env="${CG_DETERMINISTIC_SMOKE_OPENROUTER_API_KEY_ENV:-OPENROUTER_API_KEY}"
  if [[ -z "${!key_env:-}" ]]; then
    die "missing required OpenRouter API key env var named by CG_DETERMINISTIC_SMOKE_OPENROUTER_API_KEY_ENV: $key_env"
  fi

  local endpoint="${CG_DETERMINISTIC_SMOKE_OPENROUTER_ENDPOINT:-https://openrouter.ai/api/v1}"
  local model="${CG_DETERMINISTIC_SMOKE_OPENROUTER_MODEL:-anthropic/claude-haiku-4.5}"
  local connection="${CG_DETERMINISTIC_SMOKE_OPENROUTER_CONNECTION:-live-openrouter-haiku}"
  local marker="${CG_DETERMINISTIC_SMOKE_MARKER:-OR-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
  local connect_json="$tmp_dir/openrouter-connect.json"
  local selftest_json="$tmp_dir/openrouter-selftest.json"
  local record_json="$tmp_dir/openrouter-record.json"
  local query_txt="$tmp_dir/openrouter-query.txt"
  local transcript="$tmp_dir/openrouter-transcript.txt"

  write_openrouter_timeout_override

  note "Connecting cg to OpenRouter model $model"
  "$cg_bin" model connect "$connection" \
    --provider openrouter \
    --endpoint "$endpoint" \
    --model "$model" \
    --env "$key_env" \
    --consent \
    --format json >"$connect_json"
  assert_json_field "$connect_json" "/connection" "$connection"
  assert_json_field "$connect_json" "/provider" "openrouter"
  assert_json_field "$connect_json" "/model" "$model"

  note "Running cg model selftest for $connection"
  "$cg_bin" model selftest --connection "$connection" --format json >"$selftest_json"
  assert_model_selftest_passed "$selftest_json" "$connection" "openrouter"

  printf '%s\n' \
    "Goal G_${marker}: verify deterministic smoke marker ${marker} is captured and queryable." \
    "Entity E_${marker}: smoke receipt file path=smoke/${marker}.json." \
    "Decision D_${marker}: record achieved Native deterministic LLM capture for marker ${marker} serves G_${marker} based_on E_${marker}." \
    "Observation O_${marker}: OpenRouter Haiku structured output produced parseable memory for marker ${marker}." \
    >"$transcript"

  note "Recording marker-bearing session through cg"
  "$cg_bin" record "$transcript" >"$record_json"
  assert_record_receipt "$record_json" "Native" "captured" "$marker"

  note "Reading captured marker back through cg query"
  "$cg_bin" query "$marker" >"$query_txt"
  assert_query_contains "$query_txt" "$marker"
  printf 'PASS query readback: marker=%s\n' "$marker"
}

run_claude_direct_smoke() {
  if [[ "${CG_DETERMINISTIC_SMOKE_SKIP_CLAUDE_DIRECT:-0}" == "1" ]]; then
    note "Skipping direct claude -p schema smoke"
    return
  fi

  command -v claude >/dev/null 2>&1 || die "claude CLI not found on PATH"

  local model="${CG_DETERMINISTIC_SMOKE_CLAUDE_MODEL:-haiku}"
  local marker="${CG_DETERMINISTIC_SMOKE_CLAUDE_MARKER:-CL-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
  local schema_file="$tmp_dir/claude-schema.json"
  local claude_json="$tmp_dir/claude-direct.json"
  local claude_err="$tmp_dir/claude-direct.stderr"
  local schema

  printf '%s\n' \
    '{"type":"object","additionalProperties":false,"required":["marker","status","confidence"],"properties":{"marker":{"type":"string"},"status":{"type":"string","enum":["ok"]},"confidence":{"type":"string","enum":["high"]}}}' \
    >"$schema_file"
  schema="$(python3 - "$schema_file" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.dumps(json.load(handle), separators=(",", ":")))
PY
)"

  local prompt="Return JSON matching the supplied schema. Use marker ${marker}, status ok, and confidence high. Do not add prose."
  local claude_cmd=(claude -p --model "$model" --output-format json --json-schema "$schema")
  if [[ -n "${CG_DETERMINISTIC_SMOKE_CLAUDE_MAX_BUDGET_USD:-}" ]]; then
    claude_cmd+=(--max-budget-usd "$CG_DETERMINISTIC_SMOKE_CLAUDE_MAX_BUDGET_USD")
  fi

  note "Running direct claude -p schema smoke with requested model $model"
  set +e
  "${claude_cmd[@]}" "$prompt" >"$claude_json" 2>"$claude_err"
  local code=$?
  set -e
  if [[ "$code" != "0" ]]; then
    printf '%s\n' '--- claude stderr ---' >&2
    sed -n '1,120p' "$claude_err" >&2
    printf '%s\n' '--- claude stdout ---' >&2
    sed -n '1,120p' "$claude_json" >&2
    die "direct claude -p schema smoke failed with exit $code"
  fi

  assert_claude_direct "$claude_json" "$marker"
}

run_claude_cg_smoke() {
  if [[ "${CG_DETERMINISTIC_SMOKE_SKIP_CLAUDE_CG:-0}" == "1" ]]; then
    note "Skipping optional cg installed-Claude smoke"
    return
  fi
  if [[ "${CG_DETERMINISTIC_SMOKE_RUN_CLAUDE_CG:-0}" != "1" ]]; then
    note "Optional cg installed-Claude smoke is off; set CG_DETERMINISTIC_SMOKE_RUN_CLAUDE_CG=1 to run it"
    return
  fi

  command -v claude >/dev/null 2>&1 || die "claude CLI not found on PATH"

  local marker="${CG_DETERMINISTIC_SMOKE_CLAUDE_CG_MARKER:-CGCL-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
  local auth_json="$tmp_dir/claude-cg-auth.json"
  local selftest_json="$tmp_dir/claude-cg-selftest.json"
  local record_json="$tmp_dir/claude-cg-record.json"
  local query_txt="$tmp_dir/claude-cg-query.txt"
  local transcript="$tmp_dir/claude-cg-transcript.txt"

  note "Connecting cg to the installed Claude CLI host session"
  "$cg_bin" model auth login --connection claude --consent --format json >"$auth_json"
  assert_json_field "$auth_json" "/connection" "claude"
  assert_json_field "$auth_json" "/provider" "claude"
  assert_json_field "$auth_json" "/consent_recorded" "true"

  "$cg_bin" model selftest --connection claude --format json >"$selftest_json"
  assert_model_selftest_passed "$selftest_json" "claude" "claude"

  printf '%s\n' \
    "Goal G_${marker}: verify installed Claude CLI Floor capture marker ${marker} is captured and queryable." \
    "Entity E_${marker}: installed Claude smoke receipt path=smoke/${marker}.json." \
    "Decision D_${marker}: installed Claude CLI default path lands deterministic smoke memory for marker ${marker} serves G_${marker} based_on E_${marker}." \
    "Observation O_${marker}: installed Claude CLI observation lane lands for marker ${marker}." \
    >"$transcript"

  "$cg_bin" record "$transcript" >"$record_json"
  assert_record_receipt "$record_json" "Floor" "captured" "$marker"

  "$cg_bin" query "$marker" >"$query_txt"
  assert_query_contains "$query_txt" "$marker"
  printf 'PASS cg installed-Claude Floor readback: marker=%s\n' "$marker"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

cg_bin="$(resolve_cg)"

if [[ -z "${CG_DETERMINISTIC_SMOKE_ROOT:-}" ]]; then
  smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/cg-deterministic-llm-live-smoke.XXXXXX")"
  cleanup_root=1
else
  smoke_root="$CG_DETERMINISTIC_SMOKE_ROOT"
  mkdir -p "$smoke_root"
  cleanup_root=0
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/cg-deterministic-llm-live-smoke-output.XXXXXX")"
cleanup() {
  if [[ "${CG_DETERMINISTIC_SMOKE_KEEP:-0}" == "1" ]]; then
    printf 'Keeping smoke output: %s\n' "$tmp_dir" >&2
    printf 'Keeping CG_DATA_ROOT: %s\n' "$smoke_root" >&2
    return
  fi
  rm -rf "$tmp_dir"
  if [[ "$cleanup_root" == "1" ]]; then
    rm -rf "$smoke_root"
  fi
}
trap cleanup EXIT

export CG_DATA_ROOT="$smoke_root"
if [[ "${CG_DETERMINISTIC_SMOKE_USE_PROD_EMBEDDER:-0}" != "1" ]]; then
  export CG_USE_TEST_LEXICAL_EMBEDDER=1
fi

printf 'Using cg binary: %s\n' "$cg_bin"
printf 'Using isolated CG_DATA_ROOT: %s\n' "$CG_DATA_ROOT"
if [[ -n "${CG_USE_TEST_LEXICAL_EMBEDDER:-}" ]]; then
  printf 'Using lexical test embedder; set CG_DETERMINISTIC_SMOKE_USE_PROD_EMBEDDER=1 for production embedder coverage.\n'
fi

run_openrouter_smoke
run_claude_direct_smoke
run_claude_cg_smoke

printf '\nPASS deterministic LLM live smoke completed\n'
