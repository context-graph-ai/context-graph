#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  CG_LLM_SMOKE_HOSTED_ENDPOINT=https://api.openai.com/v1 \
  CG_LLM_SMOKE_HOSTED_API_KEY_ENV=OPENAI_API_KEY \
  OPENAI_API_KEY=... \
  CG_LLM_SMOKE_LOCAL_ENDPOINT=http://127.0.0.1:8000/v1 \
  scripts/llm-byok-live-smoke.sh

Optional:
  CG_BIN=target/release/cg
  CG_DATA_ROOT=/tmp/cg-live-smoke
  CG_LLM_SMOKE_HOSTED_PROVIDER=openai
  CG_LLM_SMOKE_HOSTED_MODEL=gpt-4.1-mini
  CG_LLM_SMOKE_HOSTED_CONNECTION=live-smoke-hosted
  CG_LLM_SMOKE_LOCAL_PROVIDER=local-vllm
  CG_LLM_SMOKE_LOCAL_MODEL=llama-local
  CG_LLM_SMOKE_LOCAL_CONNECTION=live-smoke-local
  CG_LLM_SMOKE_BAD_ENDPOINT=http://127.0.0.1:9/v1
  CG_LLM_SMOKE_BAD_CONNECTION=live-smoke-unreachable

The hosted API key is read by cg through --env. This script checks that
the named environment variable is present, but it never prints the key value.
USAGE
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    die "missing required environment variable: $name"
  fi
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

run_json() {
  local out_file="$1"
  shift
  "$@" --format json >"$out_file"
}

run_json_expect_failure() {
  local out_file="$1"
  local expected_code="$2"
  shift 2
  set +e
  "$@" --format json >"$out_file"
  local code=$?
  set -e
  if [[ "$code" != "$expected_code" ]]; then
    die "$out_file expected exit $expected_code, got $code"
  fi
}

json_get() {
  local file="$1"
  local key="$2"
  python3 - "$file" "$key" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)
value = data
for part in sys.argv[2].split("."):
    value = value[part]
if value is None:
    raise SystemExit(1)
if isinstance(value, bool):
    print("true" if value else "false")
    raise SystemExit(0)
print(value)
PY
}

assert_json_field() {
  local file="$1"
  local key="$2"
  local expected="$3"
  local actual
  actual="$(json_get "$file" "$key")"
  if [[ "$actual" != "$expected" ]]; then
    die "$file expected $key=$expected, got $actual"
  fi
}

assert_selftest_pass() {
  local file="$1"
  local connection="$2"
  local provider="$3"
  assert_json_field "$file" status pass
  assert_json_field "$file" connection "$connection"
  assert_json_field "$file" provider "$provider"
  local reply
  reply="$(json_get "$file" reply)"
  if [[ -z "$reply" ]]; then
    die "$file selftest reply is empty"
  fi
  printf 'PASS %s selftest reply: %s\n' "$connection" "$reply"
}

assert_status_names_connection() {
  local file="$1"
  local connection="$2"
  local provider="$3"
  local model="$4"
  local endpoint="$5"
  assert_json_field "$file" state ready
  assert_json_field "$file" degraded false
  assert_json_field "$file" connection_works true
  assert_json_field "$file" proof_status pass
  assert_json_field "$file" connection "$connection"
  assert_json_field "$file" provider "$provider"
  assert_json_field "$file" model "$model"
  assert_json_field "$file" endpoint "$endpoint"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

require_env CG_LLM_SMOKE_HOSTED_ENDPOINT
require_env CG_LLM_SMOKE_HOSTED_API_KEY_ENV
require_env CG_LLM_SMOKE_LOCAL_ENDPOINT

hosted_key_env="$CG_LLM_SMOKE_HOSTED_API_KEY_ENV"
if [[ -z "${!hosted_key_env:-}" ]]; then
  die "hosted API key env var named by CG_LLM_SMOKE_HOSTED_API_KEY_ENV is not set: $hosted_key_env"
fi

cg_bin="$(resolve_cg)"
hosted_provider="${CG_LLM_SMOKE_HOSTED_PROVIDER:-openai}"
hosted_connection="${CG_LLM_SMOKE_HOSTED_CONNECTION:-live-smoke-hosted}"
local_provider="${CG_LLM_SMOKE_LOCAL_PROVIDER:-local-vllm}"
local_connection="${CG_LLM_SMOKE_LOCAL_CONNECTION:-live-smoke-local}"
bad_endpoint="${CG_LLM_SMOKE_BAD_ENDPOINT:-http://127.0.0.1:9/v1}"
bad_connection="${CG_LLM_SMOKE_BAD_CONNECTION:-live-smoke-unreachable}"

if [[ -z "${CG_DATA_ROOT:-}" ]]; then
  smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/cg-live-smoke.XXXXXX")"
  cleanup_root=1
else
  smoke_root="$CG_DATA_ROOT"
  mkdir -p "$smoke_root"
  cleanup_root=0
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/cg-live-smoke-output.XXXXXX")"
cleanup() {
  rm -rf "$tmp_dir"
  if [[ "$cleanup_root" == "1" ]]; then
    rm -rf "$smoke_root"
  fi
}
trap cleanup EXIT

export CG_DATA_ROOT="$smoke_root"

printf 'Using cg binary: %s\n' "$cg_bin"
printf 'Using isolated CG_DATA_ROOT: %s\n' "$CG_DATA_ROOT"

hosted_connect="$tmp_dir/hosted-connect.json"
hosted_selftest="$tmp_dir/hosted-selftest.json"
hosted_status="$tmp_dir/hosted-status.json"
local_connect="$tmp_dir/local-connect.json"
local_selftest="$tmp_dir/local-selftest.json"
local_status="$tmp_dir/local-status.json"
bad_connect="$tmp_dir/bad-connect.json"
bad_selftest="$tmp_dir/bad-selftest.json"
bad_status="$tmp_dir/bad-status.json"

printf 'Connecting hosted provider %s at %s\n' "$hosted_provider" "$CG_LLM_SMOKE_HOSTED_ENDPOINT"
hosted_connect_cmd=("$cg_bin" model connect "$hosted_connection" \
  --provider "$hosted_provider" \
  --endpoint "$CG_LLM_SMOKE_HOSTED_ENDPOINT" \
  --env "$hosted_key_env" \
  --consent)
if [[ -n "${CG_LLM_SMOKE_HOSTED_MODEL:-}" ]]; then
  hosted_connect_cmd+=(--model "$CG_LLM_SMOKE_HOSTED_MODEL")
fi
run_json "$hosted_connect" "${hosted_connect_cmd[@]}"
assert_json_field "$hosted_connect" connection "$hosted_connection"
assert_json_field "$hosted_connect" provider "$hosted_provider"
assert_json_field "$hosted_connect" endpoint "$CG_LLM_SMOKE_HOSTED_ENDPOINT"
hosted_model="$(json_get "$hosted_connect" model)"
if [[ -z "$hosted_model" || "$hosted_model" == "$hosted_provider" ]]; then
  die "$hosted_connect resolved invalid hosted model: $hosted_model"
fi

run_json "$hosted_selftest" "$cg_bin" model selftest --connection "$hosted_connection"
assert_selftest_pass "$hosted_selftest" "$hosted_connection" "$hosted_provider"

run_json "$hosted_status" "$cg_bin" model status --connection "$hosted_connection"
assert_status_names_connection "$hosted_status" "$hosted_connection" "$hosted_provider" "$hosted_model" "$CG_LLM_SMOKE_HOSTED_ENDPOINT"

printf 'Connecting local provider %s at %s\n' "$local_provider" "$CG_LLM_SMOKE_LOCAL_ENDPOINT"
local_connect_cmd=("$cg_bin" model connect "$local_connection" \
  --provider "$local_provider" \
  --endpoint "$CG_LLM_SMOKE_LOCAL_ENDPOINT" \
  --no-auth \
  --consent)
if [[ -n "${CG_LLM_SMOKE_LOCAL_MODEL:-}" ]]; then
  local_connect_cmd+=(--model "$CG_LLM_SMOKE_LOCAL_MODEL")
fi
run_json "$local_connect" "${local_connect_cmd[@]}"
assert_json_field "$local_connect" connection "$local_connection"
assert_json_field "$local_connect" provider "$local_provider"
assert_json_field "$local_connect" endpoint "$CG_LLM_SMOKE_LOCAL_ENDPOINT"
local_model="$(json_get "$local_connect" model)"
if [[ -z "$local_model" || "$local_model" == "$local_provider" ]]; then
  die "$local_connect resolved invalid local model: $local_model"
fi

run_json "$local_selftest" "$cg_bin" model selftest --connection "$local_connection"
assert_selftest_pass "$local_selftest" "$local_connection" "$local_provider"

run_json "$local_status" "$cg_bin" model status --connection "$local_connection"
assert_status_names_connection "$local_status" "$local_connection" "$local_provider" "$local_model" "$CG_LLM_SMOKE_LOCAL_ENDPOINT"

printf 'Checking typed unreachable failure at %s\n' "$bad_endpoint"
bad_connect_cmd=("$cg_bin" model connect "$bad_connection" \
  --provider "$local_provider" \
  --endpoint "$bad_endpoint" \
  --no-auth \
  --consent)
if [[ -n "${CG_LLM_SMOKE_LOCAL_MODEL:-}" ]]; then
  bad_connect_cmd+=(--model "$CG_LLM_SMOKE_LOCAL_MODEL")
fi
run_json "$bad_connect" "${bad_connect_cmd[@]}"
assert_json_field "$bad_connect" connection "$bad_connection"
assert_json_field "$bad_connect" provider "$local_provider"
assert_json_field "$bad_connect" endpoint "$bad_endpoint"
bad_model="$(json_get "$bad_connect" model)"
if [[ -z "$bad_model" || "$bad_model" == "$local_provider" ]]; then
  die "$bad_connect resolved invalid unreachable-leg model: $bad_model"
fi

run_json_expect_failure "$bad_selftest" 6 "$cg_bin" model selftest --connection "$bad_connection"
assert_json_field "$bad_selftest" condition live_call_failed
assert_json_field "$bad_selftest" bucket unreachable
assert_json_field "$bad_selftest" degraded true

run_json_expect_failure "$bad_status" 6 "$cg_bin" model status --connection "$bad_connection"
assert_json_field "$bad_status" state degraded
assert_json_field "$bad_status" degraded true
assert_json_field "$bad_status" condition live_call_failed
assert_json_field "$bad_status" bucket unreachable
assert_json_field "$bad_status" connection "$bad_connection"
assert_json_field "$bad_status" provider "$local_provider"
assert_json_field "$bad_status" model "$bad_model"
assert_json_field "$bad_status" endpoint "$bad_endpoint"

cat <<'PASS'
Live smoke script assertions passed.
Only report the live smoke as passed if the hosted and local replies above are real model completions from the intended live endpoints, and the unreachable leg reports the typed failure fields asserted above.
PASS
