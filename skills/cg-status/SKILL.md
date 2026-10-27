---
name: cg-status
description: Inspect captured history and search progress.
---

# Inspect memory status

## Prerequisite

Install `cg`.

Choose the data root used for import.

## Recipe

1. Run `cg status --context <name>`.

2. Inspect the history and search sections.

3. Run `cg status --all-contexts` to inspect multiple Contexts.

## Installed-binary example

Install `cg` and Python 3; use a POSIX shell.

Run `cg model fetch --defaults` once before the example.

Use the synthetic session below in a disposable data root.

Set `CG_MODEL_CACHE_ROOT` before changing `HOME`.

<!-- smoke: skill -->
```sh
export CG_MODEL_CACHE_ROOT="${CG_MODEL_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/cg/models}"
cg_example=$(mktemp -d)
export HOME="$cg_example/home"
mkdir -p "$HOME"
export CG_DATA_ROOT="$cg_example/data"
cat > "$cg_example/session.jsonl" <<'JSONL'
{"type":"user","uuid":"00000000-0000-4000-8000-000000000001","sessionId":"example-auth","timestamp":"2026-09-01T09:00:00Z","message":{"role":"user","content":"Where is the authentication module?"}}
{"type":"assistant","uuid":"00000000-0000-4000-8000-000000000002","sessionId":"example-auth","timestamp":"2026-09-01T09:01:00Z","message":{"role":"assistant","content":"The authentication module moved from src/auth.py to src/authentication.py."}}
JSONL
cg import "$cg_example/session.jsonl" --context example-auth --search defer > "$cg_example/import.json"
cg status --context example-auth --format json > "$cg_example/status.json"
python3 -c 'import json,sys; h=json.load(open(sys.argv[1]))["history"]; print("captured:", h["captured_events"]); print("search:", h["search"]["status"]); print("pending:", h["search"]["pending_events"])' "$cg_example/status.json"
```

### Example output

```text
captured: 2
search: pending
pending: 2
```

## Failure to handle

Missing-store status returns `store_not_found` with the resolved database path. <!-- enforced by: `cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either` -->

Missing-store status creates nothing. <!-- enforced by: `cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either` -->

Import the intended history before retrying status.

See [CLI reference](../../docs/cli.md) and [configuration](../../docs/configuration.md).
