---
name: cg-find
description: Find stored conversation passages.
---

# Find conversation passages

## Prerequisite

Install `cg`.

Import history with `cg import <project-directory> --context <name> --search wait`.

## Recipe

1. Run `cg find "<query>" --context <name>`.

2. Use `cg find "<query>" --context app,infra` for multiple Contexts.

3. Pass a result to `cg show <passage-id> --context <name>`.

## Installed-binary example

Install `cg` and Python 3; use a POSIX shell.

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
cg import "$cg_example/session.jsonl" --context example-auth --search wait > "$cg_example/import.json"
cg find "where did authentication move?" --context example-auth --format json > "$cg_example/find.json"
python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hits"][0]["text"])' "$cg_example/find.json"
```

### Example output

```text
User [author="user"]: Where is the authentication module?
Assistant [author="assistant"]: The authentication module moved from src/auth.py to src/authentication.py.
```

## Failure to handle

Find on pending search returns `history_search_not_ready`. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

Find leaves pending search pending. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

Retry import with `--search wait`, then repeat find.

See [CLI reference](../../docs/cli.md) and [configuration](../../docs/configuration.md).
