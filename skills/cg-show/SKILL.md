---
name: cg-show
description: Read a passage’s stored events.
---

# Read a passage’s source

## Prerequisite

Install `cg`.

Choose a passage id from `cg find "<query>" --context <name>`.

## Recipe

1. Run `cg show <passage-id> --context <name>`.

2. Read the returned events.

3. Run `cg show <passage-id> --context <name> --format text`.

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
passage_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hits"][0]["passage_id"])' "$cg_example/find.json")
cg show "$passage_id" --context example-auth --format json > "$cg_example/show.json"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("\n".join(e["text"] for e in d["events"]))' "$cg_example/show.json"
```

### Example output

```text
Where is the authentication module?
The authentication module moved from src/auth.py to src/authentication.py.
```

## Failure to handle

A passage outside the selected Context returns `passage_not_found`. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

Check the Context selection before retrying.

See [CLI reference](../../docs/cli.md) and [configuration](../../docs/configuration.md).
