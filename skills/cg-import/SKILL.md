---
name: cg-import
description: Capture Claude Code history.
---

# Import conversation history

## Prerequisite

Install `cg`.

Choose a Context name for the imported history.

## Recipe

1. Run `cg import <project-directory> --context <name> --search wait`.

2. Run `cg import <session.jsonl> --context <name> --search defer`.

3. Inspect progress with `cg status --context <name>`.

## Installed-binary example

Install `cg` and Python 3; use a POSIX shell.

Use the synthetic session below in a disposable data root.

<!-- smoke: skill -->
```sh
cg_example=$(mktemp -d)
export HOME="$cg_example/home"
mkdir -p "$HOME"
export CG_DATA_ROOT="$cg_example/data"
cat > "$cg_example/session.jsonl" <<'JSONL'
{"type":"user","uuid":"00000000-0000-4000-8000-000000000001","sessionId":"example-auth","timestamp":"2026-09-01T09:00:00Z","message":{"role":"user","content":"Where is the authentication module?"}}
{"type":"assistant","uuid":"00000000-0000-4000-8000-000000000002","sessionId":"example-auth","timestamp":"2026-09-01T09:01:00Z","message":{"role":"assistant","content":"The authentication module moved from src/auth.py to src/authentication.py."}}
JSONL
cg import "$cg_example/session.jsonl" --context example-auth --search defer --format json > "$cg_example/import.json"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["state"]); print("inserted:", d["capture"]["events"]["inserted"])' "$cg_example/import.json"
```

### Example output

```text
captured_search_deferred
inserted: 2
```

## Failure to handle

An exact-file import without `--context` exits 2. <!-- enforced by: `import_usage_errors_name_the_token_and_do_no_work` -->

Missing local Nomic artifacts leave capture complete and search blocked. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

Run `cg model fetch --defaults`; retry import with `--search wait`.

See [CLI reference](../../docs/cli.md) and [configuration](../../docs/configuration.md).
