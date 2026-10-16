---
name: cg-forget
description: Erase selected imported history.
---

# Forget imported history

## Prerequisite

Install `cg`.

Obtain authorization for the selected session or Context.

## Recipe

1. Check the selection with `cg status --context <name>`.

2. Run `cg forget --session <session-ref> --context <name>` for one session.

3. Run `cg forget --context <name>` for the whole Context.

4. Use `--yes` only after authorizing that selection.

5. Inspect the returned counts and search status.

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
cg import "$cg_example/session.jsonl" --context example-auth --search defer > "$cg_example/import.json"
cg forget --context example-auth --yes --format json > "$cg_example/forget.json"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["state"]); print("erased events:", d["counts"]["events"]); print("remaining:", d["remaining_events"])' "$cg_example/forget.json"
```

### Example output

```text
forgotten
erased events: 2
remaining: 0
```

## Failure to handle

Unattended forget without `--yes` returns `confirmation_required`. <!-- enforced by: `cg_forget_confirms_on_the_terminal_and_enter_changes_nothing` -->

Verify the selection before retrying with `--yes`.

See [CLI reference](../../docs/cli.md) and [configuration](../../docs/configuration.md).
