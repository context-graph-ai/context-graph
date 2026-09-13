---
name: cg-import
description: Import a project's Claude Code conversation history into the local Context Graph store with `cg import`, and read its completion state. Use when an agent or operator wants past sessions captured and searchable.
---
<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

# Import Claude Code history with `cg import`

## Run it

```sh
cd <project-directory>
cg import . --context <name>          # non-interactive: always name the Context
cg import . --search wait --context <name>   # return only when search is ready
cg import <session.jsonl> --context <name>   # one exact session file
cg import . --dry-run --context <name>       # resolve sources and Context, verify, write nothing
```

On a terminal, `cg import .` without `--context` asks once for the Context name and remembers it
for the project and its sub-directories. An agent has no terminal: pass `--context`, or read the
`suggested_context` and `retry` fields of a `context_binding_required` answer and run the retry.<!-- enforced by: `first_interactive_project_import_accepts_suggested_name_on_enter_descendant_reads_reuse_binding_and_noninteractive_import_returns_retry_without_writes`, `noninteractive_unbound_project_import_returns_the_retry_without_writing` -->

## Read the answer

The JSON receipt's `state` is the whole contract:

| `state` | Exit | Do next |
| --- | ---: | --- |
| `complete` | 0 | Nothing; history is captured and searchable. |
| `captured_search_scheduled` | 0 | Poll `cg status` until `history.search.status` is `ready`. |
| `captured_search_deferred` | 0 | Run the `retry` command when search is wanted. |
| `captured_search_behind` | 1 | Run the `retry` command (installs the model with `cg model fetch --defaults` if `search.status` is `blocked`). |
| `source_partial` | 1 | Fix or remove the files in `capture.failed_sources`; everything else was captured. |
| `schema_mismatch` | 1 | The store holds an older schema; it is left untouched. Point `CG_DATA_ROOT` at a new empty directory to start a fresh store. |
| `tenant_policy_blocked` | 1 | Finish tenant enrollment for this data root, then run `retry`; nothing was written. |
| `project_history_not_found` | 2 | Import the directory the sessions were started in, or name the file. |
| `context_binding_required` | 2 | Run `retry`. |
| `context_binding_conflict` | 2 | Import without `--context`; the project is linked elsewhere. |
| `store_busy` | 4 | Retry after the other writer exits. |
| `dry_run` | 0 | `--dry-run` answer: `capture` lists new, grown, unchanged and diverged sources and bytes to capture. |

Capture makes no model or hosted call. Import never downloads a model.<!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight`, `explicit_context_import_defers_search_and_keeps_source_facts_out_of_its_receipt` -->
