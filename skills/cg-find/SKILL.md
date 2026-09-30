---
name: cg-find
description: Search imported Claude Code history for stored passages behind a question or decision.
---
<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

# Find conversation passages

## Prerequisites

Install `cg` from the repository README and capture history using [cg-import](../cg-import/SKILL.md).
Install the search model with `cg model fetch --defaults`, then run the original import with
`--search wait`; find itself never indexes.<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets`, `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

## Recipe

1. Run `cg find "your question"` in the bound project, or pass `--context NAME` explicitly.
2. To search a wider authorized scope, use `--context app,infra` or `--all-contexts`.
3. Read `hits[].text` as source passages, then cite `context_name` and `passage_id`.
4. Expand a hit with `cg show <passage-id>` using the same Context selection.<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

Find ranks across the selected Context set. It reports omitted unready Contexts rather than treating
their missing coverage as a complete search. Its shared coverage reads do not multiply with the
number of selected Contexts.<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets`, `a_find_over_ready_contexts_issues_the_same_statements_at_one_eight_and_sixty_four` -->

## Reproducible missing-store example

This POSIX-shell/Python 3 example opens a fresh data root with no store. The command exits 1;
its JSON says `store_not_found` instead of inventing results. The final command prints that asserted
state. Use the numbered recipe above for an existing imported store.<!-- enforced by: `cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either` -->

```sh
cg_read_example=$(mktemp -d)
CG_DATA_ROOT="$cg_read_example/data" cg find anything --format json > "$cg_read_example/read.json" || test "$?" -eq 1
python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$cg_read_example/read.json"
```

Exact final stdout:

```text
store_not_found
```

## Failure to handle

`history_search_not_ready` (exit 1) means captured history needs search work; follow its retry or
inspect status. `no_matching_history` (exit 0) is a successful empty result. An unknown explicit
Context refuses the selection instead of silently dropping it. Reads use `ConsumerReader` and leave
the store untouched even while another process writes it.<!-- enforced by: `find_in_a_context_with_no_imported_history_answers_no_matching_history`, `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets`, `the_read_only_commands_answer_an_idle_store_and_a_held_one_without_changing_a_byte` -->
