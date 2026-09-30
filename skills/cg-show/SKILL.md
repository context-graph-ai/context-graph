---
name: cg-show
description: Expand a search passage into its exact stored conversation events with cg show.
---
<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

# Read a passage's source

## Prerequisites

Install `cg` from the repository README. Obtain a `passage_id` and its `context_name` from
[cg-find](../cg-find/SKILL.md); keep the same Context selection when showing it.<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

## Recipe

1. Run `cg show <passage-id> --context <context-name>`.
2. Read `events` in source order, including `text`, `payload`, `author`, `kind` and `seq`.
3. Use `session_ref` and `source_remainder` when tracing the original record. The command reads
   retained database events even when the source JSONL is gone.<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

## Reproducible missing-store example

This POSIX-shell/Python 3 example opens a fresh data root with no store. The command exits 1;
its JSON says `store_not_found` instead of inventing results. The final command prints that asserted
state. Use the numbered recipe above for an existing imported store.<!-- enforced by: `cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either` -->

```sh
cg_read_example=$(mktemp -d)
CG_DATA_ROOT="$cg_read_example/data" cg show 0190a3b2-0000-7000-8000-000000000000 --format json > "$cg_read_example/read.json" || test "$?" -eq 1
python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$cg_read_example/read.json"
```

Exact final stdout:

```text
store_not_found
```

## Failure to handle

`passage_not_found` (exit 1) includes a passage outside the selected Contexts; use the Context
returned by find. Status/find/show stay read-only under a live writer and never repair a store;
`store_needs_recovery` directs you to the intended import command.<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets`, `the_read_only_commands_answer_an_idle_store_and_a_held_one_without_changing_a_byte`, `import_recovers_a_store_that_needs_a_writable_open_and_status_names_that_command` -->
