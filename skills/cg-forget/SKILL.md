---
name: cg-forget
description: Delete imported conversation history with `cg forget --session` or `cg forget --context`. Use when a session or a whole Context's imported history must be erased.
---
<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

# Delete imported history with `cg forget`

```sh
cg forget --session <session-ref> --yes          # one session of the current project's Context
cg forget --session <session-ref> --context <name> --yes
cg forget --context <name> --yes                 # every imported session in that Context
```

Forgetting is deleting: the events, their anchors, checkpoints and every search passage and vector
derived from them go in one erasure, or nothing changes. The Context and its project link remain.<!-- enforced by: `forgetting_a_session_erases_its_events_and_everything_derived_while_keeping_the_context`, `a_refused_forget_leaves_the_search_head_passages_and_find_answer_unchanged` -->

Without `--yes` and without a terminal the answer is `confirmation_required` (exit 2) with the exact
`retry` command, and nothing changes; on a terminal it asks `[y/N]` and Enter means no. After a
session is forgotten, forget schedules the re-indexing of the Context's remaining history in the
background (`search.status: rebuild_scheduled`); poll `cg status` until search is `ready`. If
`search.status` is `blocked`, run `cg model fetch --defaults` first.<!-- enforced by: `cg_forget_confirms_on_the_terminal_and_enter_changes_nothing`, `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

| `state` | Exit |
| --- | ---: |
| `forgotten` | 0 |
| `nothing_to_forget` | 0 |
| `confirmation_required` | 2 |
| `purge_refused`, `discard_denied` | 1 |
