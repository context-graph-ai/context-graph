---
name: cg-find
description: Search imported conversation history with `cg find` and use the returned passage handles. Use when an agent needs the source discussion behind a past decision or question.
---
<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

# Search imported history with `cg find`

```sh
cg find "why did we choose Iroh instead of NATS?"
cg find "<question>" --context app,infra      # several named Contexts, ranked together
cg find "<question>" --all-contexts           # every Context in the store
```

With no selector it searches the Context linked to the current project. It returns passages of
the stored conversation, never an answer: read `hits[].text`, and cite `hits[].passage_id` and
`hits[].context_name`. Pass a `passage_id` to `cg show` for the exact events.<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

- `state: no_matching_history` (exit 0): the selected Contexts hold no imported history.
- `state: history_search_not_ready` (exit 1): history is captured but not searchable; run
  `cg import <project-directory> --search wait`, or wait for `cg status` to report `ready`.
- `omitted_contexts`: selected Contexts that were not searched because their search is not ready.<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets`, `find_in_a_context_with_no_imported_history_answers_no_matching_history` -->
- `state: unknown_context` (exit 2): a named Context does not exist; nothing was searched.

`cg find` never starts or waits for indexing and never changes the store.<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->
