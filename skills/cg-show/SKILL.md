---
name: cg-show
description: Expand a `cg find` passage into the exact stored conversation events with `cg show`. Use when an agent needs the verbatim source behind a search hit.
---
<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

# Read the exact source of a passage with `cg show`

```sh
cg show <passage_id>
cg show <passage_id> --context <name>     # a passage found through a wider search
```

`events` lists the stored events behind the passage in source order, each with its `text`,
structured `payload`, `author`, `kind`, `seq`, `session_ref` and the retained `source_remainder`.
It reads the database, so it works after the original session file is gone.<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

A passage outside the selected Contexts answers `passage_not_found` (exit 1) and reveals nothing
about it; supply the `context_name` that `cg find` reported, or `--all-contexts`.<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->
