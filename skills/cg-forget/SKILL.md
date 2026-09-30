---
name: cg-forget
description: Erase an imported session or a Context's imported history with cg forget.
---
<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

# Forget imported history

## Prerequisites

Install `cg` from the repository README and identify the intended Context/session from status or
show. This recipe deletes history; use it for the selection the user asked to erase.<!-- enforced by: `forgetting_a_session_erases_its_events_and_everything_derived_while_keeping_the_context` -->

## Recipe

1. Select `cg forget --session <ref> --context <name>` for one session, or
   `cg forget --context <name>` for all history in that Context.
2. Confirm on the terminal; for an authorized noninteractive deletion add `--yes`.
3. Read `state`, `counts` and `search.status`. The Context and its project binding remain.
4. If retained history needs rebuilding, inspect status; for `blocked`, install the local model
   using `cg model fetch --defaults` and re-run import with `--search wait`.<!-- enforced by: `cg_forget_confirms_on_the_terminal_and_enter_changes_nothing`, `forgetting_a_session_erases_its_events_and_everything_derived_while_keeping_the_context`, `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

Events, anchors, checkpoints, source locators and derived search rows are erased together. A source
containing only whitespace can have a checkpoint and locator with zero captured events; forgetting
that session or Context still requires confirmation and erases those rows. `nothing_to_forget`
means the selection has neither events nor source state.<!-- enforced by: `whitespace_only_session_forget_erases_checkpoint_and_locator`, `whitespace_only_context_forget_erases_checkpoint_and_locator` -->

Erasure uses the writer memory limit reported by `cg status`, including the share derived from a
container's memory ceiling. It does not require the larger capture proof floor. If an actual
erasure does not fit, `purge_refused` carries condition `WriteDidNotFitTheMemoryLimit`, reports
the needed and allowed memory, and leaves history unchanged. Raise the limit for one attempt with
`--write-memory-limit-bytes <BYTES>` or configure `[write] memory_limit_bytes` in
`cg/fabric.toml`.<!-- enforced by: `empty_history_can_be_forgotten_with_a_small_container_writer_share`, `whitespace_only_session_forget_erases_checkpoint_and_locator` -->

The erase-memory terminal state is `purge_refused`, with condition
`WriteDidNotFitTheMemoryLimit`. Its exact `message` template is below; `<needed>` and
`<allowed>` are the human-readable memory amounts.<!-- enforced by: `whitespace_only_session_forget_erases_checkpoint_and_locator` -->

```text
erasure needed <needed> of working memory but cg is allowed <allowed> on this machine; no history was erased. raise it with --write-memory-limit-bytes <BYTES> for this one command, or with [write] memory_limit_bytes in cg/fabric.toml for every command on this machine
```

Forgetting one session
also retires the Context's search head. The receipt reports `rebuild_scheduled` when rebuilding is
scheduled, `blocked` when the local model is missing, or `rebuild_not_started` when configuration
or worker startup prevents scheduling. A refused erasure keeps the previous searchable answer.<!-- enforced by: `forgetting_a_session_erases_its_events_and_everything_derived_while_keeping_the_context`, `a_refused_forget_leaves_the_search_head_passages_and_find_answer_unchanged`, `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets`, `cg_forget_confirms_on_the_terminal_and_enter_changes_nothing`, `forget_reports_a_configured_profile_refusal_after_erasing_without_starting_a_worker` -->

## Reproducible erasure with blocked search

With `cg`, a POSIX shell and Python 3 available, create the two pantry sessions used by the
confirmation guard in an isolated data root. The empty model cache makes rebuilding the retained
session report `blocked`; the erased session still reports two deleted events.<!-- enforced by: `cg_forget_confirms_on_the_terminal_and_enter_changes_nothing` -->

```sh
cg_forget_example=$(mktemp -d)
python3 - "$cg_forget_example" <<'PYTHON'
import json, pathlib, sys, uuid
root = pathlib.Path(sys.argv[1])
for session, question, answer in [
    ("sess-pantry-one", "pantry question", "pantry answer"),
    ("sess-pantry-two", "second question", "second answer"),
]:
    records = []
    for role, text, second in [("user", question, "00"), ("assistant", answer, "30")]:
        message = {"role": role, "content": text}
        if role == "assistant":
            message.update(id=f"{session}-m0", content=[{"type": "text", "text": text}])
        records.append({"type": role, "uuid": str(uuid.uuid4()), "sessionId": session,
                        "cwd": str(root), "timestamp": f"2026-09-01T09:00:{second}Z",
                        "message": message})
    (root / f"{session}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
PYTHON
for session in sess-pantry-one sess-pantry-two; do
  CG_DATA_ROOT="$cg_forget_example/data" CG_MODEL_CACHE_ROOT="$cg_forget_example/empty-model-cache" cg import "$cg_forget_example/$session.jsonl" --context pantry --search defer --format json > "$cg_forget_example/import.json"
done
CG_DATA_ROOT="$cg_forget_example/data" CG_MODEL_CACHE_ROOT="$cg_forget_example/empty-model-cache" cg forget --session sess-pantry-one --context pantry --yes --format json > "$cg_forget_example/forget.json"
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["state"], r["counts"]["events"], r["search"]["status"])' "$cg_forget_example/forget.json"
```

Exact final stdout:

```text
forgotten 2 blocked
```

## Failure to handle

Without a terminal and without `--yes`, a nonempty selection returns `confirmation_required` (exit 2)
and changes nothing. `discard_denied` (exit 1) means tenant-hub erasure is unsupported here;
`reader_coordination_unavailable` with condition `ReaderCoordinationUnavailable` reports that
the store could not verify reader safety for erasure and leaves history unchanged.
`purge_refused` reports erasure refusals, including an actual write memory limit refusal; inspect
its typed `condition` and message before retrying. `rebuild_not_started` concerns search after a successful
erasure: inspect its reason rather than assuming deletion failed.<!-- enforced by: `cg_forget_confirms_on_the_terminal_and_enter_changes_nothing`, `a_tenant_hub_store_is_refused_rather_than_purged_from_an_edge`, `a_refused_forget_leaves_the_search_head_passages_and_find_answer_unchanged`, `forget_reports_a_configured_profile_refusal_after_erasing_without_starting_a_worker` -->
