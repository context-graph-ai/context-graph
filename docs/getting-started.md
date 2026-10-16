# Search your first conversation

Prerequisites: installed `cg`, POSIX shell, project with Claude Code history.

See [the repository README](../README.md) for installation.

Directory import selects the chosen project's Claude Code history. <!-- enforced by: `cg_import_project_directory_never_sweeps_working_tree_jsonl_or_imports_other_project_history` -->

It excludes working-tree JSONL files and other projects' history. <!-- enforced by: `cg_import_project_directory_never_sweeps_working_tree_jsonl_or_imports_other_project_history` -->

A question about the imported history returns source passages. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

```sh
cg model fetch --defaults
cg import "$HOME/projects/my-app" --context my-app --search wait --format text
cg status --context my-app --format text
cg find "why did we choose that transport?" --context my-app --format text
```

Example output (ids and sizes differ on your machine):

Long output is trimmed with ….

```text
{"disclosed":[{"bytes":546938168,"id":"nomic-embed-text-v1.5","size":"547 MB"}],"fetched":[{"id":"nomic-embed-text-v1.5","sha":"9e7d262b1fe5ea350782829496efa831901b77486bbde1cea54a4c822d010d5c"}],"skipped":[{"id":"embeddinggemma-300m","reason":"ManualAcceptanceRequired"}]}
context: my-app (explicit)
sources: 1 files / 1 sessions
history captured: 7 new events (0 already present)
semantic search: caught up
context: my-app (explicit)

memory
  goals: 0
  decisions: 0
  observations: 0

history
  sources: 1 files / 1 sessions
  capture: complete — 7 events
  coverage: 7 complete / 0 opaque / 0 payload-refused / 0 source-declared-missing
  event placement: local_only
  search: ready — 7 events covered
write memory: <N> GiB — derived from this machine: a quarter of the <installed> GiB installed
<passage-id>  my-app  2026-09-01  session synthetic-session-amber
Assistant [author="assistant"]: The edit request is separate from this conclusion: the decisive sentence after the former preview boundary says the account notice remains effective until a signed amendment replaces it.
```text
search_document: literal source prefix, not an embedding instruction
…
```

Pass the `passage_id` printed by find to `cg show` to read the exact linked stored events. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

The first interactive import accepts the suggested Context name on Enter. <!-- enforced by: `first_interactive_project_import_accepts_suggested_name_on_enter_descendant_reads_reuse_binding_and_noninteractive_import_returns_retry_without_writes` -->

Later `status` calls from a subdirectory reuse that Context binding. <!-- enforced by: `first_interactive_project_import_accepts_suggested_name_on_enter_descendant_reads_reuse_binding_and_noninteractive_import_returns_retry_without_writes` -->

A default import completes capture without local model artifacts. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

It downloads no model. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

A missing-artifact retry names `cg model fetch --defaults`. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

It then names an import with `--search wait`. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

Next: [command recipes](cli.md#recipes), [configuration](configuration.md), or [troubleshooting](troubleshooting.md).
