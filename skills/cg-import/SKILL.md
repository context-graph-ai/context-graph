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

| `state` | Exit | Do next |
| --- | ---: | --- |
| `complete` | 0 | Nothing; history is captured and searchable. |
| `captured_search_scheduled` | 0 | Poll `cg status` until `history.search.status` is `ready`. |
| `captured_search_deferred` | 0 | Run the `retry` template with the original project directory or session file when search is wanted. |
| `captured_search_behind` | 1 | Run the `retry` template with the original input; it installs the model with `cg model fetch --defaults` if `search.status` is `blocked`. |
| `write_memory_limit` | 1 | Read committed capture progress, raise the writer budget, then retry the same import; earlier commits remain. |
| `source_partial` | 5 | Fix or remove the files in `capture.failed_sources`; everything else was captured. |
| `schema_mismatch` | 1 | The unsupported schema contents are left untouched. Move data from an earlier build aside, then set `CG_DATA_ROOT` to a new empty directory and recreate the store. |
| `tenant_policy_blocked` | 1 | Finish tenant enrollment for this data root, then run `retry`; nothing was written. |
| `project_history_not_found` | 2 | Import the directory the sessions were started in, or name the file. |
| `context_binding_required` | 2 | Run `retry`. |
| `context_binding_conflict` | 2 | Import without `--context`; the project is linked elsewhere. |
| `store_busy` | 4 | Retry after the other writer exits. |
| `store_needs_recovery` | 6 | Run a writable import with `--search defer` before retrying the dry-run. |
| `dry_run` | 0 | Configuration is valid; `capture` lists new, grown, unchanged and diverged sources and bytes to capture without writing. |
| `config_error` | 1 | Correct the named configuration setting or command-line value, then retry. |

A `write_memory_limit` receipt carries `capture.status: partial`, the count committed by this
invocation in `capture.events.inserted`, and `capture.last_committed_position` (read
`committed_bytes` when present). Earlier committed events remain durable; the refused write changes nothing,
and `search.status` is `not_run`. Inspect those fields before deciding how much remains.<!-- enforced by: `an_import_that_exceeds_writer_memory_after_a_batch_reports_committed_progress_and_resumes` -->

To continue, raise `--write-memory-limit-bytes BYTES` for that invocation or
`[write] memory_limit_bytes` in the data root's `cg/fabric.toml`, then retry the same source,
Context and import options. If the original command supplied the switch, raise that switch or
remove it to use the configured budget: the switch takes precedence. With neither a switch nor
`[write] memory_limit_bytes`, cg derives a quarter of detected machine memory. `cg status` shows
that limit and its source; a value too low for capture refuses the write. cg resumes from stored
committed progress; do not trim the source or treat `last_committed_position` as a new input
argument.<!-- enforced by: `an_import_that_exceeds_writer_memory_after_a_batch_reports_committed_progress_and_resumes`, `status_names_the_write_budget_in_force_and_which_of_the_three_answers_set_it` -->

Read budgets are separate from writer memory. In the data root's `cg/fabric.toml`, set positive
`[read].memory_limit_bytes` and `[read].active_time_ms` values for the bytes a read may hold and
the active milliseconds one statement may spend. Omission uses 1 GiB (`1073741824`) and
5,000 ms. File and live-owner reads use those budgets; malformed settings return `config_error`
before store access. Merge this table into existing configuration instead of replacing other
settings.<!-- enforced by: `default_store_config_declares_the_one_gib_read_memory_ceiling`, `read_active_time_defaults_to_five_seconds_and_accepts_only_positive_milliseconds`, `cg_status_uses_the_default_and_a_valid_custom_read_budget_then_refuses_an_actual_low_budget`, `invalid_read_config_is_a_config_error_before_status_can_open_or_write_the_store` -->

```toml
[read]
memory_limit_bytes = 67108864
active_time_ms = 5000
```

A background search request keeps its accepted read active-time allowance across restart. A
temporary active-time refusal releases the failed read, waits briefly, and resumes from committed
progress without repeating completed passages or raising any limit. Three consecutive active-time
refusals without committed coverage progress end that request as `failed`; the no-progress count
survives restart. `cg status` reports the failure with an instruction to run the same import with
`--search wait` again. Foreground `--search wait` applies the same bound to assessment and projection reads, and keeps a terminal refusal visible in `cg status` even when no generation started. Captured events and completed search batches remain saved.<!-- enforced by: `a_typed_active_time_refusal_retains_committed_passages_and_resumes`, `a_transient_typed_active_time_refusal_resumes_selected_work_without_reprojecting_the_completed_prefix`, `consecutive_typed_active_time_refusals_exhaust_one_selected_request_allowance`, `active_time_recovery_restart_keeps_the_selected_allowance_and_no_progress_count`, `assessment_active_time_refusals_resume_or_leave_a_retryable_failed_status`, `an_active_time_refusal_keeps_its_original_frontier_until_a_complete_retry`, `a_refused_aggregate_charges_only_its_assessed_low_active_time_selections`, `foreground_active_time_refusals_stop_when_capture_changes_without_coverage_progress`, `prior_committed_coverage_does_not_extend_a_new_requests_active_time_allowance`, `foreground_assessment_active_time_refusals_recover_and_exhaust_with_durable_status`, `foreground_selection_timeout_exhaustion_stays_failed_until_full_context_completion`, `large_foreground_selections_and_repeated_failures_keep_bounded_durable_status` -->

Retry templates repeat the resolved session scope and order. They repeat `--embedding-workers`
only when that command supplied the flag; `[search]` configuration stays persistent for the data
root.<!-- enforced by: `interrupted_all_scope_retry_resumes_all_sessions_in_the_same_oldest_order`, `missing_model_retry_keeps_the_resolved_scope_without_a_hosted_or_downloaded_fallback` -->

The receipt intentionally uses the quoted `<project-directory>` or `<session.jsonl>` placeholder
instead of a source path; replace the whole quoted token with the original input quoted
independently for your shell before running it.
Exact-file templates retain their separately quoted `--context` value. Capture makes no model or
hosted call. Import never downloads a model.<!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight`, `explicit_context_import_defers_search_and_keeps_source_facts_out_of_its_receipt`, `import_recovers_a_store_that_needs_a_writable_open_and_status_names_that_command`, `receipt_placeholder_substitution_and_read_retries_preserve_persisted_scope` -->

## Choose which sessions become searchable first

Capture always keeps every selected session. Search embedding defaults to the 10 most recent
sessions, processed newest first. Set `--search-sessions N|all` and
`--search-order newest|oldest`, or configure the same choices for the data root:<!-- enforced by: `recent_session_scope_builds_the_ten_most_recent_newest_first_and_widens_without_reembedding`, `search_work_policy_defaults_to_newest_ten_and_cli_overrides_config` -->

```toml
[search]
sessions = 10 # or "all"
order = "newest"
```

Command-line values win for that invocation; `[search]` values persist for the data root. `oldest`
chooses the oldest sessions and processes them oldest first, while `newest` chooses and processes
the newest sessions first. Widening the scope reuses existing vectors, and narrowing does not hide
already indexed results. The receipt separates pending,
outside-scope, and no-searchable-text event counts. Outside scope means outside the current build
selection, so retained indexed results can be outside scope; output reports that indexed subset
separately. An `all` first build can take much longer.<!-- enforced by: `recent_session_scope_builds_the_ten_most_recent_newest_first_and_widens_without_reembedding`, `oldest_order_builds_the_ten_oldest_sessions_oldest_first`, `search_work_policy_defaults_to_newest_ten_and_cli_overrides_config`, `invalid_search_config_is_refused_even_when_cli_values_are_present` -->

`store_needs_recovery` is a read-only command result. `cg status` returns exit 1 and
`cg import --dry-run` returns exit 6; both direct the user to run `cg import <project-directory> --search
defer`; neither command opens the store for recovery.<!-- enforced by: `import_recovers_a_store_that_needs_a_writable_open_and_status_names_that_command`, `import_dry_run_refuses_a_recovery_needing_store_without_writing` -->

## Choose embedding parallelism

Passage embedding uses available CPU parallelism by default. Use a positive
`--embedding-workers N` limit for one import, or configure the data root in
`$CG_DATA_ROOT/cg/fabric.toml` (or `~/.cg/cg/fabric.toml` when `CG_DATA_ROOT` is unset):
<!-- enforced by: `omitted_worker_policy_resolves_to_rust_available_parallelism`, `resolver_refuses_zero_negative_and_non_number_values_from_each_input` -->

```toml
[search]
embedding_workers = 4
```

The command-line limit wins over `[search].embedding_workers`; without either, cg uses available
CPU parallelism. `1` makes passage embedding calls serial, though it does not assign CPU affinity
and Candle may use internal threads inside each call. The limit changes execution only: it does not
change the profile, build a generation, or re-embed unchanged history. There is no `--fast` or
`--slow` flag.<!-- enforced by: `configured_worker_policy_is_used_when_the_cli_is_silent`, `explicit_cli_worker_policy_wins_over_configuration_in_both_directions`, `embedding_worker_cap_one_keeps_the_second_call_out_until_the_first_returns` -->

A background import with pending search publishes its worker selection for the existing worker
to reread before the next batch. `--search defer` and an unchanged, caught-up import leave an
active worker's request alone and start no worker.<!-- enforced by: `requesting_import_publishes_the_cli_cap_to_the_existing_worker_handoff`, `claim_callback_rereads_the_worker_policy_file_before_the_next_two_passage_batch`, `deferred_import_never_creates_or_changes_the_active_worker_policy`, `already_caught_up_import_does_not_change_an_active_worker_request` -->

A pending background request keeps its selected write-memory ceiling across restart. Shared
work uses the lower ceiling. If the selected work does not fit, `cg status` reports `failed`
with the needed and allowed memory. Captured history remains available; retry the import with an
affordable ceiling.<!-- enforced by: `overlapping_imports_keep_each_contexts_write_ceiling`, `shared_pending_work_resumes_under_the_lower_persisted_ceiling_without_reprojection`, `the_next_batch_refuses_an_insufficient_persisted_ceiling_and_keeps_its_request` -->

Invalid import-specific flags such as an invalid `--search` value return `usage_error` with exit 2.
Supply a positive whole number of bytes to the global `--write-memory-limit-bytes` switch;
correct a missing, zero, negative or nonnumeric value before retrying.
For invalid placement, `[search]`, or `[write]` configuration, correct the named setting in
`cg/fabric.toml` and retry.<!-- enforced by: `import_usage_errors_name_the_token_and_do_no_work`, `a_switch_with_no_usable_number_is_refused_before_any_store_is_opened`, `a_write_budget_that_is_not_a_byte_count_is_a_config_error_before_any_store_is_opened`, `dry_run_refuses_invalid_placement_before_new_or_existing_store_writes_or_model_access`, `a_hard_passage_ceiling_refuses_each_import_mode_before_new_or_existing_store_writes` -->

## Reproducible capture example

Run from a checkout of this repository with `cg` on `PATH`, a POSIX shell and Python 3. This uses
its checked-in roundtrip fixture plus the same additional occurrence as the deferred-import guard;
the isolated model cache is empty. The last command prints the asserted receipt fields.<!-- enforced by: `explicit_context_import_defers_search_and_keeps_source_facts_out_of_its_receipt` -->

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
