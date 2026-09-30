---
name: cg-status
description: Inspect captured history and search readiness, including read-memory configuration and low-writer-memory refusals.
---
<!-- enforced by: `cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either` -->

# Inspect memory without changing it

## Prerequisites

Install `cg` from the repository README. Import history with [cg-import](../cg-import/SKILL.md)
to create a store; status itself never creates one. Model connections are inspected separately
with `cg model status`.<!-- enforced by: `cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either`, `cg_status_help_says_it_reads_memory_and_what_a_refused_read_looks_like` -->

## Recipe

1. Run `cg status` in the bound project, or `cg status --context NAME` explicitly.
2. Compare captured counts with `history.search.status` and pending/outside-scope coverage.
3. For pending search, inspect `history.search.next`; for missing-model blockage, fetch the model
   before retrying the import.<!-- enforced by: `receipt_placeholder_substitution_and_read_retries_preserve_persisted_scope`, `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->
4. For multiple Contexts, use `--context app,infra` or `--all-contexts` and read each labelled
   entry, rather than assuming every Context has the same readiness.<!-- enforced by: `cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either`, `status_separates_capture_from_search_coverage_per_selected_context` -->

## Read the fields and choose the next action

For one Context, inspect `history`; for multiple Contexts, inspect each labelled
`contexts[].history`. `history.captured_events` counts retained capture, while
`history.search.covered_events` and `pending_events` describe search coverage. A successful status
read exits 0 even when search needs work; use the nested search status to decide readiness.<!-- enforced by: `cg_import_project_directory_never_sweeps_working_tree_jsonl_or_imports_other_project_history`, `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets`, `status_separates_capture_from_search_coverage_per_selected_context` -->

`history.search.outside_scope_events` counts events outside the current session build selection;
it is not pending work for that selection. `outside_scope_indexed_events` identifies the already
indexed subset retained when the selection narrows. `no_searchable_text_events` reports captured
events without searchable text. Ready selected coverage can therefore coexist with outside-scope
history.<!-- enforced by: `cli_status_reports_bounded_session_scope_from_the_persisted_worker_policy`, `recent_session_scope_builds_the_ten_most_recent_newest_first_and_widens_without_reembedding` -->

Common search outcomes and actions (not an exhaustive state reference):

| `history.search.status` | Meaning and action |
| --- | --- |
| `ready` | Selected search coverage is caught up; use `cg find`. |
| `pending` | Search work remains; use `history.search.next` to retry with the persisted session scope and order. |
| `blocked` | The local Nomic artifact is missing; run `cg model fetch --defaults`, then the original import with `--search wait`. |

<!-- enforced by: `status_separates_capture_from_search_coverage_per_selected_context`, `receipt_placeholder_substitution_and_read_retries_preserve_persisted_scope`, `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight`, `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

These are search states, separate from a command-level refusal such as `store_needs_recovery`.
Capture can remain durable while search is blocked. If an import with `--search wait` fails to load
the model, its receipt reports `search.status: failed` and a `search.reason`; inspect that reason
before retrying. A subsequent status read still counts the captured events and does not report
search as `ready`.<!-- enforced by: `status_separates_capture_from_search_coverage_per_selected_context`, `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight`, `cg_import_search_wait_embedding_failure_returns_durable_capture_receipt_and_typed_behind_status` -->

## Read budgets and low-memory inspection

In `$CG_DATA_ROOT/cg/fabric.toml` (or `~/.cg/cg/fabric.toml` without the override), set a positive
byte count as `[read].memory_limit_bytes` and a positive millisecond count as
`[read].active_time_ms`. Omission means 1 GiB (`1073741824`) and 5,000 ms per statement; file
and live-owner reads use the same per-invocation budgets. Merge this table into an existing config instead of replacing its
other settings.<!-- enforced by: `default_store_config_declares_the_one_gib_read_memory_ceiling`, `read_active_time_defaults_to_five_seconds_and_accepts_only_positive_milliseconds`, `file_and_live_owner_reads_honor_per_instance_memory_budgets_without_losing_a_601_row_result`, `cg_status_uses_the_default_and_a_valid_custom_read_budget_then_refuses_an_actual_low_budget` -->

```toml
[read]
memory_limit_bytes = 67108864
active_time_ms = 5000
```

Writer memory is separate: `--write-memory-limit-bytes BYTES` wins over `[write] memory_limit_bytes`,
which wins over the machine-derived budget. Status reports the writer value and source even when
the derived budget is too small for capture. Help/status/find/show remain on their read route;
raising the writer setting does not raise a read budget.<!-- enforced by: `status_names_the_write_budget_in_force_and_which_of_the_three_answers_set_it`, `a_small_container_allows_inspection_but_refuses_writable_capture`, `a_small_writer_setting_does_not_block_help_status_or_show` -->

A background search request keeps its accepted read active-time allowance across restart. A
temporary active-time refusal releases the failed read, waits briefly, and resumes from committed
progress without repeating completed passages or raising any limit. Three consecutive active-time
refusals without committed coverage progress end that request as `failed`; the no-progress count
survives restart. `cg status` reports the failure with an instruction to run the same import with
`--search wait` again. Foreground `--search wait` applies the same bound to assessment and projection reads, and keeps a terminal refusal visible in `cg status` even when no generation started. Captured events and completed search batches remain saved.<!-- enforced by: `a_typed_active_time_refusal_retains_committed_passages_and_resumes`, `a_transient_typed_active_time_refusal_resumes_selected_work_without_reprojecting_the_completed_prefix`, `consecutive_typed_active_time_refusals_exhaust_one_selected_request_allowance`, `active_time_recovery_restart_keeps_the_selected_allowance_and_no_progress_count`, `assessment_active_time_refusals_resume_or_leave_a_retryable_failed_status`, `an_active_time_refusal_keeps_its_original_frontier_until_a_complete_retry`, `a_refused_aggregate_charges_only_its_assessed_low_active_time_selections`, `foreground_active_time_refusals_stop_when_capture_changes_without_coverage_progress`, `prior_committed_coverage_does_not_extend_a_new_requests_active_time_allowance`, `foreground_assessment_active_time_refusals_recover_and_exhaust_with_durable_status`, `foreground_selection_timeout_exhaustion_stays_failed_until_full_context_completion`, `large_foreground_selections_and_repeated_failures_keep_bounded_durable_status` -->

## Reproducible example

This POSIX-shell/Python 3 example uses a fresh data root with no store. It prints the complete refusal
object after replacing only the temporary database path
with `<resolved-database-path>`; the `cg` command exits 1 and the final Python command exits 0.<!-- enforced by: `cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either` -->

```sh
cg_status_example=$(mktemp -d)
CG_DATA_ROOT="$cg_status_example/data" cg status --format json > "$cg_status_example/status.json" || test "$?" -eq 1
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); r["store"]["db_path"]="<resolved-database-path>"; print(json.dumps(r, sort_keys=True, indent=2))' "$cg_status_example/status.json"
```

Exact final stdout:

```json
{
  "ok": false,
  "retry": "create_store_with_write_command",
  "state": "store_not_found",
  "store": {
    "db_path": "<resolved-database-path>"
  }
}
```

## Failure to handle

A malformed or zero read memory or active-time allowance returns `config_error` (exit 1) before store access.
An insufficient memory budget produces a memory refusal, never zero counts; increase
`[read].memory_limit_bytes` and retry. An exhausted statement time allowance produces an
active-time refusal; increase `[read].active_time_ms` if that statement needs more time, then retry.
`store_needs_recovery` (exit 1) requires the intended writable import (`--search defer` is sufficient for
recovery); status never repairs it.<!-- enforced by: `invalid_read_config_is_a_config_error_before_status_can_open_or_write_the_store`, `cg_status_uses_the_default_and_a_valid_custom_read_budget_then_refuses_an_actual_low_budget`, `import_recovers_a_store_that_needs_a_writable_open_and_status_names_that_command` -->
