# cg-cli — package `context-graph-cli`, binary `cg`

Purpose: parse commands and render typed library answers for operators.
Owns arguments, help, output envelopes, typed exit mapping, root resolution,
configuration plumbing and detached search worker process lifecycle.
Must not own event identity, Claude JSON decoding, SQL, ranking, embedding preparation,
storage policy or a second cache-root rule.

## Seams and where a change lives

| Change | Entry point |
|---|---|
| Dispatch, help and exits | `src/main.rs`; match typed conditions |
| Capture and forgetting | `src/history/import.rs`, `src/history/forget.rs` |
| Find, show, status and Context selection | `src/history/read.rs`, `src/history/mod.rs` |
| Detached work, saved selections and retry accounting | `src/history/worker.rs` |
| Reader budgets | `src/read_config.rs` |
| Writer admission | `src/write_config.rs` |

Read commands use core `ConsumerReader`; capture uses `Store` and the Claude connector.
Node locks and bounded request IO come from `cg-node-fs`; model cache resolution delegates
to `cg-catalog`. Search consumes core `Embedder` and sealed `EventSearchEmbedder`.

The schema-version probe in `src/main.rs` borrows the engine answer through `rows()`. Shared
read decoders consume the core library’s application aggregate and page accessors; the CLI does
not construct or mutate engine result storage.

## Invariants and guards

Ownership boundaries above are review-only architectural constraints unless a guard below covers them.

- Read commands change no bytes even with a writer present:
  `tests/a_read_only_command_answers_a_store_a_writer_is_holding.rs::the_read_only_commands_answer_an_idle_store_and_a_held_one_without_changing_a_byte`.
- Read budget validation precedes store access. The per-statement active-time allowance defaults
  to 5,000 ms and accepts positive integer milliseconds under `[read].active_time_ms` alongside
  read memory. Guards:
  `src/read_config.rs::read_active_time_defaults_to_five_seconds_and_accepts_only_positive_milliseconds` and
  `tests/read_memory_policy.rs::invalid_read_config_is_a_config_error_before_status_can_open_or_write_the_store`.
  Low writer memory leaves inspection available:
  `src/write_config.rs::a_small_container_allows_inspection_but_refuses_writable_capture`.
- Accepted background requests retain their read active-time allowance and no-progress count
  across restart. A typed timeout resumes committed coverage after bounded backoff; three
  consecutive refusals without coverage progress record `failed` with a retry instruction.
  Guards in `tests/import_embedding_workers.rs`:
  `a_transient_typed_active_time_refusal_resumes_selected_work_without_reprojecting_the_completed_prefix`,
  `consecutive_typed_active_time_refusals_exhaust_one_selected_request_allowance`,
  `active_time_recovery_restart_keeps_the_selected_allowance_and_no_progress_count`,
  `assessment_active_time_refusals_resume_or_leave_a_retryable_failed_status`,
  `an_active_time_refusal_keeps_its_original_frontier_until_a_complete_retry`,
  `a_refused_aggregate_charges_only_its_assessed_low_active_time_selections`,
  `foreground_active_time_refusals_stop_when_capture_changes_without_coverage_progress`,
  `prior_committed_coverage_does_not_extend_a_new_requests_active_time_allowance`,
  `foreground_assessment_active_time_refusals_recover_and_exhaust_with_durable_status`,
  `foreground_selection_timeout_exhaustion_stays_failed_until_full_context_completion`, and
  `large_foreground_selections_and_repeated_failures_keep_bounded_durable_status`.
- Import maps partial source capture to exit 5 and writable-recovery-required to exit 6 through
  its own typed wrapper; find/show/status retain their shared refusal mappings. Guards:
  `tests/event_import_roundtrip.rs::malformed_input_reports_source_partial_at_the_failed_position_without_silently_skipping_it`
  and `tests/event_history.rs::import_dry_run_refuses_a_recovery_needing_store_without_writing`.
- Capture remains durable before search; retry preserves selected scope:
  `tests/import_embedding_workers.rs::interrupted_all_scope_retry_resumes_all_sessions_in_the_same_oldest_order`.
- Background write ceilings stay with the session identities and capture frontiers selected at
  acceptance, through later captures and restart. Completed selected work is acknowledged without
  reprojection. A replacement profile freezes every saved frontier of the actual running request,
  linked to its generation and claim owner in the same claim transaction, when that
  differing profile may publish after acceptance. Same-profile acceptances keep
  their selected scope and do not acquire unrelated pending work. Completion cleanup uses the saved
  request ceilings until durable selection retirement, including terminal refusals after restart.
  Independent selections retire under their own ceilings even when another cleanup fails; a shared
  selection uses its lower ceiling and remains until every owner completes. Global reconciliation
  retains unretired selections, and cleanup errors release the worker lock
  and recheck all unfinished accepted selections. Superseded refusals are reclaimed after retirement.
  Refused-only restarts stop after the unlock queue recheck. An unrecorded refusal suppresses only
  its exact selection for that worker, preserves the pending request and nonzero outcome, and permits
  later accepted selections in the same Context. Ordinary no-progress accounting follows the
  saved selection identity, and successful no-text selections never consume a later selection's
  allowance. Aggregate read refusals exhaust only the assessed accepted selection identities;
  after unlocking, later accepted identities keep their own bounded allowance.
  Terminal selected-work read failures retain and suppress their exact selection,
  preserve the nonzero outcome, and permit other accepted work even when aggregate search is caught up.
  A caught-up import acknowledges only the refused request identities observed after verifying its
  selected work and each refusal's original session identities and capture frontier. Completing
  those frontiers clears their refusal while unselected history and later captures remain unembedded;
  a narrower retry preserves an incomplete broader refusal, and an older completion preserves a
  later refusal in the same Context.
  <!-- enforced by: `older_completed_import_acknowledges_only_observed_refusal_identities`, `a_completed_newest_one_retry_clears_its_refusal_without_embedding_older_history` -->
  Completion-only frontiers retain each refused request's original session identities and capture
  boundary after runnable selection retirement, without keeping a writer open;
  cleanup of unrelated terminal selections cannot block that acknowledgement.
  Successful selected work reassesses original same-Context refusal frontiers through the admitted
  completion writer and supersedes only those verified complete. Inspection derives visible refusals
  from live, unsuperseded saved request identities, preserving each typed refusal's actual needed
  and allowed memory. It reports the latest active refusal's own facts, and superseding that identity
  restores any remaining older refusal's facts. An incomplete broader refusal remains visible when
  newly accepted narrower work completes. Superseded refusals retain their saved request and ceiling
  until selection retirement but no longer override search readiness.
  <!-- enforced by: `new_narrow_background_work_preserves_an_incomplete_broader_refusal` -->
  Consecutive typed reader-coordination refusals retry at most three times per
  accepted selection; queue publication or generation changes do not reset that allowance.
  A recorded typed refusal acknowledges only its served selection and permits later
  disjoint work; a rejected queue publication retires its unaccepted selection rows. Shared pending
  work uses the lower ceiling; standalone forget rebuilds retain their own queued selection alongside imports; forgetting another session preserves the selection. Writable
  acceptance and acknowledgement reclaim unpublished and completed selections in bounded pages; an unrecordable memory refusal remains inspectable through the
  bounded worker record. Guards in `tests/import_embedding_workers.rs`:
  `a_transient_typed_coordination_refusal_resumes_selected_work_without_reprojecting_the_completed_prefix`,
  `consecutive_typed_coordination_refusals_exhaust_one_selected_request_allowance`,
  `a_standalone_forget_rebuild_coexists_with_another_contexts_queued_import`,
  `profile_replacement_retains_published_frontiers_outside_the_newest_selection`,
  `profile_replacement_retains_an_earlier_queued_frontier_published_after_acceptance`,
  `forgetting_after_all_import_requests_are_acknowledged_rebuilds_surviving_history`,
  `forgetting_an_unrelated_session_preserves_the_queued_selection_and_ceiling`,
  `unpublished_selections_are_reclaimed_without_erasing_queued_work_after_reopen`,
  `a_recorded_refusal_acknowledges_only_its_selection_and_serves_later_disjoint_work`,
  `a_rejected_queue_publication_retires_its_unaccepted_selection`,
  `a_deferred_append_stays_outside_the_accepted_frontier_even_when_it_would_fail`,
  `later_capture_keeps_the_earlier_ceiling_on_its_selected_session`,
  `selected_session_ceiling_survives_restart_before_the_first_claim`,
  `work_completed_by_another_request_is_acknowledged_without_reprojection`,
  `overlapping_imports_keep_each_contexts_write_ceiling`,
  `shared_pending_work_resumes_under_the_lower_persisted_ceiling_without_reprojection`,
  `the_next_batch_refuses_an_insufficient_persisted_ceiling_and_keeps_its_request`,
  `completion_cleanup_keeps_saved_ceilings_including_shared_lower_limit`,
  `refused_only_restart_stops_preserves_refusal_and_serves_a_shutdown_arrival`,
  `older_completed_import_acknowledges_only_observed_refusal_identities`,
  `a_completed_newest_one_retry_clears_its_refusal_without_embedding_older_history`,
  `retired_refusal_remains_acknowledgeable_across_context_forget`,
  `new_narrow_background_work_preserves_an_incomplete_broader_refusal`.
- Deferred/caught-up imports leave active worker requests alone:
  `tests/import_embedding_workers.rs::deferred_import_never_creates_or_changes_the_active_worker_policy`
  and `already_caught_up_import_does_not_change_an_active_worker_request` in that file.
- Forget requires confirmation and preserves the Context:
  `tests/event_forget_terminal.rs::cg_forget_confirms_on_the_terminal_and_enter_changes_nothing`;
  domain erasure is guarded in `crates/context-graph/tests/event_forget.rs`. Forgetting an entire
  Context erases its captured session identities and frontiers while retaining a queued refusal's
  empty completion identity until a later caught-up import acknowledges that exact queued token
  and reclaims it.
  <!-- enforced by: `retired_refusal_remains_acknowledgeable_across_context_forget` -->
  With `event-search-test-support`,
  `tests/event_forget_terminal.rs::forget_keeps_the_reader_coordination_condition_when_erasure_is_refused`
  checks that a coordination failure after writable open keeps its typed JSON condition and
  leaves history unchanged.
- Environment reads match the declared inventory:
  `tests/every_environment_read_is_in_the_declared_inventory.rs::the_binary_reads_exactly_the_environment_it_declares`.
- Writable busy-store refusal is exit 4; reader-held exit code remains unsettled:
  `tests/a_busy_store_is_told_to_the_operator.rs::a_command_refused_by_a_writer_held_store_keeps_its_typed_answer`
  and `a_command_refused_by_a_reader_held_store_names_the_readers_it_waits_on` in that file.

- Read commands never open a writable Store: `tests/a_read_only_command_answers_a_store_a_writer_is_holding.rs`,
  `the_read_only_commands_answer_an_idle_store_and_a_held_one_without_changing_a_byte`.
- Validate budgets before store access: `tests/read_memory_policy.rs`,
  `invalid_read_config_is_a_config_error_before_status_can_open_or_write_the_store`.
- Worker requests retain selected session identities, capture frontiers, budgets and bounded retry
  allowances through restart. Completion acknowledges only verified request identities; later capture
  never extends an accepted frontier. Guards (feature-gated: they compile only with
  `--features event-search-test-support`, which the gate's `features` stage and CI's `features` job
  run; a default `cargo test` skips them): `tests/import_embedding_workers.rs`, especially
  `selected_session_ceiling_survives_restart_before_the_first_claim` and
  `older_completed_import_acknowledges_only_observed_refusal_identities`.
- Typed time/coordination refusals preserve committed coverage, permit independent work and remain
  inspectable until their own frontiers complete. Guard (same feature gate as above):
  `tests/import_embedding_workers.rs`:
  `consecutive_typed_coordination_refusals_exhaust_one_selected_request_allowance`.
- Busy stores use core `StoreBusy`, exit 4; reader-held conditions have no dedicated exit contract.
  Guards: `tests/a_busy_store_is_told_to_the_operator.rs`:
  `a_command_refused_by_a_writer_held_store_keeps_its_typed_answer` and
  `a_command_refused_by_a_reader_held_store_names_the_readers_it_waits_on`.
  Review-only: reader-held conditions have no dedicated numeric exit contract.
- Presented help is the public command list; guard
  `tests/help_presents_exactly_the_public_command_list.rs`: `help_presents_exactly_the_public_command_list`.

Worker fixtures use feature-gated Unix socket notifications in `tests/event_support/mod.rs`
and explicit release events from `cg-node-fs`. Always reread authoritative request state after
a notification. Worker-lock release precedes queue acknowledgment: await acknowledgment before
asserting queue membership. The reader-held-store refusal test enables the existing zero-allowance
test seam through development dependencies; ordinary CLI builds retain their configured allowance.

Fast check: `cargo test -p context-graph-cli --test read_memory_policy`.
Lifecycle proofs: `cargo nextest run -p context-graph-cli --features event-search-test-support`.
Full gate: `scripts/gate.sh`; public operator contracts: [CLI reference](../../docs/cli.md).
