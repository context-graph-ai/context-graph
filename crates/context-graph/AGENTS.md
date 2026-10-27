# context-graph

Purpose: embeddable structured memory and event-history search.
Owns ontology, Store behavior, event identity/checkpoints, projections, embedding registration,
bounded status and consumer reads, owner control, and fabric.
Must not own CLI rendering/exits, Claude JSON navigation, node config IO or a local understanding runtime.

## Seams and where a change lives

| Change | Entry point |
|---|---|
| Capture and deterministic identity | `src/store/event_import.rs`, `src/store/event.rs`; `EventImportAdapter` supplies drafts |
| Passage rendering, durable request selections, work policy, embedding cap and batch writer admission | `src/event_search.rs`, `src/store/event_search_projection.rs` |
| Find/status/forget | `src/store/event_search_read.rs`, `src/store/event_history_status.rs`, `src/store/event_forget.rs` |
| One SQL/decoder across read and write | `src/store/graph_rows.rs` (`GraphRows`); `src/consumer_read/graph.rs` |
| Application aggregates and declaration-filtered pages | `src/consumer_read/result.rs`; `ConsumerQueryResult`, `ConsumerPage` |
| Read session admission, memory and active time | `src/consumer_read.rs`, `src/owner_control.rs`, `src/status_reads.rs` |
| Writer budget / node probe | `src/write_budget.rs`; facts from `cg-node-fs` |
| Atomic startup schema and writable open | `src/store/mod.rs`, `src/schema.rs` |
| Embedding trait / Nomic / space binding | `src/embedder.rs`, `src/embedder/nomic.rs`, `src/embedding_space.rs` |
| Post-commit subscription receipts / typed Outcome snapshots | `src/store/subscriptions.rs`, `src/store/audit.rs` |

CLI and library hosts consume typed answers. ContextDB supplies `ReadSession`, owner transport,
cursors, snapshot/row visibility and typed refusals. `ConsumerReader::open_with_read_limits`
uses the declaration and engine gate; `Store` and `ConsumerGraphView` share `GraphRows` reads.
Query results expose borrowed engine data. `ConsumerQueryResult` owns the copies combined by
`GraphRows` and narrowed by the declaration gate. `ConsumerPage` retains an ordinary engine page
handle or owns a declaration-filtered copy; its equality and canonical row serialization use the
visible columns, rows and `has_more`. Those application copies do not become engine charges.
`request_owner_with_budgets` addresses administrative frames by store path; only typed owner
absence permits file fallback. In-process totals use the existing live database. The one writable
open carrying both capabilities is `Store::open_with_control_handler_and_embedder_registrations`.

## Invariants and guards

Ownership boundaries above are review-only architectural constraints unless a guard below covers them.

- Selection and source-stream status use the engine-maintained grouped event counts and
  indexed newest-event reads; erasure updates those engine-owned facts with captured events.
  Projection-job pending markers in `src/store/event_search_projection.rs` store only identity,
  target generation and a boolean. Capture sets them with events; cursor commit clears them
  only after a guarded newest-entry read. Selection and resume seek their declared pending index. Recent-session selection uses the
  declared recency index and grouped SQL window ranking; coverage probes selected progress keys.
  Guards: `tests/event_capture.rs::retries_are_noops_and_a_divergent_batch_leaves_events_and_checkpoint_unchanged`,
  `tests/event_search_projection_correctness.rs::select_work_examines_no_completed_events`,
  `tests/event_search_projection_correctness.rs::pending_marker_and_cursor_publish_together_across_process_crashes`,
  `tests/event_search_projection_correctness.rs::recent_session_selection_retains_only_the_selected_sessions`,
  `tests/event_search_projection_correctness.rs::recent_coverage_reads_only_selected_progress_keys`.
- Capture retries preserve identity and checkpoint atomicity:
  `tests/event_capture.rs::retries_are_noops_and_a_divergent_batch_leaves_events_and_checkpoint_unchanged`.
- Startup schema commits together: `tests/startup_schema_is_one_commit.rs::a_fresh_store_installs_its_whole_startup_schema_in_one_commit`.
- `Embedder` is the sole implementation trait; checked role preparation is library-sealed.
  Guard: `tests/acceptance/nomic_loads.rs::embedder_trait_and_storeconfig_public_surface_diff_keeps_surviving_signatures_and_allowed_moves`.
  Event-history space owns only `event_search_passages.vector_text`, guarded by
  `src/store/embedding.rs::canonical_event_history_declaration_has_only_the_passage_binding`.
- Session work scope is independent of vector identity; narrowing keeps indexed history:
  `tests/event_search_projection_correctness.rs::recent_session_scope_builds_the_ten_most_recent_newest_first_and_widens_without_reembedding`.
- Shared status distinguishes capture, coverage and live lease:
  `tests/event_history_status.rs::status_separates_capture_from_search_coverage_per_selected_context`.
- Coverage reads batch Contexts and use maintained counts, never per-Context recounts:
  `tests/a_find_costs_the_same_statements_at_any_context_count.rs::a_find_over_ready_contexts_issues_the_same_statements_at_one_eight_and_sixty_four`;
  `tests/a_read_that_can_see_the_counts_table_never_recounts_events.rs::status_find_and_the_work_assessment_answer_without_counting_a_single_event`.
- Read memory and per-statement active-time budgets reach both file and live owner routes.
  `StoreConfig.read_active_time_ms` is positive and defaults to 5,000 ms; omitting the field in
  serialized configuration restores that default. Cursor idle/lifetime limits stay fixed.
  Guards: `tests/read_memory_policy.rs::default_store_config_declares_the_one_gib_read_memory_ceiling` and
  `tests/read_memory_policy.rs::file_and_live_owner_reads_honor_per_instance_memory_budgets_without_losing_a_601_row_result`.
- A typed active-time refusal releases the failed projection read and preserves committed
  passages for continuation, without duplicate passage identities. Guard:
  `tests/event_search_projection_correctness.rs::a_typed_active_time_refusal_retains_committed_passages_and_resumes`.
- Projection retry retains claim and rendered work; failure is recorded only for its owner:
  `tests/event_search_projection_correctness.rs::projection_retries_real_writer_at_claim_renewal_and_commit_without_reembedding`
  and `post_commit_readback_refuses_a_competing_owner_and_a_later_run_completes` in that file.
- The worker cap bounds calls and retains the held Store's lease:
  `tests/event_search_embedding_parallelism.rs::held_store_cap_two_overlaps_two_calls_while_its_claim_lease_stays_live`.
- Request selections persist session identities and per-source capture frontiers in
  `src/store/event_search_projection.rs`; the bounded node worker queue carries their identities.
  Replacement selections also freeze previously published stream frontiers and the saved
  frontiers of the whole running request linked to the actual generation and claim owner, so
  retained searchable history survives a narrower profile rebuild after acceptance,
  without acquiring deferred captures.
  Standalone forget rebuilds queue their own selected frontiers alongside imports.
  Reaching a selected frontier publishes its searchable open tail even when deferred events
  follow; widening replaces that tail without duplicating current passages.
  Refused requests retain completion-only session frontiers after runnable selection retirement;
  the worker queue's exact identities govern their bounded-page reclamation. Context forget
  preserves empty completion headers until queued tokens are acknowledged.
  Completion is acknowledged before selected-work rows are retired. Forgetting one session
  preserves other sessions’ selections; writable queue reconciliation reclaims unpublished and
  acknowledged selections in bounded pages. CLI guards in
  `crates/cg-cli/tests/import_embedding_workers.rs`:
  `a_standalone_forget_rebuild_coexists_with_another_contexts_queued_import`,
  `profile_replacement_retains_published_frontiers_outside_the_newest_selection`,
  `profile_replacement_retains_an_earlier_queued_frontier_published_after_acceptance`,
  `forgetting_after_all_import_requests_are_acknowledged_rebuilds_surviving_history`,
  `forgetting_an_unrelated_session_preserves_the_queued_selection_and_ceiling`,
  `unpublished_selections_are_reclaimed_without_erasing_queued_work_after_reopen`,
  `a_deferred_append_stays_outside_the_accepted_frontier_even_when_it_would_fail`,
  `later_capture_keeps_the_earlier_ceiling_on_its_selected_session`,
  `selected_session_ceiling_survives_restart_before_the_first_claim` and
  `work_completed_by_another_request_is_acknowledged_without_reprojection`.
- Standalone projection batches resolve their writer ceiling before admission, release old
  writable state when it changes, and retain that ceiling through renewal, commit and failure
  recording. CLI guards in `crates/cg-cli/tests/import_embedding_workers.rs`:
  `overlapping_imports_keep_each_contexts_write_ceiling` and
  `the_next_batch_refuses_an_insufficient_persisted_ceiling_and_keeps_its_request`.
- Outcome timestamps decode identically through both doors:
  `tests/audit_outcome_typed_snapshot.rs::outcome_audit_snapshot_keeps_typed_committed_time_across_store_reopen_and_reader`.
- Subscription acknowledgement retries one action identity:
  `tests/acceptance/subscriptions.rs::lost_subscription_acknowledgement_retries_one_identity_and_one_receiver_effect`.
- Forget keeps Context/binding and refuses tenant-hub erasure:
  `tests/event_forget.rs::forgetting_a_session_erases_its_events_and_everything_derived_while_keeping_the_context`
  and `a_tenant_hub_store_is_refused_rather_than_purged_from_an_edge` in the same file.

- One embedding trait; checked role preparation and sealing belong to the library.
  Guard: `tests/acceptance/nomic_loads.rs`,
  `embedder_trait_and_storeconfig_public_surface_diff_keeps_surviving_signatures_and_allowed_moves`.
- Event-history space binds only passage vectors; guard in `src/store/embedding.rs`:
  `canonical_event_history_declaration_has_only_the_passage_binding`.
- One writable open carries the owner handler and embedders; guard
  `tests/one_writable_open_carries_the_owner_handler_and_the_embedders.rs`:
  `one_handle_serves_owner_frames_and_embeds_interleaved`.
- Owner requests use the substrate route addressed by store path.
  Guard: `tests/reading_pass_through_replaces_the_control_socket.rs`:
  `a_live_owner_answers_a_control_frame_addressed_only_by_the_store_path` and
  `a_store_nobody_holds_answers_owner_not_running_without_running_any_handler`.
  Review-only: direct fallback requires typed owner absence; timeout never licenses fallback.
- Status totals use bounded reads. Review-only: the only guard,
  `tests/status_totals_are_bounded_on_a_large_store.rs`:
  `status_totals_stay_correct_and_bounded_on_a_store_no_single_read_can_cross`, is `#[ignore]`d
  because it hangs, and neither the gate nor CI runs it.
  Status answers a store a writer holds exactly as it answers the idle store; guard
  `crates/cg-cli/tests/a_read_only_command_answers_a_store_a_writer_is_holding.rs`:
  `the_read_only_commands_answer_an_idle_store_and_a_held_one_without_changing_a_byte`.
  Failed folds close cursors; guard `tests/a_failed_status_fold_leaves_no_cursor_open.rs`:
  `a_fold_that_refuses_its_first_page_leaves_no_cursor_behind` and
  `a_fold_that_refuses_a_later_page_leaves_no_cursor_behind`.
  Review-only: a refusal must never be converted to a count.
- Consumer graph reads match writable answers and obey declarations.
  Guard: `tests/a_stopped_store_answers_the_typed_graph_reads_without_changing.rs`:
  `each_typed_graph_read_answers_what_the_writable_store_answered` and
  `a_graph_read_over_an_undeclared_table_is_refused_by_that_table_name`.
  Review-only: retain shared SQL/decoding in `src/store/graph_rows.rs`.
- Missing, recovery-required and unreadable stores have distinct typed outcomes.
  Guards: `tests/a_stopped_store_answers_the_typed_graph_reads_without_changing.rs`:
  `a_graph_read_of_a_store_that_is_not_there_is_the_typed_missing_outcome`;
  `tests/a_store_that_needs_a_writable_reopen_is_not_told_it_is_unreadable.rs`:
  `recoverable_and_unreadable_are_two_typed_answers_not_one`.
  Review-only: only recovery-required permits recovery.
- Capture derives identity and commits events/checkpoints atomically.
  Guards: `tests/event_capture.rs`:
  `append_derives_occurrence_identity_and_source_order_without_caller_control` and
  `retries_are_noops_and_a_divergent_batch_leaves_events_and_checkpoint_unchanged`.
  Retained history remains searchable as scope widens; guard
  `tests/event_search_projection_correctness.rs`:
  `recent_session_in_shared_source_survives_older_head_and_widening`.
- Review-only: disk access in the core is limited to optional read tracing and fresh-store removal in
  `src/store/mod.rs`, and feature-gated bundled artifact IO in `src/embedder/nomic.rs`.
  Ordinary node filesystem work belongs to `cg-node-fs`; adding core IO requires an explicit placement decision.

Embedding-route validation preserves typed lookup refusals; only an absent space yields
`UnknownEmbeddingSpace`. Guard: `src/store/observation.rs::tests::embedding_route_lookup_is_the_writes_own_and_keeps_its_typed_refusal`.
A write's own lookups (`Store::write_lookup` in `src/store/graph_rows.rs`) are statement reads, never
charged to the caller read budget, so a write during a reindex is not refused for time spent queued
behind its commits. Guard: `src/store/observation.rs::tests::observation_recorded_while_a_reindex_commit_holds_the_commit_boundary_is_not_refused_by_the_read_budget`.
For deterministic reads, adapt ContextDB's existing `DeadlineClock` through the test-only
`src/store/graph_rows.rs` seam. For background capture, use the synchronous test drive in
`src/capture_sync_status.rs`, with the canonical Wallclock guard on the executing thread.
Both timing-site arms in `tests/test_estate_audit.rs` include private test modules; only default-tier
test counting excludes them. Never park a fixture in a pre-commit callback while its sibling
needs an engine read fence; rendezvous before the real statements enter that commit phase.

Fast check: `cargo test -p context-graph --test event_capture --test read_memory_policy`.
Full gate: `scripts/gate.sh`. Public contracts live in [library API](../../docs/library-api.md)
and [CLI reference](../../docs/cli.md); preserve the README doctest shim in `src/lib.rs`.
