# `context-graph`

Purpose: embed durable graph memory, immutable event capture and derived semantic retrieval.

**Owns:** ontology/schema, event identity/checkpoints/placement, opaque Context bindings, projection,
interpretation, erasure, typed shared reads, read/write budgets, owner control, fabric and embedders.
**Must not own:** CLI exits, provider JSON, ordinary node filesystem IO, a local understanding
runtime, a second embedding trait or a second transport to the running owner.

## Seams and routing

| Change | Module / entry point |
|---|---|
| Capture and deterministic identity | `src/store/event_import.rs`, `src/store/event.rs`; `EventImportAdapter` supplies drafts |
| Passage rendering, durable request selections, work policy, embedding cap and batch writer admission | `src/event_search.rs`, `src/store/event_search_projection.rs` |
| Find/status/forget | `src/store/event_search_read.rs`, `src/store/event_history_status.rs`, `src/store/event_forget.rs` |
| One SQL/decoder across read and write | `src/store/graph_rows.rs` (`GraphRows`); `src/consumer_read/graph.rs` |
| Read session admission, memory and active time | `src/consumer_read.rs`, `src/owner_control.rs`, `src/status_reads.rs` |
| Writer budget / node probe | `src/write_budget.rs`; facts from `cg-node-fs` |
| Atomic startup schema and writable open | `src/store/mod.rs`, `src/schema.rs` |
| Embedding trait / Nomic / space binding | `src/embedder.rs`, `src/embedder/nomic.rs`, `src/embedding_space.rs` |
| Post-commit subscription receipts / typed Outcome snapshots | `src/store/subscriptions.rs`, `src/store/audit.rs` |

CLI and library hosts consume typed answers. ContextDB supplies `ReadSession`, owner transport,
cursors, snapshot/row visibility and typed refusals. `ConsumerReader::open_with_read_limits`
uses the declaration and engine gate; `Store` and `ConsumerGraphView` share `GraphRows` reads.
`request_owner_with_budgets` addresses administrative frames by store path; only typed owner
absence permits file fallback. In-process totals use the existing live database. The one writable
open carrying both capabilities is `Store::open_with_control_handler_and_embedder_registrations`.

## Invariants and guards

- The derived `event_search_passages` table declares `HISTORY CURRENT ONLY SYNC OFF`, so superseded passage versions from tail retirement and session publication are reclaimed by maintenance while immutable captured-event and session history remain intact: `tests/event_search_projection_correctness.rs::old_conversations_remain_searchable_after_superseded_passage_reclaim`.

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
- Projection retry retains claim and rendered work; failure is recorded only for its owner.
  Linux file-store claims hold kernel presence across Store closes; a dead process releases it
  so the next importer can reclaim its unexpired lease. Memory stores retain TTL fencing, and
  epoch/token checks continue to reject results from superseded owners. Guards:
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

Further shared owner/read invariants and their guards: root `AGENTS.md`.
Fast check: `cargo test -p context-graph --test event_history_read_contracts --test read_memory_policy`.
Full gate: `scripts/gate.sh`; update claim bindings and the test-estate baseline with changed proofs.
