# context-graph

Purpose: embeddable structured memory and event-history search.
Owns ontology, Store behavior, event identity/checkpoints, projections, embedding registration,
bounded status and consumer reads, owner control, and fabric.
Must not own CLI rendering/exits, Claude JSON navigation, node config IO or a local understanding runtime.

## Seams and where a change lives

| Change | Entry point |
|---|---|
| Memory ontology | `src/types.rs`, `src/schema.rs`, `src/store/` |
| Event capture and identity | `src/store/event_import.rs`, `src/store/event.rs` |
| Projection, leases and selections | `src/store/event_search_projection.rs`, `src/event_search.rs` |
| Shared typed graph SQL/decoding | `src/store/graph_rows.rs` |
| Consumer read declaration and executor | `src/consumer_read.rs`, `src/consumer_read/graph.rs` |
| Administrative owner requests | `src/owner_control.rs` |
| In-process status totals | `src/status_reads.rs` |
| Embedding trait and spaces | `src/embedder.rs`, `src/embedding_space.rs` |
| Bundled Nomic and its cache root | `src/embedder/nomic.rs` under default `nomic-embedder` feature |

## Invariants and guards

Ownership boundaries above are review-only architectural constraints unless a guard below covers them.

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

Fast check: `cargo test -p context-graph --test event_capture --test read_memory_policy`.
Full gate: `scripts/gate.sh`. Public contracts live in [library API](../../docs/library-api.md)
and [CLI reference](../../docs/cli.md); preserve the README doctest shim in `src/lib.rs`.
