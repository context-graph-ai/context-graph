# Library API reference

| Area | Entry points | Details |
| --- | --- | --- |
| Graph memory | `Store`, `ScopedStore`, typed create/read/update requests | [Ontology](ontology.md) |
| Event capture | `append_event_batch`, `import_event_stream` | Event identity and checkpoints below |
| Search | `find_via_digests`, event passage search | [Models](models-and-embedders.md) |
| Embedding | `Embedder`, `EmbedderRegistration`, `EventSearchEmbedder` | [Register an embedder](register-an-embedder.md) |
| Read-only | `ConsumerReader`, `ConsumerGraphView` | [Read without changing a store](read-a-store-read-only.md) |
| Administrative reads | `owner_control::request_owner` | Store ownership below |
| Totals | `StatusReads` | Bounded totals below |

## Events and checkpoints

Append assigns source-stream sequence numbers in draft order. <!-- enforced by: `append_derives_occurrence_identity_and_source_order_without_caller_control` -->

The same source occurrence derives the same event id on another node. <!-- enforced by: `append_derives_occurrence_identity_and_source_order_without_caller_control` -->

Context identity participates in the derived event identity. <!-- enforced by: `append_derives_occurrence_identity_and_source_order_without_caller_control` -->

Repeating identical source occurrences inserts no new events. <!-- enforced by: `retries_are_noops_and_a_divergent_batch_leaves_events_and_checkpoint_unchanged` -->

Changed content under the same source identity returns `SourceEventDiverged`. <!-- enforced by: `retries_are_noops_and_a_divergent_batch_leaves_events_and_checkpoint_unchanged` -->

That divergence rolls back the batch's new event. <!-- enforced by: `retries_are_noops_and_a_divergent_batch_leaves_events_and_checkpoint_unchanged` -->

A caught-up import hashes the stored prefix without decoding it again. <!-- enforced by: `import_verifies_prefix_then_adds_only_tail_or_refuses_rewritten_history` -->

Rewriting a committed prefix returns `SourceHistoryDiverged`. <!-- enforced by: `import_verifies_prefix_then_adds_only_tail_or_refuses_rewritten_history` -->

The refusal leaves accepted events unchanged. <!-- enforced by: `import_verifies_prefix_then_adds_only_tail_or_refuses_rewritten_history` -->

An absent opaque binding key resolves to no Context. <!-- enforced by: `binding_is_absent_then_persists_across_reopen_and_same_context_is_a_noop` -->

Binding the same key to the same Context again is a no-op. <!-- enforced by: `binding_is_absent_then_persists_across_reopen_and_same_context_is_a_noop` -->

A binding persists across reopen. <!-- enforced by: `binding_is_absent_then_persists_across_reopen_and_same_context_is_a_noop` -->

A key already attached to another Context returns `ContextBindingConflict`. <!-- enforced by: `conflicting_or_missing_context_binding_changes_neither_binding_nor_context_rows` -->

The conflict preserves the binding rows. <!-- enforced by: `conflicting_or_missing_context_binding_changes_neither_binding_nor_context_rows` -->

The conflict preserves the Context rows. <!-- enforced by: `conflicting_or_missing_context_binding_changes_neither_binding_nor_context_rows` -->

A failed create-and-bind transaction leaves no new binding. <!-- enforced by: `create_and_bind_commits_success_and_rolls_back_on_a_real_second_write_constraint` -->

It rolls back the new Context. <!-- enforced by: `create_and_bind_commits_success_and_rolls_back_on_a_real_second_write_constraint` -->

The core preserves opaque namespace/reference pairs. <!-- enforced by: `opaque_binding_keys_are_full_node_local_pairs_in_local_and_tenant_bound_stores` -->

Context bindings remain `SYNC OFF` in tenant-bound stores. <!-- enforced by: `opaque_binding_keys_are_full_node_local_pairs_in_local_and_tenant_bound_stores` -->

Event reads return Store-assigned source sequence order. <!-- enforced by: `event_reads_are_context_scoped_and_return_source_sequence_order` -->

An event point read in a different Context returns no event. <!-- enforced by: `event_reads_are_context_scoped_and_return_source_sequence_order` -->

## Subscriptions

A new-Decision subscription identifies the committed Decision. <!-- enforced by: `subscribe_new_decision_fires_post_commit_carries_served_intention_ids_and_descriptions` -->

Its payload includes served Intention descriptions. <!-- enforced by: `subscribe_new_decision_fires_post_commit_carries_served_intention_ids_and_descriptions` -->

An Outcome-conflict subscription identifies threatened Intentions. <!-- enforced by: `subscribe_conflict_detected_fires_on_outcome_divergence_carries_older_newer_ids_and_intention_intersection` -->

A precedent-match subscription identifies the cited successful Decision. <!-- enforced by: `subscribe_precedent_match_fires_for_cites_walked_successful_precedent_carries_intention_descriptions` -->

Its payload includes served Intention descriptions. <!-- enforced by: `subscribe_precedent_match_fires_for_cites_walked_successful_precedent_carries_intention_descriptions` -->

An invalidation subscription identifies the affected Decision. <!-- enforced by: `invalidation_detected_silent_when_no_decisions_cite_entity_with_positive_control` -->

The Decision a new-Decision subscription delivers is readable after the store is reopened. <!-- enforced by: `subscribe_new_decision_fires_post_commit_carries_served_intention_ids_and_descriptions` -->

A rolled-back Observation transaction emits no subscription event. <!-- enforced by: `subscription_does_not_fire_when_observation_transaction_rolls_back` -->

A redelivered action with a saved receipt skips a second visible handler effect. <!-- enforced by: `a_redelivered_action_keeps_one_visible_effect_and_one_receipt_after_reopen` -->

The receipt remains single after reopen. <!-- enforced by: `a_redelivered_action_keeps_one_visible_effect_and_one_receipt_after_reopen` -->

Lost acknowledgement retries the same action identity. <!-- enforced by: `lost_subscription_acknowledgement_retries_one_identity_and_one_receiver_effect` -->

The subscription's `uncertain` metric counts the loss. <!-- enforced by: `lost_subscription_acknowledgement_retries_one_identity_and_one_receiver_effect` -->

Receiver deduplication preserves one visible effect. <!-- enforced by: `lost_subscription_acknowledgement_retries_one_identity_and_one_receiver_effect` -->

A handler error leaves delivery queued. <!-- enforced by: `external_handler_without_dedup_reports_uncertain_delivery` -->

## Store ownership

An owner request for an unheld store returns typed `OwnerNotRunning` naming that store. <!-- enforced by: `a_store_nobody_holds_answers_owner_not_running_and_leaves_its_file_alone` -->

The request leaves the store bytes unchanged. <!-- enforced by: `a_store_nobody_holds_answers_owner_not_running_and_leaves_its_file_alone` -->

It creates no companion files. <!-- enforced by: `a_store_nobody_holds_answers_owner_not_running_and_leaves_its_file_alone` -->

An unsupported inspection returns typed `OwnerRouteUnsupported`. <!-- enforced by: `owner_route_unsupported_classifies_to_a_typed_variant_naming_the_inspection` -->

That refusal is not owner absence. <!-- enforced by: `owner_route_unsupported_classifies_to_a_typed_variant_naming_the_inspection` -->

One writable handle serves owner frames in order. <!-- enforced by: `one_handle_serves_owner_frames_and_embeds_interleaved` -->

The same handle embeds each recorded text through its registered embedder. <!-- enforced by: `one_handle_serves_owner_frames_and_embeds_interleaved` -->

## Bounded totals

A status fold that fails on a later page closes its cursor. <!-- enforced by: `a_fold_that_refuses_a_later_page_leaves_no_cursor_behind` -->

A refused CLI status read reports `ok:false`. <!-- enforced by: `a_refused_status_read_is_reported_as_a_refusal_and_never_as_an_answer` -->

It does not render unestablished counts. <!-- enforced by: `a_refused_status_read_is_reported_as_a_refusal_and_never_as_an_answer` -->

Budgets: [configuration](configuration.md#read-budgets). First write: [library quickstart](library-quickstart.md).

## Consumer graph reads

`ConsumerReader::graph()` returns the writable Store's audit answer. <!-- enforced by: `each_typed_graph_read_answers_what_the_writable_store_answered` -->

It returns the same Context list. <!-- enforced by: `each_typed_graph_read_answers_what_the_writable_store_answered` -->

It returns the same Entity. <!-- enforced by: `each_typed_graph_read_answers_what_the_writable_store_answered` -->

It returns the same Intention. <!-- enforced by: `each_typed_graph_read_answers_what_the_writable_store_answered` -->

It returns the same Decision with its nested relationships. <!-- enforced by: `each_typed_graph_read_answers_what_the_writable_store_answered` -->

It returns the same Observation. <!-- enforced by: `each_typed_graph_read_answers_what_the_writable_store_answered` -->

The full graph read set preserves the store bytes. <!-- enforced by: `the_whole_graph_read_set_leaves_the_stopped_store_byte_identical` -->

It preserves companion files and the containing directory. <!-- enforced by: `the_whole_graph_read_set_leaves_the_stopped_store_byte_identical` -->

Paged Observation lists remain complete. <!-- enforced by: `an_observation_list_bigger_than_one_answer_is_the_writable_store_s_list` -->

Paged audit trails remain complete. <!-- enforced by: `an_audit_trail_bigger_than_one_answer_is_the_writable_store_s_trail` -->

Outcome audit rows retain their committed times in the typed timestamp column. <!-- enforced by: `outcome_audit_snapshot_keeps_typed_committed_time_across_store_reopen_and_reader` -->

The JSON snapshot has no duplicate `recorded_at` field. <!-- enforced by: `outcome_audit_snapshot_keeps_typed_committed_time_across_store_reopen_and_reader` -->

The read-only audit answer matches the writable answer. <!-- enforced by: `outcome_audit_snapshot_keeps_typed_committed_time_across_store_reopen_and_reader` -->

An undeclared-table read is refused naming that table. <!-- enforced by: `a_query_reaching_a_table_the_consumer_did_not_declare_is_refused` -->

Joins and subqueries cannot bypass that refusal. <!-- enforced by: `an_undeclared_table_reached_through_a_join_or_a_subquery_is_still_refused` -->

A truthful declaration returns the same rows through the direct and live-owner routes. <!-- enforced by: `a_truthful_declaration_answers_the_same_rows_on_the_direct_and_the_live_owner_routes` -->

A consumer audit read without declared `audit_log` is refused. <!-- enforced by: `a_graph_read_over_an_undeclared_table_is_refused_by_that_table_name` -->

A consumer Context list without declared `contexts` is refused. <!-- enforced by: `a_graph_read_over_an_undeclared_table_is_refused_by_that_table_name` -->

A consumer Entity read without declared `entities` is refused. <!-- enforced by: `a_graph_read_over_an_undeclared_table_is_refused_by_that_table_name` -->

A consumer Intention read without declared `intentions` is refused. <!-- enforced by: `a_graph_read_over_an_undeclared_table_is_refused_by_that_table_name` -->

A consumer Decision read without declared `decisions` is refused. <!-- enforced by: `a_graph_read_over_an_undeclared_table_is_refused_by_that_table_name` -->

A consumer Decision read without declared `edges` is refused. <!-- enforced by: `a_graph_read_over_an_undeclared_table_is_refused_by_that_table_name` -->

A consumer Observation read without declared `observations` is refused. <!-- enforced by: `a_graph_read_over_an_undeclared_table_is_refused_by_that_table_name` -->

A consumer Observation read without declared `evidence` is refused. <!-- enforced by: `a_graph_read_over_an_undeclared_table_is_refused_by_that_table_name` -->

A Decision read without a declared `edges` table refuses that table. <!-- enforced by: `a_graph_read_over_an_undeclared_table_is_refused_by_that_table_name` -->

An Observation read without declared evidence refuses the `evidence` table. <!-- enforced by: `a_graph_read_over_an_undeclared_table_is_refused_by_that_table_name` -->

The reader's inventory exposes only its declared tables. <!-- enforced by: `the_readers_inventory_exposes_nothing_beyond_its_declaration` -->

## Library projection workers

A held-Store worker cap of two admits no third overlapping embedding call. <!-- enforced by: `held_store_cap_two_overlaps_two_calls_while_its_claim_lease_stays_live` -->

Its projection lease stays live during the calls. <!-- enforced by: `held_store_cap_two_overlaps_two_calls_while_its_claim_lease_stays_live` -->

A held-Store worker cap of one serializes embedding calls. <!-- enforced by: `held_store_cap_one_serializes_calls_while_its_claim_lease_stays_live` -->

Its projection lease stays live during the call. <!-- enforced by: `held_store_cap_one_serializes_calls_while_its_claim_lease_stays_live` -->

## Library read budgets

A live-owner read with custom per-instance budgets returns the complete 601-row answer. <!-- enforced by: `file_and_live_owner_reads_honor_per_instance_memory_budgets_without_losing_a_601_row_result` -->

A file read with its own budgets also returns the complete answer. <!-- enforced by: `file_and_live_owner_reads_honor_per_instance_memory_budgets_without_losing_a_601_row_result` -->

Read session options built with custom memory and active-time budgets keep a five-minute cursor idle ceiling. <!-- enforced by: `default_store_config_declares_the_one_gib_read_memory_ceiling` -->

They keep a thirty-minute cursor lifetime ceiling. <!-- enforced by: `default_store_config_declares_the_one_gib_read_memory_ceiling` -->

## Projection assessment

Requested projection assessment excludes coverage with the wrong projection identity. <!-- enforced by: `requested_projection_assessment_uses_full_identity_on_store_and_actual_consumer_reader_without_artifacts` -->

ConsumerReader and Store assessments report equal covered counts. <!-- enforced by: `requested_projection_assessment_uses_full_identity_on_store_and_actual_consumer_reader_without_artifacts` -->

They report equal pending counts. <!-- enforced by: `requested_projection_assessment_uses_full_identity_on_store_and_actual_consumer_reader_without_artifacts` -->

Consumer assessment creates no schema, model or projection artifacts. <!-- enforced by: `requested_projection_assessment_uses_full_identity_on_store_and_actual_consumer_reader_without_artifacts` -->

## Event-search slice bound

The default external-embedder slice token-loss bound is `None`. <!-- enforced by: `external_embedder_implements_preparation_through_the_hook_and_produces_a_prepared_value` -->

Bundled Nomic declares `Some(200)` for the label `User: `. <!-- enforced by: `nomic_document_preparer_adds_exactly_one_system_task_prefix` -->

It declares no bound for `User:` without a trailing space. <!-- enforced by: `nomic_document_preparer_adds_exactly_one_system_task_prefix` -->

Document boundaries equal the cached tokenizer's content offsets. <!-- enforced by: `oversized_event_split_uses_selected_tokenizer_and_has_utf8_safe_exact_bidirectional_source_offsets` -->

Those boundaries land on UTF-8 character boundaries. <!-- enforced by: `oversized_event_split_uses_selected_tokenizer_and_has_utf8_safe_exact_bidirectional_source_offsets` -->

Splitting at the returned boundaries preserves the complete Unicode source. <!-- enforced by: `oversized_event_split_uses_selected_tokenizer_and_has_utf8_safe_exact_bidirectional_source_offsets` -->