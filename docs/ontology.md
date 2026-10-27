# A memory of why

A Decision retains its frozen Entity basis after an Entity update. <!-- enforced by: `decision_based_on_snapshot_frozen_after_entity_update` -->

A later basis-changing Observation creates an Invalidation naming the affected Decision. <!-- enforced by: `invalidation_created_by_engine_on_basis_change` -->

The Invalidation names the triggering Observation. <!-- enforced by: `invalidation_created_by_engine_on_basis_change` -->

<!-- concept -->
Consider a memory saying “the auth module is at src/auth.py.” After a rename, that sentence can remain a close semantic match to a search for authentication. The missing question is whether the choice that used that path still has a valid basis.

## Two layers: what was said, and what it means

See [event identity and checkpoints](library-api.md#events-and-checkpoints).

A Decision retains its serving Intention ids. <!-- enforced by: `decision_create_with_serves_and_based_on_edges_persisted` -->

It retains its Entity basis snapshots. <!-- enforced by: `decision_create_with_serves_and_based_on_edges_persisted` -->

A Digest cannot be overwritten. <!-- enforced by: `digest_immutable_no_update_symbol` -->

Digest search returns linked Intentions. <!-- enforced by: `find_via_digests_traverses_extracted_from_to_derived_intentions_and_decisions` -->

It returns linked Decisions. <!-- enforced by: `find_via_digests_traverses_extracted_from_to_derived_intentions_and_decisions` -->

It returns linked Observations. <!-- enforced by: `find_via_digests_traverses_extracted_from_to_derived_intentions_and_decisions` -->

<!-- concept -->
See the [current tool limits](../README.md#current-limits) before choosing between history search and library graph writes.

```text
Intention <- SERVES -- Decision -- BASED_ON -> Entity snapshot
                         |                         ^
                        CITES                     |
                         v                    Observation
                    prior Decision                 |
                         |                     Invalidation
                       Outcome
```

## Goals outlive the choices that serve them

Creating a Decision without an Intention returns `MissingIntention`. <!-- enforced by: `decision_create_requires_intention` -->

A Decision's type cannot be changed. <!-- enforced by: `decision_update_immutable_fields_rejected` -->

Its description cannot be changed. <!-- enforced by: `decision_update_immutable_fields_rejected` -->

Superseding a Decision retires the old choice. <!-- enforced by: `decision_supersede_creates_edge_and_transitions_status` -->

It links the replacement. <!-- enforced by: `decision_supersede_creates_edge_and_transitions_status` -->

An active Decision prevents its Intention from being archived. <!-- enforced by: `intention_archive_blocked_by_active_decision` -->

Earlier Intention wording remains readable at the Decision's creation transaction. <!-- enforced by: `intention_prior_language_queryable_at_any_txid` -->

## What a choice relied on

Entity types include File and Service. <!-- enforced by: `test_entity_type_variants` -->

Entity updates append a snapshot history that survives reopen. <!-- enforced by: `entity_update_appends_immutable_snapshot_chain_after_reopen` -->

Earlier snapshot properties remain unchanged. <!-- enforced by: `entity_update_appends_immutable_snapshot_chain_after_reopen` -->

Archiving retains Entity history. <!-- enforced by: `entity_archive_retains_history` -->

Entity updates do not rewrite a Decision's basis snapshots. <!-- enforced by: `decision_based_on_snapshot_frozen_after_entity_update` -->

An unchanged selected basis field produces no Invalidation. <!-- enforced by: `impact_analysis_basis_field_listed_but_unchanged_no_invalidation` -->

Changing that selected field creates an Invalidation. <!-- enforced by: `impact_analysis_basis_field_listed_but_unchanged_no_invalidation` -->

A selected field's first appearance can invalidate the Decision. <!-- enforced by: `impact_analysis_basis_fields_listed_field_absent_at_decision_time_triggers_on_first_appearance` -->

## What happened next

Recorded evidence links to its Observation. <!-- enforced by: `combination_record_observation_observation_evidence_snapshot_embedding_atomic_one_tx` -->

The new snapshot shares the Observation's committed time. <!-- enforced by: `combination_record_observation_observation_evidence_snapshot_embedding_atomic_one_tx` -->

The evidence embedding shares that committed time. <!-- enforced by: `combination_record_observation_observation_evidence_snapshot_embedding_atomic_one_tx` -->

A no-op state delta creates no snapshot. <!-- enforced by: `record_observation_state_delta_no_op_merge_does_not_create_snapshot_real_change_does` -->

A real state change adds one snapshot. <!-- enforced by: `record_observation_state_delta_no_op_merge_does_not_create_snapshot_real_change_does` -->

The new snapshot links back to the Observation. <!-- enforced by: `record_observation_state_delta_real_change_creates_snapshot_with_evidenced_by_pointing_back` -->

Observations cannot be overwritten. <!-- enforced by: `observation_immutable_mutation_rejected` -->

Adding a text projection preserves the evidence's original media kind. <!-- enforced by: `record_observation_evidence_text_projection_does_not_replace_original_kind` -->

Replay reports expired evidence retention status. <!-- enforced by: `replay_survives_media_expiry_returns_lineage_with_retention_status_expired` -->

The core has no source-reference-keyed filesystem read path. <!-- enforced by: `cg_library_crate_context_graph_carries_no_outbound_http_or_subprocess_or_filesystem_source_ref_dereference_path` -->

## When the world moves under a choice

Impact analysis persists the Invalidations returned by Observation recording. <!-- enforced by: `impact_analysis_direct_invalidation_by_engine` -->

An Invalidation identifies the affected Decision. <!-- enforced by: `invalidation_created_by_engine_on_basis_change` -->

It identifies the triggering Observation. <!-- enforced by: `invalidation_created_by_engine_on_basis_change` -->

It reports the changed basis field. <!-- enforced by: `invalidation_created_by_engine_on_basis_change` -->

It identifies threatened Intentions. <!-- enforced by: `invalidation_created_by_engine_on_basis_change` -->

An unchanged basis creates no Invalidation. <!-- enforced by: `impact_analysis_no_basis_change_no_invalidation` -->

Impact analysis excludes inactive Decisions. <!-- enforced by: `impact_analysis_respects_decision_status` -->

It leaves other Contexts' Invalidations unchanged. <!-- enforced by: `impact_analysis_per_context_isolation` -->

Low-confidence basis changes produce Info severity. <!-- enforced by: `impact_analysis_severity_info_for_low_confidence` -->

Middle-confidence basis changes produce Warning severity. <!-- enforced by: `impact_analysis_severity_warning_for_mid_confidence` -->

Unrated basis changes produce Warning severity. <!-- enforced by: `impact_analysis_severity_warning_for_unrated_decision` -->

A sole server of at least one threatened Intention can produce Critical severity. <!-- enforced by: `impact_analysis_severity_critical_when_sole_server_on_at_least_one_threatened_intention` -->

A stronger active sibling can reduce severity to Warning. <!-- enforced by: `impact_analysis_severity_downgraded_when_higher_confidence_sibling_exists` -->

A citing Decision receives an Info-level cascade Invalidation. <!-- enforced by: `impact_analysis_cascade_via_cites` -->

The cascade stops after ten hops. <!-- enforced by: `cg_cascade_depth_cap_below_engine_cap` -->

It does not descend through an inactive Decision. <!-- enforced by: `impact_analysis_cascade_skips_non_active_decisions_and_does_not_descend_through_them` -->

A cascade row names the citing Decision's own threatened Intentions. <!-- enforced by: `cascade_row_threatens_citing_decisions_own_intentions_under_fan_in` -->

A file-path change from `src/auth.py` to `src/authentication.py` appears in the Invalidation's basis difference. <!-- enforced by: `invalidation_created_by_engine_on_basis_change` -->

```text
Application records an Intention: keep authentication working.
Application records an Entity snapshot: path = src/auth.py.
Application records a Decision based on that snapshot.
Application records an Observation: path = src/authentication.py.
Impact analysis records an Invalidation:
  basis path:   src/auth.py
  current path: src/authentication.py
  threatened goal: keep authentication working
```

## Answering an invalidation and recording outcomes

Acknowledgement moves a pending Invalidation to acknowledged. <!-- enforced by: `acknowledge_invalidation_pending_to_acknowledged` -->

Resolving a pending Invalidation is refused. <!-- enforced by: `resolve_invalidation_on_pending_row_refused` -->

An acknowledged Invalidation can resolve with a replacement Decision. <!-- enforced by: `resolve_invalidation_acknowledged_with_new_decision_succeeds` -->

The replacement must serve a threatened Intention. <!-- enforced by: `resolve_invalidation_with_decision_not_serving_threatened_intention_refused` -->

An acknowledged Invalidation can be dismissed. <!-- enforced by: `dismiss_invalidation_acknowledged_with_reason_succeeds` -->

Terminal Invalidations cannot move back to acknowledged. <!-- enforced by: `regression_invalidation_status_cannot_go_backward_full_sweep` -->

An invalidated Decision cannot return to active. <!-- enforced by: `combination_decision_status_state_machine_invalidated_to_active_rejected` -->

“Basis still valid” is refused for an invalidated Decision. <!-- enforced by: `resolve_invalidation_basis_still_valid_on_invalidated_decision_refused` -->

A Decision has one current Outcome relationship. <!-- enforced by: `outcome_record_idempotent` -->

That single relationship survives synchronized contention and restart. <!-- enforced by: `contended_outcome_keeps_one_virtual_relationship_after_sync_restart_and_restore` -->

It survives restore. <!-- enforced by: `contended_outcome_keeps_one_virtual_relationship_after_sync_restart_and_restore` -->

Evidence-derived confidence distinguishes Decisions with different Outcome and Observation histories. <!-- enforced by: `decision_confidence_becomes_evidence_derived_after_outcomes_accrue` -->

## Asking what is connected

Intention discovery returns serving Decisions. <!-- enforced by: `by_intention_returns_serving_decisions_active_only` -->

Entity discovery returns connected Decisions. <!-- enforced by: `by_entity_returns_connected_decisions_active_only` -->

Entity discovery returns connected Intentions. <!-- enforced by: `by_entity_returns_connected_intentions` -->

The default discovery filter excludes inactive Decisions. <!-- enforced by: `discovery_default_active_filter_excludes_invalidated_superseded_archived` -->

Traversal refuses depths above ten. <!-- enforced by: `traverse_depth_cap_rejects_over_ten` -->

Traversal terminates a cycle without duplicate nodes. <!-- enforced by: `traverse_cycle_terminates_without_duplicates` -->

Historical Entity reads return the earlier snapshot properties. <!-- enforced by: `state_at_returns_historical_snapshot_for_each_moment` -->

Replay reports the Decision's creation-time basis value. <!-- enforced by: `replay_uses_creation_time_snapshot_not_current` -->

It separately reports the current value. <!-- enforced by: `replay_uses_creation_time_snapshot_not_current` -->

Replay includes the Intention wording at Decision creation. <!-- enforced by: `replay_includes_cited_intention_wording_at_decision_time` -->

## Being told instead of asking

<!-- concept -->
See [subscription delivery](library-api.md#subscriptions) for event contents, commit boundaries, retries and external receiver responsibilities.

## Walls you choose

Entity lists do not expose another Context's entities. <!-- enforced by: `context_isolation_entity_not_visible_cross_context` -->

A cross-Context Observation is refused. <!-- enforced by: `context_isolation_observation_leak_blocked` -->

Traversal excludes another Context's nodes. <!-- enforced by: `graph_operations_respect_context_isolation` -->

Impact analysis leaves other Contexts unchanged. <!-- enforced by: `impact_analysis_per_context_isolation` -->

Opt-in multi-Context Digest search labels each result with its Context. <!-- enforced by: `find_via_digests_cross_context_opt_in_returns_union_with_per_row_context_label` -->

## Vectors stay inside their space

Conflicting model identity declarations are refused. <!-- enforced by: `store_open_extra_embedding_spaces_conflicting_model_identity_returns_embedding_space_mismatch_naming_field` -->

Conflicting dimensions are refused. <!-- enforced by: `store_open_extra_embedding_spaces_conflicting_dimension_returns_embedding_space_mismatch_naming_field` -->

Conflicting metrics are refused. <!-- enforced by: `store_open_extra_embedding_spaces_conflicting_metric_returns_embedding_space_mismatch_naming_field` -->

Different embedding spaces are not compared merely because their dimensions match. <!-- enforced by: `dedicated_event_search_space_refuses_old_unlabelled_space_comparison` -->

Digests refuse non-text evidence kinds. <!-- enforced by: `create_digest_with_non_text_evidence_kind_rejected_with_unsupported_evidence_kind` -->

## Known limitations

An invalidated Decision cannot be reactivated. <!-- enforced by: `combination_decision_status_state_machine_invalidated_to_active_rejected` -->

It cannot be resolved with “basis still valid.” <!-- enforced by: `resolve_invalidation_basis_still_valid_on_invalidated_decision_refused` -->

A failed duplicate Observation transaction rolls back its auto-created Entity. <!-- enforced by: `record_observation_duplicate_observation_id_rolls_back_auto_created_context_and_entity` -->

It rolls back its auto-created Context. <!-- enforced by: `record_observation_duplicate_observation_id_rolls_back_auto_created_context_and_entity` -->

Inline data-URI media is refused as `RawMediaNotStored`. <!-- enforced by: `record_observation_rejects_inline_data_uri_with_raw_media_not_stored_before_any_db_write` -->

The refusal leaves no Observation row. <!-- enforced by: `record_observation_rejects_inline_data_uri_with_raw_media_not_stored_before_any_db_write` -->

Inline evidence media payloads are refused. <!-- enforced by: `record_observation_rejects_inline_data_uri_with_raw_media_not_stored_before_any_db_write` -->

Base64 payload keys are refused. <!-- enforced by: `record_observation_rejects_inline_data_uri_with_raw_media_not_stored_before_any_db_write` -->

Evidence accepts s3, gs, azure, cas and webhook source references. <!-- enforced by: `store_open_camera_source_s3_gs_azure_cas_webhook_schemes_accepted_unconditionally` -->

HTTPS references are denied by default. <!-- enforced by: `store_open_https_scheme_accepted_only_when_host_in_allowlist_case_insensitive_per_rfc3986` -->

An allowed HTTPS host is accepted. <!-- enforced by: `store_open_https_scheme_accepted_only_when_host_in_allowlist_case_insensitive_per_rfc3986` -->

File references are accepted when file URIs are allowed. <!-- enforced by: `store_open_file_uri_accepted_only_when_file_uri_allowed_true` -->

They are refused without that permission. <!-- enforced by: `store_open_file_uri_accepted_only_when_file_uri_allowed_true` -->

Blob, javascript, vbscript and about schemes are refused. <!-- enforced by: `record_observation_rejects_blob_javascript_vbscript_about_schemes_with_invalid_source_ref_scheme` -->

No public Store method or its local helpers names signatures, blueprints or approvals in read or write SQL. <!-- enforced by: `no_public_store_method_names_the_signature_blueprint_or_approval_tables` -->

## What it is not

<!-- concept -->
The ontology is a model for remembering goals, choices, their evidence and consequences. It is not a general-purpose graph vocabulary, a project-management method or a replacement for operational monitoring.

Passage search returns stored source text. <!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

The evidence API refuses inline media payloads. <!-- enforced by: `record_observation_rejects_inline_data_uri_with_raw_media_not_stored_before_any_db_write` -->

## Where next

<!-- concept -->
Start with the [library quickstart](library-quickstart.md), consult the [library API](library-api.md), or follow the [command recipes](cli.md#recipes) for the evidence layer.
