# Privacy and data movement

## The standalone path

The default library event placement is local-only. <!-- enforced by: `local_and_unbound_tenant_placement_keep_event_schema_and_append_behind_real_policy_binding` -->

A default import completes capture even when local model artifacts are missing. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

A default import downloads no missing model. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

Explicit default prefetch installs the local Nomic artifact. <!-- enforced by: `prefetch_defaults_writes_artifacts_on_disk_for_apache_models_only` -->

With the local Nomic artifact missing, the import reports search as blocked, names `cg model fetch --defaults` as the retry, and leaves the model cache empty. <!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

## Hosted calls outside the history journey

Default local event placement disables delivery. <!-- enforced by: `local_and_unbound_tenant_placement_keep_event_schema_and_append_behind_real_policy_binding` -->

Conversation condensation makes no paid backend call without recorded consent. <!-- enforced by: `paid_understand_call_is_refused_unless_consent_is_on_record_for_the_connection` -->

One consented condensation request invokes the paid backend once. <!-- enforced by: `paid_understand_call_is_refused_unless_consent_is_on_record_for_the_connection` -->

The permanent raw store declares `SYNC OFF`. <!-- enforced by: `permanent_raw_store_never_syncs` -->

Hosted calls use credentials resolved through the named model connection. <!-- enforced by: `llm_client_hosted_implementation_pulls_credentials_via_principal_aware_config_not_global_env` -->

The runtime does not consult per-feature credential environment variables. <!-- enforced by: `llm_client_hosted_implementation_pulls_credentials_via_principal_aware_config_not_global_env` -->

Revoking consent prevents the next paid backend call. <!-- enforced by: `consent_revoked_between_calls_is_rechecked_on_every_paid_call` -->

Restoring consent permits the next paid backend call. <!-- enforced by: `consent_revoked_between_calls_is_rechecked_on_every_paid_call` -->

Supplied tokens live behind credential references. <!-- enforced by: `cli_connect_then_status_then_selftest_success_state_exit_zero_and_json_stable` -->

Model registration metadata contains no supplied token. <!-- enforced by: `cli_connect_then_status_then_selftest_success_state_exit_zero_and_json_stable` -->

The catalog file contains no raw supplied token. <!-- enforced by: `cli_connect_then_status_then_selftest_success_state_exit_zero_and_json_stable` -->

## Where your data lives

Normalized events default to local-only placement. <!-- enforced by: `local_and_unbound_tenant_placement_keep_event_schema_and_append_behind_real_policy_binding` -->

Unbound tenant placement refuses Context creation until policy binding. <!-- enforced by: `local_and_unbound_tenant_placement_keep_event_schema_and_append_behind_real_policy_binding` -->

It refuses event append until policy binding. <!-- enforced by: `local_and_unbound_tenant_placement_keep_event_schema_and_append_behind_real_policy_binding` -->

Tenant placement keeps Context bindings node-local. <!-- enforced by: `opaque_binding_keys_are_full_node_local_pairs_in_local_and_tenant_bound_stores` -->

With no data-root override, the store is `~/.cg/context-graph.db`. <!-- enforced by: `with_no_data_root_override_the_store_lives_at_dot_cg_under_home` -->

Show reads the exact events linked by a passage. <!-- enforced by: `search_wait_round_trip_reads_linked_exact_events_from_the_store_with_real_nomic` -->

It does not need the original session file. <!-- enforced by: `search_wait_round_trip_reads_linked_exact_events_from_the_store_with_real_nomic` -->

<!-- concept -->
For selected-history erasure, follow the [forget recipe](../skills/cg-forget/SKILL.md). For full local removal, follow [local removal](troubleshooting.md#remove-local-data).

For where model artifacts are stored, see [cache layout](models-and-embedders.md#cache-layout).

Handle provider-side deletion separately under your provider’s terms.

See [forget recipe](../skills/cg-forget/SKILL.md) and [security reporting](../SECURITY.md).
