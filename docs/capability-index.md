# Capability index

See the [current tool limits](../README.md#current-limits) and the surface index below.

| Need | Guide |
| --- | --- |
| Capture history | [Import](cli.md#cg-import) |
| Find evidence | [Find and show](cli.md#cg-find-and-cg-show) |
| Inspect memory | [Status](cli.md#cg-status) |
| Erase evidence | [Forget](cli.md#cg-forget) |
| Inspect and prepare models | [Models and diagnostics](cli.md#models-and-diagnostics) |
| Record graph knowledge | [Ontology](ontology.md) |
| Read without mutation | [Read-only guide](read-a-store-read-only.md) |

The library's default search build processes the ten newest sessions first. <!-- enforced by: `recent_session_scope_builds_the_ten_most_recent_newest_first_and_widens_without_reembedding` -->

Captured sessions outside that selection remain stored. <!-- enforced by: `recent_session_scope_builds_the_ten_most_recent_newest_first_and_widens_without_reembedding` -->

Widening to all sessions embeds only the additional selected sessions. <!-- enforced by: `recent_session_scope_builds_the_ten_most_recent_newest_first_and_widens_without_reembedding` -->

Narrowing keeps the already indexed sessions searchable. <!-- enforced by: `recent_session_scope_builds_the_ten_most_recent_newest_first_and_widens_without_reembedding` -->

The default library read memory limit is 1 GiB. <!-- enforced by: `default_store_config_declares_the_one_gib_read_memory_ceiling` -->

The default CLI read active-time allowance is 5,000 ms. <!-- enforced by: `read_active_time_defaults_to_five_seconds_and_accepts_only_positive_milliseconds` -->

Graph traversal refuses a depth above ten. <!-- enforced by: `traverse_depth_cap_rejects_over_ten` -->

An invalidated Decision cannot return to active. <!-- enforced by: `combination_decision_status_state_machine_invalidated_to_active_rejected` -->

Recording “basis still valid” for an invalidated Decision is refused. <!-- enforced by: `resolve_invalidation_basis_still_valid_on_invalidated_decision_refused` -->

See [CLI](cli.md), [configuration](configuration.md), and [ontology limitations](ontology.md#known-limitations).
