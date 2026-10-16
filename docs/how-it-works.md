# How it works

The core exposes no caller-supplied socket transport API. <!-- enforced by: `the_crate_carries_no_socket_transport_and_no_path_taking_control_api` -->

A control frame addressed by the store path reaches the live owner's handler. <!-- enforced by: `a_live_owner_answers_a_control_frame_addressed_only_by_the_store_path` -->

That handler receives exactly one frame. <!-- enforced by: `a_live_owner_answers_a_control_frame_addressed_only_by_the_store_path` -->

See [event identity and checkpoints](library-api.md#events-and-checkpoints).

<!-- concept -->
Follow [getting started](getting-started.md) for history search and consult the [current limits](../README.md#current-limits) for what that journey builds.

```text
Claude Code session files
    -> connector -> immutable events -> local passage embedding -> cg find
                                   \-> exact stored events     -> cg show

Application -> Intentions, Decisions, Entity snapshots, Observations
            -> impact analysis -> Invalidations -> subscriptions
```

A [new-Decision subscription](library-api.md#subscriptions) receives the created Decision. <!-- enforced by: `subscribe_new_decision_fires_post_commit_carries_served_intention_ids_and_descriptions` -->

<!-- concept -->
The core owns the memory model and store behavior. The Claude Code connector decodes history; the CLI selects inputs and renders results; the catalog manages model artifacts and connections; node filesystem helpers own configuration and worker locks. Optional embedding crates supply models through the core's Embedder trait.

See [ontology](ontology.md), [models](models-and-embedders.md), and [privacy](privacy-and-data-movement.md).

## Projection and worker coordination

Deferred import leaves another build's active worker policy unchanged. <!-- enforced by: `deferred_import_never_creates_or_changes_the_active_worker_policy` -->

A caught-up background import leaves another build's active worker policy unchanged. <!-- enforced by: `already_caught_up_import_does_not_change_an_active_worker_request` -->

A later background import without a worker flag replaces the earlier one-off cap with the CPU default. <!-- enforced by: `later_requesting_import_without_a_flag_replaces_the_one_off_override` -->

The next claim adopts an updated worker limit. <!-- enforced by: `claim_callback_rereads_the_worker_policy_file_before_the_next_two_passage_batch` -->

A pending background import can publish to an already running worker. <!-- enforced by: `requesting_import_publishes_the_cli_cap_to_the_existing_worker_handoff` -->

It publishes the selected worker cap. <!-- enforced by: `requesting_import_publishes_the_cli_cap_to_the_existing_worker_handoff` -->

Its pending request preserves the selected order. <!-- enforced by: `requesting_import_publishes_the_cli_cap_to_the_existing_worker_handoff` -->

Its pending request preserves the selected session scope. <!-- enforced by: `requesting_import_publishes_the_cli_cap_to_the_existing_worker_handoff` -->

A writable open refused by live direct readers reports their observed count. <!-- enforced by: `a_real_context_graph_open_is_refused_by_the_live_readers_holding_the_store` -->

It reports their verified identities. <!-- enforced by: `a_real_context_graph_open_is_refused_by_the_live_readers_holding_the_store` -->

Projection retries at claim, renewal and commit retain the rendered batch without a second embedding call. <!-- enforced by: `projection_retries_real_writer_at_claim_renewal_and_commit_without_reembedding` -->

Exhausted writer contention remains a typed `StoreLocked` refusal. <!-- enforced by: `shared_store_returns_real_writer_busy_immediately_and_projection_exhaustion_stays_typed` -->

Continuation over a long metadata suffix loads at most the unfinished passage's one text member alongside fresh rows. <!-- enforced by: `open_tail_loads_only_text_members_across_a_long_metadata_suffix` -->

Metadata-only events do not become passage sources. <!-- enforced by: `open_tail_loads_only_text_members_across_a_long_metadata_suffix` -->

Changing the worker cap preserves committed passage order, content, vectors and coverage. <!-- enforced by: `serial_and_concurrent_runs_commit_identical_passages_vectors_order_and_coverage` -->

An active-time refusal can leave a committed passage prefix. <!-- enforced by: `a_typed_active_time_refusal_retains_committed_passages_and_resumes` -->

The stored failure supplies a `--search wait` retry instruction. <!-- enforced by: `a_typed_active_time_refusal_retains_committed_passages_and_resumes` -->

Resuming projection does not embed that completed prefix again. <!-- enforced by: `a_typed_active_time_refusal_retains_committed_passages_and_resumes` -->

A successful caught-up run clears its prior failure. <!-- enforced by: `a_projection_run_that_fails_records_why_and_a_later_successful_run_clears_it` -->

A failed replacement leaves the ready head visible. <!-- enforced by: `building_progress_is_visible_without_replacing_the_ready_head` -->

A late failed attempt reports `Stale` after a newer attempt takes ownership. <!-- enforced by: `a_stale_attempt_failure_cannot_overwrite_a_newer_owned_refusal` -->

It does not overwrite the newer attempt's failure metadata. <!-- enforced by: `a_stale_attempt_failure_cannot_overwrite_a_newer_owned_refusal` -->

The observed runner reports `Unrecorded` when its refusal cannot be persisted. <!-- enforced by: `a_projection_run_that_fails_records_why_and_a_later_successful_run_clears_it` -->

It reports `Recorded` when its failure is persisted. <!-- enforced by: `a_projection_run_that_fails_records_why_and_a_later_successful_run_clears_it` -->