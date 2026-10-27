# Why context-graph?

<!-- concept -->
An agent can remember a path long after the file has moved. Finding the old conversation is useful, but similarity alone does not explain whether the choice made there still makes sense. A useful memory needs both the evidence and a record of the facts a choice depended on.

Decisions retain frozen Entity basis snapshots. <!-- enforced by: `decision_based_on_snapshot_frozen_after_entity_update` -->

A basis-changing Observation creates an Invalidation identifying the affected Decision. <!-- enforced by: `invalidation_created_by_engine_on_basis_change` -->

<!-- concept -->
Choose the [history-search journey](getting-started.md) with its [current limits](../README.md#current-limits) when you want to recover what a conversation actually said.

<!-- concept -->
Choose the tool when you want to recover what a conversation actually said. Choose the library when your application needs to record why a choice was made and check its dependencies later.

Read [getting started](getting-started.md), [the ontology](ontology.md), or [the capability index](capability-index.md).
