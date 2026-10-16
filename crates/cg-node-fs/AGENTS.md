# `cg-node-fs`

Purpose: provide the filesystem primitives a deployed cg node needs outside the embeddable core.

**Owns:** config-file reads, OS worker and policy-update locks, bounded atomic worker-request IO,
and machine/cgroup memory probes, all in `src/lib.rs`.
**Must not own:** TOML/JSON policy interpretation, scheduling, store access, budget selection,
event identity, SQL, or embedding. This crate consumes no ContextDB read route or embedding trait.

## Seams and routing

| Change | Entry point / caller |
|---|---|
| Config-file IO | `load_text_if_present`; CLI parses the returned text |
| Worker exclusivity | `acquire_worker_lock`; `context_graph::fabric` and CLI worker consume it |
| Latest request replacement | `read_event_search_worker_policy`, `write_event_search_worker_policy` |
| Published request size ceiling | `EVENT_SEARCH_WORKER_POLICY_MAX_BYTES`; callers reserve space before publication |
| Serialized request update | `acquire_event_search_worker_policy_update` |
| Installed/container memory discovery | `machine_memory`; `context_graph::write_budget` chooses the budget |

## Invariants and guards

Ownership boundaries above are review-only architectural constraints unless a guard below covers them.

| Invariant | Guarding test (paths relative to repository root) |
|---|---|
| Config IO failures other than absence surface to the caller. | `crates/cg-cli/tests/read_memory_policy.rs::unreadable_read_config_is_a_config_error_before_status_can_open_or_write_the_store` |
| A held worker lock and an unusable lock remain distinct `WorkerLockRefusal` variants: the CLI reports `already_running` for the former and `spawn_failed` for the latter. | Held: `crates/cg-cli/tests/import_embedding_workers.rs::requesting_import_publishes_the_cli_cap_to_the_existing_worker_handoff`; unusable: `crates/cg-cli/tests/event_history.rs::cg_import_reports_search_behind_when_the_worker_lock_cannot_be_opened` |
| Missing requests are absent; writes and replacements use the fixed request path. | `crates/cg-node-fs/tests/event_search_worker_policy.rs::missing_policy_is_absent_then_writes_and_replacements_round_trip_at_the_fixed_path` |
| Requests over 4 KiB are refused without replacing the last valid value. | `crates/cg-node-fs/tests/event_search_worker_policy.rs::policy_record_refuses_more_than_four_kibibytes_without_replacing_the_last_value` |
| The update guard reads and replaces one bounded record. | `crates/cg-node-fs/tests/event_search_worker_policy.rs::update_guard_reads_and_replaces_one_bounded_record` |
| Given host and container memory facts, the core budget consumer sizes from the smaller container allowance. | `crates/context-graph/src/write_budget.rs::a_container_on_a_large_host_is_sized_from_the_container_not_the_host` |

## Review-only authoring rule

When changing the serialized request-update flow, release its guard before opening a store,
embedding, starting a process, or acquiring the worker lock. This ordering is an architectural
code-review constraint; no automated test currently proves it.

Fast check: `cargo test -p cg-node-fs`. Full gate: `scripts/gate.sh` at repository root.
