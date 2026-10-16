# cg-cli — package `context-graph-cli`, binary `cg`

Purpose: parse commands and render typed library answers for operators.
Owns arguments, help, output envelopes, typed exit mapping, root resolution,
configuration plumbing and detached search worker process lifecycle.
Must not own event identity, Claude JSON decoding, SQL, ranking, embedding preparation,
storage policy or a second cache-root rule.

## Seams and where a change lives

| Change | Entry point |
|---|---|
| Dispatch, help and exits | `src/main.rs`; match typed conditions |
| Capture and forgetting | `src/history/import.rs`, `src/history/forget.rs` |
| Find, show, status and Context selection | `src/history/read.rs`, `src/history/mod.rs` |
| Detached work, saved selections and retry accounting | `src/history/worker.rs` |
| Reader budgets | `src/read_config.rs` |
| Writer admission | `src/write_config.rs` |

Read commands use core `ConsumerReader`; capture uses `Store` and the Claude connector.
Node locks and bounded request IO come from `cg-node-fs`; model cache resolution delegates
to `cg-catalog`. Search consumes core `Embedder` and sealed `EventSearchEmbedder`.

## Invariants and guards

Ownership boundaries above are review-only architectural constraints unless a guard below covers them.

- Read commands never open a writable Store: `tests/a_read_only_command_answers_a_store_a_writer_is_holding.rs`,
  `the_read_only_commands_answer_an_idle_store_and_a_held_one_without_changing_a_byte`.
- Validate budgets before store access: `tests/read_memory_policy.rs`,
  `invalid_read_config_is_a_config_error_before_status_can_open_or_write_the_store`.
- Worker requests retain selected session identities, capture frontiers, budgets and bounded retry
  allowances through restart. Completion acknowledges only verified request identities; later capture
  never extends an accepted frontier. Guards (feature-gated: they compile only with
  `--features event-search-test-support`, which the gate's `features` stage and CI's `features` job
  run; a default `cargo test` skips them): `tests/import_embedding_workers.rs`, especially
  `selected_session_ceiling_survives_restart_before_the_first_claim` and
  `older_completed_import_acknowledges_only_observed_refusal_identities`.
- Typed time/coordination refusals preserve committed coverage, permit independent work and remain
  inspectable until their own frontiers complete. Guard (same feature gate as above):
  `tests/import_embedding_workers.rs`:
  `consecutive_typed_coordination_refusals_exhaust_one_selected_request_allowance`.
- Busy stores use core `StoreBusy`, exit 4; reader-held conditions have no dedicated exit contract.
  Guards: `tests/a_busy_store_is_told_to_the_operator.rs`:
  `a_command_refused_by_a_writer_held_store_keeps_its_typed_answer` and
  `a_command_refused_by_a_reader_held_store_names_the_readers_it_waits_on`.
  Review-only: reader-held conditions have no dedicated numeric exit contract.
- Presented help is the public command list; guard
  `tests/help_presents_exactly_the_public_command_list.rs`: `help_presents_exactly_the_public_command_list`.

Fast check: `cargo test -p context-graph-cli --test read_memory_policy`.
Lifecycle proofs: `cargo nextest run -p context-graph-cli --features event-search-test-support`.
Full gate: `scripts/gate.sh`; public operator contracts: [CLI reference](../../docs/cli.md).
