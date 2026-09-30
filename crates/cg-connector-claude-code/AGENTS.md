# `cg-connector-claude-code`

Purpose: locate Claude Code project history, decode JSONL occurrences, and assemble provider-neutral
`EventDraft` groups for the event core.

**Owns:** Claude field navigation, project-store keys, connector and adapter declarations, and
provider-to-render-role/kind mapping.
**Must not own:** event identity, sequence, hashing, checkpoints, bindings, projection, search or
store opens. Those belong to `context-graph`, which never depends on this connector.

## Seams and where a change lives

| Change | Entry point / boundary |
|---|---|
| History-root and project-file selection | `src/history.rs`: `effective_history_root`, `ClaudeHistoryLocator::resolve_project`; CLI supplies paths and opens selected source files |
| JSONL record navigation and source identity | `src/jsonl.rs`: `ClaudeCodeJsonlAdapter` implements core `EventSourceAdapter`; `src/occurrence.rs`: `decode_jsonl_record` produces `ClaudeOccurrence` |
| Historical or typed occurrence mapping | `src/assembler.rs`: `ClaudeOccurrenceAssembler::assemble`, with `ClaudeAssemblyContext`, returns the same ordered draft groups for both inputs |
| Prior tool-request correlation | Core `EventCorrelationResolver` is passed into the adapter/assembler; consume its typed answers rather than opening a store or writing SQL |
| Mapping/anchor version and connector id | `src/declaration.rs`: `event_adapter_declaration`, `ADAPTER_VERSION`, `ANCHOR_DECLARATION_VERSION`, `CONNECTOR_ID` |

The CLI caller lives in `crates/cg-cli/src/history/`; Claude JSON navigation stays here. The core
owns append, checkpoint and interpretation behavior. This crate consumes no ContextDB owner or
consumer read route directly, and no embedding trait.

## Invariants and guarding tests

Test paths below are relative to this crate.

| Invariant | Guard (file and test) |
|---|---|
| Only matching immediate regular project JSONL files are selected in name order | `tests/source_fidelity/history_and_reuse.rs::history_locator_selects_only_matching_immediate_regular_jsonl_in_name_order` |
| Introductory records do not hide the first project cwd | `tests/history_prefix.rs::history_locator_finds_the_first_cwd_after_introductory_records` |
| Historical and typed occurrences use one assembler and persist equivalent events | `tests/source_fidelity/history_and_reuse.rs::typed_occurrences_use_the_same_assembler_then_append_to_fresh_stores` |
| Mapping preserves source facts, order and resolver-backed correlation | `tests/source_fidelity/mapping.rs::mapping_preserves_oracle_facts_one_through_nine_in_order` |
| Refused inline media retains descriptors without leaking raw bytes | `tests/source_fidelity/capture_regressions.rs::nested_media_with_text_refuses_all_inline_encodings_without_losing_descriptor` |
| Exact-file session overrides reject conflicting embedded identities | `tests/source_fidelity/identity_recovery.rs::exact_file_override_rejects_mixed_embedded_ids_and_reader_preserves_variants` |

## Authoring rules requiring review

These architectural constraints require code review; the tests above do not mechanically enforce
them. When changing a mapping, advance the adapter or anchor declaration version and leave stored
events unchanged. Author synthetic fixtures; do not copy owner transcripts, machine paths or private
labels into fixtures, defaults or constants. Treat retained `source_remainder` as data; do not derive
new mapping vocabulary, event kinds or repair rules from it.

Fast check: `cargo test -p cg-connector-claude-code`. For history selection alone:
`cargo test -p cg-connector-claude-code --test history_prefix`.
Lint: `cargo clippy -p cg-connector-claude-code --all-targets -- -D warnings`.
Full gate: `scripts/gate.sh` at repository root.
