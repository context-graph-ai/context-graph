# `cg-catalog`

Purpose: manage model registrations, local artifacts and hosted understanding connections.

**Owns:** typed catalog conditions, fetch/copy/validation/license/swap lifecycle, provider specs,
credential references and hosted calls.
**Must not own:** CLI rendering or exits, a local understanding runtime, data-root resolution,
a second cache-root rule, storage read routes or a second embedding trait.

## Seams and routing

| Change | Entry point |
|---|---|
| Fetch, artifact identity, cache adoption, swap | `Catalog` in `src/lib.rs` |
| Typed refusal | `src/error.rs` |
| Registration location | `Catalog::registration_path` and `ModelConnectionCatalog`; caller-supplied data root |
| Artifact location | `default_cache_root` delegates to `context_graph::embedder::nomic::default_cache_root` |
| Connections, provider wire contract, credentials | `src/model_connection.rs` |
| Local model embedding | `CatalogTextEmbedder` in `src/lib.rs` implements `context_graph::Embedder` |
| User message/exit | `crates/cg-cli/src/main.rs` |

The data root holds registrations; the separate model-cache root holds replaceable artifact bytes.
`open_with_scope_against_roots` keeps them apart; `open_with_scope_against_root` deliberately
uses one directory. This crate does not open a ContextDB read session; core consumers own that seam.

## Invariants and guards

- Caller owns data-root resolution:
  `tests/production_guards.rs::the_catalog_never_resolves_the_data_root_for_itself`.
- Explicit fetch adopts a complete artifact-only cache only after pinned identity validation:
  `tests/acceptance/air_gap.rs::explicit_fetch_makes_a_complete_artifact_only_cache_usable`.
- A partial/corrupt cache is refused, never silently fetched:
  `tests/acceptance/air_gap.rs::air_gap_with_partial_cache_and_no_network_returns_typed_artifact_missing_error_does_not_silently_attempt_fetch`
  and `air_gap_manifest_matching_corrupt_bytes_does_not_override_catalog_artifact_identity_validation`.
- License acceptance is local and required:
  `tests/acceptance/license_gate.rs::license_gate_acceptance_on_host_a_does_not_propagate_to_host_b_via_sync`.
- Credentials stay off model entries and output:
  `tests/acceptance/generic_llm_credential_integrity.rs::list_show_status_emit_no_secret_in_json_or_text_output`.
- Hosted requests follow the attached provider contract:
  `tests/acceptance/generic_llm_endpoint.rs::wire_schema_tracks_the_attached_contract_not_a_constant`.

Fast check: `cargo test -p cg-catalog --test acceptance`.
Full gate: `scripts/gate.sh` at repository root.
