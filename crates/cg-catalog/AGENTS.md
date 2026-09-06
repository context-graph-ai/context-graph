# AGENTS.md — `cg-catalog` crate

Model catalog state: which models this machine knows about, which of them a `cg` install may
actually use, where their artifact bytes live, and which hosted model connection answers an
understanding request. Carries no product intent — what context-graph is for lives in the root
product documents.

## What this crate owns, and must not own

Owns: the catalog entry record and its typed conditions (`src/error.rs`); the registration file an
operator writes (`<data_root>/cg/catalog-registrations.toml`) and the artifact cache lifecycle beside it —
fetch, air-gapped copy-in, identity validation, license acceptance, swap, update check, eviction;
model connections, provider specs, credential references and the hosted understanding runtime
(`src/model_connection.rs`).

Must not own: an embedding abstraction (`context_graph::Embedder` is the only one) or a second
model-cache-root rule — `context_graph::embedder::nomic::default_cache_root` is that rule and
`default_cache_root` here delegates to it. Not exit codes, help text or output rendering: a typed
condition leaves this crate and `cg-cli` maps it. Not a local understanding runtime — cg
understanding is hosted, bring-your-own-key, and changing that is a product decision.

## The two locations, which are not the same location

An install is configured with two roots, and conflating them is the defect this crate has already
shipped once:

- **The data root** holds `<data_root>/cg/catalog-registrations.toml` — the registrations an
  operator wrote. Nothing can reconstruct them, and a cache wipe must not touch them. This crate
  does **not** resolve it: the root arrives from whoever opened the catalog and is used as given.
  The rule that turns `CG_DATA_ROOT`, else `$HOME/.cg`, into a path is `data_root` in
  `crates/cg-cli/src/main.rs`, and it is the only one — a second rule here is how a command reads
  an empty directory while the operator's entries sit somewhere else. Enforced by
  `the_catalog_never_resolves_the_data_root_for_itself` in `tests/production_guards.rs`.
- **The model-cache root** (`default_cache_root`, `CG_MODEL_CACHE_ROOT`, else
  `$XDG_CACHE_HOME/cg/models`, else the platform cache directory) holds artifact bytes. Wiping it
  costs a re-download and nothing else. The rule belongs to
  `crates/context-graph/src/embedder/nomic.rs`; `default_cache_root` here only delegates to it.

`Catalog::open_with_scope_against_root` collapses both onto one directory — the self-contained
shape, and what every disposable test root uses. `Catalog::open_with_scope_against_roots` holds
them apart, which is what an operator who sets the two environment variables to different
directories gets. Resolving a registration through the cache root hides every registration the
moment those directories differ.

## Seams

- **`Catalog`** (`src/lib.rs`) — the artifact lifecycle: `list`, `show`, `fetch`, `prefetch_defaults`,
  `swap`, `record_acceptance`, `validate_artifact_identity`. Every refusal is a `CgCatalogError`
  variant, never a rendered string.
- **`ModelConnectionCatalog`** (`src/model_connection.rs`) — connections, provider specs, and
  credential references. It reads and writes the same registration file under the data root, so a
  registration written through one side is visible through the other.
- **`CatalogTextEmbedder`** (`src/lib.rs`) — implements `context_graph::Embedder` over a
  catalog-managed local model. It is an implementation of the core trait, never a second trait.
- **`UnderstandingRuntime`** (`src/model_connection.rs`) — the hosted understanding call and its
  typed `UnderstandingRuntimeError`; the wire shape is driven by the connection's provider spec,
  never by a constant.

## Invariants and their guarding tests

- **A registration `show` can read stays fetchable.** Registrations resolve at the data root
  whatever the cache root is set to. Guard:
  `crates/cg-cli/tests/cg_model_fetch_finds_a_registration_that_cg_model_show_reads_when_data_and_cache_roots_differ.rs`.
- **One model-cache root, so what a fetch seeds is what an offline load reads.** Every layer
  resolves through `context_graph::embedder::nomic::default_cache_root` — this crate delegates to
  it, and `scripts/install.sh` and `scripts/cg-uninstall.sh` expand the same rule in shell, so an
  operator whose `XDG_CACHE_HOME` points away from `$HOME/.cache` gets the model seeded where `cg`
  reads it and removed from where `cg` reads it. `DEFAULT_CACHE_ROOT_LINUX_MACOS` and
  `DEFAULT_CACHE_ROOT_WINDOWS` describe the LAST fallback, not the resolved root; call
  `default_cache_root` to learn the real one. Guards:
  `crates/context-graph/tests/catalog_and_offline_nomic_embedder_disagree_on_the_model_cache_root.rs`,
  `the_installer_seeds_the_model_cache_root_the_library_reads_when_the_cache_home_is_relocated`.
- **A license-gated model is refused until acceptance is recorded on this host, with the canonical
  message byte-for-byte.** Guards:
  `license_gate_canonical_message_byte_equal_to_pinned_const_at_all_four_invocation_points`,
  `license_gate_acceptance_on_host_a_does_not_propagate_to_host_b_via_sync`,
  `license_gate_fetch_without_acceptance_after_artifact_dropped_in_cache_still_refuses_with_canonical_message`,
  all in `tests/acceptance/license_gate.rs`.
- **Artifact identity is validated against pinned per-file hashes, and a corrupt or partial cache
  refuses with a typed error instead of silently re-fetching.** Guards:
  `air_gap_fetch_from_path_copies_artifacts_into_cache_root_and_validates_identity_against_pinned_canonical_fixture_shas_not_tautology`,
  `air_gap_with_partial_cache_and_no_network_returns_typed_artifact_missing_error_does_not_silently_attempt_fetch`,
  `air_gap_manifest_matching_corrupt_bytes_does_not_override_catalog_artifact_identity_validation`,
  in `tests/acceptance/air_gap.rs`.
- **A pre-populated cache loads with zero outbound network.** Guard:
  `air_gap_pre_populated_cache_reads_with_zero_outbound_network_on_first_load` in
  `tests/acceptance/air_gap.rs`.
- **Concurrent writers to one model directory serialize through the lockfile; one wins and the
  others observe `AlreadyCached`.** Guards:
  `concurrent_writers_to_same_model_dir_via_lockfile_serialize_one_winner_other_observes_already_cached_barrier_inside_critical_section`,
  `concurrent_writer_crash_mid_write_leaves_lockfile_releasable_for_next_writer`,
  `concurrent_read_during_write_observes_either_pre_write_state_or_post_write_state_never_torn`,
  in `tests/acceptance/concurrency.rs`.
- **`list` and `show` report real disk state, deterministically ordered, and reflect an external
  mutation.** Guards: `catalog_list_orders_deterministic_alphabetical_by_id`,
  `catalog_list_cache_status_reflects_real_disk_after_external_population`,
  `catalog_show_returns_absolute_cache_path_artifact_filenames_per_file_sha_license_url_reflects_external_disk_mutation`,
  in `tests/acceptance/list_show.rs`.
- **Swapping the default embedder to a different dimension refuses without explicit confirmation
  and records the decision it took.** Guards:
  `swap_default_different_dimension_refuses_without_explicit_user_confirmation_and_surfaces_reindex_vs_parallel_space_gate`
  and its reindex/parallel-space siblings in `tests/acceptance/swap.rs`.
- **A model arriving over sync never fetches at the receiver behind a license gate, and an unknown
  id defers with a typed error naming it.** Guards:
  `sync_arrival_headless_for_license_gated_model_defers_at_receiver_emits_observation_does_not_fetch`,
  `sync_arrival_for_unknown_model_id_defers_with_typed_error_and_emits_observation_naming_the_unknown_id`,
  in `tests/acceptance/sync_arrival.rs`.
- **Update check is report-only by default and never applies a license-gated update silently.**
  Guard: `cg_update_check_auto_flag_applies_apache_updates_and_refuses_to_apply_license_gated_updates_silently`
  in `tests/acceptance/update_check.rs`.
- **A credential never lands on a model entry, and no output prints a secret.** Guards:
  `supplied_token_lives_on_the_credential_reference_and_never_on_the_model_entry`,
  `stored_secret_credential_resolution_requires_recorded_consent`,
  `list_show_status_emit_no_secret_in_json_or_text_output`, in
  `tests/acceptance/generic_llm_credential_integrity.rs`.
- **The hosted wire shape follows the attached contract and the connection's own provider model,
  and the base URL is used exactly as given.** Guards: `wire_schema_tracks_the_attached_contract_not_a_constant`,
  `outbound_request_model_field_equals_connection_provider_model`,
  `base_url_is_used_exactly_as_given_no_v1_auto_appended_and_trailing_slash_not_doubled`, in
  `tests/acceptance/generic_llm_endpoint.rs`.
- **Test fixture artifacts are never mistaken for a production model population.** Guard:
  `tests/production_guards.rs`.

## Where a change lives

| Change | Lands in |
|---|---|
| A catalog entry field, a fetch/swap/acceptance behavior, artifact identity validation | `src/lib.rs` |
| A new refusal, or what an existing refusal carries | `src/error.rs` — a typed variant, never a rendered string |
| Where registrations are read or written | `Catalog::registration_path` in `src/lib.rs`, and `ModelConnectionCatalog` in `src/model_connection.rs` — the two must name the same file, both joined onto the data root the caller supplied |
| Where artifact bytes are placed | `Catalog::root` / `default_cache_root` in `src/lib.rs`; the rule itself belongs to `crates/context-graph/src/embedder/nomic.rs` and is only delegated to here |
| A model connection, provider spec, credential reference, or the hosted understanding call | `src/model_connection.rs` |
| An embedding behavior over a catalog-managed model | `CatalogTextEmbedder` in `src/lib.rs`, implementing `context_graph::Embedder` — never a second trait |
| How a condition is worded to an operator, or its exit code | `crates/cg-cli/src/main.rs`, not here |

## Fast test command

`cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test --workspace`. The repository gate that adds the wall-clock and documentation-claim
checks is `scripts/gate.sh`; the root `AGENTS.md` describes both.
