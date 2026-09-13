# AGENTS.md — `context-graph` crate

The embeddable memory core: ontology, store and domain behavior, administrative owner plane,
bounded status reads, fabric, and the bundled default text embedder. Carries no product intent —
what context-graph is for lives in the root product documents.

## What this crate owns, and must not own

Owns: the `Embedder` trait and every embedding-space declaration; the store's schema, its typed
graph reads, and its embedding-space registration; the owner-only administrative read channel and
its bounded status reads; the bundled default text embedder (`nomic-embed-text-v1.5`, behind the
default feature `nomic-embedder`) and the event-history search-space wrapper over it; the generic
immutable event journal, exact-anchor projection, source checkpoints and Store-level placement.

Must not own: a second embedding abstraction beside `Embedder`; a local understanding runtime (cg
understanding is hosted, bring-your-own-key — a proposal to change that is a product decision, not
an engineering one); a second transport to a running cg (the owner-control channel is the only
door); ordinary node filesystem work such as a per-node config file or the worker lock — that is
`cg-node-fs`, one layer out.

## Seams

- **Generic event core** (`src/store/event.rs`, `src/store/event_import.rs`) — adapters declare and
  normalize provider data; the Store alone derives event identity/order/hashes and source commit
  boundaries. `LocalOnly` installs complete `SYNC OFF` event schema. `TenantHub` installs no event
  schema until ContextDB's opaque authenticated binding is already persisted, then registers one
  explicit event-plus-anchor delivery manifest in the same transaction as those rows. Machine-local
  transaction IDs remain in `event_local_commits`, never in transported event or anchor rows. Event
  reads are authored once over `GraphRows` and exposed through both Store and `ConsumerGraphView`.
- **Context bindings** (`src/store/context_binding.rs`) — the core stores and compares the complete
  caller-owned namespace/reference pair without interpreting it. Binding an existing Context is
  idempotent, and creating a Context with its binding uses one transaction. The table remains
  `SYNC OFF` under both event placements. An empty local event store may transition to unbound tenant
  placement before capture; a captured store refuses the mismatch before changing policy or data.

- **`Embedder` trait** (`src/embedder.rs`) — the one embedding abstraction. Role-aware preparation
  rides it as defaulted hooks (`prepare_document_hook`/`prepare_query_hook`), never a second trait.
  The checked, non-overridable entrypoints (`prepare_document`/`prepare_query`,
  `embed_prepared_document`/`embed_prepared_query`) and the `EventSearchSeal` that mints and
  verifies preparation are library-owned, not implementation-owned. Those four entrypoints are the
  sealed `EventSearchEmbedder` extension trait: a private `Sealed` supertrait plus exactly one
  blanket `impl<T: Embedder + ?Sized>`, so every embedder has them on whatever handle a caller
  holds — the concrete type, `&dyn Embedder`, `Arc<dyn Embedder>` — and no outside crate can
  implement, override or diverge from them. An inference hook is reached
  only through those entrypoints, after they have verified the sealing profile, the token ceiling,
  and — for the canonical event-history identity — the exact concrete runtime type against the
  library's own Nomic raw type and event-history wrapper. An outside wrapper cannot inherit that
  authority from an inner Nomic field. Event-search support and its non-zero finite ceiling are one
  `EventSearchCapability` declaration; absence means unsupported. Sealing is fallible
  (`EventSearchDocumentBuilder::seal`), and so is putting the event-history space into a store
  (`EventHistoryTextEmbedder::over`/`registration`): both return typed refusals rather than
  defaulting or panicking.
- **`Store` registration** (`src/store/mod.rs`) — `Store::open_with_control_handler_and_embedder_registrations`
  is the one writable open that carries both the owner handler and every registered embedding
  space; embedding spaces are declared once (`src/embedding_space.rs`) and seeded/validated at
  open, including this library's gate on registration of the canonical event-history identity.
- **contextdb below** — storage, sync, and the owner-control channel's framing, deadlines, and
  refusals are contextdb's; this crate declares only its namespace and read ceilings
  (`owner_control::context_graph_read_limits`).
- **CLI and vigil above** — `cg-cli` maps this crate's typed conditions to exit codes and never
  invents a second vocabulary for one; vigil links this crate directly as a library and reaches a
  live process only through the owner-control channel.

- **Event-history status and forget** (`src/store/event_history_status.rs`,
  `src/store/event_forget.rs`) — `event_history_status` is one fold over the shared row seam,
  answered identically by the writable Store and the read-only `ConsumerReader` (whose declaration
  is `event_history_reader_declaration`); search is `running` only under a live projection lease.
  `forget_event_history` issues one standalone multi-table `PURGE` per session or Context on a
  local-only Store and refuses a tenant-hub Store; the Context row and its bindings survive.

## Invariants and their guarding tests

- **Event capture is deterministic, atomic, bounded and policy-gated.** Guards: all six cases in
  `tests/event_capture.rs`, covering derived occurrence identity/order, idempotent retry and batch
  rollback, streaming prefix/checkpoint recovery, Context-scoped reads, unbound authenticated
  placement, and real accepted/equivalent/refused ContextDB custody outcomes.
- **Opaque Context bindings and pre-capture placement changes are atomic and node-local.** Guards:
  all five cases in `tests/context_bindings.rs`.

- **One `Embedder` trait, never a second abstraction.** Guard:
  `tests/acceptance/nomic_loads.rs::embedder_trait_and_storeconfig_public_surface_diff_keeps_surviving_signatures_and_allowed_moves`,
  which reads `src/embedder.rs` and requires its public traits to be exactly `Embedder` and the
  sealed `EventSearchEmbedder` extension, implemented once by one blanket impl.
- **Role-aware preparation as trait hooks, with library-owned non-overridable checked entrypoints
  and profile verification.** Bypass guard:
  `tests/prepared_event_search_document_sealed_under_a_different_profile_is_refused_even_by_an_embedder_designed_to_bypass_the_check.rs`.
  Hook guard: `tests/event_search_preparation_hook_supports_external_embedders.rs`.
  Handle guard (the four entrypoints answer identically on the concrete type, on `&dyn Embedder`
  and on `Arc<dyn Embedder>`):
  `tests/event_search_role_methods_are_callable_on_any_embedder_handle.rs`.
- **The event-history search space is never compared with the text-default space.** Guard:
  `dedicated_event_search_space_refuses_old_unlabelled_space_comparison` in `src/store/embedding.rs`.
- **Digest/Record/Evidence embedding stays byte-for-byte unchanged.** Guard:
  `event_search_uses_role_preparation_while_digest_record_and_evidence_paths_remain_byte_for_byte_unchanged`
  in `tests/event_search_uses_role_preparation_while_digest_record_and_evidence_paths_remain_byte_for_byte_unchanged.rs`,
  which compares live vectors against `tests/fixtures/text_default_space_goldens.json`.
  `tests/text_default_space_goldens_capture.rs` is NOT that guard: it is the ignored, manually run
  utility that refreshes the golden fixture, and it passes whatever the code currently produces.
- **An unsupported embedder never falls back to plain embedding.** Guard:
  `tests/embedder_without_exact_preparation_answers_unsupported_and_never_delegates_to_plain_embed.rs`
  (same file as the one-trait guard above — it proves both).
- **Real-model proofs load only from the local cache, offline.** Guard:
  `tests/nomic_real_model_helper_refuses_locally_on_an_empty_cache_and_never_reaches_the_network.rs`.
- **One model-cache root, so what a fetch seeds is what an offline load reads.**
  `default_cache_root` in `src/embedder/nomic.rs` is the whole rule — `CG_MODEL_CACHE_ROOT`, else
  `$XDG_CACHE_HOME/cg/models`, else the platform cache directory — and `cg-catalog` and the CLI
  delegate to it rather than re-deriving it. Guard:
  `tests/catalog_and_offline_nomic_embedder_disagree_on_the_model_cache_root.rs`.
- **Test-estate ratchet** (sleep/raw-clock call sites only ever decrease; per-crate test counts
  track the estate exactly and are raised in the same change that adds tests, lowered only as a
  named exception). Guard: `tests/test_estate_audit.rs`.
- **Documentation-claim binding** (every `<!-- enforced by: -->` tag resolves to a real test, and
  this crate's README carries no unbound behavior claim). Both validators accept only live
  default-tier Rust test functions, including parameterized Tokio tests — never a test-file stem,
  ignored test, or internal-only test. Default-tier membership is decided from the test function's
  own attributes AND from every enclosing `mod` declaration reachable from its test target's root
  (`tests/<target>/main.rs`), file declarations and inline `mod` blocks alike: a test inside a
  module declared behind `#[cfg(context_graph_internal_test_support)]` never compiles into the
  gate, carries no marker of its own, and does not enforce a claim. That configuration flag is the
  only one either validator evaluates — no feature gates, no target gates, no `#[cfg(test)]`.
  Guard: `tests/doc_claim_bindings.rs`; the
  whole-repository, change-scoped arm is `scripts/validate-doc-coverage.py --base <merge-base>`,
  which CI runs on every pull request.
- **Status totals stay bounded and leave no cursor open.** Guards:
  `tests/status_totals_are_bounded_on_a_large_store.rs` and
  `tests/a_failed_status_fold_leaves_no_cursor_open.rs`. A change to what a status total counts —
  event-history vectors included — must keep both green.
- **One embedding space owns a given vector column, and the event-history declaration claims
  none.** Guards: `second_owner_of_evidence_vector_text_is_refused` and
  `canonical_event_history_declaration_has_no_evidence_or_digest_binding`, both in
  `src/store/embedding.rs`. Adding or moving a vector-column binding must keep both green.
- **Registration and inference on a prepared value are library-gated, and construction of the
  event-history registration is fallible.** Store registration and the checked entrypoints accept
  the canonical event-history identity only from the exact library Nomic raw type or exact
  event-history wrapper at canonical dimension. The checked entrypoints also require one
  event-search capability carrying a finite ceiling, verify the sealing profile, and refuse a
  prepared value over that ceiling — all before any implementer hook runs.
  `EventHistoryTextEmbedder::over`/`registration` reject only configurations that can fetch, an
  explicitly different revision, or an invalid installed artifact; the canonical `REVISION`
  marker is sufficient when the optional config pin is absent. That same admission is what raw
  `NomicTextEmbedder` preparation runs on, and it separates two answers that must never be
  conflated. An embedder whose INSTALLED artifact does not verify declares no event-search
  capability at all, so both roles answer `EventSearchPreparationUnsupported` rather than stamping
  same-shaped, unverified weights with the reserved identity. An embedder whose model is simply not
  installed yet declares the capability — support is about what kind of embedder this is, whatever
  the configuration's fetch policy — and both roles answer `EmbedderModelNotAvailable` naming the
  resolved directory, the same answer plain `embed` gives. Neither role ever installs it: they load
  under the never-install policy, so preparing a value, or embedding one the caller already holds,
  can never start a download. The verdict is decided once per embedder as soon as an artifact is
  present and never re-read per passage thereafter, and it reads the revision marker, the model
  config and the tokenizer — never the weights, which the loader reads exactly once. Guards:
  `tests/role_prepared_values_share_one_event_history_profile_across_the_raw_embedder_and_its_wrapper.rs`,
  `tests/checked_entrypoints_enforce_the_token_limit_for_every_embedder_even_when_the_hook_does_not.rs`,
  `tests/event_history_registration_reuses_the_loaded_model_and_refuses_an_unfit_underlying_embedder.rs`,
  `tests/raw_nomic_preparation_over_noncanonical_same_dimension_weights_refuses_the_reserved_identity.rs`,
  `tests/registration_plus_many_preparations_reads_the_weights_once.rs`.

- **Status and erasure of imported history.** Guards: `tests/event_history_status.rs` (one fold on
  both doors, capture and search coverage kept apart, unknown Contexts refused, read-only binding
  resolution) and `tests/event_forget.rs` (one purge per selection erases events, anchors and every
  derived search row, keeps the Context and binding, and refuses a tenant-hub Store).

## Where a change lives

| Change | Lands in |
|---|---|
| A new embedder | Its own module under `src/embedder/`, implementing `Embedder`; register it where `maybe_register_default_text_embedder` or the caller's own registration list does today. |
| A change to preparation or a search space | `src/embedder.rs` (hooks/entrypoints), `src/embedder/nomic.rs` (the bundled Nomic preparer that implements them) or `src/embedding_space.rs` (declaration) — never a new trait or registry. |
| A change to the `Store`'s embedding-space registration | `src/store/mod.rs` around `open_with_control_handler_and_embedder_registrations`. |
| A Context-binding API or pre-capture event-placement transition | `src/store/context_binding.rs`, `src/store/context.rs`, and the placement bootstrap in `src/store/mod.rs`; keep `tests/context_bindings.rs` green. |
| A status read, or a change to what a total counts | `src/status_reads.rs`, bounded by `owner_control::context_graph_read_limits()`. Must not break `tests/status_totals_are_bounded_on_a_large_store.rs` or `tests/a_failed_status_fold_leaves_no_cursor_open.rs`. |
| A vector-column binding on an embedding space | `src/embedding_space.rs`, in that space's declaration. Must not break `second_owner_of_evidence_vector_text_is_refused` or `canonical_event_history_declaration_has_no_evidence_or_digest_binding` in `src/store/embedding.rs`. |
| An owner-plane read | `src/owner_control.rs`; see the root `AGENTS.md` for the transport boundary. |
| Anything else touching CLI exit codes, help text, or a second crate | See the root `AGENTS.md` crate map. |

## Fast test command

`cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test --workspace`.
