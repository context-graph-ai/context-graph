# context-graph

The embeddable memory core behind `cg`. It gives an agent or an application a
durable, queryable memory of the work it has done — intentions, decisions,
observations and the digests that tie them together — stored in a local
contextdb database and searched by meaning as well as by structure. Link it as a
library and you get the ontology, the store and its domain behavior, embedding
registration, status reads, and the administrative owner plane; the `cg`
binary is one consumer of it.<!-- enforced by: `one_handle_serves_owner_frames_and_embeds_interleaved`, `a_fold_that_refuses_a_later_page_leaves_no_cursor_behind` -->

| Version | License | Maintainer | Repository |
| --- | --- | --- | --- |
| 0.1.0 | Apache-2.0 | [John Cyriac](mailto:lucidprogrammer@hotmail.com) | [context-graph-ai/context-graph](https://github.com/context-graph-ai/context-graph) |

## Add it from source

This crate is not yet published on crates.io.

Prerequisites: Git and Rust 1.94.1 via rustup.

```sh
rustup toolchain install 1.94.1
```

From an empty parent directory:

```sh
git clone https://github.com/context-graph-ai/contextdb.git
git clone https://github.com/context-graph-ai/context-graph.git
cargo +1.94.1 new my-app
cd my-app
cargo +1.94.1 add context-graph --path ../context-graph/crates/context-graph
```

Replace `src/main.rs` with this first offline memory write and read:

```rust,no_run
use std::collections::BTreeMap;
use context_graph::{CreateContext, CreateIntention, EmbedderConfig, Store, StoreConfig};

fn main() -> context_graph::Result<()> {
    let store = Store::open(StoreConfig {
        db_path: std::env::temp_dir()
            .join(format!("context-graph-example-{}.db", std::process::id())),
        default_text_embedder: Some(EmbedderConfig::disabled()),
        ..Default::default()
    })?;
    let context = store.create_context(CreateContext {
        name: "example".into(),
        labels: vec![],
        properties: BTreeMap::new(),
    })?;
    let item = store.create_intention(CreateIntention {
        description: "preserve one memory item".into(),
        context_id: context.id,
        ..Default::default()
    })?;
    let read_back = store.get_intention(item.id)?.expect("persisted intention");
    assert_eq!(read_back.description, item.description);
    Ok(())
}
```

The example creates the Context required by an intention, persists and reads the
intention back, and disables the default embedder so this first path does not
load a model.<!-- enforced by: `test_fk_intention_invalid_context_id_rejected`, `context_create_persists_with_labels`, `intention_create_persists_across_reopen`, `store_open_with_default_config_against_freshly_created_path_writes_cg_schema_version_row` -->

```sh
cargo +1.94.1 run
```

Successful completion proves the local store persisted and read back the memory
item.<!-- enforced by: `intention_create_persists_across_reopen` -->

The example intentionally prints nothing.

## How the embedded store runs

The crate is embedded rather than served: there is no separate daemon and no
caller-configured socket address, one process holds the store's writable handle,
and everything else is a typed call. When another process owns the store,
administrative reads reach that process over contextdb's local owner channel,
which is selected from the store path and the runtime root rather than from any
address a caller supplies; there is no second transport in this repository.<!-- enforced by: `the_crate_carries_no_socket_transport_and_no_path_taking_control_api`, `a_live_owner_answers_a_control_frame_addressed_only_by_the_store_path`, `one_provisioned_root_reaches_the_owner_and_a_separate_read_only_command` -->

## Capturing normalized event history

Schema 12 adds a provider-neutral immutable event journal. An adapter supplies an
`EventAdapterDeclaration` and ordered `EventDraft` values; `Store::append_event_batch` derives the
event IDs, source-stream sequence numbers and SHA-256 integrity fields. Repeating an identical
source occurrence is a no-op, while changing one under the same source key refuses the whole batch.
`Store::import_event_stream` adds bounded record streaming, raw-prefix verification and atomic source
checkpoints without exposing its commit-boundary constructor to adapters.<!-- enforced by: `append_derives_occurrence_identity_and_source_order_without_caller_control`, `retries_are_noops_and_a_divergent_batch_leaves_events_and_checkpoint_unchanged`, `import_verifies_prefix_then_adds_only_tail_or_refuses_rewritten_history` -->

The default `StoreConfig::event_placement` is `None`: a fresh Store resolves that to
`EventPlacementPolicy::LocalOnly` and declares event and exact-anchor rows `SYNC OFF`. A fresh
`TenantHub` Store remains `TenantHubUnbound`, has no event tables and refuses capture until
`bind_authenticated_tenant_policy` receives the opaque binding that ContextDB already authenticated
and persisted. Bound appends register ContextDB delivery manifests containing the event root and its
complete anchor membership; `event_delivery_status` reports ContextDB's durable accepted,
equivalent, pending and refused outcomes.<!-- enforced by: `local_and_unbound_tenant_placement_keep_event_schema_and_append_behind_real_policy_binding`, `authenticated_custody_counts_equivalent_full_units_and_refuses_changed_members` -->

An empty standalone Store may be reopened with explicit `TenantHub` placement before its first event
is captured. That choice persists, and the Store reopens unbound with no event tables until the
authenticated policy arrives. Once capture has written an event, a different configured placement is
refused without changing the persisted policy, event-table declaration, or event rows.<!-- enforced by: `placement_change_is_allowed_before_capture_or_after_failure_but_not_after_erasure` -->

`ContextBindingKey { namespace, opaque_reference }` is the generic node-local route from a
caller-owned reference to a Context. `resolve_context_binding` reads it, `bind_context` attaches an
existing Context idempotently, and `create_context_with_binding` creates both rows in one transaction.
The core stores both key fields as opaque text in the `SYNC OFF` `context_bindings` table. A key
already attached to another Context returns `ContextBindingConflict`; an atomic create whose name
already exists returns `ContextAlreadyExists` and writes neither row.<!-- enforced by: `binding_is_absent_then_persists_across_reopen_and_same_context_is_a_noop`, `conflicting_or_missing_context_binding_changes_neither_binding_nor_context_rows`, `create_and_bind_commits_success_and_rolls_back_on_a_real_second_write_constraint`, `opaque_binding_keys_are_full_node_local_pairs_in_local_and_tenant_bound_stores`, `placement_change_is_allowed_before_capture_or_after_failure_but_not_after_erasure` -->

`get_event` and `list_stream_events` always require a Context and are implemented once over the
shared graph-row seam. They are available on both the writable `Store` and a declared read-only
`ConsumerGraphView`; stream ranges are bounded and always return Store-assigned sequence order.<!-- enforced by: `append_derives_occurrence_identity_and_source_order_without_caller_control`, `retries_are_noops_and_a_divergent_batch_leaves_events_and_checkpoint_unchanged`, `import_verifies_prefix_then_adds_only_tail_or_refuses_rewritten_history`, `event_reads_are_context_scoped_and_return_source_sequence_order`, `local_and_unbound_tenant_placement_keep_event_schema_and_append_behind_real_policy_binding`, `authenticated_custody_counts_equivalent_full_units_and_refuses_changed_members` -->

## Choosing an Embedder

The bundled text embedder is `nomic-embed-text-v1.5`, exposed through the
default text space `text_default_nomic_v1_5_768`. It runs locally through
Candle, does not require an API key, and stores first-load artifacts in the
cg embedder cache. It lives in this crate, behind the default feature
`nomic-embedder`. The upstream artifact is published at
`huggingface.co/nomic-ai/nomic-embed-text-v1.5`; the model weights are about
547MB, which is roughly three minutes on a 25Mbps developer connection before
the cache is warm. Turning default features off gives a core that fetches
nothing.<!-- enforced by: `text_default_space_id_resolves_to_text_default_nomic_v1_5_768_after_default_store_open`, `cg_readme_choosing_an_embedder_section_discloses_nomic_artifact_huggingface_url_approx_548mb_size_and_approx_three_minutes_first_load_on_25mbps`, `embedding_over_an_isolated_empty_local_cache_refuses_locally_with_a_typed_error_and_never_reaches_the_network`, `the_core_library_builds_with_default_features_off_and_drops_every_bundled_embedder_dependency` -->

### Encoding event history for search

Nomic answers differently depending on whether text is being *stored* or *asked
for*, so the same embedder offers both roles: `prepare_document` then
`embed_prepared_document` for a passage of history you want found later, and
`prepare_query` then `embed_prepared_query` for what somebody is searching for.
You always hand in plain text; preparation adds exactly one system-side task
instruction outside your content and returns an opaque prepared value carrying
the exact final token count and the exact byte span each token occupies in your
original text — enough to split one oversized unit on real token boundaries
without guessing at characters.
<!-- enforced by: `nomic_document_preparer_adds_exactly_one_system_task_prefix` -->
<!-- enforced by: `nomic_query_preparer_adds_exactly_one_system_task_prefix` -->
<!-- enforced by: `oversized_event_split_uses_selected_tokenizer_and_has_utf8_safe_exact_bidirectional_source_offsets` -->

Content that itself begins with the words `search_document:` or `search_query:`
is never inspected or rejected for it; it is passed through untouched behind the
one system-added instruction.
<!-- enforced by: `literal_source_and_query_text_starting_with_nomic_prefix_words_is_preserved_and_searchable` -->

Preparing an oversized unit always succeeds — you get the full count and the
full boundary map, which is what you need to cut it up — and only embedding a
still-oversized prepared value is refused, by exact token count against the
ceiling carried by the embedder's `event_search_capability`, never silently
truncated. No capability means exact event-search preparation is unsupported.
<!-- enforced by: `oversized_event_split_uses_selected_tokenizer_and_has_utf8_safe_exact_bidirectional_source_offsets` -->
<!-- enforced by: `an_oversized_event_search_query_is_prepared_in_full_and_refused_only_at_embed_time` -->

These vectors live in their own space,
`event_history_nomic_v1_5_document_query_v1_768`, registered through
`EventHistoryTextEmbedder::registration` over the same loaded model as the
text-default space, so both spaces cost one set of weights. Ordinary
`Store::open` does not auto-register the event-history space. Fetch the weights
explicitly before opening the store:
<!-- enforced by: `store_open_default_storeconfig_on_empty_path_routes_to_initialize_fresh_seeds_default_text_embedding_space` -->
<!-- enforced by: `event_history_registration_reuses_the_loaded_model_and_refuses_an_unfit_underlying_embedder` -->

```sh
cg model fetch --defaults
```

Then construct one `Arc<NomicTextEmbedder>` with the pinned, download-denying
`EmbedderConfig::default_nomic_offline()` and pass both registrations to
`Store::open_with_embedder_registrations`. A download-enabled configuration is
deliberately refused for event history, so registration and later search cannot
silently fetch.
<!-- enforced by: `store_open_default_storeconfig_on_empty_path_routes_to_initialize_fresh_seeds_default_text_embedding_space` -->
<!-- enforced by: `public_default_nomic_offline_configuration_uses_the_canonical_installed_model_for_event_history` -->
<!-- enforced by: `event_history_registration_reuses_the_loaded_model_and_refuses_an_unfit_underlying_embedder` -->

```rust,no_run
use std::sync::Arc;

use context_graph::embedder::nomic::{EventHistoryTextEmbedder, NomicTextEmbedder};
use context_graph::{
    Embedder, EmbedderConfig, EmbedderRegistration, EventSearchEmbedder, Store, StoreConfig,
};

fn main() -> context_graph::Result<()> {
    let model = Arc::new(NomicTextEmbedder::new(
        EmbedderConfig::default_nomic_offline(),
    ));
    let text_default = EmbedderRegistration {
        space: model.embedding_space(),
        embedder: Arc::clone(&model) as Arc<dyn Embedder>,
    };
    let event_history = EventHistoryTextEmbedder::registration(Arc::clone(&model))?;

    let mut config = StoreConfig::default();
    config
        .extra_embedding_spaces
        .push(event_history.space.clone());
    let _store = Store::open_with_embedder_registrations(
        config,
        vec![text_default, event_history],
    )?;

    let passage = model.prepare_document("the incident review closed today")?;
    let _vector = model.embed_prepared_document(&passage)?;
    let asked = model.prepare_query("what happened to the incident review")?;
    let _query_vector = model.embed_prepared_query(&asked)?;
    Ok(())
}
```

The four checked role methods come from `EventSearchEmbedder`, a sealed
extension trait this library implements once for every embedder — which is what
stops any implementation from overriding them. Bring it into scope and call
them on whatever handle you hold: the concrete `Arc<NomicTextEmbedder>` above, a
`&dyn Embedder` view, or a shared `Arc<dyn Embedder>`, all reaching the same
library-owned code.
<!-- enforced by: `the_four_role_methods_give_byte_identical_results_across_every_embedder_handle_route`, `a_noncanonical_external_embedder_answers_unsupported_directly_on_its_concrete_handle`, `document_prepared_by_the_raw_embedder_is_accepted_by_the_wrapper_and_produces_the_same_vector`, `external_embedder_cannot_borrow_canonical_event_history_authority` -->

The two spaces are never compared: a query in one against candidates in the
other is refused, even though both are 768-dimensional. Digest, Record and
Evidence embedding is untouched by any of this and keeps producing the exact
vectors it produced before.
<!-- enforced by: `dedicated_event_search_space_refuses_old_unlabelled_space_comparison` -->
<!-- enforced by: `event_search_uses_role_preparation_while_digest_record_and_evidence_paths_remain_byte_for_byte_unchanged` -->

An embedder that cannot do this preparation exactly says so —
`EventSearchPreparationUnsupported` — instead of falling back to plain text
embedding and handing you a differently-encoded vector under a space that
promises the role-aware encoding.
<!-- enforced by: `embedder_without_exact_preparation_answers_unsupported_and_never_delegates_to_plain_embed` -->

`candle-embeddinggemma` remains available as a sibling opt-in for consumers
that want the prior `text_default_embeddinggemma_300m_768` space. Register it
through `Store::open_with_embedder_registrations`, then use `reindex_space`
when moving stored rows between registered spaces. context-graph ships the
English Nomic v1.5 default; multilingual embedders are consumer-declared
through the same registration lane.<!-- enforced by: `candle_embeddinggemma_consumer_registers_via_storeconfig_extra_embedding_spaces_under_text_default_embeddinggemma_300m_768_space_id`, `runtime_cross_embedder_switch_from_text_default_nomic_v1_5_768_to_text_default_embeddinggemma_300m_768_via_existing_reindex_no_new_api_surface` -->

```rust,no_run
# use cg_catalog::embeddinggemma_test_support as candle_embeddinggemma;
use std::sync::Arc;

use candle_embeddinggemma::EmbeddingGemmaTextEmbedder;
use context_graph::{Embedder, EmbedderRegistration, Store, StoreConfig};

fn main() -> context_graph::Result<()> {
    let mut config = StoreConfig::default();
    let embedder = EmbeddingGemmaTextEmbedder::new(
        candle_embeddinggemma::default_embeddinggemma_offline_config(),
    );
    let space = embedder.embedding_space();
    config.extra_embedding_spaces.push(space.clone());
    let _store = Store::open_with_embedder_registrations(
        config,
        vec![EmbedderRegistration {
            space,
            embedder: Arc::new(embedder),
        }],
    )?;
    Ok(())
}
```

## Conversation history from the command line: `cg import`, `cg find`, `cg show`, `cg status`, `cg forget`

These five commands are the operator surface over the event-history library calls above. Every
rule — event identity, bindings, projection, ranking, erasure — lives in this crate and the Claude
Code connector; `cg` parses arguments, opens files, calls the library and renders its answer. JSON
is the default output of every one of them and `--format text` selects the human rendering.<!-- enforced by: `import_usage_errors_name_the_token_and_do_no_work`, `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

### `cg import`

```text
cd ~/projects/my-app
cg import .
```

A directory argument is the project working directory. `cg import` reaches exactly the Claude Code
session files Claude recorded for that directory — under `~/.claude/projects`, or
`$CLAUDE_CONFIG_DIR/projects` — and never searches the project tree for `.jsonl` files or picks up
another project's sessions. Helper-agent transcripts are counted and reported, not imported; naming
one imports it under its parent session.<!-- enforced by: `cg_import_project_directory_never_sweeps_working_tree_jsonl_or_imports_other_project_history` -->

The first import of an unlinked project asks once, on the controlling terminal and never on stdout,
`Context name [<project-basename>]:`; Enter accepts the suggestion. A name that already belongs to a
Context needs an explicit `y` before the project is linked to it. The answer is stored as an opaque
node-local binding, so later imports, finds, shows and statuses from the project or any
sub-directory use that Context with no flag, and `.`, an absolute path and a symlink alias all reach
the same binding.<!-- enforced by: `first_interactive_project_import_accepts_suggested_name_on_enter_descendant_reads_reuse_binding_and_noninteractive_import_returns_retry_without_writes`, `an_existing_same_name_context_requires_confirmation_and_enter_means_no`, `cg_import_project_directory_never_sweeps_working_tree_jsonl_or_imports_other_project_history` -->

Without a controlling terminal an unlinked project returns `context_binding_required` with the
suggested name and the exact retry command, and writes nothing; `--context NAME` creates or links
the Context without a prompt. A project with no matching history returns
`project_history_not_found` and creates no store, Context or binding. A project already linked to a
different Context than `--context` names returns `context_binding_conflict` and changes
nothing.<!-- enforced by: `noninteractive_unbound_project_import_returns_the_retry_without_writing`, `cg_import_project_without_matching_claude_history_creates_no_context_or_binding_and_reports_not_found`, `a_bound_project_refuses_a_different_explicit_context_without_changing_either` -->

A single `.jsonl` file imports exactly that session and requires one `--context NAME`; it never
reads or changes a project binding. The command accepts one path; a second path, an unknown flag, a
missing flag value or a `--search` value other than `background`, `wait` or `defer` is a usage error
that names the token and does no work.<!-- enforced by: `import_usage_errors_name_the_token_and_do_no_work` -->

Capture is durable and model-free. What happens to search afterwards is `--search`:<!-- enforced by: `explicit_context_import_defers_search_and_keeps_source_facts_out_of_its_receipt`, `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

| Mode | What it does | State when events are pending |
| --- | --- | --- |
| `background` (default) | checks the local Nomic model without downloading, releases the store and hands search to a detached worker | `captured_search_scheduled`, exit 0 |
| `wait` | runs the same projection in the foreground until search is caught up | `complete`, exit 0 |
| `defer` | captures only; no model check and no worker | `captured_search_deferred`, exit 0 |

A caught-up import returns `complete` under every mode. When the local model is not installed,
`background` and `wait` still capture, return `captured_search_behind` (exit 1) and name
`cg model fetch --defaults` followed by the same import with `--search wait`; nothing is downloaded.
A projection that fails after capture leaves every captured event durable and also returns
`captured_search_behind`.<!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight`, `cg_import_search_wait_embedding_failure_returns_durable_capture_receipt_and_typed_behind_status`, `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

A malformed or divergent file stops only itself: every other selected file is captured, the failed
file's committed prefix stays durable, no worker starts, and the receipt returns `source_partial`
(exit 1) naming the failed file, its position and the reason.<!-- enforced by: `cg_import_continues_past_a_malformed_file_captures_every_other_file_and_reports_source_partial_naming_the_failed_file` -->

The JSON receipt carries `ok`, `state`, the Context and whether it came from the project binding or
an explicit name, separate `capture` and `search` objects, pending counts and a `retry` command. It
contains no source text and no source path.<!-- enforced by: `cg_import_project_directory_never_sweeps_working_tree_jsonl_or_imports_other_project_history`, `explicit_context_import_defers_search_and_keeps_source_facts_out_of_its_receipt` -->

### `cg find` and `cg show`

```text
cg find "why was Iroh selected instead of NATS?" --format text
cg show <passage_id> --format text
```

`cg find` returns ranked conversation passages, never a generated answer, from the Context linked
to the current project; `--context a,b` searches several named Contexts and `--all-contexts` every
Context in the store, ranked together. Each hit carries its Context and the `passage_id` that
`cg show` accepts verbatim. An unknown name fails the whole selection before anything is read.
History captured but not yet searchable is named as omitted, and when no selected Context is
searchable the command returns `history_search_not_ready`; `find` never starts or waits for
projection.<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

`cg show` expands a passage to the exact stored events behind it, in order, from the database — the
original session file is not needed. A passage outside the selected Contexts returns
`passage_not_found` without revealing its Context or text.<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets`, `search_wait_round_trip_reads_linked_exact_events_from_the_store_with_real_nomic` -->

### `cg status`

`cg status` reads the existing store through the read-only reader and never opens it for writing,
so it answers while an import or a projection worker holds the store. For each selected Context it
reports graph memory totals and history separately: captured events, source files and sessions,
coverage, event placement and search. Search is exactly one of `ready`, `running` (only while a
projection holds a live lease), `pending`, `blocked` (the local model is not installed) or
`failed`.<!-- enforced by: `status_separates_capture_from_search_coverage_per_selected_context`, `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

When the store does not exist, `cg status` writes nothing and creates nothing. Text output is
exactly two lines on stderr, and JSON is exactly
`{"ok":false,"state":"store_not_found","store":{"db_path":"…"},"retry":"create_store_with_write_command"}`,
both with exit 1. `cg find` and `cg show` answer a missing store the same way and point at
`cg import`.<!-- enforced by: `cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either` -->

Model connections are reported by `cg model status`.

### `cg forget`

```text
cg forget --session <ref>
cg forget --context my-app --yes
```

Forgetting is deleting. One session, or every imported session of one Context, is erased with
everything derived from it — events, exact anchors, checkpoints, locators, coverage links, search
passages and vectors — in one authoritative ContextDB `PURGE`, or nothing changes. Search
generations summarise a whole Context, so forgetting one session also clears that Context's search
index. `cg forget` then schedules the re-indexing of the remaining history on the same background
worker a background import uses, and says so in its receipt; without the local model it reports the
rebuild as blocked with `cg model fetch --defaults` as the next action. `cg status` shows that
Context as pending, running and then ready. The Context and its project binding remain.<!-- enforced by: `forgetting_a_session_erases_its_events_and_everything_derived_while_keeping_the_context`, `forgetting_a_context_erases_all_of_its_history_and_the_next_capture_lands_in_the_same_context`, `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets`, `cg_forget_confirms_on_the_terminal_and_enter_changes_nothing` -->

On a terminal it asks `Forget <N> sessions (<M> events) from "<context>"? [y/N]` and Enter means
no; without a terminal it requires `--yes` and otherwise returns `confirmation_required` (exit 2)
with the exact retry command. A store that places events at a tenant hub is refused as
`discard_denied`; this build does not issue the engine's edge-local discard.<!-- enforced by: `cg_forget_confirms_on_the_terminal_and_enter_changes_nothing`, `a_tenant_hub_store_is_refused_rather_than_purged_from_an_edge` -->

### When a writing command meets a held store

`cg import` and `cg forget` open the store for writing. When another process is writing it they
return `store_busy`, name the holder and the store, and exit 4. When direct readers are holding it
they print the sentence the condition writes: which store, how many readers were observed, and each
verified reader by process id and process name; a reader that was observed and published no
identity is disclosed rather than left out.<!-- enforced by: `the_crate_readme_promises_only_the_readers_the_store_could_verify`, `a_command_refused_by_a_reader_held_store_names_the_readers_it_waits_on` -->

### Asking a running context-graph a question

Administrative questions are asked of the *process* that holds the store, not of
the file — the store path is the whole address, and no caller configures a socket
address; the channel is contextdb's local owner channel, selected from that path
and the runtime root. The
contributor-facing rules for that owner plane (which typed results license
opening the file directly, why the handler and the embedders ride one writable
open, and which single door in-process totals go through) are stated once, in
`AGENTS.md` at the repository root. They are not repeated here; this file is the
operator's and the library consumer's.<!-- enforced by: `a_live_owner_answers_a_control_frame_addressed_only_by_the_store_path`, `every_path_the_root_agents_file_names_is_in_the_tree`, `the_agents_routing_file_promises_only_the_readers_the_store_could_verify` -->

## Reading a store you must not change

A runtime that cannot take the store — another process owns it, or its own job
forbids it to risk changing one — still opens it read-only through
`ConsumerReader::open`. That open answers three ways, and each sends you
somewhere different:<!-- enforced by: `an_absent_store_is_refused_as_missing_and_leaves_the_directory_untouched`, `a_store_needing_writable_recovery_is_its_own_typed_outcome`, `a_store_this_process_cannot_look_at_is_refused_as_unreadable_never_as_missing` -->

- **`StoreMissing`** — there is nothing at that path, and nothing was created.
  Restore a backup, or point at the right store.
- **`StoreNeedsWritableRecovery`** — the store is there and its committed image
  needs one writable open to settle: the last writer left state only a writable
  hydration performs. This is the one refusal you may answer by opening the
  store WRITABLE (`Store::open_existing_for_change`), recovering it, and
  carrying on. The substrate's own words for the condition travel in `reason`.
- **`StoreUnreadable`** — the store is there and cannot be read at all: a
  permission, a mount, real damage. Reaching for a writable handle here is the
  worst available move; diagnose it first.<!-- enforced by: `an_absent_store_is_refused_as_missing_and_leaves_the_directory_untouched`, `a_store_needing_writable_recovery_is_its_own_typed_outcome`, `a_genuinely_damaged_store_is_still_reported_unreadable` -->

The three are separate typed answers, not one answer with three strings in it,
so nothing downstream has to parse prose to decide whether to open a store for
writing.<!-- enforced by: `recoverable_and_unreadable_are_two_typed_answers_not_one` -->

Once open, the reader answers SQL and cursors — and the typed graph reads, over
`ConsumerReader::graph()`: the audit trail, contexts, entities, intentions,
decisions and observations, with the same signatures and the same answer types
as the writable `Store`'s. They are the same code: one SQL text and one decoder
per read, run over whichever handle you reached it through, so the read-only
answer cannot drift from the writable one. No writable handle is anywhere in the
path, so no sequence of view reads changes the store file, its companions, or
the directory they sit in.<!-- enforced by: `each_typed_graph_read_answers_what_the_writable_store_answered`, `the_whole_graph_read_set_leaves_the_stopped_store_byte_identical`, `an_observation_list_bigger_than_one_answer_is_the_writable_store_s_list`, `an_audit_trail_bigger_than_one_answer_is_the_writable_store_s_trail` -->

The declaration is the whole of what the view may reach: a read whose table you
did not declare is refused as `UndeclaredTableRefused` naming that table, before
the engine is asked. Declare them from context-graph's own published specs
(`context_graph::schema::TABLES`), never from a copy of the DDL text. Per read:<!-- enforced by: `a_query_reaching_a_table_the_consumer_did_not_declare_is_refused`, `an_undeclared_table_reached_through_a_join_or_a_subquery_is_still_refused`, `a_truthful_declaration_answers_the_same_rows_on_the_direct_and_the_live_owner_routes` -->

| Read | Tables it needs |
| --- | --- |
| `audit_query` | `audit_log` |
| `list_contexts`, `get_context` | `contexts` |
| `get_entity` | `entities` |
| `get_intention` | `intentions` |
| `get_decision` | `decisions`, `edges`, `decision_basis_snapshots`, `outcomes` |
| `get_observation`, `list_observations` | `observations`, `evidence`, `evidence_availability_log` |

The nested tables are there because the answers are nested — a decision carries
the intentions it serves, the precedents it cites, its basis snapshots and its
outcome; an observation carries its evidence, and each piece of evidence the
retention status its availability log last recorded. Declare only the headline
table and you are told which table you still owe, rather than handed a stripped
row.<!-- enforced by: `a_graph_read_over_an_undeclared_table_is_refused_by_that_table_name`, `the_readers_inventory_exposes_nothing_beyond_its_declaration` -->

```rust,no_run
use std::collections::BTreeSet;

use context_graph::schema::TABLES;
use context_graph::{
    CgError, ConsumerDeclaration, ConsumerReader, ConsumerSchema, ConsumerTable, DecisionId,
};

/// Why a decision fired, read out of a store this process must not change.
fn explain(store_path: &std::path::Path, decision: DecisionId) {
    let declaration = ConsumerDeclaration {
        schema: ConsumerSchema {
            namespace: "vigil".to_string(),
            tables: ["decisions", "edges", "decision_basis_snapshots", "outcomes"]
                .iter()
                .map(|wanted| {
                    let spec = TABLES
                        .iter()
                        .find(|spec| spec.name == *wanted)
                        .expect("context-graph publishes a spec for every table it owns");
                    ConsumerTable {
                        name: spec.name.to_string(),
                        ddl: spec.ddl.to_string(),
                    }
                })
                .collect(),
        },
        scope_labels: BTreeSet::new(),
    };

    let reader = match ConsumerReader::open(store_path, declaration) {
        Ok(reader) => reader,
        // The store is fine and only needs settling: take the writable
        // existing-only door, recover it, and carry on serving.
        Err(CgError::StoreNeedsWritableRecovery { path, reason }) => {
            eprintln!("{} needs a writable open first: {reason}", path.display());
            return;
        }
        Err(other) => {
            eprintln!("no read-only answer from this store: {other}");
            return;
        }
    };

    match reader.graph().get_decision(decision) {
        Ok(Some(decision)) => println!("{} — {}", decision.decision_type, decision.description),
        Ok(None) => println!("no such decision in this store"),
        Err(CgError::UndeclaredTableRefused { table }) => {
            eprintln!("declare `{table}` to read this");
        }
        Err(other) => eprintln!("{other}"),
    }

    reader.close().expect("let the read session go");
}
```

## Tenant Fabric (`<data_root>/cg/fabric.toml`)

A node with two unequal machines can offload session *understanding* onto a
stronger enrolled machine in its own tenant fabric. Enrollment lives in a
per-node `<data_root>/cg/fabric.toml` (separate from the per-work-root
`capture.toml`), read through the `cg-node-fs` helper. A node with no
`fabric.toml` is a plain single-node install and never contacts a hub.<!-- enforced by: `no_fabric_toml_is_exactly_single_node_behavior` -->

```toml
[fabric]
# Pasted hub enrollment ticket / endpoint spec for your tenant fabric.
hub_endpoint = "enrollment-ticket"
# The tenant this node enrolls under. Cross-tenant movement is impossible at
# the engine layer.
tenant = "tenant-alpha"
# Input-sharing switch. DEFAULT ON (omitting the key is ON). The one visible
# off knob; set false to keep every understanding input on this machine.
sharing = true
# Milliseconds an ephemeral input copy survives after its result lands before
# each node sweeps its own copy. Defaults to about a day.
grace_ms = 86400000
# Per-source opt-out: a source named here keeps its understanding input local
# (never copied off the machine) regardless of the global sharing switch.
per_source_opt_out = ["cg-record"]
```

`event_placement` in the same file (`"local_only"` or `"tenant_hub"`) is the event placement a
fresh store is created with; omitting it keeps a new store local-only and an existing store's
persisted policy. `cg import` passes it to the library and never widens a store that already holds
events.<!-- enforced by: `local_and_unbound_tenant_placement_keep_event_schema_and_append_behind_real_policy_binding` -->

### What moves, and what never does

The permanent raw store (`cg_source_chunks`) **never syncs** — it stays on the
capturing machine, always. What may move is a separate, **ephemeral copy of the
understanding *input*** (the verbatim chunk a worker needs to run the model),
and only to the user's own **same-tenant** machines. That movement is **on by
default**, governed by the single visible `sharing` knob, with a **per-source
opt-out** for individual sources, and each copy is **retention-deleted** on
every node after its result lands plus the `grace_ms` window. Node id, tokens,
and wall-clock ride back as an operational receipt only, never a decision basis.
`cg sync` speaks the pasted ticket / endpoint of your fabric.<!-- enforced by: `permanent_raw_store_never_syncs`, `omitting_sharing_key_defaults_on_and_actually_pushes_input`, `per_source_opt_out_keeps_that_sources_input_local_under_sharing_on`, `input_copies_deleted_after_result_lands_plus_grace`, `cg_sync_surface_uses_ticket_endpoint_idiom_without_removed_transport` -->

## Continuous Integration

CI runs the workspace test gate on `macos-latest` and `ubuntu-latest`.
`windows-latest` is excluded deliberately: the contextdb substrate's durable
large-request sync staging needs descriptor-relative filesystem operations and
is Unix-only by design, so the sync test surface cannot be built there.<!-- enforced by: `cg_github_actions_workflow_runs_on_matrix_of_macos_latest_ubuntu_latest_and_documents_the_deliberate_windows_exclusion` -->

CI points `CG_MODEL_CACHE_ROOT` at runner scratch space and gives that exact
directory to `actions/cache`; runner scratch does not persist independently
between jobs, so only the cache action carries those files forward. On a cold
miss, CI seeds the same root with `cg model fetch --defaults` before running the
offline real-model proofs. On an ordinary Linux or macOS machine without the
override, the cache root falls back to `~/.cache/cg/models` (with
`$XDG_CACHE_HOME` replacing `~/.cache` when set). Key the cache on the embedder
source file rather than on `Cargo.lock`: the model id and cache layout live in
that file, and the lockfile does not change when they do.<!-- enforced by: `ci_real_model_job_seeds_and_reuses_one_declared_offline_cache_root`, `ci_workflow_model_cache_key_hashes_pinned_revision_constant_source_file_not_cargo_lock_parsed_yaml`, `nomic_weights_first_load_writes_to_xdg_compliant_cache_path_under_cg_namespace_on_linux_and_macos` -->

```yaml
env:
  CG_MODEL_CACHE_ROOT: ${{ runner.temp }}/cg-model-cache
steps:
  - uses: actions/cache@v4
    id: nomic-model-cache
    with:
      path: ${{ env.CG_MODEL_CACHE_ROOT }}
      key: cg-nomic-${{ runner.os }}-${{ hashFiles('workspace/context-graph/crates/context-graph/src/embedder/nomic.rs') }}
  - name: Seed Nomic model cache on cold miss
    if: steps.nomic-model-cache.outputs.cache-hit != 'true'
    run: |
      rustup run stable cargo run -p context-graph-cli --bin cg -- model fetch --defaults
      rustup run stable cargo test -p context-graph --test nomic_document_preparer_adds_exactly_one_system_task_prefix
  - run: rustup run stable cargo test --workspace
  - name: Gate wall-clock baseline never rises without review
    env:
      DOCUMENTATION_BASE_SHA: ${{ github.event.pull_request.base.sha || github.event.before }}
    run: |
      base="${DOCUMENTATION_BASE_SHA:-HEAD^}"
      if [[ -z "$base" ]] || [[ "$base" =~ ^0+$ ]] || ! git rev-parse --verify --quiet "${base}^{commit}" > /dev/null; then
        base="HEAD^"
      fi
      git rev-parse --verify --quiet "${base}^{commit}" > /dev/null
      printf 'comparing the gate wall-clock baseline against %s\n' "$base"
      python3 scripts/check-gate-wall-clock.py --base "$base"
  - name: Measure this run's own test wall-clock
    run: rustup run stable cargo nextest run --workspace --no-fail-fast --profile gate
  - name: Gate wall-clock stays inside the frozen baseline
    env:
      CG_GATE_JUNIT: ${{ github.workspace }}/workspace/context-graph/target/nextest/gate/junit.xml
    run: rustup run stable cargo test -p context-graph --test test_estate_audit gate_wall_seconds_never_exceed_the_frozen_baseline -- --ignored --exact
```

The wall-clock arm compares seconds someone else already measured, so CI produces a measurement of
its own: `cargo test --workspace` writes no JUnit, and the nextest step above writes one under the
`gate` profile checked in at `.config/nextest.toml` — the same profile `scripts/gate.sh` uses. The
arm then runs unconditionally with `CG_GATE_JUNIT` naming that file and refuses a test binary that
took longer than its ceiling, which is its recorded seconds times the declared tolerance factor
plus the declared slack. The factor covers drift that scales with the binary; the slack covers the
part that does not — one process spawn per test, and a first file read a later run finds already
cached — which is what keeps a binary whose whole run is a handful of spawns from being failed on
the machine's scheduling, and what makes a measurement taken in one test pool comparable against a
value frozen from a differently-sized one. `scripts/gate.sh`
runs the same two steps and additionally runs `scripts/check-gate-wall-clock.py` to refuse a
baseline value that
rose.<!-- enforced by: `the_ignored_wall_clock_arm_is_invoked_by_name_in_the_gate_script_and_ci`, `the_readme_ci_snippet_names_the_wall_clock_steps_the_workflow_actually_runs` -->
