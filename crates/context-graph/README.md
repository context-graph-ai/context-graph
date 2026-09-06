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

## Administrative reads and `cg status`

`cg status` answers about two things in one command: whether cg is connected to
a model, and what state this machine's Context Graph memory store is in. `no
model connection is active` is a statement about the model — the store beside it
may be perfectly healthy — and a store that could not be read is reported as
exactly that, never as a store with nothing in it.<!-- enforced by: `status_on_a_store_readers_are_holding_says_so_instead_of_reporting_it_empty`, `cli_status_without_connection_says_so_and_points_at_connect`, `the_top_level_help_entry_for_status_names_the_memory_store_beside_the_model` -->

The two are not two fields. The JSON carries **one** `condition`. When the store
could not be read, the store's refusal takes that field and the document has a
store path and no totals; otherwise the field is left to whatever the model half
has to say, and the store's answer is the totals sitting beside it. Those totals
are the context totals, the understanding queue, the raw footprint, and — for a
node enrolled in a tenant fabric — the fabric posture. The `fabric` object
appears only once `<data_root>/cg/fabric.toml` exists; a plain single-node
install has `fabric_jobs: []` and no `fabric` key.<!-- enforced by: `a_refused_status_read_is_reported_as_a_refusal_and_never_as_an_answer`, `a_store_that_really_holds_nothing_still_answers_with_its_absences`, `a_captured_store_answers_with_its_real_totals_when_no_read_refuses` -->

### What it writes, and how a script reads it

`cg status` writes one JSON document on stdout by default; `--format text`
writes the same answer as prose. That is true of the refusals below as well as
of the successful reads — the format flag chooses how an answer is written,
never how much of it you are told.<!-- enforced by: `an_unreadable_store_refusal_is_rendered_in_the_format_the_caller_asked_for`, `a_reader_held_refusal_is_rendered_in_the_format_the_caller_asked_for` -->

The field a script branches on is `condition`, not the exit code:<!-- enforced by: `cli_status_when_not_connected_is_a_distinct_not_connected_condition`, `corrupted_connections_file_fails_loud_and_typed` -->

```sh
cg status | jq -r .condition
```

That is the whole domain, and what each value tells you:

| `condition` | What it says | `bucket` |
| --- | --- | --- |
| `null` | A model is connected and has recorded no failure, and the store was read. | `null` |
| `StoreBusy` | The store could not be read: another process is writing it. Exit 4. | `memory` |
| `StoreHeldByReaders` | The store could not be read: direct readers are still holding it. | `memory` |
| `MemoryError` | The store could not be read at all — damage, a permission, a mount. Exit 1. | `memory` |
| `not_connected` | The store was read; no model connection is active. | `not_connected` |
| `connection_not_found` | The store was read; the `--connection` you named is not registered. | `not_found` |
| `provider_spec_invalid` | A provider spec on this machine is malformed. | `config` |
| `connections_file_unreadable` | `model-connections.toml` could not be read. | `unknown` |
| `no_host_session`, `consent_not_recorded`, `host_tool_missing`, `host_login_expired`, `live_call_failed` | The store was read; this is the connected model's own last recorded failure. | the model's own |

The casing is the short way to read that table. The store's own conditions are
Pascal-case; the model's and the configuration's are snake_case. A Pascal-case
value — equivalently, `"bucket": "memory"` — is the store refusing, and that
document carries `store_path` and no totals. Any other value means the store was
read successfully. So `"bucket": "memory"` marks a store REFUSAL rather than
every statement about the store: a healthy store on a machine with no model
connected reports `"bucket": "not_connected"` and carries its totals in the same
document.<!-- enforced by: `cli_status_when_not_connected_is_a_distinct_not_connected_condition`, `status_on_a_store_readers_are_holding_says_so_instead_of_reporting_it_empty`, `status_on_a_store_nobody_is_holding_still_answers_with_its_totals` -->

Write the branch accordingly. A `case` over the three store conditions alone
falls through on a fresh install, where the value is `not_connected`; give it a
default arm that reads "no store condition" as "the store answered." The totals
are in the document on every non-store condition except the three configuration
refusals — `connection_not_found`, `provider_spec_invalid` and
`connections_file_unreadable` — which answer before the store fields are added.<!-- enforced by: `cli_status_when_not_connected_is_a_distinct_not_connected_condition`, `corrupted_connections_file_fails_loud_and_typed` -->

The reader-held case's exit number is deliberately unsettled — it is not yet a
fixed contract you can branch on. The condition name is: it is what tells
"readers are finishing, retry" apart from "this store is
damaged."<!-- enforced by: `a_reader_held_refusal_is_rendered_in_the_format_the_caller_asked_for` -->

### `cg status` opens the store for writing, and creates one if none is there

`cg status` is not a read-only probe of the filesystem. It opens the store for
writing — that is how it takes the totals.<!-- enforced by: `cli_status` -->

| Today | Scheduled for replacement |
|---|---|
| Creates an empty store (the database file and its lock) when none exists at the path, and reports `goals: 0 decisions: 0 observations: 0` at exit 0. | A missing store answers `store_not_found` instead of being created. |

Zero totals therefore say exactly one thing: the store at the resolved path was
opened and no counted rows were found in it. They do not say whether that store
pre-existed this command or was materialized by it a moment ago, and they do not
say why it is empty — a store you have kept for months and have simply not
captured into is a legitimate zero. This command alone settles neither question.
What it does give you is the `store_path` field of the JSON document, which names
the store the totals are about: if `cg status` says your memory is empty, read
`store_path` first — a mistyped or unmounted `CG_DATA_ROOT` (the default is
`~/.cg`, and the store is `context-graph.db` under it) puts a fresh store
somewhere you were not looking, and the totals are telling the truth about the
wrong store.<!-- enforced by: `a_store_that_really_holds_nothing_still_answers_with_its_absences`, `status_on_a_store_readers_are_holding_says_so_instead_of_reporting_it_empty` -->

### What `cg status` says when the store cannot be read

- **Another process is writing the store** — a running cg, or a runtime that owns
  it. `cg status` gives this CLI's settled busy answer: condition `StoreBusy`, the
  words `store busy / already in use`, the holding process and the store path, and
  exit **4**. That is the same answer and the same code every other cg command that
  opens the store gives; status has no second vocabulary here.
- **Direct readers are holding the store.** Nothing is wrong with the machine —
  readers finish and the next attempt succeeds. The command fails under condition
  `StoreHeldByReaders` and prints the sentence the condition itself writes: which
  store, how many readers were observed, and each verified reader by process id
  and process name. The count and the names are separate answers: a reader can be
  observed and not identified, and such a holder is disclosed as one that
  published no identity rather than left out — so the names you are given are the
  readers the store could *verify*, and the sentence says when there are more. The
  numeric exit code for this case is deliberately unsettled, so branch a script on
  the failure and on the condition name rather than on a number.
- **Any other unreadable or refused store** surfaces its typed condition as the
  answer.<!-- enforced by: `status_on_a_store_another_process_is_writing_gives_this_cli_s_settled_busy_answer`, `a_command_refused_by_a_reader_held_store_names_the_readers_it_waits_on`, `a_store_whose_readers_could_none_of_them_be_named_still_presents_them_as_holders`, `the_crate_readme_promises_only_the_readers_the_store_could_verify` -->

In none of these cases does `cg status` print `goals: 0 decisions: 0
observations: 0`. Those totals are an *answer*: they say the store was read and it
is empty. Printing them for a store that was never read would tell a developer
whose memory is intact that it is gone, and tell an agent deciding whether to
capture that there is nothing worth preserving. A refused read has no totals, and
its refusal is its summary.<!-- enforced by: `status_on_a_store_readers_are_holding_says_so_instead_of_reporting_it_empty`, `a_refused_status_read_is_reported_as_a_refusal_and_never_as_an_answer` -->

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
