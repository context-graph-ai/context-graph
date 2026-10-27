# Register an embedder

Prerequisites: Rust application, library dependency, installed artifacts for the chosen model.

Add `context-graph` from a git checkout using the [path dependency instructions](library-quickstart.md#add-workspace-crates).

An opt-in EmbeddingGemma registration exposes its sibling embedding space. <!-- enforced by: `candle_embeddinggemma_consumer_registers_via_storeconfig_extra_embedding_spaces_under_text_default_embeddinggemma_300m_768_space_id` -->

A conflicting declared dimension returns `EmbeddingSpaceConfigMismatch`. <!-- enforced by: `store_open_extra_embedding_spaces_conflicting_dimension_returns_embedding_space_mismatch_naming_field` -->

Store registration accepts the EmbeddingGemma sibling space. <!-- enforced by: `candle_embeddinggemma_consumer_registers_via_storeconfig_extra_embedding_spaces_under_text_default_embeddinggemma_300m_768_space_id` -->

`reindex_space` reports completion in the registered destination space. <!-- enforced by: `runtime_cross_embedder_switch_from_text_default_nomic_v1_5_768_to_text_default_embeddinggemma_300m_768_via_existing_reindex_no_new_api_surface` -->

Searching that space returns the reindexed evidence. <!-- enforced by: `runtime_cross_embedder_switch_from_text_default_nomic_v1_5_768_to_text_default_embeddinggemma_300m_768_via_existing_reindex_no_new_api_surface` -->

```rust,ignore
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

## Register event-history encoding

`EmbedderConfig::default_nomic_offline()` selects offline operation. <!-- enforced by: `public_default_nomic_offline_configuration_uses_the_canonical_installed_model_for_event_history` -->

It pins the canonical installed revision. <!-- enforced by: `public_default_nomic_offline_configuration_uses_the_canonical_installed_model_for_event_history` -->

It declares `deny-download`. <!-- enforced by: `public_default_nomic_offline_configuration_uses_the_canonical_installed_model_for_event_history` -->

Event-history registration reuses the caller's model Arc. <!-- enforced by: `event_history_registration_reuses_the_loaded_model_and_refuses_an_unfit_underlying_embedder` -->

A download-enabled underlying configuration is refused. <!-- enforced by: `event_history_registration_reuses_the_loaded_model_and_refuses_an_unfit_underlying_embedder` -->

```rust,ignore
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

See [event-history encoding](models-and-embedders.md#encoding-event-history) for the sealed role methods.

These examples are illustrative; this page does not run doctests.

See [models and embedders](models-and-embedders.md) for preparation and cache requirements.
