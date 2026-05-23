# context-graph

## Choosing an Embedder

The bundled text embedder is moving to `nomic-embed-text-v1.5`. EmbeddingGemma remains available through the `candle-embeddinggemma` sibling crate for consumers that declare the prior `text_default_embeddinggemma_300m_768` space and bind an embedder through `Store::open_with_embedder_registrations`.

```rust
use std::sync::Arc;
use candle_embeddinggemma::EmbeddingGemmaTextEmbedder;
use context_graph::{Embedder, EmbedderConfig, EmbedderRegistration, Store, StoreConfig};

# fn main() -> context_graph::Result<()> {
let mut config = StoreConfig::default();
let embedder = EmbeddingGemmaTextEmbedder::new(EmbedderConfig::disabled());
let space = embedder.embedding_space();
config.extra_embedding_spaces.push(space.clone());
let _store = Store::open_with_embedder_registrations(
    config,
    vec![EmbedderRegistration {
        space,
        embedder: Arc::new(embedder),
    }],
)?;
# Ok(())
# }
```

## Continuous Integration

The project CI should exercise Linux and macOS runners and cache local embedder artifacts.

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/cg/embedders
    key: cg-embedders-${{ hashFiles('Cargo.lock') }}
```
