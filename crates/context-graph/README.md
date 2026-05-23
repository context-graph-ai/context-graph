# context-graph

## Choosing an Embedder

The bundled text embedder is `nomic-embed-text-v1.5`, exposed through the
default text space `text_default_nomic_v1_5_768`. It runs locally through
Candle, does not require an API key, and stores first-load artifacts in the
cg embedder cache. The upstream artifact is published at
`huggingface.co/nomic-ai/nomic-embed-text-v1.5`; the model weights are about
547MB, which is roughly three minutes on a 25Mbps developer connection before
the cache is warm.

`candle-embeddinggemma` remains available as a sibling opt-in for consumers
that want the prior `text_default_embeddinggemma_300m_768` space. Register it
through `Store::open_with_embedder_registrations`, then use `reindex_space`
when moving stored rows between registered spaces. context-graph ships the
English Nomic v1.5 default; multilingual embedders are consumer-declared
through the same registration lane.

```rust
use std::sync::Arc;

use candle_embeddinggemma::EmbeddingGemmaTextEmbedder;
use context_graph::{Embedder, EmbedderRegistration, Store, StoreConfig};

# fn main() -> context_graph::Result<()> {
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
# Ok(())
# }
```

## Continuous Integration

CI runs the workspace test gate across `windows-latest`, `macos-latest`, and
`ubuntu-latest`. Cache the local embedder artifact directory by `Cargo.lock`
so each runner keeps the Nomic cache warm across jobs.

```yaml
- uses: actions/cache@v4
  with:
    path: |
      ~/.cache/cg/embedders/
      ~/AppData/Local/cg/embedders/
    key: cg-embedders-${{ runner.os }}-${{ hashFiles('Cargo.lock') }}
```
