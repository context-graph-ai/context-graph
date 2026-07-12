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

## Tenant Fabric (`<data_root>/cg/fabric.toml`)

A node with two unequal machines can offload session *understanding* onto a
stronger enrolled machine in its own tenant fabric. Enrollment lives in a
per-node `<data_root>/cg/fabric.toml` (separate from the per-work-root
`capture.toml`), read through the `cg-node-fs` helper. A node with no
`fabric.toml` is a plain single-node install and never contacts a hub.

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
`cg sync` speaks the pasted ticket / endpoint of your fabric.

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
