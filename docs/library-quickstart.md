# Library quickstart

Use the Git dependency below.

Prerequisites: Git, and Rust 1.94.1 via rustup; 1.94.1 is the toolchain the repository's `rust-toolchain.toml` pins. <!-- enforced by: `readme_states_the_rust_version_rust_toolchain_toml_pins` -->

```text
rustup toolchain install 1.94.1
```

## Add workspace crates

The workspace manifests and lockfile pin ContextDB to one exact commit, which the [README](../README.md#download-a-development-build) names. <!-- enforced by: `manifests_lockfile_and_readme_name_the_same_exact_contextdb_revision` -->

```text
git clone --branch dev https://github.com/context-graph-ai/context-graph.git
cargo +1.94.1 new my-app
cd my-app
cargo +1.94.1 add context-graph --path ../context-graph/crates/context-graph
```

Add the checked-out crate to your application’s `Cargo.toml`:

```toml
[dependencies]
context-graph = { path = "../context-graph/crates/context-graph" }
```

Optional [EmbeddingGemma registration](register-an-embedder.md): add its sibling workspace path.

```toml
[dependencies]
candle-embeddinggemma = { path = "../context-graph/crates/candle-embeddinggemma" }
```

## Write and read

Replace `src/main.rs` with this first offline memory write and read:

```rust,ignore
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

The store schema refuses an Intention row whose Context id does not exist. <!-- enforced by: `test_fk_intention_invalid_context_id_rejected` -->

A created Intention can be read back after reopening the store. <!-- enforced by: `intention_create_persists_across_reopen` -->

A store configured with the disabled embedder records the disabled loader route. <!-- enforced by: `store_open_with_default_config_against_freshly_created_path_writes_cg_schema_version_row` -->

```text
cargo +1.94.1 run
```

A created Intention persists across store reopen. <!-- enforced by: `intention_create_persists_across_reopen` -->

These examples are illustrative; this page does not run doctests.

Next: [library API](library-api.md) and [ontology](ontology.md).
