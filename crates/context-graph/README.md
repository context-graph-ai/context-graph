# context-graph

The embeddable memory core behind `cg`, kept in a local store.

Intentions and decisions written to the store persist across a reopen, and a decision keeps the intention it serves and the entities it is based on. <!-- enforced by: `intention_create_persists_across_reopen`, `decision_create_with_serves_and_based_on_edges_persisted` -->

## Add from source

From an application beside this repository:

```sh
cargo add context-graph --path ../context-graph/crates/context-graph
```

Follow the [library quickstart](../../docs/library-quickstart.md) for an offline write and read.

| Guide | Purpose |
|---|---|
| [Library quickstart](../../docs/library-quickstart.md) | First application |
| [Library API](../../docs/library-api.md) | Store, typed reads and ownership |
| [CLI reference](../../docs/cli.md) | Operator command contracts |
| [Models and embedders](../../docs/models-and-embedders.md) | Choosing an embedder |
| [Contributing](../../CONTRIBUTING.md) | CI and verification |
