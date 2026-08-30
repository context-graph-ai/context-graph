# context-graph

Durable, queryable memory for an agent or an application: the intentions it was
serving, the decisions it made, the observations it recorded, and the digests
that tie them together — stored in a local contextdb database and searched by
meaning as well as by structure.

This repository ships two things. `context-graph` is the embeddable memory core:
link it as a library and you get the ontology, the store and its domain
behavior, embedder registration, bounded status reads, the administrative owner
plane, and a read-only view for a process that must not change the store. `cg`
is the command-line binary built on top of it — one consumer of the core, not a
privileged one.

The core is embedded rather than served. There is no daemon and no
caller-configured socket address: one process holds the store's writable handle,
everything else is a typed call, and a question asked of a *running*
context-graph travels contextdb's local owner channel, addressed by the store
path itself.

## Build it

You need:

- Rust **1.94.1**. `rust-toolchain.toml` pins it, so `rustup` installs the right
  toolchain on first build.
- A **contextdb checkout beside this one**. The workspace depends on contextdb by
  path (`Cargo.toml`, the `contextdb-*` entries), so the layout is:

  ```
  <parent>/
    context-graph/     # this repository
    contextdb/         # https://github.com/context-graph-ai/contextdb
  ```

  This is the layout CI arranges too (`.github/workflows/ci.yml`), which is the
  place to look if you want the exact revisions a green run used.

Then, from this repository's root:

```sh
cargo build --release -p context-graph-cli
```

That produces the `cg` binary at `target/release/cg`. Put it on your `PATH`, or
call it by path.

## First run

`cg status` is the command to start with. It reports whether cg is connected to
a model and what state this machine's memory store is in — and on a machine that
has neither, it tells you so plainly rather than failing:

```
$ cg status
{"bucket":"not_connected","condition":"not_connected", … ,
 "store_path":"~/.cg/context-graph.db",
 "summary":"context: default goals: 0 decisions: 0 observations: 0 | pending understanding: 0 | raw kept: 0 chunks / 0 bytes"}
$ echo $?
0
```

Two things to know about that first run:

- **It creates the store.** `cg status` opens the store for writing — that is how
  it takes the totals — and materializes an empty one (the database file and its
  lock) where none exists. The store lives under `~/.cg` unless `CG_DATA_ROOT`
  says otherwise, and the `store_path` field always names the store the totals
  are about.
- **`not_connected` is an answer about the model, not about the store.** The
  store beside it was read successfully; the zeros are real zeros. Connecting a
  model is the next step: `cg model auth login --connection <provider>
  --consent`, then `cg selftest` to spend one consented proof call.

`cg --help` lists the full command surface.

## Where to read next

| You are | Read |
| --- | --- |
| Running `cg`, or embedding the crate | [`crates/context-graph/README.md`](crates/context-graph/README.md) — choosing an embedder, the `cg status` contract in full, reading a store you must not change, and tenant fabric configuration. It is also the crate's rendered API documentation. |
| Changing this repository | [`AGENTS.md`](AGENTS.md) — which crate a change belongs in, and the invariants a change may never break, each with the file that enforces it. |

## License

Apache-2.0. See [`LICENSE`](LICENSE).
