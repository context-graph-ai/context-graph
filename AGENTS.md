# AGENTS.md — where a change lives in this repository, and what it may never do

This file is routing and architecture only. It carries no product intent: what context-graph is
for, what it promises, and what it refuses to become live in the root product documents, and a
change that needs intent reads those, not this.

Read this before editing. Every rule below is a description of what the source does today, with the
file that does it named beside it.

## The eight crates

| Crate | Path | What it owns |
|---|---|---|
| `context-graph` | `crates/context-graph/` | The embeddable memory core: the ontology, the store and its domain behavior, the administrative owner plane, bounded status reads, fabric, and the bundled default text embedder (`crates/context-graph/src/embedder/nomic.rs`, behind the default feature `nomic-embedder`). |
| `cg-cli` | `crates/cg-cli/` | The `cg` binary: argument parsing, help text, output rendering, and the mapping from a typed library condition to an exit code. |
| `cg-cli-conventions` | `crates/cg-cli-conventions/` | The command-rendering conventions the CLI surfaces share: severities, findings, the doctor/selftest check registry and the report shapes. |
| `cg-catalog` | `crates/cg-catalog/` | Model catalog state: registered models, model connections, provider specs, and the artifact cache lifecycle. |
| `cg-node-fs` | `crates/cg-node-fs/` | The node deployment's filesystem roots: reading a per-node config file, and holding the per-data-root worker lock. |
| `cg-bench` | `crates/cg-bench/` | The correctness bench: the gold answer-key format and the frozen scoring method. Not a shipped product surface. |
| `candle-embeddinggemma` | `crates/candle-embeddinggemma/` | Loading and running the EmbeddingGemma text embedder. |
| `cg-vision-embedder` | `crates/cg-vision-embedder/` | Loading and running the bundled SigLIP-family joint image+text embedder. |

Routing, in the words a change usually arrives in:

- Store behavior, ontology, an owner-plane question, a status total — `context-graph`.
- A new command, a changed message, a changed exit code, help text — `cg-cli`.
- The shape a check, finding, severity or report is rendered in — `cg-cli-conventions`.
- A model registration, a connection, a provider spec, a cached artifact — `cg-catalog`.
- Reading a file on a deployed node, or the worker lock — `cg-node-fs`.
- Gold answers, scoring, banding — `cg-bench`. Its thresholds *are* the scoring registration; they
  are held fixed so per-lever numbers stay comparable, and changing one is a product-frame change,
  not engineering.
- A new model to embed with — `candle-embeddinggemma` or `cg-vision-embedder`, or a crate of the
  consumer's own. Every embedder implements `context_graph::Embedder` and is registered into a
  store as an embedding space. Never author a second embedding abstraction beside that trait.
- The **bundled default** text embedder — `nomic-embed-text-v1.5`, the one a `cg` install gets
  without asking — is not a sibling crate: it lives inside `context-graph` itself, at
  `crates/context-graph/src/embedder/nomic.rs`, behind the default feature `nomic-embedder`
  (`crates/context-graph/Cargo.toml`). Changing which model ships by default, its space name, or
  how it fetches and caches its weights is a change to that file, not to `candle-embeddinggemma`.

The core is what vigil and WASM targets link, so a node's ordinary on-disk work — reading a
per-node config file, holding the worker lock — lives one layer out, in
`crates/cg-node-fs/src/lib.rs`, and a change that reaches for the filesystem to do that kind of
work is in the wrong crate. Three places inside `crates/context-graph/src/` do touch the disk
today, and they are the whole list:

- `crates/context-graph/src/store/mod.rs` — `StoreReadProbe` appends its read trace
  (`create_dir_all` plus an appending `OpenOptions` open), and only when `CG_READ_PROBE_PATH` is
  set.
- `crates/context-graph/src/store/mod.rs` — `Store::initialize_fresh` removes the store file it is
  about to recreate.
- `crates/context-graph/src/embedder/nomic.rs` — the bundled default text embedder downloads its
  weights over `ureq` and writes them into the embedder cache. It is behind the `nomic-embedder`
  feature, which `crates/context-graph/Cargo.toml` makes a **default** feature: a consumer that
  wants a fetch-free core turns default features off.

Adding a fourth is a placement decision, not a detail: say which module and under which feature
here, or move the code. Note that a grep for `std::fs` does not find all of these —
`crates/context-graph/src/embedder/nomic.rs` imports it as `fs as file_system`.

## Invariants

### Administrative requests go through the owner-only door

Everything that asks a *running* context-graph a question lives in
`crates/context-graph/src/owner_control.rs`. A client calls `request_owner(store_path, frame)`; the
store path is the whole address. There is no socket path in this API and no second transport in
this repository — the channel, its framing, its deadlines and its refusals are contextdb's bounded
read surface, and this crate declares only its namespace and its read ceilings.

Only an explicitly owner-absent typed result licenses doing the work directly against the file:
`OwnerControlError::OwnerNotRunning` and `NoStore`, which are what `owner_absent()` matches. Every
other variant means an owner is there and did not answer, and is never a license to open the store
behind it. `OwnerRouteUnsupported` is the one narrow middle case, and it is a distinct type for
exactly that reason.

- Forbidden: licensing a fallback on a message string, a timeout, or "any error".
- Forbidden: adding a socket, a named pipe, an HTTP port, or a second framing to reach a running
  cg. `crates/context-graph/tests/reading_pass_through_replaces_the_control_socket.rs` is the proof
  that this route replaced the control socket rather than joining it.

### The owner handler and the embedders ride the original writable open

A runtime that both answers administrative frames and recognizes what it stores opens its store
once, with `Store::open_with_control_handler_and_embedder_registrations`
(`crates/context-graph/src/store/mod.rs`). One store, one open, both capabilities.

- Forbidden: opening a second `Store` or `Database` on the same path just to attach control
  handling. A handle opened for one caller's handler answers to that caller alone and is never
  shared with a plain opener, and a separate process opening the same path while the owner is live
  is refused outright.
- Proof: `crates/context-graph/tests/one_writable_open_carries_the_owner_handler_and_the_embedders.rs`.

### In-process status totals are bounded, and refusal is never a count

`crates/context-graph/src/status_reads.rs` is the one door status totals go through. The session is
opened on the LIVE database the calling process already holds, never on the store's path: a process
asking its own store a question is not a client of itself, and routing it back through the owner
channel would spend an owner admission slot on the owner's own work.

- Every total is an aggregate, or a page of rows opened as a cursor and read to completion, under
  the ceilings `owner_control::context_graph_read_limits()` declares once for the whole crate. Only
  one page is ever held.
- Every cursor is closed on every exit path, including the error paths;
  `StatusReads::open_cursors()` is the counter that makes that checkable.
- Never raise the ceilings to make a scan fit. Page instead.
- A read that failed is never turned into a number. It travels the typed boundary and the caller
  surfaces the condition.
- Proofs: `crates/context-graph/tests/status_totals_are_bounded_on_a_large_store.rs`,
  `crates/context-graph/tests/a_failed_status_fold_leaves_no_cursor_open.rs`,
  `crates/cg-cli/tests/cg_status_never_reports_a_refused_store_as_empty.rs`.

### The runtime root variable locates the channel; it decides nothing

`CONTEXTDB_OWNER_READ_RUNTIME_DIR` is the substrate's own name, read in one place
(`crates/context-graph/src/owner_control.rs`) with the substrate's own meaning: it names a
DIRECTORY and nothing else. One directory has to serve both halves — the owner puts its channel
there and a client looks for it there — so cg invents no name of its own for it. Empty is the same
as unset.

- Forbidden: treating it as a switch. It never says whether owner inspection runs, and never what a
  read may spend; those are `owner_read_config` and `context_graph_read_limits`.
- Proof: `crates/context-graph/tests/one_runtime_root_reaches_both_sides_of_the_owner_plane.rs`.

### Conditions travel typed; only the CLI turns them into exit codes

The library hands back conditions, not rendered messages —
`MemoryCliError::StoreHeldByReaders` carries the `CgError` itself so an operator reads the sentence
the condition writes, naming which store, how many readers were observed, and each verified reader
by process id and name. The count and the identities are separate on purpose — the substrate can
observe a reader it cannot identify — so the sentence also discloses any holder that published no
identity rather than stopping at the verified list. The mapping from condition to exit code lives only in `crates/cg-cli/src/main.rs`
(`map_store_open_err`, `memory_error`), matched by type and never by substring-matching an engine
Debug dump.

- A store another process is writing is `StoreBusy` and exit 4, on every command that opens the
  store. Do not give status a second vocabulary for it.
- The numeric exit code for a reader-held store is deliberately unsettled. Do not assign one, and
  do not let a test assert one.
- Proof: `crates/cg-cli/tests/a_busy_store_is_told_to_the_operator.rs`.

### A read-only consumer gets the typed graph reads, and gets them from one place

A downstream runtime reading a store it must not change (`crates/context-graph/src/consumer_read.rs`)
gets more than SQL: `ConsumerReader::graph()` carries the typed graph reads — `audit_query`,
`list_contexts`, `get_context`, `get_entity`, `get_intention`, `get_decision`, `get_observation`,
`list_observations` — at the same signatures and answer types as the writable `Store`'s.

- **One SQL text, one decoder, two executors.** Each read is authored once as a free function over
  `crate::store::graph_rows::GraphRows` (`crates/context-graph/src/store/graph_rows.rs`); the
  writable `Store` and the read-only view are the two implementations. Adding a second decoder,
  copying a statement, or letting the two answer differently is the defect this seam exists to
  prevent. When a typed read changes, change the shared function — never one door's copy.
- **The view rides the consumer session and nothing else.** It reaches the store only through
  `ConsumerReader::execute`, so the declaration gate and the engine's row gate apply to a typed
  read exactly as they do to SQL, and a read over an undeclared table is
  `CgError::UndeclaredTableRefused` naming the table. No writable handle may appear in this path.
  What each read needs declared is stated per method in
  `crates/context-graph/src/consumer_read/graph.rs` and must stay true.
- **Proof:** `crates/context-graph/tests/a_stopped_store_answers_the_typed_graph_reads_without_changing.rs`.

### A store that needs recovering is not a store that cannot be read

The read-only and existing-only doors answer an open three ways, and the three are separate typed
values: `CgError::StoreMissing`, `CgError::StoreNeedsWritableRecovery` and
`CgError::StoreUnreadable`. The middle one is the only refusal a consumer may answer by opening the
store WRITABLE and recovering it; the last is where doing that is the worst available move. Folding
the recoverable case back into `StoreUnreadable` — or leaving it to be read out of a `reason`
string — costs a consumer either a store it could have taken back or a mutating open on damage
nobody diagnosed. The one conversion is `map_consumer_open_err` in
`crates/context-graph/src/store/convert.rs`; do not add a second, and do not widen `StoreUnreadable`
over the recoverable case.

- **Proof:** `crates/context-graph/tests/a_store_that_needs_a_writable_reopen_is_not_told_it_is_unreadable.rs`.

### Docs ship with the behavior

The public operator surface for administrative reads and `cg status` is
`crates/context-graph/README.md`, and the help text in `crates/cg-cli/src/main.rs` must agree with
it — `crates/cg-cli/tests/status_help_says_it_reads_memory_too.rs` requires status help to name
both the model connection and the memory store. A change to either behavior updates both in the
same change.

### Runtime and model work

cg understanding is hosted (bring your own key). A proposal to add a local understanding runtime is
a product decision, not an engineering choice: take it to the governing hosted-understanding
product documents and the workspace model-and-runtime strategy reference, and name the fact that
changed. Do not restate or re-derive that policy in this file.
