# AGENTS.md — where a change lives in this repository, and what it may never do

This file is routing and architecture only. It carries no product intent: what context-graph is
for, what it promises, and what it refuses to become live in the root product documents, and a
change that needs intent reads those, not this.

Read this before editing. Every rule below is a description of what the source does today, with the
file that does it named beside it.

## The nine crates

| Crate | Path | What it owns |
|---|---|---|
| `context-graph` | `crates/context-graph/` | The embeddable memory core: the ontology, the store and its domain behavior, the administrative owner plane, bounded status reads, fabric, and the bundled default text embedder (`crates/context-graph/src/embedder/nomic.rs`, behind the default feature `nomic-embedder`). |
| `cg-cli` | `crates/cg-cli/` | The `cg` binary: argument parsing, help text, output rendering, and the mapping from a typed library condition to an exit code. |
| `cg-cli-conventions` | `crates/cg-cli-conventions/` | The command-rendering conventions the CLI surfaces share: severities, findings, the doctor/selftest check registry and the report shapes. |
| `cg-catalog` | `crates/cg-catalog/` | Model catalog state: registered models, model connections, provider specs, and the artifact cache lifecycle. |
| `cg-node-fs` | `crates/cg-node-fs/` | The node deployment's filesystem roots: reading node configuration and machine-memory limits, holding worker locks, and atomically replacing bounded worker requests. |
| `cg-bench` | `crates/cg-bench/` | The correctness bench: the gold answer-key format and the frozen scoring method. Not a shipped product surface. |
| `candle-embeddinggemma` | `crates/candle-embeddinggemma/` | Loading and running the EmbeddingGemma text embedder. |
| `cg-vision-embedder` | `crates/cg-vision-embedder/` | Loading and running the bundled SigLIP-family joint image+text embedder. |
| `cg-connector-claude-code` | `crates/cg-connector-claude-code/` | The Claude Code event-history connector: finding a project's Claude Code history, decoding its JSONL records, and assembling the provider-neutral event drafts the core stores. Owns no event identity, checkpoint, projection or search. |

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
- The **event-history search space** — `event_history_nomic_v1_5_document_query_v1_768`, where
  imported history is encoded with Nomic's document/query task instruction — is declared in
  `crates/context-graph/src/embedding_space.rs` (`event_history_embedding_space`) and answered for
  by the thin wrapper `EventHistoryTextEmbedder` over the same loaded model, in
  `crates/context-graph/src/embedder/nomic.rs`. It binds only
  `event_search_passages.vector_text`, leaving `evidence.vector_text` and `digests.vector_text`
  to the text-default space. Guard: `canonical_event_history_declaration_has_only_the_passage_binding`
  in `crates/context-graph/src/store/embedding.rs`. Change the prefix, identifier or binding in
  those owning modules.
- The **bundled default** text embedder — `nomic-embed-text-v1.5`, the one a `cg` install gets
  without asking — is not a sibling crate: it lives inside `context-graph` itself, at
  `crates/context-graph/src/embedder/nomic.rs`, behind the default feature `nomic-embedder`
  (`crates/context-graph/Cargo.toml`). Changing which model ships by default, its space name, or
  how it fetches and caches its weights is a change to that file, not to `candle-embeddinggemma`.
  Where those weights live is one rule in that same file, `default_cache_root`; `cg-catalog` and
  the `cg model fetch` command delegate to it, so a second cache-root rule anywhere is the defect
  that makes a fetched model unloadable.

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

## Where a change lives

| Change | Owner and entry point |
|---|---|
| Event capture, checkpoints, identity, tenant placement | `crates/context-graph/src/store/event_import.rs` and `crates/context-graph/src/store/event.rs` |
| Passage projection, leases, retry ownership, session work policy | `crates/context-graph/src/store/event_search_projection.rs` and `crates/context-graph/src/event_search.rs` |
| Shared typed read SQL and decoding | `crates/context-graph/src/store/graph_rows.rs`; consumers use `crates/context-graph/src/consumer_read.rs` |
| CLI read budget and writer admission | `crates/cg-cli/src/read_config.rs` and `crates/cg-cli/src/write_config.rs`; library defaults in `crates/context-graph/src/owner_control.rs` and `crates/context-graph/src/write_budget.rs` |
| Detached worker handoff and command rendering | `crates/cg-cli/src/history/worker.rs` and `crates/cg-cli/src/history/read.rs` |
| Node memory probe, locks and bounded request IO | `crates/cg-node-fs/src/lib.rs` |
| Explicit fetch adopts verified artifact-only cache | `crates/cg-catalog/src/lib.rs` |

## Skills

| Task | Recipe |
|---|---|
| Capture Claude Code history and schedule search | `skills/cg-import/SKILL.md` |
| Find stored conversation passages | `skills/cg-find/SKILL.md` |
| Expand a passage to its stored source | `skills/cg-show/SKILL.md` |
| Inspect capture/search and configure read budgets | `skills/cg-status/SKILL.md` |
| Delete imported history | `skills/cg-forget/SKILL.md` |
| Fetch the default search model, including an artifact-only cache | `skills/cg-model-fetch/SKILL.md` |

## Invariants

### One embedding trait; roles ride it; sealing is the library's

Owned by the `context-graph` crate. See `crates/context-graph/AGENTS.md` for the trait, its
role-aware preparation hooks, the sealing mechanism, and every guarding test.

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
  the ceilings `owner_control::context_graph_read_limits_with_budgets()` declares once for the
  whole crate, using the held Store's immutable memory and active-time allowances. Only one page
  is ever held.
- Every cursor is closed on every exit path, including the error paths;
  `StatusReads::open_cursors()` is the counter that makes that checkable.
- Never raise the ceilings to make a scan fit. Page instead.
- A read that failed is never turned into a number. It travels the typed boundary and the caller
  surfaces the condition.
- Proofs: `crates/context-graph/tests/status_totals_are_bounded_on_a_large_store.rs`,
  `crates/context-graph/tests/a_failed_status_fold_leaves_no_cursor_open.rs`,
  `crates/cg-cli/tests/a_failed_status_read_is_never_rendered_as_an_answer.rs`.

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

## Fast test command and gate

`cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and
`cargo test --workspace` are the gate every change passes before a commit, release, or "done"
claim. `scripts/gate.sh` runs that sequence end to end and adds the checks that need a
measurement or a base ref: it writes a JUnit file from its own test run, hands it to the
wall-clock arm through `CG_GATE_JUNIT`, then runs `scripts/check-gate-wall-clock.py` in both
modes, `scripts/validate-doc-coverage.py --base` (`GATE_BASE`, default `origin/dev`), and
`scripts/check-doc-validator-negative-control.sh`, which requires that validator to still fail on
a repository carrying a dead reference and an unbound claim.

The event-search lifecycle tests run only with
`--features context-graph-cli/event-search-test-support` in both the integration tests and their
`CARGO_BIN_EXE_cg` fixture. Their exact test command is
`cargo nextest run --workspace --no-fail-fast -P default --features context-graph-cli/event-search-test-support`.
A run without the feature skips those tests.
`cargo test --workspace` carries four mechanically-enforced arms alongside the ordinary
behavior proofs:

- **Test-estate ratchet** — `crates/context-graph/tests/test_estate_audit.rs` freezes the current
  sleep/raw-clock call sites and the default-tier test count per crate. The two arms move in
  different directions: sleep/raw-clock sites only ever decrease, and a new site fails the audit;
  per-crate test counts track the estate exactly, so a change that adds tests raises the count in
  the same change — which is what puts the new tests under the ratchet — and only a deliberate,
  named removal lowers one.
- **Documentation-claim binding** — `crates/context-graph/tests/doc_claim_bindings.rs` reimplements
  the always-on, subprocess-free subset of `scripts/validate-doc-coverage.py`: it resolves every
  `<!-- enforced by: `test_name` -->` tag in `README.md`, `crates/*/README.md`, `docs/*.md` (if
  present), and `skills/**/SKILL.md` (if present) against a live default-tier Rust test function
  (including parameterized Tokio tests), and enforces the unbound-claim rule for
  `crates/context-graph/README.md`. The Python script is run separately by CI or humans with
  `--changed-files` or `--base` for the repository-wide unbound-claim check on a changed-file/base
  diff. They implement the same rule and must be updated together; the Rust test does not launch
  the Python script.
- **Agent-facing docs pointer check** —
  `crates/context-graph/tests/agents_md_names_only_paths_that_exist.rs` requires every path this
  file names to exist, and every path each crate's own `AGENTS.md` names, and requires the crate
  table above to name every crate in `crates/` and no directory that is not one. The current guides cover `context-graph`,
  `cg-cli`, `cg-catalog`, `cg-node-fs` and `cg-connector-claude-code`.
- **Gate wall-clock ratchet** — `[gate_wall_seconds]` in
  `crates/context-graph/tests/fixtures/test_estate_baseline.toml` freezes the wall seconds each
  test binary took, and `[gate_wall_seconds_policy]` declares how far a fresh measurement may
  drift. The comparison never reads a clock: it reads seconds a JUnit file already recorded.
  `scripts/gate.sh` runs `scripts/check-gate-wall-clock.py` in both modes: `--base <ref>` refuses
  a baseline value that rose, and `--junit` refuses a binary over its ceiling. Both
  `scripts/gate.sh` and `.github/workflows/ci.yml` run the ignored arm
  `gate_wall_seconds_never_exceed_the_frozen_baseline` by name against the JUnit named in
  `CG_GATE_JUNIT`. Both measure their own run with the `gate` nextest profile checked in at
  `.config/nextest.toml`, so the arm always has real seconds to read; an unreadable measurement
  fails the arm rather than passing it.
