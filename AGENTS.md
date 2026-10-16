# context-graph — agent guide

Two jobs bring an agent here:

- **Use context-graph**: run the presented commands or embed the library. Start below; no contributor gate applies.
- **Contribute**: change this repository. Read the ownership map and matching crate guide, then run the gate.

## Use context-graph

Start with [getting started](docs/getting-started.md) for local Claude Code history search,
or [library quickstart](docs/library-quickstart.md) for structured memory.
`cg` builds events and passages; imported history alone creates no structured graph rows.
Use JSON named fields for automation, and read [CLI contracts](docs/cli.md) for conditions and exits.
Only forget history when the user authorizes the selected session or Context.
Hosted understanding uses user-chosen provider credentials; read [privacy](docs/privacy-and-data-movement.md).

| Skill | Recipe |
|---|---|
| `cg-import` | [Recipe](skills/cg-import/SKILL.md) |
| `cg-find` | [Recipe](skills/cg-find/SKILL.md) |
| `cg-show` | [Recipe](skills/cg-show/SKILL.md) |
| `cg-status` | [Recipe](skills/cg-status/SKILL.md) |
| `cg-forget` | [Recipe](skills/cg-forget/SKILL.md) |
| `cg-model-connect` | [Recipe](skills/cg-model-connect/SKILL.md) |
| `cg-model-fetch` | [Recipe](skills/cg-model-fetch/SKILL.md) |
| `cg-model-list` | [Recipe](skills/cg-model-list/SKILL.md) |
| `cg-model-show` | [Recipe](skills/cg-model-show/SKILL.md) |
| `cg-model-status` | [Recipe](skills/cg-model-status/SKILL.md) |
| `cg-model-auth-login` | [Recipe](skills/cg-model-auth-login/SKILL.md) |
| `cg-doctor` | [Recipe](skills/cg-doctor/SKILL.md) |
| `cg-selftest` | [Recipe](skills/cg-selftest/SKILL.md) |

## Documentation

| Page | Link |
|---|---|
| SUMMARY | [Read](docs/SUMMARY.md) |
| getting-started | [Read](docs/getting-started.md) |
| library-quickstart | [Read](docs/library-quickstart.md) |
| why-context-graph | [Read](docs/why-context-graph.md) |
| how-it-works | [Read](docs/how-it-works.md) |
| ontology | [Read](docs/ontology.md) |
| privacy-and-data-movement | [Read](docs/privacy-and-data-movement.md) |
| cli | [Read](docs/cli.md) |
| configuration | [Read](docs/configuration.md) |
| models-and-embedders | [Read](docs/models-and-embedders.md) |
| library-api | [Read](docs/library-api.md) |
| capability-index | [Read](docs/capability-index.md) |
| register-an-embedder | [Read](docs/register-an-embedder.md) |
| read-a-store-read-only | [Read](docs/read-a-store-read-only.md) |
| troubleshooting | [Read](docs/troubleshooting.md) |

## Contribute

Preserve public behavior and compatibility in the linked docs. [CONTRIBUTING.md](CONTRIBUTING.md)
describes the full `scripts/gate.sh` gate, claim bindings, deterministic test-estate checks,
recorded test-duration limits, and live smokes. A crate guide adds to this file.

### Crate map

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

### Where a change lives

| Change | Owner and entry point |
|---|---|
| Event capture, checkpoints, identity, tenant placement | `crates/context-graph/src/store/event_import.rs` and `crates/context-graph/src/store/event.rs` |
| Passage projection, leases, retry ownership, session work policy | `crates/context-graph/src/store/event_search_projection.rs` and `crates/context-graph/src/event_search.rs` |
| Shared typed read SQL and decoding | `crates/context-graph/src/store/graph_rows.rs`; consumers use `crates/context-graph/src/consumer_read.rs` |
| CLI read budget and writer admission | `crates/cg-cli/src/read_config.rs` and `crates/cg-cli/src/write_config.rs`; library defaults in `crates/context-graph/src/owner_control.rs` and `crates/context-graph/src/write_budget.rs` |
| Detached worker handoff and command rendering | `crates/cg-cli/src/history/worker.rs` and `crates/cg-cli/src/history/read.rs` |
| Node memory probe, locks and bounded request IO | `crates/cg-node-fs/src/lib.rs` |
| Explicit fetch adopts verified artifact-only cache | `crates/cg-catalog/src/lib.rs` |

### Shared invariants

- Core `Embedder` is the only embedding abstraction; role preparation and sealing remain library-owned.
- Administrative requests use `request_owner(store_path, frame)` in `crates/context-graph/src/owner_control.rs`.
  Only typed `OwnerNotRunning` or `NoStore` licenses direct fallback; timeout never does.
  `OwnerRouteUnsupported` remains distinct. No additional transport or socket-address API.
- One original writable open carries both owner handling and embedder registrations.
- In-process status closes its cursors when a read fails, and a refusal never becomes a count. Its
  bounded-totals guard is ignored today; see `crates/context-graph/AGENTS.md`.
- `CONTEXTDB_OWNER_READ_RUNTIME_DIR` locates the directory both sides of the substrate channel use;
  unset, the platform locator applies. Review-only: an empty value counts as unset, and the variable
  controls neither admission nor budgets.
- Conditions remain typed; only CLI rendering maps exits. `StoreBusy` renders as exit 4;
  reader-held stores have no dedicated numeric contract. Refusals name each verified reader by
  process id and name, and disclose any observed holder that could not be identified.
- Writable and consumer graph reads share SQL and decoding in `crates/context-graph/src/store/graph_rows.rs`.
  The consumer view uses its declared session; never introduce a writable handle there.
- Missing, recovery-required and unreadable stores remain distinct; conversion lives in
  `crates/context-graph/src/store/convert.rs`.
- Node config IO, worker locks and request IO belong to `crates/cg-node-fs/src/lib.rs`.
  Review-only: core disk access is restricted to optional read tracing, fresh-store removal, and bundled
  Nomic artifact IO.
- Local models provide embeddings; hosted connections provide understanding. Default Nomic and its cache
  rule live in `crates/context-graph/src/embedder/nomic.rs`, behind `nomic-embedder`.
- Public help and behavior documentation change together; operator contracts live in [CLI reference](docs/cli.md).

The guarding tests and local seams are named in each crate's `AGENTS.md`.
Fast iteration uses the owning guide's command; the full contributor gate remains binding before a done claim.
