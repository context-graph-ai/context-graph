# context-graph

The embeddable memory core behind `cg`. It gives an agent or an application a
durable, queryable memory of the work it has done — intentions, decisions,
observations and the digests that tie them together — stored in a local
contextdb database and searched by meaning as well as by structure. Link it as a
library and you get the ontology, the store and its domain behavior, embedding
registration, bounded status reads, and the administrative owner plane; the `cg`
binary is one consumer of it.

The crate is embedded rather than served: there is no separate daemon and no
caller-configured socket address, one process holds the store's writable handle,
and everything else is a typed call. When another process owns the store,
administrative reads reach that process over contextdb's local owner channel,
which is selected from the store path and the runtime root rather than from any
address a caller supplies; there is no second transport in this repository.

## Choosing an Embedder

The bundled text embedder is `nomic-embed-text-v1.5`, exposed through the
default text space `text_default_nomic_v1_5_768`. It runs locally through
Candle, does not require an API key, and stores first-load artifacts in the
cg embedder cache. It lives in this crate, behind the default feature
`nomic-embedder`. The upstream artifact is published at
`huggingface.co/nomic-ai/nomic-embed-text-v1.5`; the model weights are about
547MB, which is roughly three minutes on a 25Mbps developer connection before
the cache is warm. Turning default features off gives a core that fetches
nothing.

`candle-embeddinggemma` remains available as a sibling opt-in for consumers
that want the prior `text_default_embeddinggemma_300m_768` space. Register it
through `Store::open_with_embedder_registrations`, then use `reindex_space`
when moving stored rows between registered spaces. context-graph ships the
English Nomic v1.5 default; multilingual embedders are consumer-declared
through the same registration lane.

```rust,no_run
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

## Administrative reads and `cg status`

`cg status` answers about two things in one command: whether cg is connected to
a model, and what state this machine's Context Graph memory store is in. `no
model connection is active` is a statement about the model — the store beside it
may be perfectly healthy — and a store that could not be read is reported as
exactly that, never as a store with nothing in it.

The two are not two fields. The JSON carries **one** `condition`. When the store
could not be read, the store's refusal takes that field and the document has a
store path and no totals; otherwise the field is left to whatever the model half
has to say, and the store's answer is the totals sitting beside it. Those totals
are the context totals, the understanding queue, the raw footprint, and — for a
node enrolled in a tenant fabric — the fabric posture. The `fabric` object
appears only once `<data_root>/cg/fabric.toml` exists; a plain single-node
install has `fabric_jobs: []` and no `fabric` key.

### What it writes, and how a script reads it

`cg status` writes one JSON document on stdout by default; `--format text`
writes the same answer as prose. That is true of the refusals below as well as
of the successful reads — the format flag chooses how an answer is written,
never how much of it you are told.

The field a script branches on is `condition`, not the exit code:

```sh
cg status | jq -r .condition
```

That is the whole domain, and what each value tells you:

| `condition` | What it says | `bucket` |
| --- | --- | --- |
| `null` | A model is connected and has recorded no failure, and the store was read. | `null` |
| `StoreBusy` | The store could not be read: another process is writing it. Exit 4. | `memory` |
| `StoreHeldByReaders` | The store could not be read: direct readers are still holding it. | `memory` |
| `MemoryError` | The store could not be read at all — damage, a permission, a mount. Exit 1. | `memory` |
| `not_connected` | The store was read; no model connection is active. | `not_connected` |
| `connection_not_found` | The store was read; the `--connection` you named is not registered. | `not_found` |
| `provider_spec_invalid` | A provider spec on this machine is malformed. | `config` |
| `connections_file_unreadable` | `model-connections.toml` could not be read. | `unknown` |
| `no_host_session`, `consent_not_recorded`, `host_tool_missing`, `host_login_expired`, `live_call_failed` | The store was read; this is the connected model's own last recorded failure. | the model's own |

The casing is the short way to read that table. The store's own conditions are
Pascal-case; the model's and the configuration's are snake_case. A Pascal-case
value — equivalently, `"bucket": "memory"` — is the store refusing, and that
document carries `store_path` and no totals. Any other value means the store was
read successfully. So `"bucket": "memory"` marks a store REFUSAL rather than
every statement about the store: a healthy store on a machine with no model
connected reports `"bucket": "not_connected"` and carries its totals in the same
document.

Write the branch accordingly. A `case` over the three store conditions alone
falls through on a fresh install, where the value is `not_connected`; give it a
default arm that reads "no store condition" as "the store answered." The totals
are in the document on every non-store condition except the three configuration
refusals — `connection_not_found`, `provider_spec_invalid` and
`connections_file_unreadable` — which answer before the store fields are added.

Both the reader-held case and an ordinary unreadable store exit `1`, so the
number cannot tell "readers are finishing, retry" apart from "this store is
damaged." The condition name can.

### `cg status` opens the store for writing, and creates one if none is there

`cg status` is not a read-only probe of the filesystem. It opens the store for
writing — that is how it takes the totals — and if there is no store at the path
it is pointed at, it **materializes a new empty one** (the database file and its
lock) and reports `goals: 0 decisions: 0 observations: 0` at exit 0.

Zero totals therefore say exactly one thing: the store at the resolved path was
opened and no counted rows were found in it. They do not say whether that store
pre-existed this command or was materialized by it a moment ago, and they do not
say why it is empty — a store you have kept for months and have simply not
captured into is a legitimate zero. This command alone settles neither question.
What it does give you is the `store_path` field of the JSON document, which names
the store the totals are about: if `cg status` says your memory is empty, read
`store_path` first — a mistyped or unmounted `CG_DATA_ROOT` (the default is
`~/.cg`, and the store is `context-graph.db` under it) puts a fresh store
somewhere you were not looking, and the totals are telling the truth about the
wrong store.

### What `cg status` says when the store cannot be read

- **Another process is writing the store** — a running cg, or a runtime that owns
  it. `cg status` gives this CLI's settled busy answer: condition `StoreBusy`, the
  words `store busy / already in use`, the holding process and the store path, and
  exit **4**. That is the same answer and the same code every other cg command that
  opens the store gives; status has no second vocabulary here.
- **Direct readers are holding the store.** Nothing is wrong with the machine —
  readers finish and the next attempt succeeds. The command fails under condition
  `StoreHeldByReaders` and prints the sentence the condition itself writes: which
  store, how many readers were observed, and each verified reader by process id
  and process name. The count and the names are separate answers: a reader can be
  observed and not identified, and such a holder is disclosed as one that
  published no identity rather than left out — so the names you are given are the
  readers the store could *verify*, and the sentence says when there are more. The
  numeric exit code for this case is deliberately unsettled, so branch a script on
  the failure and on the condition name rather than on a number.
- **Any other unreadable or refused store** surfaces its typed condition as the
  answer.

In none of these cases does `cg status` print `goals: 0 decisions: 0
observations: 0`. Those totals are an *answer*: they say the store was read and it
is empty. Printing them for a store that was never read would tell a developer
whose memory is intact that it is gone, and tell an agent deciding whether to
capture that there is nothing worth preserving. A refused read has no totals, and
its refusal is its summary.

### Asking a running context-graph a question

Administrative questions are asked of the *process* that holds the store, not of
the file — the store path is the whole address, and no caller configures a socket
address; the channel is contextdb's local owner channel, selected from that path
and the runtime root. The
contributor-facing rules for that owner plane (which typed results license
opening the file directly, why the handler and the embedders ride one writable
open, and which single door in-process totals go through) are stated once, in
`AGENTS.md` at the repository root. They are not repeated here; this file is the
operator's and the library consumer's.

## Reading a store you must not change

A runtime that cannot take the store — another process owns it, or its own job
forbids it to risk changing one — still opens it read-only through
`ConsumerReader::open`. That open answers three ways, and each sends you
somewhere different:

- **`StoreMissing`** — there is nothing at that path, and nothing was created.
  Restore a backup, or point at the right store.
- **`StoreNeedsWritableRecovery`** — the store is there and its committed image
  needs one writable open to settle: the last writer left state only a writable
  hydration performs. This is the one refusal you may answer by opening the
  store WRITABLE (`Store::open_existing_for_change`), recovering it, and
  carrying on. The substrate's own words for the condition travel in `reason`.
- **`StoreUnreadable`** — the store is there and cannot be read at all: a
  permission, a mount, real damage. Reaching for a writable handle here is the
  worst available move; diagnose it first.

The three are separate typed answers, not one answer with three strings in it,
so nothing downstream has to parse prose to decide whether to open a store for
writing.

Once open, the reader answers SQL and cursors — and the typed graph reads, over
`ConsumerReader::graph()`: the audit trail, contexts, entities, intentions,
decisions and observations, with the same signatures and the same answer types
as the writable `Store`'s. They are the same code: one SQL text and one decoder
per read, run over whichever handle you reached it through, so the read-only
answer cannot drift from the writable one. No writable handle is anywhere in the
path, so no sequence of view reads changes the store file, its companions, or
the directory they sit in.

The declaration is the whole of what the view may reach: a read whose table you
did not declare is refused as `UndeclaredTableRefused` naming that table, before
the engine is asked. Declare them from context-graph's own published specs
(`context_graph::schema::TABLES`), never from a copy of the DDL text. Per read:

| Read | Tables it needs |
| --- | --- |
| `audit_query` | `audit_log` |
| `list_contexts`, `get_context` | `contexts` |
| `get_entity` | `entities` |
| `get_intention` | `intentions` |
| `get_decision` | `decisions`, `edges`, `decision_basis_snapshots`, `outcomes` |
| `get_observation`, `list_observations` | `observations`, `evidence`, `evidence_availability_log` |

The nested tables are there because the answers are nested — a decision carries
the intentions it serves, the precedents it cites, its basis snapshots and its
outcome; an observation carries its evidence, and each piece of evidence the
retention status its availability log last recorded. Declare only the headline
table and you are told which table you still owe, rather than handed a stripped
row.

```rust,no_run
use std::collections::BTreeSet;

use context_graph::schema::TABLES;
use context_graph::{
    CgError, ConsumerDeclaration, ConsumerReader, ConsumerSchema, ConsumerTable, DecisionId,
};

/// Why a decision fired, read out of a store this process must not change.
fn explain(store_path: &std::path::Path, decision: DecisionId) {
    let declaration = ConsumerDeclaration {
        schema: ConsumerSchema {
            namespace: "vigil".to_string(),
            tables: ["decisions", "edges", "decision_basis_snapshots", "outcomes"]
                .iter()
                .map(|wanted| {
                    let spec = TABLES
                        .iter()
                        .find(|spec| spec.name == *wanted)
                        .expect("context-graph publishes a spec for every table it owns");
                    ConsumerTable {
                        name: spec.name.to_string(),
                        ddl: spec.ddl.to_string(),
                    }
                })
                .collect(),
        },
        scope_labels: BTreeSet::new(),
    };

    let reader = match ConsumerReader::open(store_path, declaration) {
        Ok(reader) => reader,
        // The store is fine and only needs settling: take the writable
        // existing-only door, recover it, and carry on serving.
        Err(CgError::StoreNeedsWritableRecovery { path, reason }) => {
            eprintln!("{} needs a writable open first: {reason}", path.display());
            return;
        }
        Err(other) => {
            eprintln!("no read-only answer from this store: {other}");
            return;
        }
    };

    match reader.graph().get_decision(decision) {
        Ok(Some(decision)) => println!("{} — {}", decision.decision_type, decision.description),
        Ok(None) => println!("no such decision in this store"),
        Err(CgError::UndeclaredTableRefused { table }) => {
            eprintln!("declare `{table}` to read this");
        }
        Err(other) => eprintln!("{other}"),
    }

    reader.close().expect("let the read session go");
}
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

CI runs the workspace test gate on `macos-latest` and `ubuntu-latest`.
`windows-latest` is excluded deliberately: the contextdb substrate's durable
large-request sync staging needs descriptor-relative filesystem operations and
is Unix-only by design, so the sync test surface cannot be built there.

Cache the embedder artifact directory so each runner keeps the Nomic weights
warm across jobs. The cached directory is `~/.cache/cg/embedders` —
`$XDG_CACHE_HOME` takes the place of `~/.cache` when it is set, and the Windows
equivalent is `%LOCALAPPDATA%\cg\embedders` — and the model sits one level
under it, in `nomic-embed-text-v1.5/`. A machine that has never held the weights
writes them to the sibling `models/` directory instead, so point
`CG_MODEL_CACHE_ROOT` at the directory you cache if you want a cold runner to
seed it. Key the cache on the embedder source file rather than on `Cargo.lock`:
the model id and the cache layout live in that file, and the lockfile does not
change when they do.

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/cg/embedders
    key: cg-nomic-${{ runner.os }}-${{ hashFiles('crates/context-graph/src/embedder/nomic.rs') }}
```
