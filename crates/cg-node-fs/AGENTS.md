# AGENTS.md — `cg-node-fs` crate

The two filesystem touches a Context Graph node makes that are not part of the embeddable memory
core (`src/lib.rs`): reading a per-node config file (`load_text_if_present`) and holding the
per-data-root worker lock (`acquire_worker_lock`). Carries no product intent — what context-graph
is for lives in the root product documents.

## What this crate owns, and must not own

Owns: the on-disk read of a node config file, mapping only a missing file to "no config", and the
OS advisory exclusive lock on a data root's worker lockfile, with its two distinct refusals.

Must not own: config parsing, store access, scheduling, or any decision about what a node does
with the lock. `context-graph` re-exports these primitives from `crates/context-graph/src/fabric.rs`
so its own core sources stay free of `std::fs`; the `cg` binary and a real node deployment consume
them, an embed never does.

## Rules for maintenance

- A lock somebody else holds (`WorkerLockRefusal::HeldByAnotherWorker`) and a lockfile that cannot
  be used (`WorkerLockRefusal::Unusable`) stay separate variants: the first means a worker is doing
  the work, the second means nothing is, and a caller owes its user that difference.
- Only `NotFound` becomes `Ok(None)` when reading a config file; every other IO error surfaces.
- New filesystem access the embeddable core needs lands here or in another node-side crate, never
  in `crates/context-graph/src/`.

## Fast test command

`cargo clippy -p cg-node-fs --all-targets -- -D warnings`. The crate carries no tests of its own;
its behavior is proven through `crates/context-graph/tests/integration/fabric_capture.rs` and
`crates/cg-cli/tests/event_history.rs`.
