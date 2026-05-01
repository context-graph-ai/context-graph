//! Journey B vertical-slice spike — R0 Track 2.
//!
//! ICP: U3 AI Agent Developer.
//! JTBD (verbatim from `context-graph-intent.md`): "When I'm building an AI
//! agent that needs to remember and reason across sessions, I want a
//! structured memory layer that tracks what's current, what's stale, and
//! how decisions connect, so my agent can build on prior work instead of
//! hallucinating from outdated context."
//!
//! Walks the four moments-of-pain end-to-end against the released
//! contextdb v1.0.0 surface, stubbing engine-blocked / wrapper plans per
//! the Track 2 stub policy. Real engine, real disk (TempDir), real reopen
//! between session boundaries, real multi-thread contention where the
//! contract calls for it.
//!
//! Spike-only: lives on the `engine-spikes` branch; never merged to main.

pub mod connect;
pub mod embedder;
pub mod hook_runtime;
pub mod ingest_history;
pub mod schema;
pub mod search;
pub mod store;
pub mod subscriptions;
pub mod trigger;
pub mod types;
