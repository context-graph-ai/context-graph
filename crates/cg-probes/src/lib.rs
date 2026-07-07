//! cg-probes — verifier-trunk probe machinery for the cg-verifier-probes measurement run.
//!
//! Probes P0–P3 answer: can the trunk judgments (decision-vs-discussion, decision→intention
//! attachment, basis support) be rendered by extractive capture + small CPU-only verifier
//! models, with no generative model in the trunk? Deterministic parts (corpus loading, window
//! enumeration, scoring math, accretion, pair generation, the verdict rule) live here under the
//! test wall; verifier inference runs on candle; model judgment quality is the owner live
//! smoke's job, never CI's.
//!
//! Measurement-only: this crate never opens a `Store` and never writes a cg data root. Verifier
//! scores never write active memory. Statement matching reuses cg-bench's frozen method
//! (`cg_bench::scoring::band` + the production nomic embedder) so the method is shared code,
//! not a re-implementation.

pub mod accretion;
pub mod corpus;
pub mod pairs;
pub mod scoring;
pub mod verdict;
pub mod verifier;
pub mod windows;
