//! Correctness bench: gold answer-key format + the frozen scoring method.
//!
//! The scoring registration lives in `docs/cg-correctness-criteria.md` (criteria commit
//! `99d99ee`). The thresholds and banding here ARE that registration rendered as code; they are
//! held fixed across every recipe/model/chunking variant so the per-lever numbers stay
//! comparable. Changing them is a product-frame change requiring an owner decision.

pub mod driver;
pub mod gold;
pub mod report;
pub mod scoring;
