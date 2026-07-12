//! Correctness bench: gold answer-key format + the frozen scoring method.
//!
//! The thresholds and banding here ARE the scoring registration rendered as code; they are
//! held fixed across every recipe/model/chunking variant so the per-lever numbers stay
//! comparable. Changing them is a product-frame change requiring a deliberate product decision.

pub mod driver;
pub mod gold;
pub mod report;
pub mod scoring;
