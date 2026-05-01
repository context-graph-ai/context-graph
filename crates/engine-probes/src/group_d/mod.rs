//! Group D — server + correlate + signature plans.
//!
//! Plans: 12 cg-server-correlate, 13 cg-server-signature-detect,
//! 14 cg-server-signature-propagate, 15 cg-server-blueprint-catalog,
//! 23 cg-server-core. Plan 10 cg-subscriptions is engine-blocked entirely
//! on §5 EventBus (no probe module; one-line catalogue entry only).

pub mod plan_12;
pub mod plan_13;
pub mod plan_14;
pub mod plan_15;
pub mod plan_23;
