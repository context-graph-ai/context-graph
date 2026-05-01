//! Group B — invalidation + impact-analysis plans.
//!
//! Plans:
//! - 7 cg-impact-analysis: ENGINE-BLOCKED entirely on §3 ObservationTrigger
//!   (engine-gaps-for-cg.md). No Store-API surface runnable today; no probes.
//!   Catalogued as a one-line "blocked on" entry in track1-results/.
//! - 8 cg-invalidation: primitive-owner with partial blockage on §3
//!   ObservationTrigger (transitive — no Invalidation rows arrive without
//!   the trigger) and §4 PreCommitContextGuard
//!   (`context_isolation_every_op_enforces` requires engine-systematic
//!   context_id check). Unblocked single + multi-primitive +
//!   concurrent-semantics surface IS probed in plan_8.

pub mod plan_8;
