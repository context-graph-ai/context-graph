//! Engine-probe sweep for R0 derisking of the cg plan manifest.
//!
//! Each `group_x` module hosts probes for one plan-group as defined by R0
//! Track 1.2. Probes target the released contextdb v1.0.0 engine surface and
//! exercise contention / edge / failure-mode semantics that the cg plans
//! assume. Probes that pass mean the engine delivers the contract; probes
//! that fail are GAP-CANDIDATE entries for the contextdb release bundle.
//!
//! Spike code: lives only on the `engine-spikes` branch. Never merged to
//! `main`. Never published.

#[cfg(test)]
pub mod group_a;
#[cfg(test)]
pub mod group_b;
#[cfg(test)]
pub mod group_c;
#[cfg(test)]
pub mod group_d;
#[cfg(test)]
pub mod group_e;
