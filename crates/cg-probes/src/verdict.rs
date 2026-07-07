//! The pre-registered decision rule (criteria checksum f401c2db94e75fff), as code. Inputs are
//! per-corpus pass/fail signals at the frozen bars; the output maps to the owner fork recorded
//! in PROJECT-STATUS. Once registered, this mapping is never iterated against without an owner
//! decision.

/// Per-corpus outcomes at the frozen operating points. `p1_pass` = some candidate model class
/// clears BOTH P1-a and P1-b bars on this corpus.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorpusSignals {
    pub p1_pass: bool,
    pub p2_pass: bool,
    pub p3_known_pass: bool,
    pub p3_new_local_pass: bool,
    pub p3_new_hosted_pass: bool,
}

/// How the rare new-intention slot is filled when the trunk is adopted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewIntentionSlot {
    /// The small local generative arm clears the bar on both corpora — fully $0.
    ZeroCostLocal,
    /// Only the hosted arm clears it — $0 trunk except a rare, priced, per-new-intention event.
    PricedHosted,
    /// Neither arm clears it — HITL confirm is the slot (already the curation design).
    HitlOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// P1 failed on every candidate class on BOTH corpora: the verifier-trunk architecture is
    /// dead; the fork is decided by the extension arm + the measured split-fallback.
    Killed,
    /// Full pass (P1+P2+P3-known) on both corpora.
    Adopt(NewIntentionSlot),
    /// Full pass on the dev corpus only — a materially different product verdict, never averaged.
    DevScopeOnly,
    /// Full pass on the non-dev corpus only.
    NonDevScopeOnly,
    /// P1 survived somewhere but the trunk does not clear its bars end-to-end: the measured
    /// split-fallback is the shape.
    SplitFallback,
}

pub fn decide(dev: &CorpusSignals, nondev: &CorpusSignals) -> Verdict {
    let _ = (dev, nondev);
    todo!("probe impl commit")
}

/// Owner-readable verdict block (plain language first, mechanism second).
pub fn render(verdict: &Verdict) -> String {
    let _ = verdict;
    todo!("probe impl commit")
}
