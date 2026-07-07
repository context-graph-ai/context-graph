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

impl CorpusSignals {
    /// The full verifier trunk holds on this corpus: oracle judgments, end-to-end scan, and
    /// attachment to the known roster all at bar.
    fn full_trunk(&self) -> bool {
        self.p1_pass && self.p2_pass && self.p3_known_pass
    }
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
    if !dev.p1_pass && !nondev.p1_pass {
        return Verdict::Killed;
    }
    match (dev.full_trunk(), nondev.full_trunk()) {
        (true, true) => {
            let slot = if dev.p3_new_local_pass && nondev.p3_new_local_pass {
                NewIntentionSlot::ZeroCostLocal
            } else if dev.p3_new_hosted_pass && nondev.p3_new_hosted_pass {
                NewIntentionSlot::PricedHosted
            } else {
                NewIntentionSlot::HitlOnly
            };
            Verdict::Adopt(slot)
        }
        (true, false) => Verdict::DevScopeOnly,
        (false, true) => Verdict::NonDevScopeOnly,
        (false, false) => Verdict::SplitFallback,
    }
}

/// Owner-readable verdict block (plain language first, mechanism second).
pub fn render(verdict: &Verdict) -> String {
    match verdict {
        Verdict::Killed => "VERDICT: KILLED — the verifier models cannot render the trunk \
            judgments even on gold evidence spans, on any candidate class, on either corpus. \
            The verifier-trunk architecture is dead; the owner fork is decided by the \
            local-model extension arm (DL-16) and the measured split-fallback only."
            .to_string(),
        Verdict::Adopt(slot) => {
            let slot_line = match slot {
                NewIntentionSlot::ZeroCostLocal => {
                    "the rare new-intention event is also $0 \
                    (the small local generative arm clears the intention bar)"
                }
                NewIntentionSlot::PricedHosted => {
                    "the rare new-intention event is priced \
                    (~one hosted call per new intention); everything else is $0"
                }
                NewIntentionSlot::HitlOnly => {
                    "the rare new-intention event goes to human \
                    confirm (the $0 curation path); everything else is $0"
                }
            };
            format!(
                "VERDICT: ADOPT — the extractive + verifier trunk clears the correctness bars \
                 on BOTH corpora at $0 marginal cost on CPU; {slot_line}."
            )
        }
        Verdict::DevScopeOnly => "VERDICT: DEVELOPER-SCOPE ONLY — the trunk clears its bars on \
            the owner's developer sessions but NOT on the non-developer domains. The \
            domain-neutral product promise is not met; the product claim narrows or the failing \
            domain drives the next measurement. Never averaged."
            .to_string(),
        Verdict::NonDevScopeOnly => "VERDICT: NON-DEV-SCOPE ONLY — the trunk clears its bars on \
            the authored non-developer domains but NOT on the owner's real developer sessions. \
            The daily-driver promise is not met; the failing corpus drives the next measurement. \
            Never averaged."
            .to_string(),
        Verdict::SplitFallback => "VERDICT: SPLIT-FALLBACK — verifiers judge gold spans on at \
            least one corpus, but the end-to-end trunk does not clear its bars. The measured \
            split-fallback (deterministic anchors + local retrieval + one hosted trunk-judgment \
            pass per session) is the product shape."
            .to_string(),
    }
}
