//! RED tests — the pre-registered decision rule as code: kill, adopt (with the new-intention
//! slot grading), honest scoping, and the split-fallback default. This mapping is the frozen
//! criteria's decision rule; changing it is a product-frame change.

use cg_probes::verdict::{CorpusSignals, NewIntentionSlot, Verdict, decide};

fn signals(p1: bool, p2: bool, known: bool, new_local: bool, new_hosted: bool) -> CorpusSignals {
    CorpusSignals {
        p1_pass: p1,
        p2_pass: p2,
        p3_known_pass: known,
        p3_new_local_pass: new_local,
        p3_new_hosted_pass: new_hosted,
    }
}

const FULL_LOCAL: CorpusSignals = CorpusSignals {
    p1_pass: true,
    p2_pass: true,
    p3_known_pass: true,
    p3_new_local_pass: true,
    p3_new_hosted_pass: true,
};

#[test]
fn p1_dead_on_both_corpora_kills_the_architecture() {
    let dead = signals(false, true, true, true, true);
    assert_eq!(decide(&dead, &dead), Verdict::Killed);
    // P1 surviving on ONE corpus is not a kill — it is a scoped verdict.
    assert_ne!(decide(&FULL_LOCAL, &dead), Verdict::Killed);
}

#[test]
fn full_pass_adopts_with_the_new_intention_slot_graded() {
    assert_eq!(
        decide(&FULL_LOCAL, &FULL_LOCAL),
        Verdict::Adopt(NewIntentionSlot::ZeroCostLocal)
    );
    let hosted_only = signals(true, true, true, false, true);
    assert_eq!(
        decide(&hosted_only, &hosted_only),
        Verdict::Adopt(NewIntentionSlot::PricedHosted)
    );
    // The local arm must clear BOTH corpora to claim the fully-$0 slot.
    assert_eq!(
        decide(&FULL_LOCAL, &hosted_only),
        Verdict::Adopt(NewIntentionSlot::PricedHosted)
    );
    let hitl = signals(true, true, true, false, false);
    assert_eq!(
        decide(&hitl, &hitl),
        Verdict::Adopt(NewIntentionSlot::HitlOnly)
    );
}

#[test]
fn scoped_verdicts_are_honest_never_averaged() {
    let nondev_p2_fails = signals(true, false, true, true, true);
    assert_eq!(decide(&FULL_LOCAL, &nondev_p2_fails), Verdict::DevScopeOnly);
    assert_eq!(
        decide(&nondev_p2_fails, &FULL_LOCAL),
        Verdict::NonDevScopeOnly
    );
}

#[test]
fn surviving_p1_without_a_full_trunk_falls_to_the_split_fallback() {
    let p2_dead = signals(true, false, true, true, true);
    assert_eq!(decide(&p2_dead, &p2_dead), Verdict::SplitFallback);
    let known_dead = signals(true, true, false, true, true);
    assert_eq!(decide(&known_dead, &known_dead), Verdict::SplitFallback);
}
