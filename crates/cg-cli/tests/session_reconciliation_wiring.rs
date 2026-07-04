//! RED test for the reconciliation wiring gap (external review finding, 2026-07-04 — an
//! implementation blocker): `reconcile_session_goals` existed in the library with green tests
//! but NO production caller — the exact "well-tested pieces, absent connective wiring" failure
//! the workspace's correction-loop lesson names. The drain must invoke it after a context's
//! digests are structured. Source-scan guard (same class as the delegation guard); the live
//! smoke exercises the behavior end to end. Committed failing before the fix.

use std::path::Path;

#[test]
fn understand_drain_invokes_session_goal_reconciliation() {
    let main_rs = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    let source = std::fs::read_to_string(&main_rs).expect("cg-cli main.rs readable");
    assert!(
        source.contains("reconcile_session_goals("),
        "the understand drain must call the library session-goal reconciliation after a \
         context's digests are structured — a library capability with no production caller is \
         not production behavior"
    );
}
