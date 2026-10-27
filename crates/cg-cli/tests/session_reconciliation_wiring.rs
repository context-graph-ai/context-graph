//! The understanding drain reconciles session goals after structuring each context.
//! The source guard verifies that the CLI invokes the shared library reconciliation.

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
