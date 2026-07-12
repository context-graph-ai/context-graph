//! Fabric first-light — the frozen source rule: no transport name in ledger/payload/job/apply code.
//!
//! The ledger treats cg's payload as opaque and knows no transport. Transport identifiers
//! (`iroh`/`nats`) live only in the `[fabric]` config surface and the contextdb adapter — never in
//! cg's submission, payload, job, apply, or sync-contract sources.

use std::path::PathBuf;

#[test]
fn no_transport_name_in_ledger_or_payload_sources() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    // The ledger/payload/job/apply layer of cg — the code that must name no transport.
    let scanned = [
        "fabric.rs",
        "tier0.rs",
        "understanding.rs",
        "harness.rs",
        "capture_sync_status.rs",
    ];
    let mut offenders = Vec::new();
    for file in scanned {
        let path = src.join(file);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_default()
            .to_ascii_lowercase();
        for name in ["iroh", "nats"] {
            if text.contains(name) {
                offenders.push(format!("{file} names transport `{name}`"));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "transport identifiers leaked into ledger/payload/job/apply sources: {offenders:?}"
    );
}
