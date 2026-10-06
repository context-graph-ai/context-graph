//! Prompt recovery of an interrupted `cg understand` drain — background work recovered by the same
//! shared claim mechanism as search projection and interpretation.
//!
//! The user-facing contract: a digest whose drain process died after claiming it is offered again
//! and claimable at once by the next drain, from any process; it is counted as interrupted, never
//! as in flight; a digest claimed by a live drain is never handed to a second one.
//!
//! The dead drain is a real child copy of this test binary that claims the digest and exits
//! without releasing anything. Nothing sleeps and no clock is advanced.

#[path = "common/fake_extractor.rs"]
#[allow(dead_code)]
mod fake_extractor;

use context_graph::tier0::{Tier0CaptureRequest, capture_tier0, queue_counts};
use context_graph::understanding::{claim_digest, list_pending_digests, mark_structured};
use context_graph::{ContextId, DigestId, Store, StoreConfig};

use fake_extractor::open_store_with_lexical_embedder;

const CONTEXT: &str = "understanding-crash-recovery";
const CHILD_PATH: &str = "CG_UNDERSTANDING_CRASH_CHILD_PATH";
const CHILD_DIGEST: &str = "CG_UNDERSTANDING_CRASH_CHILD_DIGEST";
const EXIT_AFTER_CLAIM: i32 = 72;
const NOT_CLAIMED: i32 = 71;

fn store_at(path: std::path::PathBuf) -> Store {
    open_store_with_lexical_embedder(StoreConfig {
        db_path: path,
        default_text_embedder: None,
        ..Default::default()
    })
}

fn context_id(store: &Store) -> ContextId {
    store
        .list_contexts()
        .expect("contexts")
        .into_iter()
        .find(|context| context.name == CONTEXT)
        .expect("capture Context exists")
        .id
}

#[test]
fn understanding_crash_child() {
    let Some(path) = std::env::var_os(CHILD_PATH) else {
        return;
    };
    let store = store_at(path.into());
    let digest = DigestId::from(
        uuid::Uuid::parse_str(&std::env::var(CHILD_DIGEST).expect("child digest"))
            .expect("child digest id"),
    );
    if !claim_digest(&store, digest).expect("claim the pending digest") {
        std::process::exit(NOT_CLAIMED);
    }
    // Die holding the claim: no release, no failure record, no destructor.
    std::process::exit(EXIT_AFTER_CLAIM);
}

#[test]
fn a_digest_whose_drain_was_killed_is_offered_and_claimed_again_at_once() {
    // A drain killed mid-digest strands nothing: the digest is interrupted and the next drain claims it.
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("understanding.db");
    let store = store_at(path.clone());
    let report = capture_tier0(
        &store,
        Tier0CaptureRequest {
            context_label: CONTEXT.to_string(),
            session_ref: "crash-session".to_string(),
            transcript: "User: we chose sqlite for the local cache".to_string(),
            source: "test".to_string(),
            chunk_window: 0,
        },
    )
    .expect("tier-0 capture");
    assert_eq!(report.digest_ids.len(), 1);
    let digest = report.digest_ids[0];
    let context = context_id(&store);
    drop(store);

    let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args(["--exact", "understanding_crash_child", "--nocapture"])
        .env(CHILD_PATH, &path)
        .env(CHILD_DIGEST, digest.as_uuid().to_string())
        .output()
        .expect("run the claiming drain");
    assert_eq!(
        output.status.code(),
        Some(EXIT_AFTER_CLAIM),
        "the drain claimed the digest and died holding it: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let store = store_at(path);
    let counts = queue_counts(&store, context).expect("queue counts after the crash");
    assert_eq!(
        counts.interrupted, 1,
        "the dead drain's digest is interrupted"
    );
    assert_eq!(
        counts.in_flight, 0,
        "nobody is understanding it, so it is not in flight"
    );
    let offered: Vec<DigestId> = list_pending_digests(&store, context, None)
        .expect("list drainable digests")
        .into_iter()
        .map(|pending| pending.digest_id)
        .collect();
    assert_eq!(
        offered,
        vec![digest],
        "the interrupted digest is offered again"
    );

    assert!(
        claim_digest(&store, digest).expect("claim the interrupted digest"),
        "the next drain claims the interrupted digest at once"
    );
    let counts = queue_counts(&store, context).expect("queue counts while claimed");
    assert_eq!(counts.in_flight, 1, "a live claimer's digest is in flight");
    assert_eq!(counts.interrupted, 0);
    assert!(
        !claim_digest(&store, digest).expect("second claim attempt"),
        "a digest held by a live drain is never handed out twice"
    );
    assert!(
        list_pending_digests(&store, context, None)
            .expect("list while claimed")
            .is_empty(),
        "a live claim is not offered to another drain"
    );

    mark_structured(&store, context, digest, "fixture-model", None)
        .expect("the resumed drain completes the digest");
    let counts = queue_counts(&store, context).expect("queue counts after completion");
    assert_eq!(counts.structured, 1);
    assert_eq!(counts.in_flight + counts.interrupted + counts.pending, 0);
}
