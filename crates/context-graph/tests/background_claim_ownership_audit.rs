//! Who may publish a unit of background work is decided in one place.
//!
//! The contract: crash recovery of background work is one shared mechanism. Search projection,
//! connector interpretation and the understanding drain all claim, take over, fence and release
//! through the background-claim module; none of them keeps its own lease, expiry, renewal or
//! polling code, and only that module touches the kernel holder lock or writes claim rows. A new
//! kind of background work therefore inherits prompt crash recovery instead of re-implementing it.

use std::path::{Path, PathBuf};

/// The one module allowed to hold presence and write claim rows.
const CLAIM_MODULE: &str = "crates/context-graph/src/store/background_claim.rs";

/// Every job that runs as background work and must claim through the shared module.
const BACKGROUND_JOBS: &[&str] = &[
    "crates/context-graph/src/store/event_search_projection.rs",
    "crates/context-graph/src/store/interpretation_build.rs",
    "crates/context-graph/src/understanding.rs",
];

/// Vocabulary of the per-feature timer leases the shared mechanism replaced.
const RETIRED_LEASE_VOCABULARY: &[&str] = &[
    "event_search_leases",
    "lease_duration_ms",
    "lease_expires_at",
    "lease_token",
    "renew_claim",
    "renewal_due",
    "LeaseRenewal",
    "MAX_EXPIRED_RECLAIMS",
    "ExpiredClaim",
    "live_projection_lease",
];

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate sits two levels below the repository root")
        .to_path_buf()
}

fn rust_sources(directory: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            out.push(path);
        }
    }
}

/// Every shipped source file (`src` and `benches` of every crate), repo-relative, with contents.
fn shipped_sources() -> Vec<(String, String)> {
    let root = repository_root();
    let mut files = Vec::new();
    for entry in std::fs::read_dir(root.join("crates"))
        .expect("list crates")
        .flatten()
    {
        for sub in ["src", "benches"] {
            rust_sources(&entry.path().join(sub), &mut files);
        }
    }
    rust_sources(&root.join("benches"), &mut files);
    files
        .into_iter()
        .map(|path| {
            let relative = path
                .strip_prefix(&root)
                .expect("source under the repository")
                .to_string_lossy()
                .replace('\\', "/");
            let contents = std::fs::read_to_string(&path).expect("read source");
            (relative, contents)
        })
        .collect()
}

fn writes_claim_rows(contents: &str) -> bool {
    let normalized = contents.split_whitespace().collect::<Vec<_>>().join(" ");
    [
        "INSERT INTO background_claims",
        "DELETE FROM background_claims",
        "UPDATE background_claims",
    ]
    .iter()
    .any(|statement| normalized.contains(statement))
}

#[test]
fn only_the_background_claim_module_holds_presence_or_writes_claim_rows() {
    // One module decides who may publish background work; no job holds the lock or writes claims itself.
    let sources = shipped_sources();
    assert!(
        sources.iter().any(|(path, _)| path == CLAIM_MODULE),
        "the shared background-claim module exists at {CLAIM_MODULE}"
    );
    let presence_holders: Vec<&str> = sources
        .iter()
        .filter(|(path, contents)| path != CLAIM_MODULE && contents.contains("holder_presence"))
        .map(|(path, _)| path.as_str())
        .collect();
    assert!(
        presence_holders.is_empty(),
        "only {CLAIM_MODULE} touches the kernel holder lock: {presence_holders:?}"
    );
    let claim_writers: Vec<&str> = sources
        .iter()
        .filter(|(path, contents)| path != CLAIM_MODULE && writes_claim_rows(contents))
        .map(|(path, _)| path.as_str())
        .collect();
    assert!(
        claim_writers.is_empty(),
        "only {CLAIM_MODULE} writes background claim rows: {claim_writers:?}"
    );
}

#[test]
fn every_background_job_claims_through_the_shared_module_and_keeps_no_lease_of_its_own() {
    // Each background job is a thin caller of the shared claim; no timer lease vocabulary survives.
    let sources = shipped_sources();
    for job in BACKGROUND_JOBS {
        let (_, contents) = sources
            .iter()
            .find(|(path, _)| path == job)
            .unwrap_or_else(|| panic!("background job source {job} exists"));
        assert!(
            contents.contains("background_claim"),
            "{job} claims its work through the shared background-claim module"
        );
    }
    let retired: Vec<(String, &str)> = sources
        .iter()
        .flat_map(|(path, contents)| {
            RETIRED_LEASE_VOCABULARY
                .iter()
                .filter(|word| contents.contains(**word))
                .map(move |word| (path.clone(), *word))
        })
        .collect();
    assert!(
        retired.is_empty(),
        "no shipped source keeps a per-feature timer lease: {retired:?}"
    );
}
