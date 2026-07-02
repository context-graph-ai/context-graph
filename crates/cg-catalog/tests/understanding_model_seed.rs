//! Verifies the bundled understanding model's cache-seed path: seeding from an already-downloaded
//! GGUF (no re-download) produces a `LocalPath`, cached catalog entry that `show`/`list` disclose
//! with the right size and single-file sha, and that `declared_local_model` will accept.
//!
//! Weights-gated: skips cleanly when the real GGUF is absent (CI has no weights). Point it at the
//! GGUF with `CG_LOCAL_LLM_TEST_GGUF=/path/to/model.gguf`, or drop it at the staging path.

use std::path::PathBuf;

use cg_catalog::{
    Catalog, CacheStatus, Source, UNDERSTANDING_DEFAULT_ID, understanding_default_disclosure,
};

fn locate_gguf() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("CG_LOCAL_LLM_TEST_GGUF") {
        let path = PathBuf::from(explicit);
        return path.exists().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let staged = PathBuf::from(home)
        .join(".cache/cg/staging")
        .join("gemma-4-E2B-it-Q4_K_M.gguf");
    staged.exists().then_some(staged)
}

#[test]
fn seed_from_path_produces_cached_local_entry() {
    let Some(gguf) = locate_gguf() else {
        eprintln!("SKIP seed_from_path_produces_cached_local_entry: no GGUF (set CG_LOCAL_LLM_TEST_GGUF)");
        return;
    };

    // Disclosure is available without any download.
    let disclosure = understanding_default_disclosure();
    assert_eq!(disclosure.id, UNDERSTANDING_DEFAULT_ID);
    assert_eq!(disclosure.byte_size, 3_106_736_256);

    let temp = tempfile::tempdir().expect("temp cache root");
    let catalog = Catalog::open_with_scope_against_root(temp.path(), "understanding-seed-test")
        .expect("catalog opens");

    assert!(
        !catalog.understanding_model_cached(),
        "fresh cache should not report the model as cached"
    );

    catalog
        .seed_understanding_model_from_path(&gguf)
        .expect("seed from staged GGUF succeeds");

    assert!(
        catalog.understanding_model_cached(),
        "after seeding, the model is cached with the pinned sha"
    );

    // show() returns a cached, local-path LlmClient entry with the disclosed size and one artifact.
    let entry = catalog.show(UNDERSTANDING_DEFAULT_ID).expect("entry shown");
    assert_eq!(entry.cache_status, CacheStatus::Cached);
    assert!(matches!(entry.source, Source::LocalPath { .. }));
    assert_eq!(entry.byte_size, disclosure.byte_size);
    assert_eq!(entry.artifact_filenames, vec![disclosure.file.clone()]);
    assert_eq!(entry.per_file_sha256.len(), 1);
    assert!(entry.local_path.is_some());

    // list() includes it (so `cg model fetch`/`list` disclose it with size).
    let listed = catalog.list().expect("list");
    assert!(
        listed
            .iter()
            .any(|e| e.id == UNDERSTANDING_DEFAULT_ID && e.byte_size == disclosure.byte_size),
        "catalog list must include the understanding model with its size"
    );

    // Seeding again is idempotent (already-in-place copy is skipped, sha re-verified).
    catalog
        .seed_understanding_model_from_path(&gguf)
        .expect("re-seed is idempotent");
}
