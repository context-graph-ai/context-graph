//! The two bundled local understanding tiers (e2b default + e4b escalation) are runtime-selectable:
//! both are registered specs the catalog can look up, disclose, and seed/verify against THEIR OWN
//! pinned sha under THEIR OWN model dir. e2b stays the default. These are deterministic (no weights,
//! no network); the real seed of each tier is covered weights-gated below.

use std::path::PathBuf;

use cg_catalog::{
    Catalog, UNDERSTANDING_DEFAULT_ID, UNDERSTANDING_MODELS, understanding_default_disclosure,
    understanding_default_spec, understanding_disclosures, understanding_model_disclosure,
    understanding_model_ids, understanding_model_spec,
};

const E2B_ID: &str = "gemma4-e2b-it-q4";
const E4B_ID: &str = "gemma4-e4b-it-q4";
const E2B_SHA: &str = "9378bc471710229ef165709b62e34bfb62231420ddaf6d729e727305b5b8672d";
const E4B_SHA: &str = "519b9793ed6ce0ff530f1b7c96e848e08e49e7af4d57bb97f76215963a54146d";

#[test]
fn default_is_e2b_and_both_tiers_resolve() {
    assert_eq!(UNDERSTANDING_DEFAULT_ID, E2B_ID);
    assert_eq!(understanding_default_spec().id, E2B_ID);
    assert_eq!(understanding_default_spec().sha, E2B_SHA);

    let e2b = understanding_model_spec(E2B_ID).expect("e2b resolves");
    let e4b = understanding_model_spec(E4B_ID).expect("e4b resolves");
    assert_eq!(e2b.repo, "unsloth/gemma-4-E2B-it-GGUF");
    assert_eq!(e2b.file, "gemma-4-E2B-it-Q4_K_M.gguf");
    assert_eq!(e2b.byte_size, 3_106_736_256);
    assert_eq!(e2b.sha, E2B_SHA);
    assert_eq!(e4b.repo, "unsloth/gemma-4-E4B-it-GGUF");
    assert_eq!(e4b.file, "gemma-4-E4B-it-Q4_K_M.gguf");
    assert_eq!(e4b.byte_size, 4_977_169_568);
    assert_eq!(e4b.sha, E4B_SHA);

    // The two tiers are genuinely distinct (a leak from one to the other would collapse them).
    assert_ne!(e2b.sha, e4b.sha);
    assert_ne!(e2b.file, e4b.file);
    assert_ne!(e2b.byte_size, e4b.byte_size);
}

#[test]
fn unknown_id_is_none_and_ids_list_is_default_first() {
    assert!(understanding_model_spec("gemma4-e8b-it-q4").is_none());
    assert!(understanding_model_spec("nomic-embed-text-v1.5").is_none());
    assert!(understanding_model_disclosure("bogus").is_none());
    assert_eq!(understanding_model_ids(), vec![E2B_ID, E4B_ID]);
    assert_eq!(UNDERSTANDING_MODELS.len(), 2);
}

#[test]
fn disclosures_cover_both_tiers_with_honest_sizes() {
    let default = understanding_default_disclosure();
    assert_eq!(default.id, E2B_ID);
    assert_eq!(default.byte_size, 3_106_736_256);

    let all = understanding_disclosures();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].id, E2B_ID);
    assert_eq!(all[0].byte_size, 3_106_736_256);
    assert_eq!(all[1].id, E4B_ID);
    assert_eq!(all[1].byte_size, 4_977_169_568);
}

#[test]
fn fresh_cache_reports_neither_tier_cached() {
    let temp = tempfile::tempdir().expect("temp root");
    let catalog =
        Catalog::open_with_scope_against_root(temp.path(), "tiers-test").expect("catalog");
    assert!(!catalog.understanding_model_cached(E2B_ID));
    assert!(!catalog.understanding_model_cached(E4B_ID));
    // An unknown id is never "cached".
    assert!(!catalog.understanding_model_cached("gemma4-e8b-it-q4"));
}

#[test]
fn seed_verifies_each_tier_against_its_own_sha() {
    // Seeding a NON-matching file must fail with THAT tier's expected sha — proving the per-model
    // sha (and, with it, the per-model file/dir/manifest) is wired through, not the default's.
    let temp = tempfile::tempdir().expect("temp root");
    let catalog =
        Catalog::open_with_scope_against_root(temp.path(), "tiers-test").expect("catalog");
    let fake = temp.path().join("not-the-real-gguf.bin");
    std::fs::write(&fake, b"not a real gemma gguf").expect("write fake");

    let e2b_err = catalog
        .seed_understanding_model_from_path(E2B_ID, &fake)
        .expect_err("a non-matching file cannot seed e2b");
    let e4b_err = catalog
        .seed_understanding_model_from_path(E4B_ID, &fake)
        .expect_err("a non-matching file cannot seed e4b");
    assert!(
        format!("{e2b_err:?}").contains(E2B_SHA),
        "e2b seed must verify against the e2b sha: {e2b_err:?}"
    );
    assert!(
        format!("{e4b_err:?}").contains(E4B_SHA),
        "e4b seed must verify against the e4b sha: {e4b_err:?}"
    );
    // Neither wrote a manifest (the sha gate rejected before publish).
    assert!(!catalog.understanding_model_cached(E2B_ID));
    assert!(!catalog.understanding_model_cached(E4B_ID));
}

/// Weights-gated: seed each real tier from its staged GGUF and assert it caches under its OWN model
/// dir with its OWN manifest (e4b never overwrites e2b). Skips cleanly when the GGUFs are absent.
#[test]
fn real_seed_caches_each_tier_independently() {
    let Some((e2b, e4b)) = locate_both() else {
        eprintln!("SKIP real_seed_caches_each_tier_independently: staged GGUFs absent");
        return;
    };
    let temp = tempfile::tempdir().expect("temp root");
    let catalog =
        Catalog::open_with_scope_against_root(temp.path(), "tiers-test").expect("catalog");

    catalog
        .seed_understanding_model_from_path(E2B_ID, &e2b)
        .expect("seed e2b");
    catalog
        .seed_understanding_model_from_path(E4B_ID, &e4b)
        .expect("seed e4b");

    assert!(catalog.understanding_model_cached(E2B_ID));
    assert!(catalog.understanding_model_cached(E4B_ID));

    let e2b_entry = catalog.show(E2B_ID).expect("e2b entry");
    let e4b_entry = catalog.show(E4B_ID).expect("e4b entry");
    assert_eq!(e2b_entry.byte_size, 3_106_736_256);
    assert_eq!(e4b_entry.byte_size, 4_977_169_568);
    assert_eq!(
        e2b_entry.artifact_filenames,
        vec!["gemma-4-E2B-it-Q4_K_M.gguf".to_string()]
    );
    assert_eq!(
        e4b_entry.artifact_filenames,
        vec!["gemma-4-E4B-it-Q4_K_M.gguf".to_string()]
    );
    assert_eq!(e2b_entry.per_file_sha256[0].1, E2B_SHA);
    assert_eq!(e4b_entry.per_file_sha256[0].1, E4B_SHA);
}

fn locate_both() -> Option<(PathBuf, PathBuf)> {
    let home = std::env::var("HOME").ok()?;
    let staging = PathBuf::from(home).join(".cache/cg/staging");
    let e2b = staging.join("gemma-4-E2B-it-Q4_K_M.gguf");
    let e4b = staging.join("gemma-4-E4B-it-Q4_K_M.gguf");
    (e2b.exists() && e4b.exists()).then_some((e2b, e4b))
}
