#![cfg(unix)]
//! Editing a store a consumer already has never brings one into existence.
//!
//! ## What a consumer is entitled to
//!
//! A downstream runtime that means to CHANGE a store it already has must be able to say so in the
//! open itself. Without that it has two bad choices: look the path up first and race whoever moves
//! it between the look and the open, or accept that a typo, an unmounted volume or a link whose
//! target is gone quietly materializes a brand-new empty deployment and reports success. The second
//! is the dangerous one — the consumer is told everything worked and writes its data into a store
//! nobody will ever look for.
//!
//! `Store::open_existing_for_change` is the atomic door. The open is ATTEMPTED — context-graph asks
//! no existence question of its own, so there is no answer that could go stale between the asking
//! and the opening — and the engine's typed outcome is the sole authority. A store that is not
//! there comes back missing, one that is there and cannot be looked at comes back unreadable, and
//! every refusal leaves the store, its companion and the directory holding them byte-for-byte as
//! they were found.
//!
//! The door takes the same consumer declaration and scope labels as create-capable openers,
//! which create an absent store.
//!
//! Every assertion below is about a typed value, a row, or a pathname. Nothing sleeps.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use context_graph::{
    CgError, ConsumerDeclaration, ConsumerSchema, ConsumerTable, ScopeLabel, Store, StoreConfig,
};
use contextdb_core::{Error as EngineError, Value};

/// A consumer table whose rows carry the role that may write them.
const SETTINGS_DDL: &str = "CREATE TABLE vigil_settings (name TEXT PRIMARY KEY, requested TEXT, scope_label TEXT SCOPE_LABEL ('edge', 'server'))";

fn declared_schema() -> ConsumerSchema {
    ConsumerSchema {
        namespace: "vigil".to_string(),
        tables: vec![ConsumerTable {
            name: "vigil_settings".to_string(),
            ddl: SETTINGS_DDL.to_string(),
        }],
    }
}

/// A declaration with no role narrowing on it: the consumer may write every row its own
/// declaration accepts.
fn unrestricted_declaration() -> ConsumerDeclaration {
    ConsumerDeclaration {
        schema: declared_schema(),
        scope_labels: BTreeSet::new(),
    }
}

/// The same declaration opened as the edge, which owns none of the rows reserved for the hub.
fn edge_declaration() -> ConsumerDeclaration {
    ConsumerDeclaration {
        schema: declared_schema(),
        scope_labels: BTreeSet::from([ScopeLabel::new("edge")]),
    }
}

fn config(path: &Path) -> StoreConfig {
    StoreConfig {
        db_path: path.to_path_buf(),
        ..Default::default()
    }
}

/// One entry in a directory, with enough of it to notice any change.
#[derive(Debug, PartialEq, Eq)]
enum EntryImage {
    File(Vec<u8>),
    Symlink(PathBuf),
    Directory,
}

/// Everything a directory holds, by name, down to the bytes: no store, no companion, no lock and
/// no residue of any kind may appear across a refusal.
fn directory_image(directory: &Path) -> BTreeMap<String, EntryImage> {
    std::fs::read_dir(directory)
        .expect("read the directory holding the store")
        .map(|entry| {
            let entry = entry.expect("read one directory entry");
            let name = entry.file_name().to_string_lossy().into_owned();
            let kind = entry.file_type().expect("classify one directory entry");
            let image = if kind.is_symlink() {
                EntryImage::Symlink(
                    std::fs::read_link(entry.path()).expect("read one link's target"),
                )
            } else if kind.is_dir() {
                EntryImage::Directory
            } else {
                EntryImage::File(std::fs::read(entry.path()).expect("read one file's bytes"))
            };
            (name, image)
        })
        .collect()
}

fn scratch_root(directory: &tempfile::TempDir) -> PathBuf {
    std::fs::canonicalize(directory.path()).expect("resolve the scratch directory")
}

fn write_setting(store: &Store, name: &str, scope: &str) -> Result<(), EngineError> {
    store
        .sync_database()
        .execute(
            "INSERT INTO vigil_settings (name, requested, scope_label) VALUES ($name, $requested, $scope)",
            &HashMap::from([
                ("name".to_string(), Value::Text(name.to_string())),
                ("requested".to_string(), Value::Text("4".to_string())),
                ("scope".to_string(), Value::Text(scope.to_string())),
            ]),
        )
        .map(|_| ())
}

fn setting_names(store: &Store) -> Vec<String> {
    store
        .sync_database()
        .execute(
            "SELECT name FROM vigil_settings ORDER BY name",
            &HashMap::new(),
        )
        .expect("read the consumer table back")
        .rows
        .iter()
        .map(|row| match &row[0] {
            Value::Text(name) => name.clone(),
            other => panic!("unexpected consumer row shape: {other:?}"),
        })
        .collect()
}

/// A real store the consumer already has, closed and let go.
fn seeded_store(directory: &Path) -> PathBuf {
    let path = directory.join("consumer.db");
    let store = Store::open_with_consumer_schema(config(&path), Vec::new(), declared_schema())
        .expect("the consumer's first run creates the store and its declared table");
    write_setting(&store, "detector_sample_frames", "edge")
        .expect("an unrestricted handle owns the row");
    drop(store);
    path
}

/// An open that must refuse, with its refusal.
fn expect_refusal(result: context_graph::Result<Store>, context: &str) -> CgError {
    match result {
        Ok(_store) => panic!("{context}: the edit door opened instead of refusing"),
        Err(error) => error,
    }
}

// The ordinary case: the store is there, the door opens it for change, and the write lands.
#[test]
fn an_existing_store_opens_for_change_and_the_write_lands() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let path = seeded_store(&root);

    let store =
        Store::open_existing_for_change(config(&path), Vec::new(), unrestricted_declaration())
            .expect("a store the consumer already has opens for change");

    write_setting(&store, "detector_confidence_threshold", "edge")
        .expect("the door opened the store writable, so the write lands");
    assert_eq!(
        vec![
            "detector_confidence_threshold".to_string(),
            "detector_sample_frames".to_string(),
        ],
        setting_names(&store),
        "the door changed the store the consumer already had, keeping what was in it",
    );
}

// A store that is not there is a typed refusal that names it, and nothing appears where it was
// asked for.
#[test]
fn an_absent_store_is_refused_by_the_edit_door_and_nothing_is_created() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let absent = root.join("never-created.db");
    let before = directory_image(&root);

    let refusal = expect_refusal(
        Store::open_existing_for_change(config(&absent), Vec::new(), unrestricted_declaration()),
        "absent store",
    );

    let CgError::StoreMissing { path } = &refusal else {
        panic!("a store that is not there must be typed missing, got: {refusal:?}");
    };
    assert_eq!(
        &absent, path,
        "the refusal must name the store the consumer meant to change",
    );
    assert!(
        !absent.exists(),
        "the edit door must never create a store at {absent:?}",
    );
    assert_eq!(
        before,
        directory_image(&root),
        "the refusal must leave the directory byte-for-byte as it found it",
    );
}

// A link whose target is gone must refuse rather than be created through — the case that otherwise
// hands the consumer an empty deployment and reports success.
#[test]
fn a_dangling_link_is_refused_by_the_edit_door_and_never_created_through() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let target = root.join("target-that-is-gone.db");
    let link = root.join("store-link.db");
    std::os::unix::fs::symlink(&target, &link)
        .expect("plant a link where a store the consumer expects used to be");
    let before = directory_image(&root);

    let refusal = expect_refusal(
        Store::open_existing_for_change(config(&link), Vec::new(), unrestricted_declaration()),
        "dangling link",
    );

    let named = match &refusal {
        CgError::StoreMissing { path } => path.clone(),
        other => panic!("a link whose target is gone must be a typed refusal, got: {other:?}"),
    };
    assert!(
        named == link || named == target,
        "the refusal must name the store it could not open; it named {named:?}",
    );
    assert!(
        !target.exists(),
        "nothing may be created at the link's missing target {target:?}",
    );
    assert!(
        std::fs::symlink_metadata(&link)
            .expect("the link itself survives the refusal")
            .file_type()
            .is_symlink(),
        "the refusal must leave the caller's own path as it found it",
    );
    assert_eq!(
        before,
        directory_image(&root),
        "the refusal must leave the directory byte-for-byte as it found it",
    );
}

// A store that IS there and cannot be looked at is unreadable, never missing: the fix is a
// permission or a mount, not a restore.
#[test]
fn a_store_this_process_cannot_look_at_is_refused_as_unreadable_by_the_edit_door() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let vault = root.join("vault");
    std::fs::create_dir(&vault).expect("create the directory the store lives in");
    let path = seeded_store(&vault);
    let before = directory_image(&vault);

    std::fs::set_permissions(&vault, std::fs::Permissions::from_mode(0o000))
        .expect("seal the directory the store lives in");
    let sealed_answer =
        Store::open_existing_for_change(config(&path), Vec::new(), unrestricted_declaration())
            .err();
    std::fs::set_permissions(&vault, std::fs::Permissions::from_mode(0o700))
        .expect("unseal the directory so the rest of the journey can look at it");

    let refusal = sealed_answer
        .expect("an edit-door open of a store this process cannot look at must refuse");
    assert!(
        !matches!(refusal, CgError::StoreMissing { .. }),
        "a store that cannot be looked at is UNREADABLE, not missing: {refusal:?}",
    );
    let CgError::StoreUnreadable { path: named, .. } = &refusal else {
        panic!("a metadata failure must be the typed unreadable refusal, got: {refusal:?}");
    };
    assert_eq!(
        &path, named,
        "the refusal must name the store it could not look at"
    );
    assert_eq!(
        before,
        directory_image(&vault),
        "the refusal must leave the directory byte-for-byte as it found it",
    );
}

// The edit door refuses the absent store that the ordinary opener creates.
#[test]
fn the_edit_door_refuses_the_absent_store_the_ordinary_opener_still_creates() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let path = root.join("first-run.db");

    let refusal = expect_refusal(
        Store::open_existing_for_change(config(&path), Vec::new(), unrestricted_declaration()),
        "absent store at the edit door",
    );
    assert!(
        matches!(refusal, CgError::StoreMissing { .. }),
        "the edit door refuses an absent store: {refusal:?}",
    );
    assert!(!path.exists(), "the refused door created nothing");

    let store = Store::open_with_consumer_schema(config(&path), Vec::new(), declared_schema())
        .expect("the ordinary opener creates an absent store");
    write_setting(&store, "detector_sample_frames", "edge").expect("the first run owns the write");
    assert_eq!(
        vec!["detector_sample_frames".to_string()],
        setting_names(&store),
    );
}

// The door is scope-aware: a consumer that opens it as the edge is held to the `SCOPE_LABEL` its
// own declaration wrote down, exactly as it is on the openers that already take a role.
#[test]
fn the_edit_door_holds_an_edge_consumer_to_the_role_its_declaration_wrote_down() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = scratch_root(&directory);
    let path = seeded_store(&root);

    let store = Store::open_existing_for_change(config(&path), Vec::new(), edge_declaration())
        .expect("the edge opens the store it already has");

    let refusal = write_setting(&store, "hub_only_setting", "server")
        .expect_err("an edge must not write a row its own declaration reserves for the hub");
    match refusal {
        EngineError::ScopeLabelViolation { requested, allowed } => {
            assert_eq!(
                ScopeLabel::new("server"),
                requested,
                "the refusal must name the role the row asked to be written as",
            );
            assert_eq!(
                BTreeSet::from([ScopeLabel::new("edge")]),
                allowed,
                "the refusal must name the role the door was opened with",
            );
        }
        other => panic!("the engine's own scope refusal must surface, got {other:?}"),
    }

    write_setting(&store, "edge_setting", "edge").expect("the edge owns its own role's row");
    assert_eq!(
        vec![
            "detector_sample_frames".to_string(),
            "edge_setting".to_string(),
        ],
        setting_names(&store),
        "only the row the edge owns landed",
    );
}
