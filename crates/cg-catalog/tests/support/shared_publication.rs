// Private unit-test support: every writer below calls the real catalog publication helper.
use super::*;
use std::io::Read;
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command};

const CHILD_ENV: &str = "CG_CATALOG_PUBLICATION_CHILD";
const POINT_ENV: &str = "CG_CATALOG_PUBLICATION_POINT";
const TEST_NAME: &str = "shared_publication_tests::interrupted_catalog_publications_refuse_catalog_and_direct_nomic_readers";

struct Children(Vec<Child>);

impl Drop for Children {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub(super) fn after_replace(index: usize) -> Result<()> {
    let Ok(point) = std::env::var(POINT_ENV) else {
        return Ok(());
    };
    let (mode, boundary) = point.split_once(':').unwrap();
    if index + 1 != boundary.parse::<usize>().unwrap() {
        return Ok(());
    }
    if mode == "fail" {
        return Err(CgCatalogError::SourceUnavailable {
            model_id: NOMIC_ID.to_string(),
            message: "injected publication failure".to_string(),
        });
    }
    let arguments = std::env::var(CHILD_ENV).unwrap();
    let address = arguments.split('|').nth(1).unwrap();
    let mut stream = TcpStream::connect(address).unwrap();
    stream.write_all(&[(index + 1) as u8]).unwrap();
    let mut release = [0];
    stream.read_exact(&mut release).unwrap();
    assert_eq!(release, [1]);
    Ok(())
}

fn catalog(root: &Path) -> Catalog {
    Catalog::open_with_scope_against_root(root, "publication-test").unwrap()
}

fn write_set(directory: &Path, identity: &str) {
    fs::create_dir_all(directory).unwrap();
    for file in ARTIFACT_FILES {
        let bytes = if file == "REVISION" {
            identity.to_string()
        } else {
            format!("{identity}:{file}")
        };
        fs::write(directory.join(file), bytes).unwrap();
    }
}

fn allow_set(root: &Path, directory: &Path, revision: &str) {
    let shas: toml::Table = shas_from_artifact_dir(directory)
        .unwrap()
        .into_iter()
        .map(|(name, sha)| (name, toml::Value::String(sha)))
        .collect();
    let source = toml::Table::from_iter([
        (
            "base_url".to_string(),
            toml::Value::String("file:///unused".to_string()),
        ),
        (
            "revision".to_string(),
            toml::Value::String(revision.to_string()),
        ),
        ("sha256".to_string(), toml::Value::Table(shas)),
    ]);
    let sources = toml::Table::from_iter([(NOMIC_ID.to_string(), toml::Value::Table(source))]);
    let config = toml::Table::from_iter([("sources".to_string(), toml::Value::Table(sources))]);
    fs::create_dir_all(root.join("cg")).unwrap();
    fs::write(
        root.join("cg/catalog-sources.toml"),
        toml::to_string(&config).unwrap(),
    )
    .unwrap();
}

fn publish(catalog: &Catalog, staging: &Path, revision: &str) -> Result<()> {
    let _lock = catalog.lock_model(NOMIC_ID)?;
    catalog.publish_temp_artifacts_with_revision_and_source(
        &spec_by_id(NOMIC_ID).unwrap(),
        staging,
        revision,
        None,
    )
}

fn setup(root: &Path) {
    let catalog = catalog(root);
    let staging = catalog.model_dir(NOMIC_ID).join(".initial");
    write_set(&staging, "old");
    allow_set(root, &staging, "old");
    publish(&catalog, &staging, "old").unwrap();
    assert_eq!(
        catalog.fetch(None, NOMIC_ID, OutputMode::Headless).unwrap(),
        FetchResult::AlreadyCached
    );
    write_set(&catalog.model_dir(NOMIC_ID).join(".next"), "new");
}

fn snapshot(directory: &Path) -> BTreeMap<String, Vec<u8>> {
    fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_file())
        .map(|path| {
            (
                path.file_name().unwrap().to_str().unwrap().to_string(),
                fs::read(&path).unwrap(),
            )
        })
        .collect()
}

fn assert_catalog_refusal<T: fmt::Debug>(result: Result<T>, marker: &str) {
    assert!(
        matches!(result, Err(CgCatalogError::SourceUnavailable { model_id, message })
        if model_id == NOMIC_ID && message.contains(marker))
    );
}

fn direct_embedder(directory: &Path, revision: Option<&str>) -> NomicTextEmbedder {
    NomicTextEmbedder::new(EmbedderConfig {
        source: ModelSource::LocalPath(directory.to_path_buf()),
        offline: true,
        cache_dir: directory.parent().map(Path::to_path_buf),
        pinned_revision: revision.map(str::to_string),
        network_policy: "offline".to_string(),
    })
}

fn assert_refused(root: &Path, marker: &str) {
    let catalog = catalog(root);
    let directory = catalog.model_dir(NOMIC_ID);
    let before = snapshot(&directory);
    assert_catalog_refusal(catalog.show(NOMIC_ID), marker);
    assert_catalog_refusal(catalog.fetch(None, NOMIC_ID, OutputMode::Headless), marker);
    assert_catalog_refusal(
        catalog.fetch_from_path(NOMIC_ID, directory.join(".next")),
        marker,
    );
    assert_catalog_refusal(validate_artifact_identity(NOMIC_ID, &directory), marker);
    assert_catalog_refusal(catalog.model_revision_for_runtime(NOMIC_ID), marker);
    let embedder = CatalogTextEmbedder::new(catalog, NOMIC_ID, EmbeddingSpace::default());
    assert!(
        matches!(embedder.embed(EmbeddingInput::Text("probe".to_string())),
        Err(CgError::Engine(message)) if message.contains(marker))
    );
    // Exercise real loading, including both pinned and unpinned admission. Tiny fixture bytes
    // must be refused for the pending marker BEFORE any model parser or inference is reached.
    for revision in [None, Some("old"), Some("new")] {
        let embedder = direct_embedder(&directory, revision);
        for _ in 0..2 {
            assert!(
                matches!(embedder.embed(EmbeddingInput::Text("probe".to_string())),
                Err(CgError::Engine(message)) if message.contains(marker))
            );
            assert!(!embedder.structured_loader_events().iter().any(|event| matches!(event,
                context_graph::embedder::load_observer::EmbedderLoadEvent::LoadStarted { .. }
                | context_graph::embedder::load_observer::EmbedderLoadEvent::CacheHit { .. })));
        }
    }
    assert_eq!(
        snapshot(&directory),
        before,
        "refusal must never reset metadata or replay artifacts"
    );
}

fn assert_committed(root: &Path, revision: &str, expected_fetch: FetchResult) {
    let catalog = catalog(root);
    let directory = catalog.model_dir(NOMIC_ID);
    assert_eq!(
        catalog.fetch(None, NOMIC_ID, OutputMode::Headless).unwrap(),
        expected_fetch
    );
    // Admission waits for the publisher's writer lock; only then can the marker be absent.
    assert!(!directory.join(".catalog-publication").exists());
    assert_eq!(
        catalog.show(NOMIC_ID).unwrap().cache_status,
        CacheStatus::Cached
    );
    validate_artifact_identity(NOMIC_ID, &directory).unwrap();
    assert_eq!(
        catalog.model_revision_for_runtime(NOMIC_ID).unwrap(),
        revision
    );
    for file in ARTIFACT_FILES {
        let expected = if file == "REVISION" {
            revision.to_string()
        } else {
            format!("{revision}:{file}")
        };
        assert_eq!(fs::read(directory.join(file)).unwrap(), expected.as_bytes());
    }
    // This is successful catalog admission, not successful model loading. Tiny local artifacts
    // deliberately fail the real Nomic parser; a wrong pin fails earlier at revision admission.
    assert!(matches!(
        direct_embedder(&directory, Some(revision))
            .embed(EmbeddingInput::Text("probe".to_string())),
        Err(CgError::EmbedderModelCorrupt { .. })
    ));
    assert!(matches!(
        direct_embedder(&directory, Some("different"))
            .embed(EmbeddingInput::Text("probe".to_string())),
        Err(CgError::EmbedderModelRevisionMismatch { .. })
    ));
}

fn run_child(arguments: &str) {
    let fields: Vec<_> = arguments.splitn(3, '|').collect();
    let root = Path::new(fields[2]);
    if fields[0] == "writer" || fields[0] == "failed-writer" {
        let catalog = catalog(root);
        let staging = catalog.model_dir(NOMIC_ID).join(".next");
        allow_set(root, &staging, "new");
        let result = publish(&catalog, &staging, "new");
        if fields[0] == "failed-writer" {
            assert!(
                matches!(result, Err(CgCatalogError::SourceUnavailable { message, .. })
                if message == "injected publication failure")
            );
        } else {
            result.unwrap();
        }
        return;
    }
    let mut stream = TcpStream::connect(fields[1]).unwrap();
    if fields[0] != "independent" {
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .open(root.join(NOMIC_ID).join(".lock"))
            .unwrap();
        assert!(!FileExt::try_lock_exclusive(&lock).unwrap());
        stream.write_all(b"R").unwrap();
        drop(lock);
    }
    if fields[0] == "direct-refused" {
        let directory = root.join(NOMIC_ID);
        let before = snapshot(&directory);
        assert!(
            matches!(direct_embedder(&directory, Some("old")).embed(EmbeddingInput::Text("probe".to_string())),
            Err(CgError::Engine(message)) if message.contains(".catalog-publication"))
        );
        assert_eq!(snapshot(&directory), before);
    } else if fields[0] == "refused" {
        assert_refused(root, ".catalog-publication");
    } else {
        assert_committed(
            root,
            if fields[0] == "independent" {
                "old"
            } else {
                "new"
            },
            if fields[0] == "independent" {
                FetchResult::AlreadyCached
            } else {
                FetchResult::AlreadyCachedByConcurrentWinner
            },
        );
    }
    stream.write_all(b"D").unwrap();
}

fn spawn(children: &mut Children, role: &str, root: &Path, address: &str, point: &str) -> usize {
    let index = children.0.len();
    children.0.push(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST_NAME, "--nocapture"])
            .env(CHILD_ENV, format!("{role}|{address}|{}", root.display()))
            .env(POINT_ENV, point)
            .spawn()
            .unwrap(),
    );
    index
}

fn receive(stream: &mut TcpStream, expected: u8) {
    let mut byte = [0];
    stream.read_exact(&mut byte).unwrap();
    assert_eq!(byte, [expected]);
}

/// A killed catalog publisher leaves every catalog and direct Nomic reader closed at all four replacement boundaries.
#[test]
fn interrupted_catalog_publications_refuse_catalog_and_direct_nomic_readers() {
    if let Ok(arguments) = std::env::var(CHILD_ENV) {
        run_child(&arguments);
        return;
    }
    for boundary in 1..=4 {
        let root = tempfile::tempdir().unwrap();
        let independent = tempfile::tempdir().unwrap();
        setup(root.path());
        setup(independent.path());
        let directory = root.path().join(NOMIC_ID);
        let old_manifest = fs::read(directory.join("catalog.toml")).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let mut children = Children(Vec::new());
        let writer = spawn(
            &mut children,
            "writer",
            root.path(),
            &address,
            &format!("pause:{boundary}"),
        );
        let (mut paused, _) = listener.accept().unwrap();
        receive(&mut paused, boundary as u8);
        // An unrelated selected cache remains usable while the first one is mid-publication.
        let other = spawn(
            &mut children,
            "independent",
            independent.path(),
            &address,
            "",
        );
        let (mut other_stream, _) = listener.accept().unwrap();
        receive(&mut other_stream, b'D');
        assert!(children.0[other].wait().unwrap().success());
        let readers = [
            spawn(&mut children, "refused", root.path(), &address, ""),
            spawn(&mut children, "direct-refused", root.path(), &address, ""),
        ];
        let mut waiting = Vec::new();
        for _ in readers {
            let (mut connection, _) = listener.accept().unwrap();
            receive(&mut connection, b'R');
            waiting.push(connection);
        }
        children.0[writer].kill().unwrap();
        assert!(!children.0[writer].wait().unwrap().success());
        for connection in &mut waiting {
            receive(connection, b'D');
        }
        for reader in readers {
            assert!(children.0[reader].wait().unwrap().success());
        }
        assert!(directory.join(".catalog-publication").is_file());
        assert_eq!(
            fs::read(directory.join("catalog.toml")).unwrap(),
            old_manifest
        );
        for (index, file) in ARTIFACT_FILES.into_iter().enumerate() {
            let identity = if index < boundary { "new" } else { "old" };
            let expected = if file == "REVISION" {
                identity.to_string()
            } else {
                format!("{identity}:{file}")
            };
            assert_eq!(fs::read(directory.join(file)).unwrap(), expected.as_bytes());
        }
        // Repeated admission after termination must leave the abandoned state untouched.
        assert_refused(root.path(), ".catalog-publication");
    }
}

/// A catalog commit releases blocked readers only after all four files and the matching manifest are published.
#[test]
fn completed_catalog_publication_admits_the_new_complete_identity() {
    let root = tempfile::tempdir().unwrap();
    setup(root.path());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    let mut children = Children(Vec::new());
    let writer = spawn(&mut children, "writer", root.path(), &address, "pause:4");
    let (mut paused, _) = listener.accept().unwrap();
    receive(&mut paused, 4);
    let reader = spawn(&mut children, "committed", root.path(), &address, "");
    let (mut waiting, _) = listener.accept().unwrap();
    receive(&mut waiting, b'R');
    paused.write_all(&[1]).unwrap();
    receive(&mut waiting, b'D');
    for index in [writer, reader] {
        assert!(children.0[index].wait().unwrap().success());
    }
    assert_committed(root.path(), "new", FetchResult::AlreadyCached);
}

/// Catalog replacement errors and abandoned Nomic journals refuse later publishers without changing retained bytes.
#[test]
fn failed_catalog_publications_and_abandoned_nomic_journals_preserve_refusal() {
    for boundary in 1..=4 {
        let root = tempfile::tempdir().unwrap();
        setup(root.path());
        let mut children = Children(Vec::new());
        let writer = spawn(
            &mut children,
            "failed-writer",
            root.path(),
            "unused",
            &format!("fail:{boundary}"),
        );
        assert!(children.0[writer].wait().unwrap().success());
        assert_refused(root.path(), ".catalog-publication");
    }
    // An actual rename failure after a partial prefix also leaves admission closed.
    let failed = tempfile::tempdir().unwrap();
    setup(failed.path());
    let failed_catalog = catalog(failed.path());
    let staging = failed.path().join(NOMIC_ID).join(".next");
    fs::remove_file(staging.join("config.json")).unwrap();
    assert!(publish(&failed_catalog, &staging, "new").is_err());
    assert_refused(failed.path(), ".catalog-publication");

    // Failure to write catalog metadata after replacing every artifact cannot commit the set.
    let metadata = tempfile::tempdir().unwrap();
    setup(metadata.path());
    let metadata_catalog = catalog(metadata.path());
    let manifest = metadata.path().join(NOMIC_ID).join("catalog.toml");
    fs::remove_file(&manifest).unwrap();
    fs::create_dir(&manifest).unwrap();
    assert!(
        publish(
            &metadata_catalog,
            &metadata.path().join(NOMIC_ID).join(".next"),
            "new"
        )
        .is_err()
    );
    assert_refused(metadata.path(), ".catalog-publication");

    let root = tempfile::tempdir().unwrap();
    setup(root.path());
    let directory = root.path().join(NOMIC_ID);
    fs::create_dir(directory.join(".nomic-rollback")).unwrap();
    fs::write(directory.join(".nomic-rollback/READY"), []).unwrap();
    assert_refused(root.path(), ".nomic-rollback");
    assert!(directory.join(".nomic-rollback/READY").exists());
}

// Compile the production Nomic publisher privately: no test API or cyclic crate dependency.
mod direct_nomic {
    use context_graph::{CgError, Result};
    pub(super) mod publication {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../context-graph/src/embedder/nomic_publication.rs"
        ));
    }
}

fn replacement_metadata(root: &Path) {
    let catalog = catalog(root);
    let directory = catalog.model_dir(NOMIC_ID);
    let before = catalog.show(NOMIC_ID).unwrap();
    let staging = directory.join(".next");
    let new_shas = shas_from_artifact_dir(&staging).unwrap();
    let old_manifest = read_toml(&directory.join("catalog.toml")).unwrap().unwrap();
    direct_nomic::publication::NomicCacheAccess::with(&directory, |access| {
        access.publish(
            &staging,
            |path| {
                // Tiny bytes are validated for exact fixture identity; no model-load claim.
                assert_eq!(shas_from_artifact_dir(path).unwrap(), new_shas);
                Ok(())
            },
            |_| Ok(()),
        )
    })
    .unwrap();
    let installed = read_toml(&directory.join("catalog.toml")).unwrap().unwrap();
    assert_eq!(toml_str(&installed, "source_revision"), Some("new"));
    assert_eq!(
        installed.get("acceptance_records"),
        old_manifest.get("acceptance_records")
    );
    assert_eq!(
        installed.get("consumer_extension"),
        old_manifest.get("consumer_extension")
    );
    assert!(!directory.join(".nomic-rollback").exists());
    let shown = catalog.show(NOMIC_ID).unwrap();
    assert_eq!(shown.per_file_sha256, new_shas);
    assert_ne!(shown.per_file_sha256, before.per_file_sha256);
    // Source override is desired identity, not a claim that its bytes are installed.
    assert!(matches!(shown.source, Source::HuggingFace { revision, .. } if revision == "old"));
    assert_eq!(shown.cache_status, CacheStatus::NotCached);
    fs::remove_file(root.join("cg/catalog-sources.toml")).unwrap();
    assert!(matches!(catalog.show(NOMIC_ID).unwrap().source,
        Source::HuggingFace { revision, .. } if revision == "new"));
}

/// Direct replacement updates installed catalog identity across processes and retains consumer metadata.
#[test]
fn direct_replacement_updates_installed_metadata_and_preserves_configured_identity() {
    const ENV: &str = "CG_CATALOG_METADATA_CHILD";
    if let Ok(root) = std::env::var(ENV) {
        replacement_metadata(Path::new(&root));
        return;
    }
    let root = tempfile::tempdir().unwrap();
    setup(root.path());
    let manifest = root.path().join(NOMIC_ID).join("catalog.toml");
    let mut value = read_toml(&manifest).unwrap().unwrap();
    value.as_table_mut().unwrap().insert(
        "consumer_extension".to_string(),
        toml::Value::String("retain this value".to_string()),
    );
    append_acceptance_record(
        &mut value,
        &AcceptanceRecord {
            accepted_by: Principal::System,
            accepted_at: TxId(17),
            license_url_sha: "license".to_string(),
            artifact_sha: "artifact".to_string(),
            scope_key: "publication-test".to_string(),
        },
    );
    write_toml(&manifest, &value).unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "shared_publication_tests::direct_replacement_updates_installed_metadata_and_preserves_configured_identity", "--nocapture"])
        .env(ENV, root.path()).spawn().unwrap();
    assert!(child.wait().unwrap().success());
}

/// Direct publication restores installed bytes and metadata on every precommit error, including metadata replacement.
#[test]
fn direct_replacement_failures_restore_the_previous_catalog_metadata() {
    for unwind in [false, true] {
        for boundary in 0..=5 {
            let root = tempfile::tempdir().unwrap();
            setup(root.path());
            let directory = root.path().join(NOMIC_ID);
            let before = snapshot(&directory);
            let result = std::panic::catch_unwind(|| {
                direct_nomic::publication::NomicCacheAccess::with(&directory, |access| {
                    access.publish(
                        &directory.join(".next"),
                        |_| Ok(()),
                        |index| {
                            if index == boundary {
                                assert!(!unwind, "injected publication unwind");
                                return Err(CgError::Engine(
                                    "injected publication failure".to_string(),
                                ));
                            }
                            Ok(())
                        },
                    )
                })
            });
            if unwind {
                assert!(result.is_err());
            } else {
                assert!(matches!(result.unwrap(), Err(CgError::Engine(message))
                    if message == "injected publication failure"));
            }
            assert_eq!(snapshot(&directory), before);
            assert_committed(root.path(), "old", FetchResult::AlreadyCached);
        }
    }
    // A real metadata rename failure after all four artifact replacements also rolls back.
    let root = tempfile::tempdir().unwrap();
    setup(root.path());
    let directory = root.path().join(NOMIC_ID);
    let before = snapshot(&directory);
    let staging = directory.join(".next");
    let result = direct_nomic::publication::NomicCacheAccess::with(&directory, |access| {
        access.publish(
            &staging,
            |_| Ok(()),
            |index| {
                if index == 4 {
                    fs::remove_file(staging.join("catalog.toml")).unwrap();
                }
                Ok(())
            },
        )
    });
    assert!(
        matches!(result, Err(CgError::Engine(message)) if message.starts_with("nomic artifact move failed "))
    );
    assert_eq!(snapshot(&directory), before);
    assert_committed(root.path(), "old", FetchResult::AlreadyCached);

    // Unreadable/malformed metadata fails preparation before replacement or journal creation.
    let root = tempfile::tempdir().unwrap();
    setup(root.path());
    let directory = root.path().join(NOMIC_ID);
    fs::write(directory.join("catalog.toml"), "invalid = [").unwrap();
    let before = snapshot(&directory);
    assert!(
        direct_nomic::publication::NomicCacheAccess::with(&directory, |access| {
            access.publish(&directory.join(".next"), |_| Ok(()), |_| Ok(()))
        })
        .is_err()
    );
    assert_eq!(snapshot(&directory), before);
    assert!(!directory.join(".nomic-rollback").exists());
}
