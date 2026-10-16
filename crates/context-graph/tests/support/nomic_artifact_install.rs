// Compiled only as the private Nomic unit-test module; these bytes never load a model.
use super::*;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream as LoopbackStream};
use std::process::{Child, Command};

const CHILD_ENV: &str = "CG_NOMIC_ARTIFACT_CHILD";
const TEST_NAME: &str = "embedder::nomic::artifact_install_tests::separate_processes_publish_and_read_complete_sets_and_failures_restore_the_previous_set";

struct Children(Vec<Child>);

impl Drop for Children {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn write_set(path: &Path, id: u8) {
    for (index, name) in NOMIC_ARTIFACT_FILES.into_iter().enumerate() {
        file_system::write(path.join(name), [id, index as u8]).unwrap();
    }
}

// This validator tests set coherence only. It deliberately makes no claim that tiny bytes
// are Nomic weights. The real parser is covered by invalid_download_preserves_the_existing_cache_and_cleans_its_own_directory.
fn assert_set(path: &Path, allowed: &[u8]) -> u8 {
    let first = file_system::read(path.join(NOMIC_ARTIFACT_FILES[0])).unwrap();
    assert_eq!(first.len(), 2);
    let id = first[0];
    assert!(allowed.contains(&id), "unexpected set {id}");
    for (index, name) in NOMIC_ARTIFACT_FILES.into_iter().enumerate() {
        assert_eq!(
            file_system::read(path.join(name)).unwrap(),
            [id, index as u8]
        );
    }
    id
}

fn run_child(arguments: &str) {
    let fields: Vec<_> = arguments.splitn(4, '|').collect();
    let role = fields[0];
    let failure: usize = fields[1].parse().unwrap();
    let directory = Path::new(fields[3]);
    let mut stream = LoopbackStream::connect(fields[2]).unwrap();
    if role == "abandon" {
        let access = NomicCacheAccess::acquire(directory).unwrap();
        let staging = NomicDownloadDirectory::new(directory).unwrap();
        write_set(&staging.0, 1);
        let _ = access.publish(
            &staging.0,
            |path| {
                assert_set(path, &[1]);
                Ok(())
            },
            |index| {
                if index == 1 {
                    // Terminate without any destructor, after the first actual replacement.
                    std::process::exit(0);
                }
                Ok(())
            },
        );
        panic!("the process must terminate during publication");
    }
    if role != "first" {
        // The first publisher is stopped after replacing only the weights. Prove the
        // exact OS lock rejects admission before entering the production blocking acquire.
        let lock = file_system::OpenOptions::new()
            .read(true)
            .write(true)
            .open(directory.join(".lock"))
            .unwrap();
        assert!(matches!(
            lock.try_lock(),
            Err(file_system::TryLockError::WouldBlock)
        ));
        stream
            .write_all(if role == "reader" { b"R" } else { b"W" })
            .unwrap();
        drop(lock);
    }
    if role == "reader" {
        // Admission is the same path all production model reads use.
        let allowed: &[u8] = if failure == 4 { &[1, 2] } else { &[0, 2] };
        NomicCacheAccess::with(directory, |_access| {
            assert_set(directory, allowed);
            Ok(())
        })
        .unwrap();
        stream.write_all(b"D").unwrap();
        return;
    }
    let access = NomicCacheAccess::acquire(directory).unwrap();
    let id = if role == "first" { 1 } else { 2 };
    let staging = NomicDownloadDirectory::new(directory).unwrap();
    write_set(&staging.0, id);
    let result = access.publish(
        &staging.0,
        |path| {
            assert_set(path, &[id]);
            Ok(())
        },
        |index| {
            if role == "first" && index == 1 {
                stream.write_all(b"F").unwrap();
                let mut release = [0];
                stream.read_exact(&mut release).unwrap();
                assert_eq!(release, [1]);
            }
            if role == "first" && index == failure {
                return Err(CgError::Engine(
                    "injected set publication failure".to_string(),
                ));
            }
            Ok(())
        },
    );
    if role == "first" && failure < 4 {
        assert!(matches!(result, Err(CgError::Engine(message))
            if message == "injected set publication failure"));
        assert_set(directory, &[0]);
    } else {
        result.unwrap();
        assert_set(directory, &[id]);
    }
    stream.write_all(b"D").unwrap();
}

fn spawn_child(
    children: &mut Children,
    role: &str,
    failure: usize,
    address: &str,
    directory: &Path,
) {
    children.0.push(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST_NAME, "--nocapture"])
            .env(
                CHILD_ENV,
                format!("{role}|{failure}|{address}|{}", directory.display()),
            )
            .spawn()
            .unwrap(),
    );
}

/// Separate processes read complete four-file sets and failed replacements restore every prior file.
#[test]
fn separate_processes_publish_and_read_complete_sets_and_failures_restore_the_previous_set() {
    if let Ok(arguments) = std::env::var(CHILD_ENV) {
        run_child(&arguments);
        return;
    }
    // Fail after each possible partial prefix, or commit the entire competing set.
    for failure in [1, 2, 3, 4] {
        let directory = tempfile::tempdir().unwrap();
        {
            let access = NomicCacheAccess::acquire(directory.path()).unwrap();
            let staging = NomicDownloadDirectory::new(directory.path()).unwrap();
            write_set(&staging.0, 0);
            access
                .publish(
                    &staging.0,
                    |path| {
                        assert_set(path, &[0]);
                        Ok(())
                    },
                    |_| Ok(()),
                )
                .unwrap();
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let mut children = Children(Vec::new());
        spawn_child(&mut children, "first", failure, &address, directory.path());
        let (mut first, _) = listener.accept().unwrap();
        let mut phase = [0];
        first.read_exact(&mut phase).unwrap();
        assert_eq!(phase, *b"F");
        spawn_child(&mut children, "reader", failure, &address, directory.path());
        spawn_child(&mut children, "second", failure, &address, directory.path());
        let mut waiting = Vec::new();
        let mut roles = Vec::new();
        for _ in 0..2 {
            let (mut connection, _) = listener.accept().unwrap();
            connection.read_exact(&mut phase).unwrap();
            roles.push(phase[0]);
            waiting.push(connection);
        }
        roles.sort();
        assert_eq!(roles, b"RW");
        first.write_all(&[1]).unwrap();
        first.read_exact(&mut phase).unwrap();
        assert_eq!(phase, *b"D");
        for connection in &mut waiting {
            connection.read_exact(&mut phase).unwrap();
            assert_eq!(phase, *b"D");
        }
        for child in &mut children.0 {
            assert!(child.wait().unwrap().success());
        }
        let _access = NomicCacheAccess::acquire(directory.path()).unwrap();
        assert_set(directory.path(), &[2]);
        assert_eq!(file_system::read_dir(directory.path()).unwrap().count(), 5);
    }
}

/// Failed staging reads and a refusal immediately before replacement leave the prior complete set intact.
#[test]
fn staging_and_prepublication_failures_preserve_the_complete_set() {
    struct FailedRead;
    impl Read for FailedRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("injected staging read failure"))
        }
    }
    let directory = tempfile::tempdir().unwrap();
    let access = NomicCacheAccess::acquire(directory.path()).unwrap();
    write_set(directory.path(), 0);
    let staging = NomicDownloadDirectory::new(directory.path()).unwrap();
    let destination = staging.0.join("model.safetensors");
    let result = install_artifact(&mut FailedRead, &destination, "local bytes");
    assert!(
        matches!(result, Err(CgError::EmbedderModelNotAvailable {source, path})
        if source == "local bytes" && path == destination.display().to_string())
    );
    assert_eq!(file_system::read_dir(&staging.0).unwrap().count(), 0);
    assert_set(directory.path(), &[0]);
    write_set(&staging.0, 1);
    for fail_validation in [true, false] {
        let result = access.publish(
            &staging.0,
            |path| {
                assert_set(path, &[1]);
                if fail_validation {
                    Err(CgError::Engine("validation refused".to_string()))
                } else {
                    Ok(())
                }
            },
            |index| {
                assert_eq!(index, 0);
                Err(CgError::Engine("prepublication refused".to_string()))
            },
        );
        assert!(matches!(result, Err(CgError::Engine(message))
            if message == if fail_validation { "validation refused" } else { "prepublication refused" }));
        assert_set(directory.path(), &[0]);
        assert_set(&staging.0, &[1]);
        assert!(!directory.path().join(".nomic-rollback").exists());
    }
    // Let the actual third rename fail after two successful destination replacements.
    // This is an OS error from the production operation, not just a hook returning an error.
    let result = access.publish(
        &staging.0,
        |path| {
            assert_set(path, &[1]);
            Ok(())
        },
        |index| {
            if index == 2 {
                file_system::remove_file(staging.0.join("config.json")).unwrap();
            }
            Ok(())
        },
    );
    assert!(matches!(result, Err(CgError::Engine(message))
        if message.starts_with("nomic artifact move failed ")
            && message.contains(&staging.0.join("config.json").display().to_string())));
    assert_set(directory.path(), &[0]);
    assert!(!directory.path().join(".nomic-rollback").exists());
}

/// Unwinding restores the prior set and an abandoned publication refuses readers without overwriting later writes.
#[test]
fn unwinding_restores_the_prior_set_and_abandoned_publications_refuse_readers() {
    let directory = tempfile::tempdir().unwrap();
    {
        let access = NomicCacheAccess::acquire(directory.path()).unwrap();
        write_set(directory.path(), 0);
        let staging = NomicDownloadDirectory::new(directory.path()).unwrap();
        write_set(&staging.0, 1);
        let outcome = std::panic::catch_unwind(|| {
            let _ = access.publish(
                &staging.0,
                |_| Ok(()),
                |index| {
                    assert!(index < 2, "injected interruption");
                    Ok(())
                },
            );
        });
        assert!(outcome.is_err());
        assert_set(directory.path(), &[0]);
        assert!(!directory.path().join(".nomic-rollback").exists());
    }
    {
        let _access = NomicCacheAccess::acquire(directory.path()).unwrap();
        assert_set(directory.path(), &[0]);
        assert!(!directory.path().join(".nomic-rollback").exists());
    }
    let abandoned = tempfile::tempdir().unwrap();
    write_set(abandoned.path(), 0);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    let mut children = Children(Vec::new());
    spawn_child(&mut children, "abandon", 0, &address, abandoned.path());
    let (_connection, _) = listener.accept().unwrap();
    assert!(children.0[0].wait().unwrap().success());
    assert!(abandoned.path().join(".nomic-rollback/READY").exists());
    assert_set(&abandoned.path().join(".nomic-rollback"), &[0]);
    // Model a later writer that uses the shared lock but does not understand the journal.
    // Its complete replacement must never be undone by stale automatic recovery.
    {
        let lock = file_system::OpenOptions::new()
            .read(true)
            .write(true)
            .open(abandoned.path().join(".lock"))
            .unwrap();
        lock.lock().unwrap();
        let staging = NomicDownloadDirectory::new(abandoned.path()).unwrap();
        write_set(&staging.0, 2);
        for name in NOMIC_ARTIFACT_FILES {
            publish_artifact(&staging.0.join(name), &abandoned.path().join(name)).unwrap();
        }
    }
    let result = NomicCacheAccess::with(abandoned.path(), |_access| -> Result<()> {
        panic!("a pending transaction must refuse admission before reading");
    });
    assert!(matches!(result, Err(CgError::Engine(message))
        if message.starts_with("nomic cache has an unfinished publication; prior artifacts retained at ")));
    assert_set(abandoned.path(), &[2]);
    assert_set(&abandoned.path().join(".nomic-rollback"), &[0]);
}

/// Invalid downloaded artifacts leave every existing cache file intact and clean only their staging directory.
#[test]
fn invalid_download_preserves_the_existing_cache_and_cleans_its_own_directory() {
    let directory = tempfile::tempdir().unwrap();
    let names = [
        "model.safetensors",
        "tokenizer.json",
        "config.json",
        "REVISION",
    ];
    for name in names {
        file_system::write(directory.path().join(name), name).unwrap();
    }
    let other = NomicDownloadDirectory::new(directory.path()).unwrap();
    file_system::write(other.0.join("model.safetensors"), b"another writer").unwrap();
    let staging_path;
    {
        let staging = NomicDownloadDirectory::new(directory.path()).unwrap();
        staging_path = staging.0.clone();
        for name in names {
            file_system::write(staging.0.join(name), b"truncated").unwrap();
        }
        let access = NomicCacheAccess::acquire(directory.path()).unwrap();
        let result = publish_download(&access, &staging.0, directory.path(), None);
        assert!(matches!(result, Err(CgError::EmbedderModelCorrupt { path })
            if path == staging.0.join("model.safetensors").display().to_string()));
        let reported = artifact_error_at_destination(
            CgError::EmbedderModelNotAvailable {
                source: "local bytes".to_string(),
                path: staging.0.join("model.safetensors").display().to_string(),
            },
            &staging.0,
            directory.path(),
        );
        assert!(
            matches!(reported, CgError::EmbedderModelNotAvailable { source, path }
            if source == "local bytes"
                && path == directory.path().join("model.safetensors").display().to_string())
        );
        for name in names {
            assert_eq!(
                file_system::read(directory.path().join(name)).unwrap(),
                name.as_bytes()
            );
        }
    }
    assert!(!staging_path.exists());
    assert_eq!(
        file_system::read(other.0.join("model.safetensors")).unwrap(),
        b"another writer"
    );
}
