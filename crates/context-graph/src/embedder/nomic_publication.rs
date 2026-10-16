// Private Nomic cache publication; catalog tests compile this same implementation with tiny fixtures.
use super::{CgError, Result};
use sha2::{Digest, Sha256};
use std::{
    fs as file_system, io,
    path::{Path, PathBuf},
};

pub(crate) struct NomicTemporaryArtifact(pub(crate) PathBuf);

impl Drop for NomicTemporaryArtifact {
    fn drop(&mut self) {
        let _ = file_system::remove_file(&self.0);
    }
}

pub(crate) const NOMIC_ARTIFACT_FILES: [&str; 4] = [
    "model.safetensors",
    "tokenizer.json",
    "config.json",
    "REVISION",
];

pub(crate) fn cache_io_error(path: &Path, error: io::Error) -> CgError {
    CgError::Engine(format!(
        "nomic cache IO failed at {}: {error}",
        path.display()
    ))
}

/// The file stays in place: unlinking it would let waiters lock different inodes.
/// All cooperating reads, validation and publication hold this guard. Catalog mutations
/// hold the same lock and reject both owners' unfinished-publication markers.
pub(crate) struct NomicCacheAccess {
    _lock: file_system::File,
    pub(crate) model_dir: PathBuf,
}

// Unwinding restores while the caller still holds the process lock. A killed process cannot
// run this destructor; its pending marker makes subsequent Nomic access fail closed.
struct NomicPublicationRollback<'a>(&'a NomicCacheAccess);

impl Drop for NomicPublicationRollback<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            let _ = self.0.recover();
        }
    }
}

impl NomicCacheAccess {
    pub(crate) fn with<T>(
        model_dir: &Path,
        operation: impl FnOnce(&Self) -> Result<T>,
    ) -> Result<T> {
        let access = Self::acquire(model_dir)?;
        operation(&access)
    }

    pub(crate) fn acquire(model_dir: &Path) -> Result<Self> {
        file_system::create_dir_all(model_dir).map_err(|err| cache_io_error(model_dir, err))?;
        let path = model_dir.join(".lock");
        let lock = file_system::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|err| cache_io_error(&path, err))?;
        lock.lock().map_err(|err| cache_io_error(&path, err))?;
        let access = Self {
            _lock: lock,
            model_dir: model_dir.to_path_buf(),
        };
        let catalog_pending = model_dir.join(".catalog-publication");
        if catalog_pending
            .try_exists()
            .map_err(|err| cache_io_error(&catalog_pending, err))?
        {
            return Err(CgError::Engine(format!(
                "nomic cache has an unfinished catalog publication at {}",
                catalog_pending.display()
            )));
        }
        let pending = model_dir.join(".nomic-rollback/READY");
        if pending
            .try_exists()
            .map_err(|err| cache_io_error(&pending, err))?
        {
            // Both publishers refuse abandoned transactions. Never replay an old journal
            // over bytes that might have been installed by a different or older writer.
            return Err(CgError::Engine(format!(
                "nomic cache has an unfinished publication; prior artifacts retained at {}",
                model_dir.join(".nomic-rollback").display()
            )));
        }
        access.recover()?;
        Ok(access)
    }

    // READY means an uncommitted replacement. Backups are never consumed by restoration.
    // Only the active publisher restores them, while still holding its original lock.
    fn recover(&self) -> Result<()> {
        let backup = self.model_dir.join(".nomic-rollback");
        if !backup
            .try_exists()
            .map_err(|err| cache_io_error(&backup, err))?
        {
            return Ok(());
        }
        if backup
            .join("READY")
            .try_exists()
            .map_err(|err| cache_io_error(&backup, err))?
        {
            for name in NOMIC_ARTIFACT_FILES.into_iter().chain(["catalog.toml"]) {
                let old = backup.join(name);
                let destination = self.model_dir.join(name);
                if old.try_exists().map_err(|err| cache_io_error(&old, err))? {
                    let temporary = NomicTemporaryArtifact(
                        self.model_dir
                            .join(format!(".artifact-{}", uuid::Uuid::new_v4())),
                    );
                    backup_artifact(&old, &temporary.0)?;
                    publish_artifact(&temporary.0, &destination)?;
                } else {
                    match file_system::remove_file(&destination) {
                        Ok(()) => {}
                        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                        Err(err) => return Err(cache_io_error(&destination, err)),
                    }
                }
            }
            file_system::remove_file(backup.join("READY"))
                .map_err(|err| cache_io_error(&backup, err))?;
        }
        file_system::remove_dir_all(&backup).map_err(|err| cache_io_error(&backup, err))
    }

    // Update only installed identity. Consumer fields, including acceptance history and
    // unknown extension tables, survive. Preparation fails before any installed byte changes.
    fn prepare_catalog_metadata(&self, staging: &Path) -> Result<Option<PathBuf>> {
        let manifest = self.model_dir.join("catalog.toml");
        let text = match file_system::read_to_string(&manifest) {
            Ok(text) => text,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(cache_io_error(&manifest, error)),
        };
        let invalid = |error| {
            CgError::Engine(format!(
                "nomic catalog metadata failed at {}: {error}",
                manifest.display()
            ))
        };
        let mut table: toml::Table =
            toml::from_str(&text).map_err(|error| invalid(error.to_string()))?;
        let revision_path = staging.join("REVISION");
        let revision = file_system::read_to_string(&revision_path)
            .map_err(|error| cache_io_error(&revision_path, error))?;
        table.insert(
            "source_revision".to_string(),
            toml::Value::String(revision.trim().to_string()),
        );
        let mut shas = toml::Table::new();
        for name in NOMIC_ARTIFACT_FILES {
            let path = staging.join(name);
            let mut input =
                file_system::File::open(&path).map_err(|error| cache_io_error(&path, error))?;
            let mut hash = Sha256::new();
            io::copy(&mut input, &mut hash).map_err(|error| cache_io_error(&path, error))?;
            shas.insert(
                name.to_string(),
                toml::Value::String(format!("{:x}", hash.finalize())),
            );
        }
        table.insert("per_file_sha256".to_string(), toml::Value::Table(shas));
        let text = toml::to_string(&table).map_err(|error| invalid(error.to_string()))?;
        let path = staging.join("catalog.toml");
        file_system::write(&path, text).map_err(|error| cache_io_error(&path, error))?;
        Ok(Some(path))
    }

    pub(crate) fn publish(
        &self,
        staging: &Path,
        validate: impl FnOnce(&Path) -> Result<()>,
        mut before_replace: impl FnMut(usize) -> Result<()>,
    ) -> Result<()> {
        validate(staging)?;
        let metadata = self.prepare_catalog_metadata(staging)?;
        let backup = self.model_dir.join(".nomic-rollback");
        file_system::create_dir(&backup).map_err(|err| cache_io_error(&backup, err))?;
        let _rollback = NomicPublicationRollback(self);
        let replacing = (|| {
            for name in NOMIC_ARTIFACT_FILES.into_iter().chain(["catalog.toml"]) {
                let old = self.model_dir.join(name);
                if old.try_exists().map_err(|err| cache_io_error(&old, err))? {
                    backup_artifact(&old, &backup.join(name))?;
                }
            }
            file_system::write(backup.join("READY"), [])
                .map_err(|err| cache_io_error(&backup, err))?;
            for (index, name) in NOMIC_ARTIFACT_FILES.into_iter().enumerate() {
                before_replace(index)?;
                publish_artifact(&staging.join(name), &self.model_dir.join(name))?;
            }
            if let Some(metadata) = &metadata {
                before_replace(NOMIC_ARTIFACT_FILES.len())?;
                publish_artifact(metadata, &self.model_dir.join("catalog.toml"))?;
                before_replace(NOMIC_ARTIFACT_FILES.len() + 1)?;
            }
            // Removing READY is the commit point. Until then every error rolls back.
            file_system::remove_file(backup.join("READY"))
                .map_err(|err| cache_io_error(&backup, err))?;
            Ok(())
        })();
        if let Err(error) = replacing {
            self.recover()?;
            return Err(error);
        }
        // Cleanup after commit cannot turn a successful publication into a reported failure.
        // If cleanup fails, the next acquisition removes the marker-free backup directory.
        let _ = file_system::remove_dir_all(&backup);
        Ok(())
    }
}

// Links avoid copying the weights on ordinary local filesystems. The streaming fallback
// keeps replacement usable on filesystems that support renaming but not hard links.
fn backup_artifact(source: &Path, destination: &Path) -> Result<()> {
    if file_system::hard_link(source, destination).is_err() {
        let mut input =
            file_system::File::open(source).map_err(|err| cache_io_error(source, err))?;
        let mut output = file_system::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)
            .map_err(|err| cache_io_error(destination, err))?;
        io::copy(&mut input, &mut output).map_err(|err| cache_io_error(destination, err))?;
    }
    Ok(())
}

pub(crate) fn publish_artifact(source: &Path, destination: &Path) -> Result<()> {
    // A same-filesystem rename replaces the directory entry atomically. Never unlink the
    // destination: readers must keep seeing a complete artifact throughout publication.
    file_system::rename(source, destination).map_err(|err| {
        CgError::Engine(format!(
            "nomic artifact move failed {} -> {}: {err}",
            source.display(),
            destination.display()
        ))
    })
}
