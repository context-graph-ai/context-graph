use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::ClaudeConnectorError;

const PROJECT_IDENTITY_PROBE_BUDGET: u64 = 4 * 1024 * 1024;

pub fn effective_history_root(home: &Path, claude_config_dir: Option<&Path>) -> PathBuf {
    #[cfg(context_graph_internal_test_support)]
    crate::internal_test_support::record_history_path_resolution();

    match claude_config_dir.filter(|directory| !directory.as_os_str().is_empty()) {
        Some(directory) => directory.join("projects"),
        None => home.join(".claude").join("projects"),
    }
}

pub struct ClaudeHistoryLocator {
    history_root: PathBuf,
}

impl ClaudeHistoryLocator {
    pub fn new(history_root: PathBuf) -> Self {
        Self { history_root }
    }

    pub fn resolve_project(
        &self,
        project: &Path,
    ) -> Result<ResolvedClaudeHistory, ClaudeConnectorError> {
        #[cfg(context_graph_internal_test_support)]
        crate::internal_test_support::record_history_path_resolution();

        let canonical_project = canonicalize("canonicalize requested project", project)?;
        let project_key = claude_project_key(&canonical_project)?;
        let candidate = self.history_root.join(project_key);
        let canonical_candidate = match fs::canonicalize(&candidate) {
            Ok(path) => path,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(ResolvedClaudeHistory {
                    canonical_project,
                    sources: Vec::new(),
                    helper_transcripts_not_imported: 0,
                });
            }
            Err(error) => return Err(io_error("canonicalize project history", candidate, error)),
        };

        let mut source_paths = immediate_jsonl_files(&canonical_candidate)?;
        source_paths.sort_by(|left, right| left.file_name().cmp(&right.file_name()));

        let mut sources = Vec::new();
        let mut helper_transcripts_not_imported = 0;
        for path in source_paths {
            if !header_proves_project(&path, &canonical_project)? {
                continue;
            }

            let canonical_source = canonicalize("canonicalize selected history source", &path)?;
            let file_stem = path
                .file_stem()
                .expect("JSONL paths have a file stem")
                .to_os_string();
            helper_transcripts_not_imported +=
                count_helper_transcripts(&canonical_candidate.join(&file_stem))?;
            sources.push(ResolvedClaudeSource {
                path: canonical_source.clone(),
                file_stem,
                source_locator_ref: sha256_os_path(&canonical_source),
            });
        }

        Ok(ResolvedClaudeHistory {
            canonical_project,
            sources,
            helper_transcripts_not_imported,
        })
    }
}

fn canonicalize(operation: &'static str, path: &Path) -> Result<PathBuf, ClaudeConnectorError> {
    fs::canonicalize(path).map_err(|error| io_error(operation, path.to_path_buf(), error))
}

fn io_error(operation: &'static str, path: PathBuf, error: std::io::Error) -> ClaudeConnectorError {
    ClaudeConnectorError::Io {
        operation,
        path,
        reason: error.to_string(),
    }
}

fn claude_project_key(project: &Path) -> Result<String, ClaudeConnectorError> {
    let project = project.to_str().ok_or_else(|| ClaudeConnectorError::Io {
        operation: "derive Claude project key",
        path: project.to_path_buf(),
        reason: "canonical project path is not valid Unicode".to_owned(),
    })?;
    let encoded: String = project
        .encode_utf16()
        .map(|unit| match unit {
            0x41..=0x5a | 0x61..=0x7a | 0x30..=0x39 => {
                char::from_u32(u32::from(unit)).expect("ASCII UTF-16 unit")
            }
            _ => '-',
        })
        .collect();
    if encoded.len() <= 200 {
        return Ok(encoded);
    }

    let hash = project.encode_utf16().fold(0_i32, |hash, unit| {
        hash.wrapping_shl(5)
            .wrapping_sub(hash)
            .wrapping_add(i32::from(unit))
    });
    Ok(format!(
        "{}-{}",
        &encoded[..200],
        base36(i64::from(hash).abs())
    ))
}

fn base36(mut value: i64) -> String {
    let mut digits = Vec::new();
    loop {
        let digit = u8::try_from(value % 36).expect("base-36 digit");
        digits.push(if digit < 10 {
            char::from(b'0' + digit)
        } else {
            char::from(b'a' + (digit - 10))
        });
        value /= 36;
        if value == 0 {
            break;
        }
    }
    digits.into_iter().rev().collect()
}

fn immediate_jsonl_files(candidate: &Path) -> Result<Vec<PathBuf>, ClaudeConnectorError> {
    let entries = fs::read_dir(candidate).map_err(|error| {
        io_error(
            "read project history directory",
            candidate.to_path_buf(),
            error,
        )
    })?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| {
            io_error(
                "read project history directory entry",
                candidate.to_path_buf(),
                error,
            )
        })?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| io_error("inspect history source", path.clone(), error))?;
        if metadata.file_type().is_file() && has_jsonl_extension(&path) {
            files.push(path);
        }
    }
    Ok(files)
}

fn has_jsonl_extension(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension == "jsonl")
}

fn header_proves_project(path: &Path, project: &Path) -> Result<bool, ClaudeConnectorError> {
    let file = File::open(path)
        .map_err(|error| io_error("open history source", path.to_path_buf(), error))?;
    let mut reader = BufReader::new(file.take(PROJECT_IDENTITY_PROBE_BUDGET));
    let mut record = Vec::new();
    loop {
        record.clear();
        let bytes_read = reader
            .read_until(b'\n', &mut record)
            .map_err(|error| io_error("read history source", path.to_path_buf(), error))?;
        if bytes_read == 0 {
            if probe_has_unread_bytes(&reader, path)? {
                return Err(project_identity_probe_exhausted(path));
            }
            return Ok(false);
        }
        if !record.ends_with(b"\n") {
            if probe_has_unread_bytes(&reader, path)? {
                return Err(project_identity_probe_exhausted(path));
            }
            return Ok(false);
        }
        if record.iter().all(u8::is_ascii_whitespace) {
            continue;
        }

        let Ok(value) = serde_json::from_slice::<serde_json::Value>(&record) else {
            return Ok(false);
        };
        let Some(cwd) = value
            .as_object()
            .and_then(|object| object.get("cwd"))
            .and_then(|cwd| cwd.as_str())
        else {
            continue;
        };
        let cwd = Path::new(cwd);
        return Ok(cwd == project || fs::canonicalize(cwd).is_ok_and(|cwd| cwd == project));
    }
}

fn probe_has_unread_bytes(
    reader: &BufReader<std::io::Take<File>>,
    path: &Path,
) -> Result<bool, ClaudeConnectorError> {
    if reader.get_ref().limit() != 0 {
        return Ok(false);
    }
    let source_length = reader
        .get_ref()
        .get_ref()
        .metadata()
        .map_err(|error| io_error("inspect history source length", path.to_path_buf(), error))?
        .len();
    Ok(source_length > PROJECT_IDENTITY_PROBE_BUDGET)
}

fn project_identity_probe_exhausted(path: &Path) -> ClaudeConnectorError {
    ClaudeConnectorError::Io {
        operation: "establish project identity from history source",
        path: path.to_path_buf(),
        reason: format!(
            "project identity could not be established within the \
             {PROJECT_IDENTITY_PROBE_BUDGET}-byte probe budget"
        ),
    }
}

fn count_helper_transcripts(root: &Path) -> Result<usize, ClaudeConnectorError> {
    let metadata = match fs::symlink_metadata(root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => {
            return Err(io_error(
                "inspect helper transcript directory",
                root.to_path_buf(),
                error,
            ));
        }
    };
    if !metadata.file_type().is_dir() {
        return Ok(0);
    }

    let mut pending = vec![root.to_path_buf()];
    let mut count = 0;
    while let Some(directory) = pending.pop() {
        let entries = fs::read_dir(&directory).map_err(|error| {
            io_error("read helper transcript directory", directory.clone(), error)
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                io_error(
                    "read helper transcript directory entry",
                    directory.clone(),
                    error,
                )
            })?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| io_error("inspect helper transcript", path.clone(), error))?;
            if metadata.file_type().is_dir() {
                pending.push(path);
            } else if metadata.file_type().is_file() && has_jsonl_extension(&path) {
                count += 1;
            }
        }
    }
    Ok(count)
}

fn sha256_os_path(path: &Path) -> String {
    let mut hash = Sha256::new();
    hash.update(path.as_os_str().as_encoded_bytes());
    format!("{:x}", hash.finalize())
}

pub struct ResolvedClaudeHistory {
    pub canonical_project: PathBuf,
    pub sources: Vec<ResolvedClaudeSource>,
    pub helper_transcripts_not_imported: usize,
}

pub struct ResolvedClaudeSource {
    pub path: PathBuf,
    pub file_stem: OsString,
    pub source_locator_ref: String,
}
