//! Verifier weight fetch: plain HTTPS downloads from Hugging Face at a PINNED revision
//! (registered in the run's working doc before any verdict scoring), cached under
//! `~/.cache/cg-probes/<repo>/` with a REVISION marker. Measurement-only machinery — no cg data
//! root is touched; the cache layout mirrors the catalog convention (files on disk, transport
//! is the operator's business).

use std::io;
use std::path::{Path, PathBuf};

const BYTE_LIMIT: u64 = 8 * 1024 * 1024 * 1024;

pub fn probe_cache_root() -> Result<PathBuf, String> {
    if let Some(root) = std::env::var_os("CG_PROBES_CACHE") {
        return Ok(PathBuf::from(root));
    }
    let home = std::env::var_os("HOME").ok_or("HOME is not set and CG_PROBES_CACHE is not set")?;
    Ok(PathBuf::from(home).join(".cache").join("cg-probes"))
}

/// Ensure `files` of `repo` at `revision` are present in the cache; download what is missing.
/// Returns the model directory.
pub fn fetch_model(repo: &str, revision: &str, files: &[&str]) -> Result<PathBuf, String> {
    let dir = probe_cache_root()?.join(repo.replace('/', "__"));
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {dir:?}: {e}"))?;
    for file in files {
        let destination = dir.join(file);
        if destination.exists() {
            continue;
        }
        eprintln!("[fetch] {repo}@{revision} {file} …");
        download(repo, revision, file, &destination)?;
    }
    std::fs::write(dir.join("REVISION"), revision).map_err(|e| format!("write REVISION: {e}"))?;
    Ok(dir)
}

fn download(repo: &str, revision: &str, file: &str, destination: &Path) -> Result<(), String> {
    let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{file}");
    let tmp = destination.with_extension("download");
    let mut response = ureq::get(&url)
        .call()
        .map_err(|e| format!("GET {url}: {e}"))?;
    let mut reader = response.body_mut().with_config().limit(BYTE_LIMIT).reader();
    let mut out = std::fs::File::create(&tmp).map_err(|e| format!("create {tmp:?}: {e}"))?;
    io::copy(&mut reader, &mut out).map_err(|e| format!("download {url}: {e}"))?;
    std::fs::rename(&tmp, destination).map_err(|e| format!("move into place: {e}"))?;
    Ok(())
}
