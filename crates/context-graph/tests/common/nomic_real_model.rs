// Included into several independent test binaries via `#[path = "common/nomic_real_model.rs"]`;
// any one binary typically uses only a subset of these helpers, so an unused item here is not a
// dead-code defect in this shared module.
#![allow(dead_code)]

//! Shared support for tests that exercise the REAL cached Nomic model, never a mock: resolves the
//! exact directory production locates the installed `nomic-embed-text-v1.5` weights under, and
//! builds an embedder from a LOCAL source, offline, with downloads denied — so a real-model test
//! reads only the artifact the user already installed and no code path here can reach the
//! network.
//!
//! Resolves the exact directory production locates the installed `nomic-embed-text-v1.5` weights
//! under by calling `cg_catalog::default_cache_root()` — the one function that owns the
//! `CG_MODEL_CACHE_ROOT` env var / platform-default resolution — rather than re-deriving that same
//! rule a second time here. Tests use the same cache root as the production loader.

use std::path::PathBuf;
use std::{fs::File, io::Read};

use sha2::{Digest, Sha256};

use context_graph::embedder::nomic::NomicTextEmbedder;
use context_graph::{Embedder, EmbedderConfig, ModelSource};

pub const NOMIC_MODEL_ID: &str = "nomic-embed-text-v1.5";
pub const NOMIC_CANONICAL_REVISION: &str = "e9b6763023c676ca8431644204f50c2b100d9aab";

/// The canonical cache root production resolves cached models under: `cg_catalog`'s own canonical
/// function, honouring the same `CG_MODEL_CACHE_ROOT` env var the catalog reads.
pub fn canonical_cache_root() -> PathBuf {
    cg_catalog::default_cache_root()
}

/// The canonical, already-installed model directory: `<canonical_cache_root>/nomic-embed-text-v1.5`.
pub fn canonical_model_dir() -> PathBuf {
    canonical_cache_root().join(NOMIC_MODEL_ID)
}

/// Bind a real-model golden to the same declared artifact set and digest shape as the
/// event-history benchmark. Catalog bookkeeping is excluded from the model identity.
pub fn model_artifacts_sha256() -> String {
    fn declared_files(prefix: &str, table: &toml::Table, files: &mut Vec<String>) {
        for (name, value) in table {
            let name = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}.{name}")
            };
            if value.is_str() {
                files.push(name);
            } else if let Some(child) = value.as_table() {
                declared_files(&name, child, files);
            }
        }
    }

    let dir = canonical_model_dir();
    let catalog = dir.join("catalog.toml");
    let mut names = if catalog.is_file() {
        let contents = std::fs::read_to_string(&catalog).expect("read model catalog manifest");
        let manifest: toml::Value =
            toml::from_str(&contents).expect("parse model catalog manifest");
        let table = manifest
            .get("per_file_sha256")
            .and_then(toml::Value::as_table)
            .expect("model catalog manifest declares per_file_sha256");
        let mut files = Vec::new();
        declared_files("", table, &mut files);
        files
    } else {
        [
            "REVISION",
            "config.json",
            "model.safetensors",
            "tokenizer.json",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    };
    names.sort();
    names.dedup();
    assert!(
        !names.is_empty(),
        "model catalog manifest declares no artifacts"
    );
    let mut binding = Sha256::new();
    for name in names {
        let mut file = File::open(dir.join(&name))
            .unwrap_or_else(|err| panic!("model catalog artifact {name} is missing: {err}"));
        let mut artifact = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let count = file.read(&mut buffer).expect("read model artifact");
            if count == 0 {
                break;
            }
            artifact.update(&buffer[..count]);
        }
        binding.update(name.as_bytes());
        binding.update(format!("{:x}", artifact.finalize()).as_bytes());
    }
    format!("{:x}", binding.finalize())
}

/// A local-source, offline, download-denying config over an arbitrary model directory, pinned to
/// the exact revision the canonical event-history encoding advertises. No branch reachable from
/// this config ever constructs an HTTP client: `ModelSource::LocalPath` is refused locally, with
/// a typed error, the moment its files are missing or invalid, before any Hugging-Face-only code
/// path is even considered.
pub fn local_offline_config(model_dir: PathBuf) -> EmbedderConfig {
    EmbedderConfig {
        source: ModelSource::LocalPath(model_dir),
        offline: true,
        cache_dir: None,
        pinned_revision: Some(NOMIC_CANONICAL_REVISION.to_string()),
        network_policy: "deny-download".to_string(),
    }
}

/// The real, installed Nomic embedder: local source over the canonical cache, offline, downloads
/// denied. Returned as `Box<dyn Embedder>` — the way every real caller in this crate holds an
/// embedder — so the checked, library-owned event-search entrypoints (`prepare_document`,
/// `prepare_query`, `embed_prepared_document`, `embed_prepared_query`) are reachable directly
/// through `Box`'s deref.
pub fn nomic_live_embedder() -> Box<dyn Embedder> {
    Box::new(NomicTextEmbedder::new(local_offline_config(
        canonical_model_dir(),
    )))
}

/// The tokenizer bundled with the canonical, already-installed model cache — loaded directly, for
/// tests that need to independently re-derive tokenizer output to compare against the embedder's
/// own.
pub fn nomic_tokenizer() -> tokenizers::Tokenizer {
    let tokenizer_path = canonical_model_dir().join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&tokenizer_path)
        .expect("cached nomic tokenizer.json must load offline from the canonical model cache")
}
