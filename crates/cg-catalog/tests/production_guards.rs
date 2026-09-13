use cg_catalog::{CgCatalogError, validate_artifact_identity};

use std::path::{Path, PathBuf};

const GEMMA_ID: &str = "embeddinggemma-300m";

const PRODUCTION_GEMMA_MODEL_SAFETENSORS_SHA: &str =
    "cbf5a78393b6a033e0b8a63a57549964f7ed5c6fbeb4ba0694214f36123f2fd2";
const PRODUCTION_GEMMA_TOKENIZER_JSON_SHA: &str =
    "6852f8d561078cc0cebe70ca03c5bfdd0d60a45f9d2e0e1e4cc05b68e9ec329e";
const PRODUCTION_GEMMA_CONFIG_JSON_SHA: &str =
    "8f863f76e2d9c710cc833dc92efa898c9adfd41031c786507cc6b0e49c2e3e68";
const PRODUCTION_GEMMA_REVISION_SHA: &str =
    "240bf6458e37b64ab66077a6e6900fdb4089c848f8372fa64b8dc490d4ef81e3";
const FIXTURE_GEMMA_MODEL_SAFETENSORS_SHA: &str =
    "0739a2fb1413bc2268f47f2109bcc473c46e43d9de003a6492f47ba4b726bc63";
const FIXTURE_GEMMA_TOKENIZER_JSON_SHA: &str =
    "2be98cd907beb6e99a60a80015937a46f5c678b572f59ff75ad55bcf3f7e73ea";
const FIXTURE_GEMMA_CONFIG_JSON_SHA: &str =
    "178e387336a35b5e234826c0e0b0e0b3f2029b3dbe9902c2e25576b13c7874de";
const FIXTURE_GEMMA_REVISION_SHA: &str =
    "fa314f5bc6abbf4ecd0b7871494525bb78e10964bc83c919b9a1615485307312";

fn write_gemma_fixture_cache(root: &Path) -> PathBuf {
    let model_dir = root.join(GEMMA_ID);
    std::fs::create_dir_all(&model_dir).expect("model dir created");
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embeddinggemma");
    for file in [
        "model.safetensors",
        "tokenizer.json",
        "config.json",
        "REVISION",
    ] {
        std::fs::copy(fixture_dir.join(file), model_dir.join(file))
            .expect("fixture copied into cache");
    }
    std::fs::write(
        model_dir.join("catalog.toml"),
        "model_id = \"embeddinggemma-300m\"\ncache_status = \"cached\"\n",
    )
    .expect("catalog stub written");
    model_dir
}

#[test]
fn normal_test_binary_rejects_embeddinggemma_fixture_artifacts_as_production_manual_population() {
    // Tiny fixture bytes require an explicit source override and must not certify manual population.
    let temp = tempfile::tempdir().expect("tempdir");
    let model_dir = write_gemma_fixture_cache(temp.path());
    let result = validate_artifact_identity(GEMMA_ID, &model_dir);

    assert!(
        matches!(
            result,
            Err(CgCatalogError::ArtifactValidationFailed {
                ref file,
                ref expected_sha,
                ref actual_sha,
                ref mismatches,
            }) if file == "model.safetensors"
                && expected_sha == PRODUCTION_GEMMA_MODEL_SAFETENSORS_SHA
                && actual_sha == FIXTURE_GEMMA_MODEL_SAFETENSORS_SHA
                && mismatches.iter().any(|mismatch| mismatch.file == "model.safetensors"
                    && mismatch.expected_sha == PRODUCTION_GEMMA_MODEL_SAFETENSORS_SHA
                    && mismatch.actual_sha == FIXTURE_GEMMA_MODEL_SAFETENSORS_SHA)
                && mismatches.iter().any(|mismatch| mismatch.file == "tokenizer.json"
                    && mismatch.expected_sha == PRODUCTION_GEMMA_TOKENIZER_JSON_SHA
                    && mismatch.actual_sha == FIXTURE_GEMMA_TOKENIZER_JSON_SHA)
                && mismatches.iter().any(|mismatch| mismatch.file == "config.json"
                    && mismatch.expected_sha == PRODUCTION_GEMMA_CONFIG_JSON_SHA
                    && mismatch.actual_sha == FIXTURE_GEMMA_CONFIG_JSON_SHA)
                && mismatches.iter().any(|mismatch| mismatch.file == "REVISION"
                    && mismatch.expected_sha == PRODUCTION_GEMMA_REVISION_SHA
                    && mismatch.actual_sha == FIXTURE_GEMMA_REVISION_SHA)
        ),
        "normal binaries must reject every committed fixture artifact as non-production Gemma bytes: {result:?}"
    );
}

/// Where an operator's registrations live is decided once, by the command the operator ran, and
/// handed down. A second rule inside this crate is how `cg model show` comes back empty while the
/// entries the operator wrote sit in another directory — this crate has shipped that defect once
/// already for the model-cache root. So the crate's own sources must not resolve the data root at
/// all: no `CG_DATA_ROOT` read, no `$HOME/.cg` fallback, no `default_data_root`. The root arrives
/// through `Catalog::open_with_scope_against_root`/`_roots` and is used as given.
#[test]
fn the_catalog_never_resolves_the_data_root_for_itself() {
    let sources = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offenders = Vec::new();
    let mut files = Vec::new();
    collect_rust_sources(&sources, &mut files);
    assert!(
        !files.is_empty(),
        "the crate's src/ directory must contain Rust sources to scan"
    );
    for file in &files {
        let content = std::fs::read_to_string(file)
            .unwrap_or_else(|error| panic!("reading {}: {error}", file.display()));
        for (index, line) in content.lines().enumerate() {
            let code = line.split_once("//").map_or(line, |(before, _)| before);
            for needle in ["CG_DATA_ROOT", "default_data_root", "\".cg\""] {
                if code.contains(needle) {
                    offenders.push(format!(
                        "{}:{}: {} — {}",
                        file.display(),
                        index + 1,
                        needle,
                        code.trim()
                    ));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "this crate must take the data root from its caller and resolve none of its own; found:\n{}",
        offenders.join("\n")
    );
}

fn collect_rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}
