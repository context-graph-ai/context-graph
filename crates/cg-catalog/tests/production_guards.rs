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
