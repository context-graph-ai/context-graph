use cg_catalog::{
    CacheLayout, CacheStatus, CatalogEntry, CgCatalogError, ManifestStub, ModelKind, Source,
};
use std::path::{Path, PathBuf};

use crate::common::{
    API_ID, CANONICAL_LICENSE_GATE_MESSAGE, CUSTOM_VISION_ID, FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA,
    FOOTNOTE_EMBEDDINGGEMMA, GEMMA_BYTES, GEMMA_ID, GEMMA_REVISION, NOMIC_BYTES, NOMIC_ID,
    catalog_at, dir_regular_file_size_sum, empty_catalog_at, fixture_sha, nomic_entry,
    write_gemma_cache, write_nomic_cache, write_source_override,
};

fn write_custom_vision_cache(root: &Path) -> (PathBuf, u64) {
    let model_dir = root.join(CUSTOM_VISION_ID);
    std::fs::create_dir_all(&model_dir).expect("custom model dir");
    let artifact = b"vigil face re-identification fixture weights\n";
    std::fs::write(model_dir.join("weights.bin"), artifact).expect("custom artifact");
    let artifact_sha = fixture_sha(&model_dir.join("weights.bin"));
    std::fs::write(
        model_dir.join("catalog.toml"),
        format!(
            "model_id = \"{CUSTOM_VISION_ID}\"\nkind = \"VisionEmbedder\"\ndimension = 512\ninput_modality = \"image\"\nlicense_spdx = \"Apache-2.0\"\nlicense_url = \"https://www.apache.org/licenses/LICENSE-2.0\"\nsource_repo = \"vigil/face-reid-v1\"\nsource_revision = \"custom-revision-before-override\"\nbyte_size = {}\ncache_status = \"cached\"\nartifact_filenames = [\"weights.bin\", \"catalog.toml\"]\n[per_file_sha256]\nweights.bin = \"{artifact_sha}\"\n",
            artifact.len()
        ),
    )
    .expect("custom manifest");
    (model_dir, artifact.len() as u64)
}

#[test]
fn catalog_list_names_bundled_default_and_alternative_and_cached_entries_with_kind_and_license_and_size_and_source()
 {
    // A developer needs the catalog list to explain all available text models before any download.
    let temp = tempfile::tempdir().expect("tempdir");
    let catalog = catalog_at(temp.path(), "host-A");
    let entries = catalog.list().expect("list returns");
    assert_eq!(entries.len(), 2);
    let nomic = entries
        .iter()
        .find(|entry| entry.id == NOMIC_ID)
        .expect("nomic list row present");
    assert_eq!(nomic.kind, ModelKind::TextEmbedder);
    assert_eq!(nomic.capability.dimension, Some(768));
    assert_eq!(nomic.capability.input_modality.as_deref(), Some("text"));
    assert_eq!(nomic.license.spdx, "Apache-2.0");
    assert_eq!(
        nomic.license.url,
        "https://www.apache.org/licenses/LICENSE-2.0"
    );
    assert!(!nomic.license.requires_manual_acceptance);
    assert_eq!(nomic.byte_size, NOMIC_BYTES);
    assert_eq!(nomic.cache_status, CacheStatus::NotCached);
    assert!(matches!(
        &nomic.source,
        Source::HuggingFace { repo, revision }
            if repo == "nomic-ai/nomic-embed-text-v1.5"
                && revision == crate::common::NOMIC_REVISION
    ));

    let gemma = entries
        .iter()
        .find(|entry| entry.id == GEMMA_ID)
        .expect("embeddinggemma list row present");
    assert_eq!(gemma.kind, ModelKind::TextEmbedder);
    assert_eq!(gemma.capability.dimension, Some(768));
    assert_eq!(gemma.license.spdx, "Gemma");
    assert_eq!(gemma.license.url, "https://ai.google.dev/gemma/terms");
    assert!(gemma.license.requires_manual_acceptance);
    assert_eq!(gemma.cache_status, CacheStatus::NotCached);
    assert_eq!(gemma.byte_size, GEMMA_BYTES);
    assert!(matches!(
        &gemma.source,
        Source::HuggingFace { repo, revision }
            if repo == "google/embeddinggemma-300m" && revision == GEMMA_REVISION
    ));
}

#[test]
fn catalog_list_orders_deterministic_alphabetical_by_id() {
    // Stable ordering keeps CLI output and sync receiver logs diffable across repeated runs.
    let temp = tempfile::tempdir().expect("tempdir");
    let mut entries = vec![
        CatalogEntry {
            id: "zeta-text-v1".to_string(),
            ..nomic_entry(temp.path())
        },
        nomic_entry(temp.path()),
        CatalogEntry {
            id: "alpha-text-v1".to_string(),
            ..nomic_entry(temp.path())
        },
        CatalogEntry {
            id: GEMMA_ID.to_string(),
            ..nomic_entry(temp.path())
        },
        CatalogEntry {
            id: API_ID.to_string(),
            source: Source::ApiEndpoint {
                base_url: "https://api.customer.example".to_string(),
                provider: "customer".to_string(),
                model: "text-embedding-v1".to_string(),
                auth_env_var: "CUSTOMER_API_KEY".to_string(),
            },
            byte_size: 0,
            local_path: None,
            artifact_filenames: Vec::new(),
            per_file_sha256: Vec::new(),
            ..nomic_entry(temp.path())
        },
    ];
    entries.rotate_left(2);
    let catalog =
        cg_catalog::Catalog::open_with_entries_against_root(temp.path(), "host-A", entries)
            .expect("catalog opens with custom registrations");
    let entries = catalog.list().expect("list returns");
    let ids = entries
        .iter()
        .map(|entry| entry.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        vec![NOMIC_ID, "alpha-text-v1", API_ID, GEMMA_ID, "zeta-text-v1"]
    );
    let api = entries
        .iter()
        .find(|entry| entry.id == API_ID)
        .expect("explicit API endpoint row listed");
    assert!(matches!(
        &api.source,
        Source::ApiEndpoint {
            base_url,
            provider,
            model,
            auth_env_var,
        } if base_url == "https://api.customer.example"
            && provider == "customer"
            && model == "text-embedding-v1"
            && auth_env_var == "CUSTOMER_API_KEY"
    ));
    assert_eq!(api.byte_size, 0);
    assert_eq!(api.local_path, None);
    assert!(api.artifact_filenames.is_empty());
    assert!(api.per_file_sha256.is_empty());
    let rendered_api = format!("{api:?}");
    assert!(rendered_api.contains("CUSTOMER_API_KEY"));
    assert!(!rendered_api.contains("sk-"));
    assert!(!rendered_api.to_ascii_lowercase().contains("secret"));
    assert_eq!(entries, catalog.list().expect("second list returns"));
}

#[test]
fn catalog_list_with_empty_bundled_and_empty_cache_returns_empty_vec() {
    // An empty catalog is a valid starting point for an embeddable downstream tool.
    let temp = tempfile::tempdir().expect("tempdir");
    let entries = empty_catalog_at(temp.path(), "host-A")
        .list()
        .expect("list returns");
    assert_eq!(entries, Vec::new());
}

#[test]
fn catalog_show_returns_absolute_cache_path_artifact_filenames_per_file_sha_license_url_reflects_external_disk_mutation()
 {
    // A developer inspecting a cached default model should see the exact files and license on disk.
    let temp = tempfile::tempdir().expect("tempdir");
    let model_dir = write_nomic_cache(temp.path());
    let expected = nomic_entry(temp.path());
    let layout = CacheLayout {
        root: temp.path().to_path_buf(),
        per_model_path: model_dir.clone(),
    };
    assert_eq!(layout.per_model_path, temp.path().join(NOMIC_ID));
    let expected_manifest_shape = ManifestStub {
        artifact_filenames: expected.artifact_filenames.clone(),
        per_file_sha256: expected.per_file_sha256.clone(),
        total_byte_budget: expected.byte_size,
        ..Default::default()
    };
    assert_eq!(
        expected_manifest_shape
            .per_file_sha256
            .iter()
            .find(|(file, _)| file == "model.safetensors")
            .map(|(_, sha)| sha.as_str()),
        Some(FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA)
    );
    let shown = catalog_at(temp.path(), "host-A")
        .show(NOMIC_ID)
        .expect("show returns");
    assert_eq!(shown, expected);
    assert!(shown.local_path.as_ref().expect("local path").is_absolute());
    assert_eq!(
        fixture_sha(&model_dir.join("model.safetensors")),
        FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA
    );
    assert_eq!(
        shown
            .per_file_sha256
            .iter()
            .find(|(file, _)| file == "model.safetensors")
            .map(|(_, sha)| sha.as_str()),
        Some(FIXTURE_NOMIC_MODEL_SAFETENSORS_SHA)
    );
    std::fs::write(
        temp.path().join(NOMIC_ID).join("catalog.toml"),
        "license_url = \"https://sentinel.invalid/marker\"",
    )
    .expect("mutated catalog stub written");
    let shown_after_mutation = catalog_at(temp.path(), "host-A")
        .show(NOMIC_ID)
        .expect("show returns");
    assert_eq!(
        shown_after_mutation.license.url,
        "https://sentinel.invalid/marker"
    );
}

#[test]
fn catalog_show_for_license_gated_model_includes_verbatim_canonical_install_instructions_byte_equal_to_pinned_const_reflects_external_disk_mutation()
 {
    // License-gated entries must tell a developer exactly how to install bytes manually.
    let temp = tempfile::tempdir().expect("tempdir");
    write_gemma_cache(temp.path());
    let catalog = catalog_at(temp.path(), "host-A");
    let shown = catalog.show(GEMMA_ID).expect("show returns");
    assert_eq!(
        shown.install_instructions.as_deref(),
        Some(CANONICAL_LICENSE_GATE_MESSAGE)
    );
    write_nomic_cache(temp.path());
    let apache = catalog.show(NOMIC_ID).expect("show returns");
    assert_eq!(apache.install_instructions, None);
    std::fs::write(
        temp.path().join(GEMMA_ID).join("catalog.toml"),
        "manual_install_url = \"https://sentinel.invalid/gemma-sentinel\"",
    )
    .expect("mutated gated catalog stub written");
    let mutated = catalog.show(GEMMA_ID).expect("show returns after mutation");
    assert_eq!(
        mutated.manual_install_url.as_deref(),
        Some("https://sentinel.invalid/gemma-sentinel")
    );
}

#[test]
fn catalog_show_for_embeddinggemma_includes_oss_purity_audit_footnote_referencing_ci_workflow_absent_for_apache_entries_reflects_disk()
 {
    // The gated sibling model needs an OSS-purity disclosure without putting policy state into the graph.
    let temp = tempfile::tempdir().expect("tempdir");
    write_gemma_cache(temp.path());
    write_nomic_cache(temp.path());
    let catalog = catalog_at(temp.path(), "host-A");
    let shown = catalog.show(GEMMA_ID).expect("show returns");
    assert_eq!(
        shown.audit_footnote.as_deref(),
        Some(FOOTNOTE_EMBEDDINGGEMMA)
    );
    let apache = catalog.show(NOMIC_ID).expect("show returns");
    assert_eq!(apache.audit_footnote, None);
    std::fs::write(
        temp.path().join(GEMMA_ID).join("catalog.toml"),
        "audit_footnote = \"sentinel-footnote-marker\"",
    )
    .expect("mutated audit footnote written");
    let mutated = catalog.show(GEMMA_ID).expect("show returns after mutation");
    assert_eq!(
        mutated.audit_footnote.as_deref(),
        Some("sentinel-footnote-marker")
    );
}

#[test]
fn catalog_show_not_found_returns_typed_error() {
    // Unknown ids should be machine-matchable errors, not fabricated records.
    let temp = tempfile::tempdir().expect("tempdir");
    let result = catalog_at(temp.path(), "host-A").show("ghost");
    assert!(matches!(result, Err(CgCatalogError::NotFound)));
}

#[test]
fn catalog_list_cache_status_reflects_real_disk_after_external_population() {
    // Air-gapped pre-population has to be visible to the next list call.
    let temp = tempfile::tempdir().expect("tempdir");
    let catalog = catalog_at(temp.path(), "host-A");
    let before = catalog.list().expect("list returns");
    assert!(
        before
            .iter()
            .any(|entry| entry.id == GEMMA_ID && entry.cache_status == CacheStatus::NotCached)
    );
    let gemma_dir = write_gemma_cache(temp.path());
    let gemma_disk_bytes = dir_regular_file_size_sum(&gemma_dir);
    let after = catalog.list().expect("list returns");
    assert!(after.iter().any(|entry| entry.id == GEMMA_ID
        && entry.cache_status == CacheStatus::Cached
        && entry.byte_size == gemma_disk_bytes));

    let (custom_dir, custom_artifact_bytes) = write_custom_vision_cache(temp.path());
    write_source_override(
        temp.path(),
        CUSTOM_VISION_ID,
        "https://models.example.invalid/vigil",
        "custom-revision-after-override",
    );
    let after_custom = catalog
        .list()
        .expect("list with custom cached model returns");
    let ids = after_custom
        .iter()
        .map(|entry| entry.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        vec![NOMIC_ID, GEMMA_ID, CUSTOM_VISION_ID],
        "list ordering must remain deterministic after arbitrary cached entries appear"
    );
    let custom = after_custom
        .iter()
        .find(|entry| entry.id == CUSTOM_VISION_ID)
        .expect("custom cached model surfaced from disk");
    assert_eq!(custom.kind, ModelKind::VisionEmbedder);
    assert_eq!(custom.capability.dimension, Some(512));
    assert_eq!(custom.capability.input_modality.as_deref(), Some("image"));
    assert_eq!(custom.license.spdx, "Apache-2.0");
    assert_eq!(custom.cache_status, CacheStatus::Cached);
    assert_eq!(
        custom.byte_size, custom_artifact_bytes,
        "arbitrary cached entries should report byte size from their own manifest/artifact, not a bundled constant"
    );
    assert_eq!(custom.local_path.as_deref(), Some(custom_dir.as_path()));
    assert_eq!(
        custom
            .per_file_sha256
            .iter()
            .find(|(file, _)| file == "weights.bin")
            .map(|(_, sha)| sha.as_str()),
        Some(fixture_sha(&custom_dir.join("weights.bin")).as_str())
    );
    assert!(matches!(
        &custom.source,
        Source::HuggingFace { repo, revision }
            if repo == "vigil/face-reid-v1" && revision == "custom-revision-after-override"
    ));
    let shown = catalog.show(CUSTOM_VISION_ID).expect("custom show returns");
    assert_eq!(shown.id, CUSTOM_VISION_ID);
    assert_eq!(shown.kind, ModelKind::VisionEmbedder);
    assert_eq!(shown.source, custom.source);
    assert_eq!(shown.artifact_filenames, custom.artifact_filenames);
}
