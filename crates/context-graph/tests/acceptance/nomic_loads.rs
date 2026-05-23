use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_embeddinggemma::EmbeddingGemmaTextEmbedder;
use context_graph::embedder::nomic::{NOMIC_MODEL_ID, NomicTextEmbedder};
use context_graph::embedding_space::{BuiltinSpaceRegistry, default_text_embedding_space};
use context_graph::{
    CG_EDGE_SYSTEM_TAG, CreateContext, CreateEntity, DistanceMetric, Embedder, EmbedderConfig,
    EmbedderRegistration, EmbeddingInput, EmbeddingSpaceDecl, EntityType, EvidenceKind,
    EvidenceProducer, ModelSource, Store, StoreConfig,
};

const NOMIC_TEXT_DEFAULT_ID: &str = "text_default_nomic_v1_5_768";
const EMBEDDINGGEMMA_TEXT_DEFAULT_ID: &str = "text_default_embeddinggemma_300m_768";
const FACE_REID_SPACE: &str = "vigil_face_reid_v1";

fn temp_store_config(name: &str) -> (tempfile::TempDir, StoreConfig) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir
        .path()
        .join(format!("{name}-{}.db", uuid::Uuid::now_v7()));
    (
        dir,
        StoreConfig {
            db_path,
            ..Default::default()
        },
    )
}

fn disabled_store_config(name: &str) -> (tempfile::TempDir, StoreConfig) {
    let (dir, mut config) = temp_store_config(name);
    config.default_text_embedder = Some(EmbedderConfig::disabled());
    (dir, config)
}

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn workspace_root() -> PathBuf {
    crate_root()
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn read_crate_file(path: &str) -> String {
    let path = crate_root().join(path);
    fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

fn read_workspace_file(path: &str) -> String {
    let path = workspace_root().join(path);
    fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

fn l2_norm(vector: &[f32]) -> f32 {
    vector
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt()
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let dot = left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>();
    dot / (l2_norm(left) * l2_norm(right))
}

fn embeddinggemma_embedder() -> EmbeddingGemmaTextEmbedder {
    EmbeddingGemmaTextEmbedder::new(EmbedderConfig::disabled())
}

fn face_space() -> EmbeddingSpaceDecl {
    EmbeddingSpaceDecl {
        embedding_space_id: FACE_REID_SPACE.to_string(),
        dimension: 512,
        metric: DistanceMetric::Cosine,
        model_family: "vision".to_string(),
        model_name: "face-reid".to_string(),
        model_version: "1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
        default_for_evidence_kind: None,
        bindings: Vec::new(),
    }
}

fn face_producer() -> EvidenceProducer {
    EvidenceProducer {
        system: CG_EDGE_SYSTEM_TAG.to_string(),
        model_name: "face-reid".to_string(),
        model_version: "1".to_string(),
        pipeline_version: "local-face".to_string(),
    }
}

fn collect_test_sources(dir: &Path, out: &mut String) {
    for entry in fs::read_dir(dir).expect("test source directory") {
        let entry = entry.expect("test source entry");
        let path = entry.path();
        if path.is_dir() {
            collect_test_sources(&path, out);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            out.push_str(&fs::read_to_string(&path).expect("test source file"));
            out.push('\n');
        }
    }
}

#[test]
fn freshly_initialized_store_with_default_storeconfig_seeds_text_default_nomic_v1_5_768_space() {
    let (_dir, config) = temp_store_config("fresh-nomic-default");
    let store = Store::open(config).expect("store opens");
    let spaces = store.list_embedding_spaces().expect("embedding spaces");
    let nomic = spaces
        .iter()
        .find(|space| space.embedding_space_id == NOMIC_TEXT_DEFAULT_ID);

    assert_eq!(
        Some(768),
        nomic.map(|space| space.dimension),
        "a new default store must seed the Nomic text space"
    );
    assert!(
        spaces
            .iter()
            .all(|space| space.embedding_space_id != EMBEDDINGGEMMA_TEXT_DEFAULT_ID),
        "the previous bundled text space must not be seeded by default"
    );
}

#[test]
fn text_default_space_id_resolves_to_text_default_nomic_v1_5_768_after_default_store_open() {
    let (_dir, config) = temp_store_config("text-default-id");
    let store = Store::open(config).expect("store opens");
    let spaces = store.list_embedding_spaces().expect("embedding spaces");
    let default_space = default_text_embedding_space();

    assert_eq!(NOMIC_TEXT_DEFAULT_ID, BuiltinSpaceRegistry::TEXT_DEFAULT_ID);
    assert_eq!(NOMIC_TEXT_DEFAULT_ID, default_space.embedding_space_id);
    assert!(
        spaces
            .iter()
            .any(|space| space.embedding_space_id == NOMIC_TEXT_DEFAULT_ID),
        "store metadata must expose the same bundled default id"
    );
}

#[test]
fn nomic_embed_text_adapter_implements_embedder_trait_with_declared_dimension_768_and_l2_normalized_output()
 {
    let embedder = NomicTextEmbedder::new(EmbedderConfig::disabled());
    let space = embedder.embedding_space();
    let output = embedder
        .embed(EmbeddingInput::Text("the quick brown fox".to_string()))
        .expect("nomic text embed");
    let hello = embedder
        .embed(EmbeddingInput::Text("hello".to_string()))
        .expect("hello embed");
    let world = embedder
        .embed(EmbeddingInput::Text("world".to_string()))
        .expect("world embed");

    assert_eq!(NOMIC_TEXT_DEFAULT_ID, space.embedding_space_id);
    assert_eq!(768, output.dimension);
    assert_eq!(768, output.vector.len());
    assert!(
        (l2_norm(&output.vector) - 1.0).abs() <= 1.0e-3,
        "Nomic embeddings must be unit-normalized for cosine search"
    );
    assert_eq!(NOMIC_TEXT_DEFAULT_ID, output.embedding_space_id);
    assert!(
        cosine_similarity(&hello.vector, &world.vector) < 0.99,
        "distinct text inputs must not collapse to one constant vector"
    );
}

#[test]
fn nomic_embed_text_matryoshka_heads_truncate_to_512_256_128_with_renormalized_unit_norm() {
    for dimension in [512, 256, 128] {
        let embedder = NomicTextEmbedder::new_for_dimension(EmbedderConfig::disabled(), dimension);
        let output = embedder
            .embed(EmbeddingInput::Text("payment idempotency keys".to_string()))
            .expect("matryoshka embed");

        assert_eq!(dimension, output.dimension);
        assert_eq!(dimension, output.vector.len());
        assert!(
            (l2_norm(&output.vector) - 1.0).abs() <= 1.0e-3,
            "selected Nomic head {dimension} must be unit-normalized"
        );
    }
}

#[test]
fn candle_embeddinggemma_workspace_sibling_crate_implements_existing_embedder_trait_signature() {
    let embedder: Box<dyn Embedder> = Box::new(embeddinggemma_embedder());
    let space = embedder.embedding_space();

    assert_eq!(EMBEDDINGGEMMA_TEXT_DEFAULT_ID, space.embedding_space_id);
    assert_eq!(768, space.dimension);
    assert_eq!(vec![EvidenceKind::Text], embedder.supported_inputs());
}

#[test]
fn candle_embeddinggemma_consumer_registers_via_storeconfig_extra_embedding_spaces_under_text_default_embeddinggemma_300m_768_space_id()
 {
    let (_dir, mut config) = temp_store_config("embeddinggemma-registration");
    let embedder = Arc::new(embeddinggemma_embedder());
    let space = embedder.embedding_space();
    config.extra_embedding_spaces.push(space.clone());

    let opened = Store::open_with_embedder_registrations(
        config,
        vec![EmbedderRegistration { space, embedder }],
    );

    assert!(
        opened.is_ok(),
        "EmbeddingGemma should be accepted as an opt-in sibling space beside the bundled default"
    );
    let store = opened.expect("asserted open succeeds");
    let spaces = store.list_embedding_spaces().expect("embedding spaces");
    assert!(
        spaces
            .iter()
            .any(|space| space.embedding_space_id == NOMIC_TEXT_DEFAULT_ID)
    );
    assert!(
        spaces
            .iter()
            .any(|space| space.embedding_space_id == EMBEDDINGGEMMA_TEXT_DEFAULT_ID)
    );
}

#[test]
fn candle_embeddinggemma_sibling_crate_carries_apache_2_0_license_file_and_workspace_members_declaration_in_root_cargo_toml()
 {
    let cargo_toml = read_workspace_file("Cargo.toml");
    let license = read_workspace_file("crates/candle-embeddinggemma/LICENSE");

    assert!(cargo_toml.contains("\"crates/candle-embeddinggemma\""));
    assert!(
        license.contains("Apache License") && license.contains("Version 2.0"),
        "sibling crate must carry the canonical Apache-2.0 license text"
    );
}

#[test]
fn cg_main_crate_has_no_embeddinggemma_runtime_code_grep_audit_against_cargo_expanded_source_plus_no_embeddinggemma_transitive_dep_in_cargo_tree()
 {
    let embedder_mod = read_crate_file("src/embedder.rs");
    let store_mod = read_crate_file("src/store/mod.rs");
    let cargo_toml = read_crate_file("Cargo.toml");
    let nomic_source = read_crate_file("src/embedder/nomic.rs");

    assert!(
        !embedder_mod.to_lowercase().contains("embeddinggemma")
            && !store_mod.to_lowercase().contains("embeddinggemma")
            && !cargo_toml.contains("embeddinggemma-runtime"),
        "main crate runtime source and feature surface must not carry EmbeddingGemma"
    );
    assert!(!cargo_toml.contains("candle-embeddinggemma = {"));
    let nomic_tensors = [
        "model.embeddings.word_embeddings.weight",
        "model.encoder.layer.0.attention.self.query.weight",
        "model.text_projection.up_proj.weight",
        "model.text_projection.down_proj.weight",
    ];
    let fingerprint_count = nomic_tensors
        .iter()
        .filter(|tensor| nomic_source.contains(**tensor))
        .count();
    assert!(
        fingerprint_count >= 3,
        "Nomic adapter must reference upstream Nomic-BERT tensor identifiers"
    );
}

#[test]
fn embedder_trait_surface_remains_byte_identical_across_default_embedder_pivot() {
    fn assert_embed_signature(embedder: &dyn Embedder, input: EmbeddingInput) {
        let _ = embedder.embed(input);
    }

    let embedder = NomicTextEmbedder::new(EmbedderConfig::disabled());
    let inputs = [
        EmbeddingInput::Text("text".to_string()),
        EmbeddingInput::ImageFrame {
            uri: "file:///frame.png".to_string(),
        },
        EmbeddingInput::AudioSegment {
            uri: "file:///clip.wav".to_string(),
        },
        EmbeddingInput::VideoSegment {
            uri: "file:///clip.mp4".to_string(),
        },
        EmbeddingInput::StructuredSignal(serde_json::json!({"kind": "sensor"})),
    ];

    for input in inputs {
        assert_embed_signature(&embedder, input);
    }
    let fixture = include_str!("../fixtures/previous_public_api.txt");
    assert!(fixture.contains("pub trait Embedder"));
    assert!(fixture.contains("pub struct EmbedderConfig"));
    assert!(fixture.contains("pub struct EmbedderRegistration"));
}

#[test]
fn bundled_default_pivot_to_nomic_preserves_embedding_substrate_acceptance_anchors() {
    let mut corpus = String::new();
    collect_test_sources(&crate_root().join("tests"), &mut corpus);
    let anchors = [
        "embedder_trait_implementation",
        "embedder_metadata_stored_on_decision",
        "reindex_updates_vectors_and_preserves_searchability",
        "reindex_changes_stored_vectors",
        "mock_embedder_deterministic",
        "embedder_trait_typed_input",
        "embedding_metadata_includes_space_id",
        "reindex_scoped_by_embedding_space",
        "embedder_metadata_distinguishes_internal_vs_external",
        "ingest_observation_with_precomputed_embedding",
        "ingest_observation_rejects_dimension_mismatch_for_known_space",
        "local_entity_face_embedding_match_overrides_class_detection",
        "screenshot_review_ingest_and_internal_search",
    ];

    for anchor in anchors {
        let marker = format!("fn {anchor}");
        let start = corpus
            .find(&marker)
            .unwrap_or_else(|| panic!("missing substrate test anchor {anchor}"));
        let body_window = &corpus[start..corpus.len().min(start + 4000)];
        assert!(
            body_window.contains("assert!")
                || body_window.contains("assert_eq!")
                || body_window.contains("assert_ne!")
                || body_window.contains("ok!("),
            "substrate test anchor {anchor} must retain a substantive assertion"
        );
    }
}

#[test]
fn nomic_bert_forward_pass_graph_shape_cache_contract_fixtures_match_main_crate() {
    let nomic_source = read_crate_file("src/embedder/nomic.rs");
    let expected_shape_terms = [
        "NomicBertModel",
        "model.encoder.layer.0.attention.self.query.weight",
        "model.encoder.layer.0.attention.self.key.weight",
        "model.encoder.layer.0.attention.self.value.weight",
        "model.text_projection.up_proj.weight",
    ];
    let matched = expected_shape_terms
        .iter()
        .filter(|term| nomic_source.contains(**term))
        .count();

    assert!(
        matched >= 4,
        "main crate must carry Nomic-specific forward-pass shape fixtures"
    );
}

#[test]
fn embeddinggemma_graph_shape_cache_contract_fixtures_relocated_to_sibling_crate() {
    let relocated = read_workspace_file(
        "crates/candle-embeddinggemma/tests/acceptance/embeddinggemma_loads.rs",
    );

    assert!(relocated.contains("EmbeddingGemmaForwardStage::AttentionQkv"));
    assert!(relocated.contains("EmbeddingGemmaForwardStage::ProjectionMlp"));
    assert!(relocated.contains("embeddinggemma_matryoshka_512"));
}

#[test]
fn runtime_cross_embedder_switch_from_text_default_nomic_v1_5_768_to_text_default_embeddinggemma_300m_768_via_existing_reindex_no_new_api_surface()
 {
    let reindex_source = read_crate_file("src/store/reindex.rs");
    let store_source = read_crate_file("src/store/mod.rs");
    let sibling = embeddinggemma_embedder();

    assert_eq!(
        EMBEDDINGGEMMA_TEXT_DEFAULT_ID,
        sibling.embedding_space().embedding_space_id
    );
    assert!(reindex_source.contains("pub fn reindex_space"));
    assert!(store_source.contains("pub fn open_with_embedder_registrations"));
    assert!(!store_source.contains("switch_embedder"));
}

#[test]
fn semantic_search_through_default_text_space_after_pivot_returns_results_under_nomic_bert_embeddings()
 {
    let embedding_source = read_crate_file("src/store/embedding.rs");
    let default_space = default_text_embedding_space();

    assert_eq!(NOMIC_TEXT_DEFAULT_ID, default_space.embedding_space_id);
    assert!(
        embedding_source.contains(NOMIC_TEXT_DEFAULT_ID),
        "semantic search must route through the Nomic bundled default id"
    );
}

#[test]
fn evidence_search_against_evidence_vector_text_column_under_text_default_nomic_v1_5_768_returns_expected_neighbours()
 {
    let embedding_source = read_crate_file("src/store/embedding.rs");
    let default_space = default_text_embedding_space();

    assert_eq!(NOMIC_TEXT_DEFAULT_ID, default_space.embedding_space_id);
    assert!(default_space.bindings.iter().any(|binding| {
        binding.table_name == "evidence" && binding.vector_column == "vector_text"
    }));
    assert!(
        embedding_source.contains(NOMIC_TEXT_DEFAULT_ID),
        "evidence search must bind the evidence vector_text column to the Nomic default id"
    );
}

#[test]
fn vector_search_with_embedding_space_id_text_default_nomic_v1_5_768_refuses_query_vectors_of_mismatched_dimension()
 {
    let source = read_crate_file("src/store/embedding.rs");

    assert!(source.contains("query_vector.len() != space.dimension"));
    assert!(source.contains("EmbeddingDimensionMismatch"));
    assert!(source.contains("CrossSpaceQueryRefused"));
}

#[test]
fn nomic_weights_first_load_writes_to_xdg_compliant_cache_path_under_cg_namespace_on_linux_and_macos()
 {
    let source = read_crate_file("src/embedder/nomic.rs");

    assert!(source.contains("XDG_CACHE_HOME") || source.contains(".cache"));
    assert!(source.contains("cg") && source.contains("embedders"));
    assert!(source.contains(NOMIC_MODEL_ID));
}

#[test]
fn nomic_weights_first_load_writes_to_localappdata_cg_namespace_on_windows() {
    let source = read_crate_file("src/embedder/nomic.rs");

    assert!(source.contains("LOCALAPPDATA"));
    assert!(source.contains("cg") && source.contains("embedders"));
    assert!(source.contains(NOMIC_MODEL_ID));
}

#[test]
fn nomic_weights_first_load_emits_log_info_and_default_stderr_line_at_fetch_start_and_completion() {
    let nomic_source = read_crate_file("src/embedder/nomic.rs");
    let observer_source = read_crate_file("src/embedder/load_observer.rs");

    assert!(observer_source.contains("FetchStarted"));
    assert!(observer_source.contains("FetchCompleted"));
    assert!(
        nomic_source.contains("eprintln!")
            && nomic_source.contains("FetchStarted")
            && nomic_source.contains("FetchCompleted"),
        "fresh Nomic fetch must emit observer events and default stderr progress lines"
    );
}

#[test]
fn embedderconfig_cache_path_override_field_routes_weight_storage_to_supplied_path_for_air_gapped_consumers()
 {
    let cache = tempfile::tempdir().expect("cache tempdir");
    let embedder = NomicTextEmbedder::new(EmbedderConfig {
        source: ModelSource::HuggingFace {
            repo: "nomic-ai/nomic-embed-text-v1.5".to_string(),
        },
        offline: false,
        cache_dir: Some(cache.path().to_path_buf()),
        pinned_revision: None,
        network_policy: "allow-download".to_string(),
    });

    let _ = embedder
        .embed(EmbeddingInput::Text("cache override check".to_string()))
        .expect("nomic embed");
    let expected_dir = cache.path().join("nomic-embed-text-v1.5");
    assert!(
        expected_dir.exists(),
        "Nomic weights must be stored under the supplied cache_dir override"
    );
}

#[test]
fn cg_readme_choosing_an_embedder_section_discloses_nomic_artifact_huggingface_url_approx_548mb_size_and_approx_three_minutes_first_load_on_25mbps()
 {
    let readme = read_crate_file("README.md");

    assert!(readme.to_lowercase().contains("choosing an embedder"));
    assert!(readme.contains("nomic-embed-text-v1.5"));
    assert!(readme.contains("candle-embeddinggemma"));
    assert!(readme.contains("reindex_space"));
    assert!(readme.contains("huggingface.co/nomic-ai/nomic-embed-text-v1.5"));
    assert!(
        ["500MB", "547MB", "548MB", "549MB", "550MB", "600MB"]
            .iter()
            .any(|size| readme.contains(size)),
        "README must disclose the approximate first-load artifact size"
    );
    assert!(readme.contains("three minutes"));
    assert!(readme.contains("25Mbps"));
    assert!(readme.contains("Store::open_with_embedder_registrations"));
}

#[test]
fn store_open_with_default_config_against_freshly_created_path_writes_cg_schema_version_row_matching_current_version_and_emits_no_legacy_or_migration_observable()
 {
    let (_dir, config) = disabled_store_config("greenfield-schema");
    let store = Store::open(config).expect("fresh store opens");
    let trace = store.last_query_trace().expect("loader trace");

    assert!(!store.list_embedding_spaces().expect("spaces").is_empty());
    assert_eq!(
        context_graph::CG_SCHEMA_VERSION,
        StoreConfig::default().expected_schema_version
    );
    assert!(
        !trace.to_lowercase().contains("legacy")
            && !trace.to_lowercase().contains("migrat")
            && !trace.to_lowercase().contains("upgrade")
            && !trace.to_lowercase().contains("prior")
    );
}

#[test]
fn cargo_tree_for_context_graph_default_dep_graph_carries_no_native_dep_sys_crates_no_python_runtime_no_api_key_dep()
 {
    let cargo_toml = read_crate_file("Cargo.toml");
    let sibling_cargo_toml = read_workspace_file("crates/candle-embeddinggemma/Cargo.toml");
    assert!(
        !cargo_toml.contains("embeddinggemma-runtime"),
        "default dependency surface must not expose an EmbeddingGemma runtime feature"
    );

    let declared = format!("{cargo_toml}\n{sibling_cargo_toml}").to_lowercase();
    for banned in [
        "pyo3",
        "cpython",
        "tch",
        "ort",
        "tensorflow",
        "tflite",
        "onnxruntime",
        "libtorch",
        "reqwest",
        "openai",
        "anthropic",
        "cohere",
    ] {
        assert!(
            !declared.contains(banned),
            "banned dependency {banned} declared on the embedding path"
        );
    }
}

#[test]
fn vigil_pattern_extra_embedding_space_declares_face_reid_v1_under_dim_512_entity_reference_lane_unaffected_by_default_swap()
 {
    let (_dir, mut config) = disabled_store_config("vigil-face");
    config.extra_embedding_spaces.push(face_space());
    let store = Store::open(config).expect("store opens with face space");
    let context = store
        .create_context(CreateContext {
            name: format!("vigil-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Person,
            name: "known visitor".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");

    let mut vector = vec![0.0; 512];
    vector[17] = 1.0;
    let embedding_id = store
        .enroll_entity_reference(
            entity.id,
            FACE_REID_SPACE,
            Some("gate-camera".to_string()),
            vector.clone(),
            face_producer(),
        )
        .expect("entity reference enrollment");
    let rows = store
        .list_entity_reference_embeddings_for_entity(entity.id)
        .expect("entity reference rows");

    assert!(
        store
            .list_embedding_spaces()
            .expect("spaces")
            .iter()
            .any(|space| space.embedding_space_id == FACE_REID_SPACE)
    );
    assert!(rows.iter().any(|row| {
        row.id == embedding_id && row.embedding_space_id == FACE_REID_SPACE && row.vector == vector
    }));
}

#[test]
fn cg_github_actions_workflow_runs_on_matrix_of_windows_latest_macos_latest_ubuntu_latest_explicitly_declared_in_yaml()
 {
    let workflow = read_workspace_file(".github/workflows/ci.yml");
    assert!(
        serde_yml::from_str::<serde_yml::Value>(&workflow).is_ok(),
        "CI workflow must be valid YAML"
    );
    for os in ["windows-latest", "macos-latest", "ubuntu-latest"] {
        assert!(workflow.contains(os), "CI matrix missing {os}");
    }
    assert!(workflow.contains("cargo test --workspace"));
}

#[test]
fn cg_readme_ci_section_documents_three_runner_matrix_and_actions_cache_snippet_keyed_off_cargo_lock_for_embedder_cache_path()
 {
    let readme = read_crate_file("README.md");
    let lower = readme.to_lowercase();

    assert!(lower.contains("continuous integration") || lower.contains("## ci"));
    for os in ["windows-latest", "macos-latest", "ubuntu-latest"] {
        assert!(readme.contains(os), "README CI section missing {os}");
    }
    assert!(readme.contains("actions/cache"));
    assert!(readme.contains("Cargo.lock"));
    assert!(readme.contains("~/.cache/cg/embedders"));
    assert!(
        readme.contains("LOCALAPPDATA")
            || readme.contains("%LOCALAPPDATA%")
            || readme.contains("cg/embedders")
    );
}

#[test]
fn embedder_trait_and_storeconfig_public_surface_diff_keeps_surviving_signatures_and_allowed_moves()
{
    let fixture = include_str!("../fixtures/previous_public_api.txt");
    let embedder_source = read_crate_file("src/embedder.rs");
    let types_source = read_crate_file("src/types.rs");
    let store_source = read_crate_file("src/store/mod.rs");
    let observer_source = read_crate_file("src/embedder/load_observer.rs");

    assert!(fixture.contains("pub trait Embedder"));
    assert!(embedder_source.contains("pub trait Embedder"));
    assert!(embedder_source.contains("fn embed(&self, input: EmbeddingInput)"));
    assert!(types_source.contains("pub default_text_embedder: Option<EmbedderConfig>"));
    assert!(types_source.contains("pub extra_embedding_spaces: Vec<EmbeddingSpaceDecl>"));
    assert!(observer_source.contains("pub trait EmbedderLoadObserver"));
    assert!(!store_source.contains("pub fn switch_embedder"));
}
