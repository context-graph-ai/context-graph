use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use cg_catalog::embeddinggemma_test_support::{
    EmbeddingGemmaTextEmbedder, default_embeddinggemma_offline_config,
};
use chrono::Utc;
use context_graph::embedder::nomic::NomicTextEmbedder;
use context_graph::embedding_space::{BuiltinSpaceRegistry, default_text_embedding_space};
use context_graph::{
    CG_EDGE_SYSTEM_TAG, CgError, ContextId, CreateContext, CreateEntity, DistanceMetric, Embedder,
    EmbedderConfig, EmbedderLoadEvent, EmbedderLoadObserver, EmbedderRegistration, EmbeddingId,
    EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceDecl, EmbeddingTarget, EntityId,
    EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, ModelSource,
    ObservationId, PrecomputedEmbedding, RecordObservation, ReindexEvent, ReindexOptions,
    RetentionStatus, Store, StoreConfig,
};
use contextdb_core::{RowId, Value, VectorIndexRef};
use contextdb_engine::Database;
use safetensors::tensor::{Dtype, TensorView};
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

const NOMIC_TEXT_DEFAULT_ID: &str = "text_default_nomic_v1_5_768";
const NOMIC_MODEL_ID: &str = "nomic-embed-text-v1.5";
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

fn read_store_public_api_sources() -> String {
    let store_dir = crate_root().join("src/store");
    let mut paths = fs::read_dir(&store_dir)
        .expect("store source dir")
        .map(|entry| entry.expect("store source entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect::<Vec<_>>();
    paths.sort();
    paths.extend([
        crate_root().join("src/producer.rs"),
        crate_root().join("src/messaging.rs"),
    ]);
    paths
        .into_iter()
        .map(|path| {
            fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn public_struct_fields(source: &str, struct_name: &str) -> Vec<String> {
    let marker = format!("pub struct {struct_name} {{");
    let block = source
        .split(&marker)
        .nth(1)
        .and_then(|after_marker| after_marker.split("\n}").next())
        .unwrap_or_else(|| panic!("missing public struct {struct_name}"));
    let mut fields = block
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            trimmed
                .strip_prefix("pub ")
                .and_then(|field| field.split(':').next())
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(ToOwned::to_owned)
        })
        .collect::<Vec<_>>();
    fields.sort();
    fields
}

fn brace_delta_outside_string_literals(line: &str) -> isize {
    let mut delta = 0;
    let mut in_string = false;
    let mut escaped = false;
    for ch in line.chars() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => delta += 1,
            '}' => delta -= 1,
            _ => {}
        }
    }
    delta
}

fn public_inherent_methods(source: &str, type_name: &str) -> Vec<String> {
    let marker = format!("impl {type_name} {{");
    let mut remaining = source;
    let mut methods = Vec::new();
    while let Some(offset) = remaining.find(&marker) {
        let block = &remaining[offset + marker.len()..];
        let mut depth = 1_isize;
        let mut pending_attrs = Vec::new();
        for line in block.lines() {
            if depth == 1 {
                let trimmed = line.trim();
                if trimmed.starts_with("#[") {
                    pending_attrs.push(trimmed.to_string());
                } else if let Some(signature) = trimmed.strip_prefix("pub fn ") {
                    if !pending_attrs
                        .iter()
                        .any(|attr| attr.contains("context_graph_internal_test_support"))
                    {
                        let name = signature
                            .split(|ch: char| ch == '(' || ch == '<' || ch.is_whitespace())
                            .next()
                            .expect("public method name");
                        methods.push(name.to_string());
                    }
                    pending_attrs.clear();
                } else if !trimmed.is_empty() && !trimmed.starts_with("//") {
                    pending_attrs.clear();
                }
            }
            depth += brace_delta_outside_string_literals(line);
            if depth <= 0 {
                break;
            }
        }
        remaining = block;
    }
    methods.sort();
    methods
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
    let config = default_embeddinggemma_offline_config();
    assert!(config.offline);
    assert_eq!("offline-cache-only", config.network_policy);
    assert!(matches!(
        &config.source,
        ModelSource::HuggingFace { repo } if repo == "google/embeddinggemma-300m"
    ));
    EmbeddingGemmaTextEmbedder::new(config)
}

#[derive(Clone, Debug)]
struct DeterministicSiblingTextEmbedder {
    space: EmbeddingSpace,
    vector: Vec<f32>,
}

impl DeterministicSiblingTextEmbedder {
    fn new(space: EmbeddingSpace, vector: Vec<f32>) -> Self {
        Self { space, vector }
    }
}

impl Embedder for DeterministicSiblingTextEmbedder {
    fn embed(&self, input: EmbeddingInput) -> context_graph::Result<EmbeddingOutput> {
        match input {
            EmbeddingInput::Text(_) => Ok(EmbeddingOutput {
                embedding_space_id: self.space.embedding_space_id.clone(),
                dimension: self.space.dimension,
                producer: EvidenceProducer {
                    system: "acceptance".to_string(),
                    model_name: "embeddinggemma-reindex-fixture".to_string(),
                    model_version: "1".to_string(),
                    pipeline_version: "reindex".to_string(),
                },
                vector: self.vector.clone(),
            }),
            EmbeddingInput::ImageFrame { .. } | EmbeddingInput::ImageBytes(..) => {
                Err(CgError::UnsupportedEvidenceKind {
                    kind: EvidenceKind::ImageFrame,
                })
            }
            EmbeddingInput::AudioSegment { .. } => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::AudioSegment,
            }),
            EmbeddingInput::VideoSegment { .. } => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::VideoSegment,
            }),
            EmbeddingInput::StructuredSignal(_) => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::StructuredSignal,
            }),
        }
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        self.space.clone()
    }
}

#[derive(Default)]
struct RecordingNomicLoadObserver {
    events: Mutex<Vec<EmbedderLoadEvent>>,
}

impl RecordingNomicLoadObserver {
    fn events(&self) -> Vec<EmbedderLoadEvent> {
        self.events.lock().expect("nomic load events").clone()
    }
}

impl EmbedderLoadObserver for RecordingNomicLoadObserver {
    fn observe(&self, event: EmbedderLoadEvent) {
        self.events.lock().expect("nomic load events").push(event);
    }
}

fn nomic_acceptance_cache_root() -> PathBuf {
    if let Some(path) = std::env::var_os("CG_NOMIC_TEST_CACHE") {
        return PathBuf::from(path);
    }
    if cfg!(target_os = "windows") {
        return std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("cg")
            .join("embedders");
    }
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".cache"))
        })
        .unwrap_or_else(std::env::temp_dir)
        .join("cg")
        .join("embedders")
}

fn nomic_live_config() -> EmbedderConfig {
    let cache_dir = nomic_acceptance_cache_root();
    fs::create_dir_all(&cache_dir).expect("nomic acceptance cache directory");
    EmbedderConfig {
        cache_dir: Some(cache_dir),
        ..EmbedderConfig::default_nomic()
    }
}

fn nomic_live_embedder() -> NomicTextEmbedder {
    NomicTextEmbedder::new(nomic_live_config())
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

fn text_producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "acceptance".to_string(),
        model_name: "text-fixture".to_string(),
        model_version: "1".to_string(),
        pipeline_version: "local-text".to_string(),
    }
}

fn text_evidence(
    observation_id: ObservationId,
    evidence_id: EvidenceId,
    context_id: ContextId,
    text: &str,
) -> EvidenceRef {
    EvidenceRef {
        id: evidence_id,
        observation_id,
        context_id,
        kind: EvidenceKind::Text,
        source_ref: "cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .to_string(),
        content_hash: Some(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
        ),
        captured_at: Some(Utc::now()),
        producer: text_producer(),
        text_projection: Some(text.to_string()),
        retention_status: RetentionStatus::RetainedExternal,
        ..Default::default()
    }
}

fn text_observation(
    id: ObservationId,
    entity_id: EntityId,
    context_id: ContextId,
    evidence: EvidenceRef,
    embeddings: Vec<PrecomputedEmbedding>,
) -> RecordObservation {
    RecordObservation {
        id,
        entity_id,
        context_id,
        observation_type: "nomic-acceptance".to_string(),
        source: "nomic-acceptance".to_string(),
        observed_at: Utc::now(),
        evidence: vec![evidence],
        observed_properties: BTreeMap::from([(
            "component".to_string(),
            serde_json::json!("embedding"),
        )]),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings,
    }
}

fn assert_first_vector_hit_id(
    db: &Database,
    table: &str,
    hits: &[(RowId, f32)],
    expected: uuid::Uuid,
) {
    assert!(
        !hits.is_empty(),
        "expected at least one vector hit for {table}.id {expected}"
    );
    let row_id = hits[0].0;
    let rows = db
        .scan(table, db.snapshot())
        .expect("scan vector hit table");
    let row = rows
        .into_iter()
        .find(|row| row.row_id == row_id)
        .expect("vector hit row must still be visible");
    assert_eq!(row.values.get("id"), Some(&Value::Uuid(expected)));
}

fn tensor_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn write_minimal_tokenizer(path: &Path) {
    let vocab = [
        ("<unk>".to_string(), 0),
        ("wrong".to_string(), 1),
        ("cache".to_string(), 2),
    ]
    .into_iter()
    .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".to_string())
        .build()
        .expect("wordlevel");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer.save(path, false).expect("tokenizer");
}

fn stage_wrong_identity_nomic_cache(model_dir: &Path) -> PathBuf {
    fs::create_dir_all(model_dir).expect("wrong nomic model dir");
    let values = vec![0.0; 256 * 1024];
    let bytes = tensor_bytes(&values);
    let tensor = TensorView::new(Dtype::F32, vec![256, 1024], &bytes).expect("wrong tensor view");
    let model_bytes = safetensors::tensor::serialize([("not_nomic_bert.weight", tensor)], None)
        .expect("wrong safetensors");
    fs::write(model_dir.join("model.safetensors"), model_bytes).expect("wrong model");
    fs::write(
        model_dir.join("config.json"),
        r#"{"model_type":"nomic_bert","n_embd":768,"n_head":12,"n_layer":12}"#,
    )
    .expect("wrong config");
    write_minimal_tokenizer(&model_dir.join("tokenizer.json"));
    fs::write(model_dir.join("REVISION"), "old-revision").expect("wrong revision");
    model_dir.to_path_buf()
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
        "the opt-in sibling text space must not be seeded by default"
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

    // The dedicated event-history space is opt-in, never something a plain store open conjures.
    // A store that listed it without being asked would advertise a space nothing in that store can
    // serve, and would make the two 768-dimensional spaces look interchangeable to anyone reading
    // the store's own metadata.
    assert!(
        !spaces.iter().any(|space| space.embedding_space_id
            == BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID),
        "a default store open must register no event-history space; the dedicated space is only \
         ever registered by a caller that asked for it, and the store listed {:?}",
        spaces
            .iter()
            .map(|space| space.embedding_space_id.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn nomic_embed_text_adapter_implements_embedder_trait_with_declared_dimension_768_and_l2_normalized_output()
 {
    let embedder = nomic_live_embedder();
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
        let embedder = NomicTextEmbedder::new_for_dimension(nomic_live_config(), dimension);
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
    let context_id = ContextId::new_v7();
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let evidence = text_evidence(
        observation_id,
        evidence_id,
        context_id,
        "embeddinggemma sibling recall path",
    );
    let mut sibling_vector = vec![0.0; 768];
    sibling_vector[0] = 1.0;
    store
        .record_observation(text_observation(
            observation_id,
            EntityId::new_v7(),
            context_id,
            evidence,
            vec![PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: EMBEDDINGGEMMA_TEXT_DEFAULT_ID.to_string(),
                vector: sibling_vector.clone(),
                target: EmbeddingTarget::Evidence(evidence_id),
                produced_by: text_producer(),
            }],
        ))
        .expect("record through sibling space");
    let db_path = store.db_path().to_path_buf();
    drop(store);
    let db = Database::open(&db_path).expect("raw contextdb");
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &sibling_vector,
            1,
            None,
            db.snapshot(),
        )
        .expect("query sibling vector");
    assert_first_vector_hit_id(&db, "evidence", &hits, evidence_id.as_uuid());
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
fn embedder_trait_surface_remains_byte_identical_across_default_embedder_pivot() {
    fn assert_embed_signature(embedder: &dyn Embedder, input: EmbeddingInput) {
        let _ = embedder.embed(input);
    }

    let embedder = nomic_live_embedder();
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
        "embeddings.word_embeddings.weight",
        "encoder.layers.0.attn.Wqkv.weight",
        "encoder.layers.0.attn.out_proj.weight",
        "encoder.layers.0.mlp.fc11.weight",
        "encoder.layers.11.attn.Wqkv.weight",
    ];
    let matched = expected_shape_terms
        .iter()
        .filter(|term| nomic_source.contains(**term))
        .count();

    assert!(
        matched == expected_shape_terms.len(),
        "main crate must carry Nomic-specific forward-pass shape fixtures"
    );
}

/// Every checked-in golden text in `tests/fixtures/text_default_space_goldens.json` is 11-15
/// tokens, nowhere near the plain-embedding truncation ceiling, so that fixture alone cannot
/// catch a change to the ceiling: lowering it would silently move every real Digest/Record/
/// Evidence vector longer than the ceiling without affecting the golden suite. This pins
/// the private ceiling constant's value and proves the plain-text `forward()` path still spends
/// it to truncate from the right before the tokens ever reach the model — a rename, a value
/// change, or a path that stops truncating with this exact constant all fail here.
#[test]
fn plain_text_embedding_truncates_at_its_frozen_token_ceiling_and_never_silently_moves() {
    let nomic_source = read_crate_file("src/embedder/nomic.rs");

    assert!(
        nomic_source.contains("const NOMIC_PLAIN_EMBED_MAX_SEQUENCE_TOKENS: usize = 8192;"),
        "the plain-embedding truncation ceiling must stay exactly 8192 tokens; a change here \
         moves every real Digest/Record/Evidence vector longer than the new ceiling without any \
         golden fixture noticing, since every golden text is 11-15 tokens"
    );

    let forward_marker =
        "fn forward(&self, loaded: &LoadedNomicModel, text: &str) -> Result<Vec<f32>> {";
    let forward_start = nomic_source
        .find(forward_marker)
        .expect("the plain-text forward() method must still exist under its known signature");
    let after_marker = forward_start + forward_marker.len();
    let forward_end = nomic_source[after_marker..]
        .find("\n    fn ")
        .map(|offset| after_marker + offset)
        .expect("forward() must still be followed by another impl method");
    let forward_body = &nomic_source[forward_start..forward_end];

    assert!(
        forward_body.contains("encoding.truncate(")
            && forward_body.contains("NOMIC_PLAIN_EMBED_MAX_SEQUENCE_TOKENS")
            && forward_body.contains("TruncationDirection::Right"),
        "the plain-text forward() path must truncate its tokenizer encoding with the frozen \
         NOMIC_PLAIN_EMBED_MAX_SEQUENCE_TOKENS ceiling, from the right, before the model ever \
         sees the tokens — not a different constant, not a hardcoded literal, and not silently \
         removed"
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
    let store_source = read_crate_file("src/store/mod.rs");
    let sibling = embeddinggemma_embedder();
    let sibling_space = sibling.embedding_space();
    let mut old_vector = vec![0.0; 768];
    old_vector[1] = 1.0;
    let mut reindexed_vector = vec![0.0; 768];
    reindexed_vector[2] = 1.0;
    let (_dir, mut config) = temp_store_config("embeddinggemma-reindex");
    config.extra_embedding_spaces.push(sibling_space.clone());
    let embedder = Arc::new(DeterministicSiblingTextEmbedder::new(
        sibling_space.clone(),
        reindexed_vector.clone(),
    ));
    let store = Store::open_with_embedder_registrations(
        config,
        vec![EmbedderRegistration {
            space: sibling_space.clone(),
            embedder,
        }],
    )
    .expect("open Nomic default with registered EmbeddingGemma sibling space");
    let spaces = store.list_embedding_spaces().expect("embedding spaces");
    let context_id = ContextId::new_v7();
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let evidence = text_evidence(
        observation_id,
        evidence_id,
        context_id,
        "embeddinggemma reindex keeps prior text searchable",
    );

    assert_eq!(
        EMBEDDINGGEMMA_TEXT_DEFAULT_ID,
        sibling_space.embedding_space_id
    );
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
    store
        .record_observation(text_observation(
            observation_id,
            EntityId::new_v7(),
            context_id,
            evidence,
            vec![PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: EMBEDDINGGEMMA_TEXT_DEFAULT_ID.to_string(),
                vector: old_vector,
                target: EmbeddingTarget::Evidence(evidence_id),
                produced_by: text_producer(),
            }],
        ))
        .expect("seed target-space handle before reindex");
    store
        .reindex_space(EMBEDDINGGEMMA_TEXT_DEFAULT_ID, ReindexOptions::default())
        .expect("reindex registered EmbeddingGemma sibling space");
    let events = store.drain_reindex_events();
    assert!(
        events.iter().any(|event| matches!(
            event,
            ReindexEvent::Completed {
                embedding_space_id,
                rows_processed: 1,
                ..
            } if embedding_space_id == EMBEDDINGGEMMA_TEXT_DEFAULT_ID
        )),
        "expected reindex completion for the sibling space, events={events:?}"
    );
    let db_path = store.db_path().to_path_buf();
    drop(store);
    let db = Database::open(&db_path).expect("raw contextdb");
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &reindexed_vector,
            1,
            None,
            db.snapshot(),
        )
        .expect("query reindexed EmbeddingGemma sibling vector");
    assert_first_vector_hit_id(&db, "evidence", &hits, evidence_id.as_uuid());
    assert!(
        hits.first().is_some_and(|(_, score)| *score > 0.99),
        "the sibling-space vector index must return the reindexed text evidence with a self-match score, hits={hits:?}"
    );
    assert!(store_source.contains("pub fn open_with_embedder_registrations"));
    assert!(!store_source.contains("switch_embedder"));
}

#[test]
fn semantic_search_through_default_text_space_after_pivot_returns_results_under_nomic_bert_embeddings()
 {
    let embedding_source = read_crate_file("src/store/embedding.rs");
    let default_space = default_text_embedding_space();
    let (_dir, mut config) = temp_store_config("nomic-semantic-recall");
    config.default_text_embedder = Some(nomic_live_config());
    let store = Store::open(config).expect("open with live Nomic default");
    let context_id = ContextId::new_v7();
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let text = "auth library choice uses argon2id password hashing";
    let evidence = text_evidence(observation_id, evidence_id, context_id, text);
    store
        .record_observation(text_observation(
            observation_id,
            EntityId::new_v7(),
            context_id,
            evidence,
            Vec::new(),
        ))
        .expect("record Nomic text observation");
    let query = NomicTextEmbedder::new(nomic_live_config())
        .embed(EmbeddingInput::Text(text.to_string()))
        .expect("query embedding");
    let db_path = store.db_path().to_path_buf();
    drop(store);
    let db = Database::open(&db_path).expect("raw contextdb");
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &query.vector,
            1,
            None,
            db.snapshot(),
        )
        .expect("query Nomic evidence vector");

    assert_eq!(NOMIC_TEXT_DEFAULT_ID, default_space.embedding_space_id);
    assert_first_vector_hit_id(&db, "evidence", &hits, evidence_id.as_uuid());
    assert!(
        embedding_source.contains("pub fn semantic_search")
            && embedding_source.contains("vector_search_with_count_mode")
            && embedding_source.contains("EmbeddingInput::Text"),
        "semantic_search must embed the query text and route through the vector search primitive"
    );
}

#[test]
fn evidence_search_against_evidence_vector_text_column_under_text_default_nomic_v1_5_768_returns_expected_neighbours()
 {
    let embedding_source = read_crate_file("src/store/embedding.rs");
    let default_space = default_text_embedding_space();
    let (_dir, mut config) = temp_store_config("nomic-evidence-recall");
    config.default_text_embedder = Some(nomic_live_config());
    let store = Store::open(config).expect("open with live Nomic default");
    let context_id = ContextId::new_v7();
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let text = "queue retry policy records idempotency key decisions";
    let evidence = text_evidence(observation_id, evidence_id, context_id, text);
    store
        .record_observation(text_observation(
            observation_id,
            EntityId::new_v7(),
            context_id,
            evidence,
            Vec::new(),
        ))
        .expect("record Nomic text observation");
    let query = NomicTextEmbedder::new(nomic_live_config())
        .embed(EmbeddingInput::Text(text.to_string()))
        .expect("query embedding");
    let db_path = store.db_path().to_path_buf();
    drop(store);
    let db = Database::open(&db_path).expect("raw contextdb");
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &query.vector,
            1,
            None,
            db.snapshot(),
        )
        .expect("query Nomic evidence vector");

    assert_eq!(NOMIC_TEXT_DEFAULT_ID, default_space.embedding_space_id);
    assert!(default_space.bindings.iter().any(|binding| {
        binding.table_name == "evidence" && binding.vector_column == "vector_text"
    }));
    assert_first_vector_hit_id(&db, "evidence", &hits, evidence_id.as_uuid());
    assert!(
        embedding_source.contains("pub fn evidence_search")
            && embedding_source.contains("row_vector_search_from_evidence")
            && embedding_source.contains("vector_search_with_count_mode"),
        "evidence_search must use persisted row vectors or the vector search primitive under the Nomic default"
    );
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
    let cache_root = nomic_acceptance_cache_root().join("observer-fresh-fetch");
    let model_dir = cache_root.join(NOMIC_MODEL_ID);
    let _ = fs::remove_dir_all(&model_dir);
    fs::create_dir_all(&cache_root).expect("observer cache root");
    let observer = Arc::new(RecordingNomicLoadObserver::default());
    let observer_trait: Arc<dyn EmbedderLoadObserver> = observer.clone();
    let embedder = NomicTextEmbedder::new_with_load_observer(
        EmbedderConfig {
            cache_dir: Some(cache_root.clone()),
            ..EmbedderConfig::default_nomic()
        },
        observer_trait,
    );

    let output = embedder
        .embed(EmbeddingInput::Text("observable first load".to_string()))
        .expect("observed nomic fetch");
    let observed = observer.events();
    let structured = embedder.structured_loader_events();
    let nomic_source = read_crate_file("src/embedder/nomic.rs");
    let observer_source = read_crate_file("src/embedder/load_observer.rs");

    assert_eq!(NOMIC_TEXT_DEFAULT_ID, output.embedding_space_id);
    assert_eq!(768, output.vector.len());
    assert!(matches!(
        observed.first(),
        Some(EmbedderLoadEvent::FetchStarted {
            model_id,
            byte_size: None,
            destination,
        }) if model_id == NOMIC_MODEL_ID && destination == &model_dir
    ));
    assert!(observed.iter().any(|event| {
        matches!(
            event,
            EmbedderLoadEvent::FetchCompleted {
                model_id,
                byte_size: Some(546_938_168),
                destination,
                elapsed,
            } if model_id == NOMIC_MODEL_ID
                && destination == &model_dir
                && elapsed.as_nanos() > 0
        )
    }));
    assert_eq!(observed, structured);
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
fn nomic_cache_and_first_load_fetch_failures_return_typed_model_errors() {
    assert_eq!(
        EmbedderConfig::default_nomic().pinned_revision.as_deref(),
        Some("e9b6763023c676ca8431644204f50c2b100d9aab"),
        "the bundled Nomic default must pin a content-stable Hugging Face revision"
    );

    let cache = tempfile::tempdir().expect("nomic corrupt cache");
    let model_dir = stage_wrong_identity_nomic_cache(&cache.path().join("prestaged-nomic-v1-5"));
    let embedder = NomicTextEmbedder::new(EmbedderConfig {
        source: ModelSource::LocalPath(model_dir.clone()),
        offline: true,
        cache_dir: Some(cache.path().to_path_buf()),
        pinned_revision: None,
        network_policy: "offline-cache-only".to_string(),
    });

    let output = embedder.embed(EmbeddingInput::Text("wrong cache".to_string()));

    assert!(
        matches!(
            output,
            Err(CgError::EmbedderModelCorrupt { ref path })
                if path == &model_dir.join("model.safetensors").display().to_string()
        ),
        "wrong-model SafeTensors caches must fail with a typed corrupt-artifact error, output={output:?}"
    );

    let revision_mismatch = NomicTextEmbedder::new(EmbedderConfig {
        source: ModelSource::LocalPath(model_dir.clone()),
        offline: true,
        cache_dir: Some(cache.path().to_path_buf()),
        pinned_revision: Some("expected-revision".to_string()),
        network_policy: "offline-cache-only".to_string(),
    });
    let output = revision_mismatch.embed(EmbeddingInput::Text("wrong revision".to_string()));

    assert!(
        matches!(
            output,
            Err(CgError::EmbedderModelRevisionMismatch { ref expected, ref actual })
                if expected == "expected-revision" && actual == "old-revision"
        ),
        "pinned revisions must reject mismatched local Nomic caches before accepting tensors by shape, output={output:?}"
    );

    let missing_cache = tempfile::tempdir().expect("nomic missing fetch cache");
    let missing_repo = "context-graph-ai/definitely-missing-nomic-test-artifact";
    let missing_revision = "never-existing-revision";
    let missing = NomicTextEmbedder::new(EmbedderConfig {
        source: ModelSource::HuggingFace {
            repo: missing_repo.to_string(),
        },
        offline: false,
        cache_dir: Some(missing_cache.path().to_path_buf()),
        pinned_revision: Some(missing_revision.to_string()),
        network_policy: "allow-download".to_string(),
    });
    let output = missing.embed(EmbeddingInput::Text("missing remote".to_string()));
    let nomic_source = read_crate_file("src/embedder/nomic.rs");

    assert!(
        matches!(
            output,
            Err(CgError::EmbedderModelNotAvailable { ref source, ref path })
                if source == &format!("HuggingFace({missing_repo})")
                    && path == &missing_cache
                        .path()
                        .join(NOMIC_MODEL_ID)
                        .join("model.safetensors")
                        .display()
                        .to_string()
        ),
        "first-load fetch failures must fail with typed model-unavailable errors, output={output:?}"
    );
    assert!(
        nomic_source.contains("resolve/{revision}"),
        "Nomic fetch URLs must honor EmbedderConfig::pinned_revision rather than hard-coding main"
    );
    assert!(
        nomic_source.contains("check_revision_marker(&revision_path")
            && nomic_source.contains("NomicModelCacheKey")
            && nomic_source.contains("pinned_revision: self.config.pinned_revision.clone()")
            && nomic_source.contains("guard.get(&cache_key)"),
        "warm in-process Nomic loads must key and validate the cache by pinned_revision before returning a cached model"
    );
}

#[test]
fn embedderconfig_cache_path_override_field_routes_weight_storage_to_supplied_path_for_air_gapped_consumers()
 {
    let cache = nomic_acceptance_cache_root().join("override-cache");
    fs::create_dir_all(&cache).expect("override cache root");
    let embedder = NomicTextEmbedder::new(EmbedderConfig {
        source: ModelSource::HuggingFace {
            repo: "nomic-ai/nomic-embed-text-v1.5".to_string(),
        },
        offline: false,
        cache_dir: Some(cache.clone()),
        pinned_revision: None,
        network_policy: "allow-download".to_string(),
    });

    let _ = embedder
        .embed(EmbeddingInput::Text("cache override check".to_string()))
        .expect("nomic embed");
    let expected_dir = cache.join("nomic-embed-text-v1.5");
    assert!(
        expected_dir.exists(),
        "Nomic weights must be stored under the supplied cache_dir override"
    );
}

/// The embedder guide discloses model artifacts, registration symbols and first-load requirements.
#[test]
fn cg_readme_choosing_an_embedder_section_discloses_nomic_artifact_huggingface_url_approx_548mb_size_and_approx_three_minutes_first_load_on_25mbps()
 {
    let readme = read_workspace_file("docs/models-and-embedders.md");

    assert!(readme.to_lowercase().contains("choosing an embedder"));
    assert!(readme.contains("nomic-embed-text-v1.5"));
    assert!(readme.contains("candle-embeddinggemma"));
    assert!(readme.contains("reindex_space"));
    assert!(readme.contains("huggingface.co/nomic-ai/nomic-embed-text-v1.5"));
    assert!(
        ["500MB", "547MB", "548MB", "549MB", "550MB", "600MB"]
            .iter()
            .any(|size| readme.contains(size)),
        "The embedder guide must disclose the approximate first-load artifact size"
    );
    assert!(readme.contains("three minutes"));
    assert!(readme.contains("25Mbps"));
    assert!(readme.contains("Store::open_with_embedder_registrations"));
    assert!(readme.contains("default_embeddinggemma_offline_config"));
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

const EXPECTED_STORE_PUBLIC_METHODS: &[&str] = &[
    // Event-history surface (event placement: event placement, adapter and
    // import, event read, Context binding; dry-run `cg import --dry-run` for the prefix check). The
    // `_for_test` hooks are internal-support seams like the `admin_raw_*` ones.
    "add_relationship",
    "admin_overwrite_digest",
    "admin_overwrite_observation",
    "admin_raw_edge_count",
    "admin_raw_edge_scan",
    "admin_raw_has_outcome_edge_count",
    "admin_raw_insert_cites",
    "admin_raw_insert_intention",
    "admin_raw_install_legacy_schema_for_tests",
    "admin_raw_observations_for_entity_in_context",
    "admin_raw_outcomes_for_decision",
    "admin_raw_scan",
    "admin_raw_scan_intention_rows",
    "admin_raw_update_decision_immutable_fields",
    "append_event_batch",
    "archive_decision",
    "archive_entity",
    "archive_intention",
    "audit_query",
    "basis_diff",
    "bind_authenticated_tenant_policy",
    "bind_context",
    "by_entity_decisions",
    "by_entity_intentions",
    "by_intention",
    "check_event_source_prefix",
    "count_vectors_in_space",
    "create_context",
    "create_context_with_binding",
    "create_decision",
    "create_digest",
    "create_entity",
    "create_intention",
    "current_entity_snapshot",
    "db_path",
    "decision_lineage_producer_systems",
    "decision_targets_entity",
    "decision_targets_intention",
    // Deliberate additive Store surface — the session-goal
    // reconciliation writer/reader.
    "decisions_without_serves_in_context",
    "dereference_evidence",
    "diff",
    "drain_reindex_events",
    "edge_exists",
    "enroll_entity_reference",
    "entity_history",
    "entity_history_graph",
    "entity_neighborhood",
    "event_delivery_status",
    "event_history_status",
    "event_placement_state",
    "evidence_availability_log",
    "find_via_digests",
    "forget_event_history",
    "get_context",
    "get_decision",
    "get_digest",
    "get_entity",
    "get_event",
    "get_evidence",
    "get_intention",
    "get_observation",
    // HAS_OUTCOME is derived from the current outcome mapping rather than a
    // retained generic edge, so consumers need one explicit relationship read.
    "has_outcome_relationship",
    "history",
    "import_event_stream",
    "incoming_supersedes",
    "initialize_fresh",
    "intention_at",
    "intention_at_on_branch",
    "intention_snapshot_links",
    "intention_tree",
    "invalidate_decision",
    "last_query_trace",
    "list_contexts",
    "list_embedding_spaces",
    "list_entities",
    "list_entity_reference_embeddings_for_entity",
    "list_events_by_anchor",
    "list_evidence_for_observation",
    "list_intentions_in_context",
    "list_invalidations",
    "list_observations",
    "list_stream_events",
    "maintain_event_search_vectors_for_test",
    "match_entity_reference",
    "open",
    "open_existing_for_change",
    "open_with_consumer_schema",
    "open_with_consumer_schema_scoped",
    "open_with_contexts",
    "open_with_control_handler",
    "open_with_control_handler_and_embedder_registrations",
    "open_with_embedder_registrations",
    "precedent_chain",
    "record_observation",
    "record_outcome",
    "register_embedding_space",
    "reindex_space",
    "remove_entity_references",
    "render_decision_producer_audit_paragraph",
    "render_entity_alert_message_body",
    "replay",
    "resolve_context_binding",
    "retire_event_context_counts_for_test",
    "retire_event_search_head_for_test",
    // Idempotent additive SERVES routing (reconciliation writer).
    "route_decision_to_intention",
    "snapshot_at",
    "state_at",
    "state_at_on_branch",
    "supersede_decision",
    "supersede_history",
    "sync_database",
    "traverse",
    "update_decision",
    "update_decision_basis",
    "update_entity",
    "update_evidence_retention_status",
    "update_intention",
    "update_intention_on_branch",
    "update_intentions_batch",
    "with_principal",
    "with_system_principal",
];

/// Every public trait `source` declares, in declaration order — the mechanical reading of "how
/// many embedding abstractions does this file define".
fn declared_public_trait_names(source: &str) -> Vec<String> {
    source
        .lines()
        .map(str::trim_start)
        .filter_map(|line| line.strip_prefix("pub trait "))
        .map(|rest| {
            rest.split(|character: char| !(character.is_alphanumeric() || character == '_'))
                .next()
                .unwrap_or_default()
                .to_string()
        })
        .collect()
}

/// How many times `source` implements the sealed extension trait — one blanket impl is the
/// contract, and any second implementation is a per-type divergence.
fn blanket_extension_impl_count(source: &str) -> usize {
    source.matches("EventSearchEmbedder for ").count()
}

/// The one-embedding-abstraction guard above is only worth its line count if it actually fires, so
/// this drives both of its detectors over source text that carries the defect. Without this, a
/// detector that silently reported nothing would leave the guard passing on any tree at all.
#[test]
fn the_one_embedding_abstraction_guard_reports_a_second_trait_and_a_second_blanket_impl() {
    let source_with_a_second_embedding_trait = "\
pub trait Embedder: Send + Sync + Any {}

pub trait EventSearchEmbedder: Embedder + sealed::Sealed {}

pub trait EventEmbedder {
    fn embed_event(&self, text: &str) -> Result<EmbeddingOutput>;
}
";
    assert_eq!(
        declared_public_trait_names(source_with_a_second_embedding_trait),
        vec![
            "Embedder".to_string(),
            "EventSearchEmbedder".to_string(),
            "EventEmbedder".to_string()
        ],
        "a second embedding abstraction added beside `Embedder` must be reported, which is what \
         makes the exact-set assertion in the public-surface guard fail instead of passing"
    );

    let source_with_a_second_blanket_impl = "\
impl<T: Embedder + ?Sized> EventSearchEmbedder for T {}

impl EventSearchEmbedder for MyOwnEmbedder {}
";
    assert_eq!(
        blanket_extension_impl_count(source_with_a_second_blanket_impl),
        2,
        "a per-type implementation of the sealed extension trait must be counted, which is what \
         makes the exactly-one assertion in the public-surface guard fail instead of passing"
    );

    // Both detectors inspect the shipped source and must find the declared traits and blanket impl.
    let embedder_source = read_crate_file("src/embedder.rs");
    assert_eq!(declared_public_trait_names(&embedder_source).len(), 2);
    assert_eq!(blanket_extension_impl_count(&embedder_source), 1);
}

#[test]
fn embedder_trait_and_storeconfig_public_surface_diff_keeps_surviving_signatures_and_allowed_moves()
{
    let lib_source = read_crate_file("src/lib.rs");
    let embedder_source = read_crate_file("src/embedder.rs");
    let observer_source = read_crate_file("src/embedder/load_observer.rs");
    let nomic_source = read_crate_file("src/embedder/nomic.rs");
    let error_source = read_crate_file("src/error.rs");
    let types_source = read_crate_file("src/types.rs");
    let store_source = read_store_public_api_sources();
    let public_sources = format!(
        "{lib_source}\n{embedder_source}\n{observer_source}\n{nomic_source}\n{types_source}\n{store_source}"
    );

    for (source, required) in [
        (&embedder_source, "pub struct EmbedderConfig"),
        (&embedder_source, "pub struct EmbedderRegistration"),
        (&embedder_source, "pub enum EmbeddingInput"),
        (&embedder_source, "pub struct EmbeddingOutput"),
        (&observer_source, "pub enum EmbedderLoadEvent"),
        (&observer_source, "pub trait EmbedderLoadObserver"),
        (&nomic_source, "pub struct NomicTextEmbedder"),
        (&types_source, "pub struct StoreConfig"),
        (&store_source, "pub struct Store"),
    ] {
        assert!(
            source.contains(required),
            "public API source missing required item {required}"
        );
    }
    assert!(lib_source.contains("EmbedderLoadEvent"));
    assert!(lib_source.contains("EmbedderLoadObserver"));
    assert!(lib_source.contains("NomicTextEmbedder") || nomic_source.contains("NomicTextEmbedder"));
    for removed_or_forbidden in [
        "EmbeddingGemma",
        "EmbeddingGemmaLoadObserver",
        "EmbeddingGemmaLoadEvent",
        "EmbeddingGemmaTextEmbedder",
        "LazyHandle",
        "default_embeddinggemma_offline",
        "tensor_fingerprints",
        "switch_embedder",
    ] {
        assert!(
            !public_sources.contains(removed_or_forbidden),
            "public API source exposes forbidden item {removed_or_forbidden}"
        );
    }

    assert!(embedder_source.contains("pub trait Embedder: Send + Sync + Any"));
    assert!(
        embedder_source
            .contains("fn embed(&self, input: EmbeddingInput) -> Result<EmbeddingOutput>;")
    );
    assert!(embedder_source.contains("fn supported_inputs(&self) -> Vec<EvidenceKind>;"));
    assert!(embedder_source.contains("fn embedding_space(&self) -> EmbeddingSpace;"));

    // The four checked event-search entrypoints must be callable on any embedder handle a caller
    // happens to be holding — an embedder's own concrete type included, not only through a
    // `dyn Embedder` handle. A sealed extension trait with a single blanket impl makes the
    // entrypoints reachable on concrete types; an inherent `impl dyn Embedder` block cannot.
    // Exact source strings constrain both the required and forbidden declarations.
    assert!(
        !embedder_source.contains("impl dyn Embedder"),
        "the four checked event-search entrypoints must be absent from an inherent `impl dyn \
         Embedder` block — that shape is reachable only through a `dyn Embedder` handle, never \
         directly on an embedder's own concrete type"
    );
    assert!(
        embedder_source.contains("pub trait EventSearchEmbedder"),
        "embedder.rs must declare `pub trait EventSearchEmbedder`, the sealed extension trait \
         that makes the four checked entrypoints callable on any embedder handle, re-exported \
         beside `Embedder`"
    );
    assert!(
        embedder_source.contains("trait Sealed") && !embedder_source.contains("pub trait Sealed"),
        "`EventSearchEmbedder` must be sealed by a private `Sealed` supertrait, never a public \
         one, so no outside crate can implement it"
    );
    assert!(
        embedder_source.contains("impl<T: Embedder + ?Sized>"),
        "`EventSearchEmbedder` must be granted through exactly one blanket impl over every \
         `Embedder`, `impl<T: Embedder + ?Sized> EventSearchEmbedder for T`, never a per-type \
         impl an implementer could diverge from"
    );

    // There is exactly ONE embedding abstraction in this crate, and a second one is a defect, not
    // an extension: two traits mean two ways to produce a vector, two places a caller can be
    // handed an embedder, and no single point where the event-history identity is vouched for.
    // The invariant `crates/context-graph/AGENTS.md` publishes is that `Embedder` is the only
    // embedding trait and `EventSearchEmbedder` its one sealed, library-owned extension — so this
    // asserts the exact set of `pub trait` declarations in the file, not merely that the two
    // expected ones are present. A third public trait violates this exact-set contract.
    let declared_public_traits = declared_public_trait_names(&embedder_source);
    assert_eq!(
        declared_public_traits,
        vec!["Embedder".to_string(), "EventSearchEmbedder".to_string()],
        "src/embedder.rs must declare exactly two public traits — the one embedding trait \
         `Embedder` and its one sealed extension `EventSearchEmbedder`. A third embedding \
         abstraction beside them is the defect this guard exists to catch; found {declared_public_traits:?}"
    );

    // Error documentation names the sealed extension trait and the import needed to call it.
    assert!(
        !error_source.contains("&dyn Embedder"),
        "src/error.rs must not describe the checked event-search entrypoints as living on          `&dyn Embedder`: they are methods of the sealed `EventSearchEmbedder` extension trait,          reachable on any embedder handle once that trait is in scope"
    );

    let blanket_extension_impls = blanket_extension_impl_count(&embedder_source);
    assert_eq!(
        blanket_extension_impls, 1,
        "src/embedder.rs must grant `EventSearchEmbedder` through exactly one blanket impl; a \
         second impl of it is a per-type implementation an embedder could diverge from, which is \
         precisely what sealing the trait is meant to make impossible; found \
         {blanket_extension_impls}"
    );

    assert_eq!(
        vec![
            "cache_dir",
            "network_policy",
            "offline",
            "pinned_revision",
            "source"
        ],
        public_struct_fields(&embedder_source, "EmbedderConfig")
    );
    assert_eq!(
        vec!["default_nomic", "default_nomic_offline", "disabled"],
        public_inherent_methods(&embedder_source, "EmbedderConfig")
    );

    // `event_placement` is the event-placement surface's frozen `StoreConfig` field
    // (event placement, "Event-placement surface").
    assert_eq!(
        vec![
            "db_path",
            "default_text_embedder",
            "event_placement",
            "expected_schema_version",
            "extra_embedding_spaces",
            "file_uri_allowed",
            "https_allowed_hosts",
            "read_active_time_ms",
            "read_memory_limit_bytes",
            "write_memory_limit_bytes",
        ],
        public_struct_fields(&types_source, "StoreConfig")
    );

    assert_eq!(
        vec![
            "new",
            "new_for_dimension",
            "new_with_load_observer",
            "structured_loader_events"
        ],
        public_inherent_methods(&nomic_source, "NomicTextEmbedder")
    );
    assert_eq!(
        Vec::<String>::new(),
        public_struct_fields(&nomic_source, "NomicTextEmbedder")
    );
    assert!(!nomic_source.contains("pub const NOMIC_"));

    let store_methods = public_inherent_methods(&store_source, "Store");
    let store_methods = store_methods.iter().map(String::as_str).collect::<Vec<_>>();
    assert_eq!(EXPECTED_STORE_PUBLIC_METHODS, store_methods.as_slice());
}
