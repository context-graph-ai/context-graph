use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Utc;
use context_graph::embedder::Embedder;
use context_graph::embedder::mock::MockEmbedder;
use context_graph::{
    CG_EDGE_SYSTEM_TAG, CgError, CreateContext, CreateEntity, EmbedderConfig, EmbedderRegistration,
    EmbeddingId, EmbeddingTarget, EntityId, EntityType, EvidenceId, EvidenceKind, EvidenceProducer,
    EvidenceRef, ModelSource, PrecomputedEmbedding, RecordObservation, RetentionStatus, Store,
    StoreConfig,
};
use contextdb_core::Value;
use contextdb_engine::Database;
use safetensors::tensor::{Dtype, TensorView};
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

const TEXT_SPACE: &str = "text_default_nomic_v1_5_768";
const CONSUMER_SPACE: &str = "consumer_trait_text_768";
const TOKEN_EMBEDDING_TENSOR: &str = "model.embed_tokens.weight";
const Q_PROJ_TENSOR: &str = "model.layers.0.self_attn.q_proj.weight";
const K_PROJ_TENSOR: &str = "model.layers.0.self_attn.k_proj.weight";
const V_PROJ_TENSOR: &str = "model.layers.0.self_attn.v_proj.weight";
const O_PROJ_TENSOR: &str = "model.layers.0.self_attn.o_proj.weight";
const Q_NORM_TENSOR: &str = "model.layers.0.self_attn.q_norm.weight";
const K_NORM_TENSOR: &str = "model.layers.0.self_attn.k_norm.weight";
const INPUT_LAYERNORM_TENSOR: &str = "model.layers.0.input_layernorm.weight";
const PRE_FEEDFORWARD_LAYERNORM_TENSOR: &str = "model.layers.0.pre_feedforward_layernorm.weight";
const POST_FEEDFORWARD_LAYERNORM_TENSOR: &str = "model.layers.0.post_feedforward_layernorm.weight";
const POST_ATTENTION_LAYERNORM_TENSOR: &str = "model.layers.0.post_attention_layernorm.weight";
const GATE_PROJ_TENSOR: &str = "model.layers.0.mlp.gate_proj.weight";
const UP_PROJ_TENSOR: &str = "model.layers.0.mlp.up_proj.weight";
const DOWN_PROJ_TENSOR: &str = "model.layers.0.mlp.down_proj.weight";
const FINAL_NORM_TENSOR: &str = "model.norm.weight";
const TEXT_PROJECTION_UP_TENSOR: &str = "model.text_projection.up_proj.weight";
const TEXT_PROJECTION_DOWN_TENSOR: &str = "model.text_projection.down_proj.weight";

fn required_graph_tensors() -> [&'static str; 17] {
    [
        TOKEN_EMBEDDING_TENSOR,
        Q_PROJ_TENSOR,
        K_PROJ_TENSOR,
        V_PROJ_TENSOR,
        O_PROJ_TENSOR,
        Q_NORM_TENSOR,
        K_NORM_TENSOR,
        INPUT_LAYERNORM_TENSOR,
        PRE_FEEDFORWARD_LAYERNORM_TENSOR,
        POST_FEEDFORWARD_LAYERNORM_TENSOR,
        POST_ATTENTION_LAYERNORM_TENSOR,
        GATE_PROJ_TENSOR,
        UP_PROJ_TENSOR,
        DOWN_PROJ_TENSOR,
        FINAL_NORM_TENSOR,
        TEXT_PROJECTION_UP_TENSOR,
        TEXT_PROJECTION_DOWN_TENSOR,
    ]
}

fn stale_11_graph_tensors() -> [&'static str; 11] {
    [
        TOKEN_EMBEDDING_TENSOR,
        Q_PROJ_TENSOR,
        K_PROJ_TENSOR,
        V_PROJ_TENSOR,
        O_PROJ_TENSOR,
        GATE_PROJ_TENSOR,
        UP_PROJ_TENSOR,
        DOWN_PROJ_TENSOR,
        FINAL_NORM_TENSOR,
        TEXT_PROJECTION_UP_TENSOR,
        TEXT_PROJECTION_DOWN_TENSOR,
    ]
}

fn wrong_identity_17_graph_tensors() -> [&'static str; 17] {
    [
        TOKEN_EMBEDDING_TENSOR,
        Q_PROJ_TENSOR,
        K_PROJ_TENSOR,
        V_PROJ_TENSOR,
        O_PROJ_TENSOR,
        "model.layers.0.self_attn.q_norm.bias",
        K_NORM_TENSOR,
        INPUT_LAYERNORM_TENSOR,
        PRE_FEEDFORWARD_LAYERNORM_TENSOR,
        POST_FEEDFORWARD_LAYERNORM_TENSOR,
        POST_ATTENTION_LAYERNORM_TENSOR,
        GATE_PROJ_TENSOR,
        UP_PROJ_TENSOR,
        DOWN_PROJ_TENSOR,
        FINAL_NORM_TENSOR,
        TEXT_PROJECTION_UP_TENSOR,
        TEXT_PROJECTION_DOWN_TENSOR,
    ]
}

fn corrupt_model_error(error: &Option<CgError>) -> bool {
    matches!(
        error,
        Some(CgError::EmbedderModelCorrupt { path })
            if path.contains("embeddinggemma-300m") && !path.trim().is_empty()
    )
}

fn tensor_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn seeded_values(seed: u128, len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            let shift = ((idx % 16) * 8) as u32;
            let byte = ((seed.rotate_left((idx % 64) as u32) >> shift) & 0xff) as f32;
            ((byte + 1.0) / 257.0) * scale
        })
        .collect()
}

struct Fixture {
    _dir: tempfile::TempDir,
    db_path: PathBuf,
    store: Store,
    context_id: context_graph::ContextId,
    entity_id: context_graph::EntityId,
}

fn producer(system: &str, version: &str) -> EvidenceProducer {
    EvidenceProducer {
        system: system.to_string(),
        model_name: "audit-encoder".to_string(),
        model_version: version.to_string(),
        pipeline_version: "pipeline-7".to_string(),
    }
}

fn bundled_config(model_dir: PathBuf, cache_dir: PathBuf) -> EmbedderConfig {
    EmbedderConfig {
        source: ModelSource::LocalPath(model_dir),
        offline: true,
        cache_dir: Some(cache_dir),
        pinned_revision: Some("metadata-rev".to_string()),
        network_policy: "offline-cache-only".to_string(),
    }
}

fn write_tokenizer_and_revision(model_dir: &Path, revision: &[u8]) {
    let vocab = [
        ("<unk>".to_string(), 0),
        ("metadata".to_string(), 1),
        ("path".to_string(), 2),
        ("bundled".to_string(), 3),
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
    tokenizer
        .save(model_dir.join("tokenizer.json"), false)
        .expect("tokenizer");
    std::fs::write(model_dir.join("REVISION"), revision).expect("revision");
}

fn stage_one_tensor_embeddinggemma_cache(cache_dir: &Path) -> PathBuf {
    let model_dir = cache_dir.join("embeddinggemma-300m");
    std::fs::create_dir_all(&model_dir).expect("model dir");
    let tensor_name = format!("metadata.runtime_probe.{}", uuid::Uuid::now_v7().simple());
    let weights = (0..16)
        .flat_map(|idx| ((idx as f32 + 1.0) / 16.0).to_le_bytes())
        .collect::<Vec<_>>();
    let tensor = TensorView::new(Dtype::F32, vec![4, 4], &weights).expect("tensor view");
    let model_bytes = safetensors::tensor::serialize([(tensor_name.as_str(), tensor)], None)
        .expect("safetensors");
    std::fs::write(model_dir.join("model.safetensors"), model_bytes).expect("model");
    write_tokenizer_and_revision(&model_dir, b"metadata-rev");
    std::fs::write(
        model_dir.join("config.json"),
        format!(
            r#"{{"model_type":"embeddinggemma","projection_dim":768,"required_tensor":"{tensor_name}"}}"#
        ),
    )
    .expect("config");
    model_dir
}

fn stage_stale_11_tensor_embeddinggemma_cache(cache_dir: &Path) -> PathBuf {
    let model_dir = cache_dir.join("embeddinggemma-300m");
    std::fs::create_dir_all(&model_dir).expect("model dir");
    let seed = uuid::Uuid::now_v7().as_u128();
    let token_embedding = seeded_values(seed, 16, 1.0);
    let q_proj = seeded_values(seed.rotate_left(1), 4 * 4, 0.2);
    let k_proj = seeded_values(seed.rotate_left(2), 4 * 4, 0.2);
    let v_proj = seeded_values(seed.rotate_left(3), 4 * 4, 0.2);
    let o_proj = seeded_values(seed.rotate_left(4), 4 * 4, 0.2);
    let gate_proj = seeded_values(seed.rotate_left(5), 4 * 8, 0.15);
    let up_proj = seeded_values(seed.rotate_left(6), 4 * 8, 0.15);
    let down_proj = seeded_values(seed.rotate_left(7), 8 * 4, 0.15);
    let final_norm = seeded_values(seed.rotate_left(8), 4, 0.5);
    let text_projection_up = seeded_values(seed.rotate_left(9), 4 * 8, 0.2);
    let text_projection_down = seeded_values(seed.reverse_bits(), 8 * 768, 0.2);

    let token_embedding_bytes = tensor_bytes(&token_embedding);
    let q_proj_bytes = tensor_bytes(&q_proj);
    let k_proj_bytes = tensor_bytes(&k_proj);
    let v_proj_bytes = tensor_bytes(&v_proj);
    let o_proj_bytes = tensor_bytes(&o_proj);
    let gate_proj_bytes = tensor_bytes(&gate_proj);
    let up_proj_bytes = tensor_bytes(&up_proj);
    let down_proj_bytes = tensor_bytes(&down_proj);
    let final_norm_bytes = tensor_bytes(&final_norm);
    let text_projection_up_bytes = tensor_bytes(&text_projection_up);
    let text_projection_down_bytes = tensor_bytes(&text_projection_down);

    let token_tensor =
        TensorView::new(Dtype::F32, vec![4, 4], &token_embedding_bytes).expect("token tensor");
    let q_tensor = TensorView::new(Dtype::F32, vec![4, 4], &q_proj_bytes).expect("q tensor");
    let k_tensor = TensorView::new(Dtype::F32, vec![4, 4], &k_proj_bytes).expect("k tensor");
    let v_tensor = TensorView::new(Dtype::F32, vec![4, 4], &v_proj_bytes).expect("v tensor");
    let o_tensor = TensorView::new(Dtype::F32, vec![4, 4], &o_proj_bytes).expect("o tensor");
    let gate_tensor =
        TensorView::new(Dtype::F32, vec![4, 8], &gate_proj_bytes).expect("gate tensor");
    let up_tensor = TensorView::new(Dtype::F32, vec![4, 8], &up_proj_bytes).expect("up tensor");
    let down_tensor =
        TensorView::new(Dtype::F32, vec![8, 4], &down_proj_bytes).expect("down tensor");
    let final_norm_tensor =
        TensorView::new(Dtype::F32, vec![4], &final_norm_bytes).expect("norm tensor");
    let text_projection_up_tensor =
        TensorView::new(Dtype::F32, vec![4, 8], &text_projection_up_bytes)
            .expect("projection up tensor");
    let text_projection_down_tensor =
        TensorView::new(Dtype::F32, vec![8, 768], &text_projection_down_bytes)
            .expect("projection down tensor");

    let model_bytes = safetensors::tensor::serialize(
        [
            (TOKEN_EMBEDDING_TENSOR, token_tensor),
            (Q_PROJ_TENSOR, q_tensor),
            (K_PROJ_TENSOR, k_tensor),
            (V_PROJ_TENSOR, v_tensor),
            (O_PROJ_TENSOR, o_tensor),
            (GATE_PROJ_TENSOR, gate_tensor),
            (UP_PROJ_TENSOR, up_tensor),
            (DOWN_PROJ_TENSOR, down_tensor),
            (FINAL_NORM_TENSOR, final_norm_tensor),
            (TEXT_PROJECTION_UP_TENSOR, text_projection_up_tensor),
            (TEXT_PROJECTION_DOWN_TENSOR, text_projection_down_tensor),
        ],
        None,
    )
    .expect("safetensors");
    std::fs::write(model_dir.join("model.safetensors"), model_bytes).expect("model");
    write_tokenizer_and_revision(&model_dir, b"metadata-rev");
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::json!({
            "model_type": "embeddinggemma",
            "architectures": ["EmbeddingGemmaTextModel"],
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "intermediate_size": 8,
            "projection_dim": 768,
            "projection_intermediate_size": 8,
            "matryoshka_dimensions": [768, 512, 256, 128],
            "required_tensors": stale_11_graph_tensors(),
        })
        .to_string(),
    )
    .expect("config");
    model_dir
}

fn stage_projection_only_embeddinggemma_cache(cache_dir: &Path) -> PathBuf {
    let model_dir = cache_dir.join("embeddinggemma-300m");
    std::fs::create_dir_all(&model_dir).expect("model dir");
    let seed = uuid::Uuid::now_v7().as_u128();
    let token_embedding = seeded_values(seed, 16, 1.0);
    let text_projection_up = seeded_values(seed.rotate_left(1), 4 * 8, 0.2);
    let text_projection_down = seeded_values(seed.reverse_bits(), 8 * 768, 0.2);

    let token_embedding_bytes = tensor_bytes(&token_embedding);
    let text_projection_up_bytes = tensor_bytes(&text_projection_up);
    let text_projection_down_bytes = tensor_bytes(&text_projection_down);

    let token_tensor =
        TensorView::new(Dtype::F32, vec![4, 4], &token_embedding_bytes).expect("token tensor");
    let text_projection_up_tensor =
        TensorView::new(Dtype::F32, vec![4, 8], &text_projection_up_bytes)
            .expect("projection up tensor");
    let text_projection_down_tensor =
        TensorView::new(Dtype::F32, vec![8, 768], &text_projection_down_bytes)
            .expect("projection down tensor");

    let model_bytes = safetensors::tensor::serialize(
        [
            (TOKEN_EMBEDDING_TENSOR, token_tensor),
            (TEXT_PROJECTION_UP_TENSOR, text_projection_up_tensor),
            (TEXT_PROJECTION_DOWN_TENSOR, text_projection_down_tensor),
        ],
        None,
    )
    .expect("safetensors");
    std::fs::write(model_dir.join("model.safetensors"), model_bytes).expect("model");
    write_tokenizer_and_revision(&model_dir, b"metadata-rev");
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::json!({
            "model_type": "embeddinggemma",
            "architectures": ["EmbeddingGemmaTextModel"],
            "hidden_size": 4,
            "projection_dim": 768,
            "projection_intermediate_size": 8,
            "matryoshka_dimensions": [768, 512, 256, 128],
            "required_tensors": [
                TOKEN_EMBEDDING_TENSOR,
                TEXT_PROJECTION_UP_TENSOR,
                TEXT_PROJECTION_DOWN_TENSOR
            ],
        })
        .to_string(),
    )
    .expect("config");
    model_dir
}

fn stage_wrong_identity_17_tensor_embeddinggemma_cache(cache_dir: &Path) -> PathBuf {
    let model_dir = cache_dir.join("embeddinggemma-300m");
    std::fs::create_dir_all(&model_dir).expect("model dir");
    let seed = uuid::Uuid::now_v7().as_u128();
    let tensor_names = wrong_identity_17_graph_tensors();
    let tensor_values = tensor_names
        .iter()
        .enumerate()
        .map(|(idx, _)| tensor_bytes(&seeded_values(seed.rotate_left((idx as u32) + 1), 4, 0.1)))
        .collect::<Vec<_>>();
    let tensor_views = tensor_names
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            (
                *name,
                TensorView::new(Dtype::F32, vec![4], &tensor_values[idx])
                    .expect("wrong identity tensor"),
            )
        })
        .collect::<Vec<_>>();
    let model_bytes = safetensors::tensor::serialize(tensor_views, None).expect("safetensors");
    std::fs::write(model_dir.join("model.safetensors"), model_bytes).expect("model");
    write_tokenizer_and_revision(&model_dir, b"metadata-rev");
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::json!({
            "model_type": "embeddinggemma",
            "architectures": ["EmbeddingGemmaTextModel"],
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "intermediate_size": 8,
            "projection_dim": 768,
            "projection_intermediate_size": 8,
            "matryoshka_dimensions": [768, 512, 256, 128],
            "required_tensors": required_graph_tensors(),
        })
        .to_string(),
    )
    .expect("config");
    model_dir
}

fn fixture(name: &str) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join(format!("{name}.db"));
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open");
    let context = store
        .create_context(CreateContext {
            name: format!("{name}-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "metadata-target".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    Fixture {
        _dir: dir,
        db_path,
        store,
        context_id: context.id,
        entity_id: entity.id,
    }
}

fn fixture_with_registered_embedder(name: &str, embedder: MockEmbedder) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join(format!("{name}.db"));
    let mut space = embedder.embedding_space();
    space.default_for_evidence_kind = Some(EvidenceKind::Text);
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: db_path.clone(),
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space,
            embedder: Arc::new(embedder),
        }],
    )
    .expect("open");
    let context = store
        .create_context(CreateContext {
            name: format!("{name}-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "metadata-target".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    Fixture {
        _dir: dir,
        db_path,
        store,
        context_id: context.id,
        entity_id: entity.id,
    }
}

fn vector(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 768];
    vector[axis] = 1.0;
    vector
}

fn record(fixture: &Fixture, system: &str, version: &str, axis: usize) -> EmbeddingId {
    let evidence_id = EvidenceId::new_v7();
    let embedding_id = EmbeddingId::new_v7();
    let result = fixture.store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id: fixture.entity_id,
        context_id: fixture.context_id,
        observation_type: "embedder-metadata".to_string(),
        source: "metadata".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::Text,
            source_ref:
                "cas://sha256:4444444444444444444444444444444444444444444444444444444444444444"
                    .to_string(),
            retention_status: RetentionStatus::RetainedExternal,
            producer: producer("capture", "raw"),
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![PrecomputedEmbedding {
            id: embedding_id,
            embedding_space_id: TEXT_SPACE.to_string(),
            vector: vector(axis),
            target: EmbeddingTarget::Evidence(evidence_id),
            produced_by: producer(system, version),
        }],
    });
    assert!(result.is_ok(), "metadata setup failed: {result:?}");
    embedding_id
}

fn raw_db(path: &Path, store: Store) -> Database {
    drop(store);
    Database::open(path).expect("raw database open")
}

#[derive(Debug, Clone)]
struct ProducerProjection {
    space: String,
    producer_json: serde_json::Value,
}

fn producer_projection_by_target(
    db: &Database,
    evidence_id: EvidenceId,
) -> Option<ProducerProjection> {
    let result = db
        .execute(
            "SELECT embedding_space_id, produced_by FROM evidence_embedding_handles WHERE target_id = $target_id",
            &params(vec![("target_id", Value::Uuid(evidence_id.as_uuid()))]),
        )
        .expect("producer projection");
    let row = result.rows.first()?;
    let space = match row.first() {
        Some(Value::Text(space)) => space.clone(),
        _ => return None,
    };
    let producer_json = match row.get(1) {
        Some(Value::Json(value)) => value.clone(),
        _ => return None,
    };
    Some(ProducerProjection {
        space,
        producer_json,
    })
}

fn record_auto_text(
    store: &Store,
    context_id: context_graph::ContextId,
    entity_id: EntityId,
    source: &str,
) -> (
    EvidenceId,
    context_graph::Result<context_graph::RecordObservationResult>,
) {
    let evidence_id = EvidenceId::new_v7();
    let result = store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id,
        context_id,
        observation_type: "embedder-metadata-auto".to_string(),
        source: source.to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::Text,
            source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
            mime_type: Some("text/plain".to_string()),
            retention_status: RetentionStatus::RetainedExternal,
            producer: producer("capture", "raw"),
            text_projection: Some(format!("metadata semantic text {source}")),
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: Vec::new(),
    });
    (evidence_id, result)
}

fn auto_path_projection(
    name: &str,
    config: StoreConfig,
    registrations: Vec<EmbedderRegistration>,
) -> (Option<ProducerProjection>, Option<CgError>, String) {
    let store = if registrations.is_empty() {
        Store::open(config)
    } else {
        Store::open_with_embedder_registrations(config, registrations)
    };
    let store = match store {
        Ok(store) => store,
        Err(error) => {
            let debug = format!("open={error:?}");
            return (None, Some(error), debug);
        }
    };
    let context = match store.create_context(CreateContext {
        name: format!("{name}-{}", uuid::Uuid::now_v7()),
        labels: Vec::new(),
        properties: BTreeMap::new(),
    }) {
        Ok(context) => context,
        Err(error) => {
            let debug = format!("context={error:?}");
            return (None, Some(error), debug);
        }
    };
    let entity = match store.create_entity(CreateEntity {
        entity_type: EntityType::Service,
        name: format!("{name}-entity"),
        properties: BTreeMap::new(),
        tags: Vec::new(),
        context_id: context.id,
    }) {
        Ok(entity) => entity,
        Err(error) => {
            let debug = format!("entity={error:?}");
            return (None, Some(error), debug);
        }
    };
    let db_path = store.db_path().to_path_buf();
    let (evidence_id, result) = record_auto_text(&store, context.id, entity.id, name);
    let output_ids = result
        .as_ref()
        .map(|output| output.embedding_ids.clone())
        .unwrap_or_default();
    let error = result.as_ref().err().cloned();
    drop(store);
    let db = Database::open(&db_path).expect("raw auto-path db");
    let projection = producer_projection_by_target(&db, evidence_id);
    (
        projection,
        error,
        format!("record={result:?}, output_embedding_ids={output_ids:?}"),
    )
}

fn producer_schema(value: &serde_json::Value) -> Option<Vec<(String, &'static str)>> {
    let object = value.as_object()?;
    let mut schema = object
        .iter()
        .map(|(key, value)| {
            let kind = match value {
                serde_json::Value::String(_) => "string",
                serde_json::Value::Number(_) => "number",
                serde_json::Value::Bool(_) => "bool",
                serde_json::Value::Array(_) => "array",
                serde_json::Value::Object(_) => "object",
                serde_json::Value::Null => "null",
            };
            (key.clone(), kind)
        })
        .collect::<Vec<_>>();
    schema.sort();
    Some(schema)
}

fn params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn count(db: &Database, sql: &str, params: &HashMap<String, Value>) -> i64 {
    let result = db.execute(sql, params).expect("count query");
    assert_eq!(result.rows.len(), 1);
    assert!(
        matches!(result.rows[0].first(), Some(Value::Int64(_))),
        "expected count integer, got {:?}",
        result.rows[0].first()
    );
    match result.rows[0][0] {
        Value::Int64(value) => value,
        _ => 0,
    }
}

#[test]
fn embedding_metadata_includes_space_id() {
    let fixture = fixture("metadata-space");
    let embedding_id = record(&fixture, "customer", "1", 0);
    let db = raw_db(&fixture.db_path, fixture.store);
    let rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id AND embedding_space_id = $space",
        &params(vec![
            ("id", Value::Uuid(embedding_id.as_uuid())),
            ("space", Value::Text(TEXT_SPACE.to_string())),
        ]),
    );

    assert!(
        rows == 1,
        "stored embedding metadata must carry its embedding_space_id, rows={rows}"
    );
}

#[test]
fn embedder_metadata_distinguishes_internal_vs_external() {
    let embedder = MockEmbedder::new(TEXT_SPACE, 768);
    let expected_internal = embedder.producer.clone();
    let fixture = fixture_with_registered_embedder("metadata-discriminant", embedder);
    let (_internal_evidence_id, internal_result) = record_auto_text(
        &fixture.store,
        fixture.context_id,
        fixture.entity_id,
        "metadata-discriminant-internal",
    );
    let internal = internal_result
        .as_ref()
        .expect("internal auto embedding")
        .embedding_ids[0];
    let external = record(&fixture, "customer", "1", 1);
    let db = raw_db(&fixture.db_path, fixture.store);
    let internal_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id AND produced_by = $producer",
        &params(vec![
            ("id", Value::Uuid(internal.as_uuid())),
            (
                "producer",
                Value::Json(serde_json::to_value(expected_internal).expect("json")),
            ),
        ]),
    );
    let external_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id AND produced_by = $producer",
        &params(vec![
            ("id", Value::Uuid(external.as_uuid())),
            (
                "producer",
                Value::Json(serde_json::to_value(producer("customer", "1")).expect("json")),
            ),
        ]),
    );

    assert!(
        internal_rows == 1 && external_rows == 1,
        "producer.system must distinguish cg-attested and consumer-claimed embeddings, internal_rows={internal_rows}, external_rows={external_rows}"
    );
}

#[test]
fn producer_metadata_first_party_vs_third_party_distinguishable() {
    let internal = producer(CG_EDGE_SYSTEM_TAG, "1").to_string();
    let external = producer("customer", "1").to_string();
    assert!(
        internal == "audit-encoder 1 (pipeline pipeline-7) - first-party, cg-attested"
            && external == "audit-encoder 1 (pipeline pipeline-7) - consumer-claimed"
            && internal != external,
        "producer Display must expose audit grade, internal={internal:?}, external={external:?}"
    );
}

#[test]
fn producer_metadata_serializes_to_human_readable_string() {
    let rendered = producer("customer", "2").to_string();
    assert!(
        rendered == "audit-encoder 2 (pipeline pipeline-7) - consumer-claimed",
        "producer Display must be exactly the producer-owned shape with no fabricated space/txid/date, rendered={rendered:?}"
    );
}

#[test]
fn producer_metadata_uniform_schema_across_three_ingest_paths() {
    let one_tensor_dir = tempfile::tempdir().expect("one tensor dir");
    let one_tensor_model = stage_one_tensor_embeddinggemma_cache(one_tensor_dir.path());
    let one_tensor_config = StoreConfig {
        db_path: one_tensor_dir.path().join("one-tensor.db"),
        default_text_embedder: Some(bundled_config(
            one_tensor_model,
            one_tensor_dir.path().to_path_buf(),
        )),
        ..Default::default()
    };
    let (one_tensor_projection, one_tensor_error, one_tensor_debug) = auto_path_projection(
        "metadata-one-tensor-rejected",
        one_tensor_config,
        Vec::new(),
    );

    let stale_11_dir = tempfile::tempdir().expect("stale 11 tensor dir");
    let stale_11_model = stage_stale_11_tensor_embeddinggemma_cache(stale_11_dir.path());
    let stale_11_config = StoreConfig {
        db_path: stale_11_dir.path().join("stale-11-tensor.db"),
        default_text_embedder: Some(bundled_config(
            stale_11_model,
            stale_11_dir.path().to_path_buf(),
        )),
        ..Default::default()
    };
    let (stale_11_projection, stale_11_error, stale_11_debug) = auto_path_projection(
        "metadata-stale-11-tensor-rejected",
        stale_11_config,
        Vec::new(),
    );

    let projection_only_dir = tempfile::tempdir().expect("projection only dir");
    let projection_only_model =
        stage_projection_only_embeddinggemma_cache(projection_only_dir.path());
    let projection_only_config = StoreConfig {
        db_path: projection_only_dir.path().join("projection-only.db"),
        default_text_embedder: Some(bundled_config(
            projection_only_model,
            projection_only_dir.path().to_path_buf(),
        )),
        ..Default::default()
    };
    let (projection_only_projection, projection_only_error, projection_only_debug) =
        auto_path_projection(
            "metadata-projection-only-rejected",
            projection_only_config,
            Vec::new(),
        );

    let wrong_identity_dir = tempfile::tempdir().expect("wrong identity dir");
    let wrong_identity_model =
        stage_wrong_identity_17_tensor_embeddinggemma_cache(wrong_identity_dir.path());
    let wrong_identity_config = StoreConfig {
        db_path: wrong_identity_dir.path().join("wrong-identity-17.db"),
        default_text_embedder: Some(bundled_config(
            wrong_identity_model,
            wrong_identity_dir.path().to_path_buf(),
        )),
        ..Default::default()
    };
    let (wrong_identity_projection, wrong_identity_error, wrong_identity_debug) =
        auto_path_projection(
            "metadata-wrong-identity-17-rejected",
            wrong_identity_config,
            Vec::new(),
        );

    let bundled_dir = tempfile::tempdir().expect("bundled dir");
    let bundled_config = StoreConfig {
        db_path: bundled_dir.path().join("bundled.db"),
        ..Default::default()
    };
    let (bundled, _bundled_error, bundled_debug) =
        auto_path_projection("metadata-bundled", bundled_config, Vec::new());

    let consumer_dir = tempfile::tempdir().expect("consumer dir");
    let consumer_embedder = MockEmbedder::new(CONSUMER_SPACE, 768);
    let mut registered_space = consumer_embedder.embedding_space();
    registered_space.default_for_evidence_kind = Some(EvidenceKind::Text);
    let consumer_config = StoreConfig {
        db_path: consumer_dir.path().join("consumer.db"),
        extra_embedding_spaces: vec![registered_space.clone()],
        ..Default::default()
    };
    let (consumer_trait, _consumer_error, consumer_debug) = auto_path_projection(
        "metadata-consumer-trait",
        consumer_config,
        vec![EmbedderRegistration {
            space: registered_space,
            embedder: Arc::new(consumer_embedder),
        }],
    );

    let fixture = fixture("metadata-uniform-precomputed");
    let precomputed_evidence_id = EvidenceId::new_v7();
    let precomputed_embedding_id = EmbeddingId::new_v7();
    let precomputed_result = fixture.store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id: fixture.entity_id,
        context_id: fixture.context_id,
        observation_type: "embedder-metadata-precomputed".to_string(),
        source: "metadata-precomputed".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: precomputed_evidence_id,
            kind: EvidenceKind::Text,
            source_ref: format!(
                "cas://sha256:{:064x}",
                precomputed_evidence_id.as_uuid().as_u128()
            ),
            retention_status: RetentionStatus::RetainedExternal,
            producer: producer("capture", "raw"),
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![PrecomputedEmbedding {
            id: precomputed_embedding_id,
            embedding_space_id: TEXT_SPACE.to_string(),
            vector: vector(2),
            target: EmbeddingTarget::Evidence(precomputed_evidence_id),
            produced_by: producer("customer", "precomputed"),
        }],
    });
    let db = raw_db(&fixture.db_path, fixture.store);
    let precomputed = producer_projection_by_target(&db, precomputed_evidence_id);
    let bundled_schema = bundled
        .as_ref()
        .and_then(|projection| producer_schema(&projection.producer_json));
    let consumer_schema = consumer_trait
        .as_ref()
        .and_then(|projection| producer_schema(&projection.producer_json));
    let precomputed_schema = precomputed
        .as_ref()
        .and_then(|projection| producer_schema(&projection.producer_json));
    let expected_schema = vec![
        ("model_name".to_string(), "string"),
        ("model_version".to_string(), "string"),
        ("pipeline_version".to_string(), "string"),
        ("system".to_string(), "string"),
    ];
    assert!(
        one_tensor_projection.is_none()
            && corrupt_model_error(&one_tensor_error)
            && stale_11_projection.is_none()
            && corrupt_model_error(&stale_11_error)
            && projection_only_projection.is_none()
            && corrupt_model_error(&projection_only_error)
            && wrong_identity_projection.is_none()
            && corrupt_model_error(&wrong_identity_error)
            && bundled
                .as_ref()
                .is_some_and(|projection| projection.space == TEXT_SPACE)
            && consumer_trait
                .as_ref()
                .is_some_and(|projection| projection.space == CONSUMER_SPACE)
            && precomputed
                .as_ref()
                .is_some_and(|projection| projection.space == TEXT_SPACE)
            && bundled_schema == Some(expected_schema.clone())
            && consumer_schema == Some(expected_schema.clone())
            && precomputed_schema == Some(expected_schema)
            && bundled.as_ref().is_some_and(|projection| !projection
                .producer_json
                .as_object()
                .is_some_and(|object| object.contains_key("producer_kind")))
            && consumer_trait.as_ref().is_some_and(|projection| !projection
                .producer_json
                .as_object()
                .is_some_and(|object| object.contains_key("producer_kind")))
            && precomputed.as_ref().is_some_and(|projection| !projection
                .producer_json
                .as_object()
                .is_some_and(|object| object.contains_key("producer_kind"))),
        "all producer paths must reject one-tensor, stale-11-tensor, projection-only, and same-cardinality wrong-identity fake EmbeddingGemma caches with typed EmbedderModelCorrupt and land real producer rows with one uniform JSON schema, one_tensor_projection={one_tensor_projection:?}, one_tensor_error={one_tensor_error:?}, one_tensor_debug={one_tensor_debug}, stale_11_projection={stale_11_projection:?}, stale_11_error={stale_11_error:?}, stale_11_debug={stale_11_debug}, projection_only_projection={projection_only_projection:?}, projection_only_error={projection_only_error:?}, projection_only_debug={projection_only_debug}, wrong_identity_projection={wrong_identity_projection:?}, wrong_identity_error={wrong_identity_error:?}, wrong_identity_debug={wrong_identity_debug}, bundled={bundled:?}, bundled_schema={bundled_schema:?}, bundled_debug={bundled_debug}, consumer_trait={consumer_trait:?}, consumer_schema={consumer_schema:?}, consumer_debug={consumer_debug}, precomputed={precomputed:?}, precomputed_schema={precomputed_schema:?}, precomputed_result={precomputed_result:?}"
    );
}
