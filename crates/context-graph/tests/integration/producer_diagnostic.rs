use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(context_graph_internal_test_support)]
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
#[cfg(context_graph_internal_test_support)]
use std::time::Duration;

use chrono::Utc;
#[cfg(context_graph_internal_test_support)]
use context_graph::ObservationId;
use context_graph::embedder::Embedder;
use context_graph::embedder::mock::MockEmbedder;
#[cfg(context_graph_internal_test_support)]
use context_graph::store::signatures::{
    SignatureAggregate, SignatureProducerContribution, SignatureProducerLineage,
    SignatureProducerLineageRow, persist_signature_aggregate, persist_signature_producer_lineage,
    signature_match_full_chain_producers,
};
use context_graph::{
    CG_EDGE_SYSTEM_TAG, CgError, CreateContext, CreateDecision, CreateEntity, CreateIntention,
    EmbedderConfig, EmbedderRegistration, EmbeddingId, EmbeddingTarget, EntityId, EntityType,
    EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, IntentionOrigin, IntentionStatus,
    ModelSource, PrecomputedEmbedding, RecordObservation, RetentionStatus, Store, StoreConfig,
};
use contextdb_core::Value;
use contextdb_engine::Database;
#[cfg(context_graph_internal_test_support)]
use contextdb_engine::plugin::{DatabasePlugin, QueryOutcome};
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

#[cfg(context_graph_internal_test_support)]
#[derive(Default)]
struct QueryCountingPlugin {
    enabled: AtomicBool,
    count: AtomicUsize,
    sql: Mutex<Vec<String>>,
}

#[cfg(context_graph_internal_test_support)]
impl QueryCountingPlugin {
    fn enable_for_next_call(&self) {
        self.count.store(0, Ordering::SeqCst);
        self.sql.lock().expect("query sql").clear();
        self.enabled.store(true, Ordering::SeqCst);
    }

    fn disable(&self) {
        self.enabled.store(false, Ordering::SeqCst);
    }

    fn count(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }

    fn sql(&self) -> Vec<String> {
        self.sql.lock().expect("query sql").clone()
    }
}

#[cfg(context_graph_internal_test_support)]
impl DatabasePlugin for QueryCountingPlugin {
    fn on_query(&self, sql: &str) -> contextdb_core::Result<()> {
        if self.enabled.load(Ordering::SeqCst) {
            self.count.fetch_add(1, Ordering::SeqCst);
            self.sql.lock().expect("query sql").push(sql.to_string());
        }
        Ok(())
    }

    fn post_query(&self, _sql: &str, _duration: Duration, _outcome: &QueryOutcome) {}
}

struct Fixture {
    _dir: tempfile::TempDir,
    db_path: PathBuf,
    store: Store,
    context_id: context_graph::ContextId,
    entity_id: context_graph::EntityId,
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
            name: "diagnostic-target".to_string(),
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

fn producer(system: &str, version: &str) -> EvidenceProducer {
    EvidenceProducer {
        system: system.to_string(),
        model_name: "embeddinggemma-300m".to_string(),
        model_version: version.to_string(),
        pipeline_version: "producer-diagnostic".to_string(),
    }
}

fn bundled_config(model_dir: PathBuf, cache_dir: PathBuf) -> EmbedderConfig {
    EmbedderConfig {
        source: ModelSource::LocalPath(model_dir),
        offline: true,
        cache_dir: Some(cache_dir),
        pinned_revision: Some("producer-rev".to_string()),
        network_policy: "offline-cache-only".to_string(),
    }
}

fn write_tokenizer_and_revision(model_dir: &Path, revision: &[u8]) {
    let vocab = [
        ("<unk>".to_string(), 0),
        ("producer".to_string(), 1),
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
    let tensor_name = format!("producer.runtime_probe.{}", uuid::Uuid::now_v7().simple());
    let weights = (0..16)
        .flat_map(|idx| ((idx as f32 + 1.0) / 16.0).to_le_bytes())
        .collect::<Vec<_>>();
    let tensor = TensorView::new(Dtype::F32, vec![4, 4], &weights).expect("tensor view");
    let model_bytes = safetensors::tensor::serialize([(tensor_name.as_str(), tensor)], None)
        .expect("safetensors");
    std::fs::write(model_dir.join("model.safetensors"), model_bytes).expect("model");
    write_tokenizer_and_revision(&model_dir, b"producer-rev");
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
    write_tokenizer_and_revision(&model_dir, b"producer-rev");
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
    write_tokenizer_and_revision(&model_dir, b"producer-rev");
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
    write_tokenizer_and_revision(&model_dir, b"producer-rev");
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

fn vector(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 768];
    vector[axis] = 1.0;
    vector
}

#[cfg(context_graph_internal_test_support)]
#[derive(Clone)]
struct RecordedEmbedding {
    observation_id: ObservationId,
    evidence_id: EvidenceId,
    embedding_id: EmbeddingId,
    embedding_space_id: String,
    producer: EvidenceProducer,
}

#[cfg(context_graph_internal_test_support)]
fn record_full_with_system(
    fixture: &Fixture,
    system: &str,
    version: &str,
    axis: usize,
) -> RecordedEmbedding {
    let evidence_id = EvidenceId::new_v7();
    let embedding_id = EmbeddingId::new_v7();
    let produced_by = producer(system, version);
    let result = fixture.store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id: fixture.entity_id,
        context_id: fixture.context_id,
        observation_type: "producer-diagnostic".to_string(),
        source: "edge".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::Text,
            source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
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
            produced_by: produced_by.clone(),
        }],
    });
    assert!(
        result.is_ok(),
        "producer diagnostic setup failed: {result:?}"
    );
    let result = result.expect("recorded embedding");
    RecordedEmbedding {
        observation_id: result.observation_id,
        evidence_id,
        embedding_id,
        embedding_space_id: TEXT_SPACE.to_string(),
        producer: produced_by,
    }
}

fn record_with_system(
    fixture: &Fixture,
    system: &str,
    version: &str,
    axis: usize,
) -> (EvidenceId, EmbeddingId) {
    let evidence_id = EvidenceId::new_v7();
    let embedding_id = EmbeddingId::new_v7();
    let result = fixture.store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id: fixture.entity_id,
        context_id: fixture.context_id,
        observation_type: "producer-diagnostic".to_string(),
        source: "edge".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::Text,
            source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
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
    assert!(
        result.is_ok(),
        "producer diagnostic setup failed: {result:?}"
    );
    (evidence_id, embedding_id)
}

fn record(fixture: &Fixture, version: &str, axis: usize) -> (EvidenceId, EmbeddingId) {
    record_with_system(fixture, "edge-memory", version, axis)
}

fn create_rendered_decision_for_snapshot(
    store: &Store,
    context_id: context_graph::ContextId,
    snapshot_id: context_graph::EntitySnapshotId,
    name: &str,
) -> Option<String> {
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: format!("{name} producer display intention"),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id,
            properties: BTreeMap::new(),
            tags: Vec::new(),
            blueprint_catalog_id: None,
        })
        .ok()?;
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "producer_display".to_string(),
            description: format!("{name} producer display decision"),
            confidence: Some(0.9),
            intention_ids: vec![intention.id],
            based_on_entity_ids: Vec::new(),
            based_on_snapshots: vec![snapshot_id],
            context_id,
            ..Default::default()
        })
        .ok()?;
    store
        .render_decision_producer_audit_paragraph(decision.id)
        .ok()
}

fn record_precomputed_basis_with_system(
    fixture: &Fixture,
    system: &str,
    version: &str,
    axis: usize,
    name: &str,
) -> (EvidenceId, EmbeddingId, Option<String>) {
    let evidence_id = EvidenceId::new_v7();
    let embedding_id = EmbeddingId::new_v7();
    let mut state_delta = BTreeMap::new();
    state_delta.insert(
        format!("producer_display_{}", evidence_id.as_uuid().simple()),
        Some(serde_json::json!(system)),
    );
    let result = fixture.store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id: fixture.entity_id,
        context_id: fixture.context_id,
        observation_type: "producer-diagnostic-display".to_string(),
        source: "edge".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::Text,
            source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
            retention_status: RetentionStatus::RetainedExternal,
            producer: producer("capture", "raw"),
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta,
        properties: BTreeMap::new(),
        embeddings: vec![PrecomputedEmbedding {
            id: embedding_id,
            embedding_space_id: TEXT_SPACE.to_string(),
            vector: vector(axis),
            target: EmbeddingTarget::Evidence(evidence_id),
            produced_by: producer(system, version),
        }],
    });
    assert!(
        result.is_ok(),
        "producer diagnostic display setup failed: {result:?}"
    );
    let rendered = result
        .as_ref()
        .ok()
        .and_then(|output| output.snapshot_id)
        .and_then(|snapshot_id| {
            create_rendered_decision_for_snapshot(
                &fixture.store,
                fixture.context_id,
                snapshot_id,
                name,
            )
        });
    (evidence_id, embedding_id, rendered)
}

#[cfg(context_graph_internal_test_support)]
fn contribution(role: &str, recorded: &RecordedEmbedding) -> SignatureProducerContribution {
    SignatureProducerContribution {
        role: role.to_string(),
        observation_id: recorded.observation_id,
        evidence_id: recorded.evidence_id,
        embedding_id: recorded.embedding_id,
        embedding_space_id: recorded.embedding_space_id.clone(),
        producer: recorded.producer.clone(),
    }
}

#[cfg(context_graph_internal_test_support)]
fn aggregate(
    signature_id: &str,
    centroid: &RecordedEmbedding,
    member_count: usize,
) -> SignatureAggregate {
    SignatureAggregate {
        signature_id: signature_id.to_string(),
        embedding_space_id: centroid.embedding_space_id.clone(),
        producer: centroid.producer.clone(),
        centroid: vector(0),
        member_count,
        scope_labels: vec![
            "org:acme".to_string(),
            "scope:signature-diagnostic".to_string(),
        ],
    }
}

#[cfg(context_graph_internal_test_support)]
fn expected_lineage_row(
    signature_id: &str,
    role: &str,
    recorded: &RecordedEmbedding,
) -> SignatureProducerLineageRow {
    SignatureProducerLineageRow {
        signature_id: signature_id.to_string(),
        role: role.to_string(),
        observation_id: recorded.observation_id,
        evidence_id: recorded.evidence_id,
        embedding_id: recorded.embedding_id,
        embedding_space_id: recorded.embedding_space_id.clone(),
        producer: recorded.producer.clone(),
        created_at: context_graph::TxId(0),
    }
}

#[cfg(context_graph_internal_test_support)]
fn same_lineage_ignoring_txid(
    actual: &SignatureProducerLineageRow,
    expected: &SignatureProducerLineageRow,
) -> bool {
    actual.signature_id == expected.signature_id
        && actual.role == expected.role
        && actual.observation_id == expected.observation_id
        && actual.evidence_id == expected.evidence_id
        && actual.embedding_id == expected.embedding_id
        && actual.embedding_space_id == expected.embedding_space_id
        && actual.producer == expected.producer
        && actual.created_at.0 > 0
}

#[cfg(context_graph_internal_test_support)]
fn raw_signature_lineage_rows(
    db: &Database,
    signature_id: &str,
) -> contextdb_core::Result<Vec<SignatureProducerLineageRow>> {
    let result = db.execute(
        "SELECT signature_id, contribution_role, observation_id, evidence_id, embedding_id, embedding_space_id, producer, created_at FROM signature_producer_lineage WHERE signature_id = $signature_id ORDER BY contribution_role ASC, observation_id ASC",
        &params(vec![("signature_id", Value::Text(signature_id.to_string()))]),
    )?;
    let mut rows = Vec::new();
    for row in result.rows {
        let (
            Some(Value::Text(signature_id)),
            Some(Value::Text(role)),
            Some(Value::Uuid(observation_id)),
            Some(Value::Uuid(evidence_id)),
            Some(Value::Uuid(embedding_id)),
            Some(Value::Text(embedding_space_id)),
            Some(Value::Json(producer_json)),
            Some(Value::TxId(created_at)),
        ) = (
            row.first(),
            row.get(1),
            row.get(2),
            row.get(3),
            row.get(4),
            row.get(5),
            row.get(6),
            row.get(7),
        )
        else {
            continue;
        };
        let Ok(producer) = serde_json::from_value::<EvidenceProducer>(producer_json.clone()) else {
            continue;
        };
        rows.push(SignatureProducerLineageRow {
            signature_id: signature_id.clone(),
            role: role.clone(),
            observation_id: (*observation_id).into(),
            evidence_id: (*evidence_id).into(),
            embedding_id: (*embedding_id).into(),
            embedding_space_id: embedding_space_id.clone(),
            producer,
            created_at: *created_at,
        });
    }
    Ok(rows)
}

fn raw_db(path: &Path, store: Store) -> Database {
    drop(store);
    Database::open(path).expect("raw database open")
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

#[derive(Debug)]
struct HandleProducerRow {
    space: String,
    producer: EvidenceProducer,
    txid: context_graph::TxId,
}

fn producer_display(space: &str, producer: &EvidenceProducer, txid: context_graph::TxId) -> String {
    let audit_grade = if producer.system == CG_EDGE_SYSTEM_TAG {
        "first-party, cg-attested"
    } else {
        "consumer-claimed"
    };
    format!(
        "{} {} ({}) - {}, pipeline {}, write txid 0x{:x}",
        producer.model_name,
        producer.model_version,
        space,
        audit_grade,
        producer.pipeline_version,
        txid.0
    )
}

fn handle_producer_row(db: &Database, embedding_id: EmbeddingId) -> Option<HandleProducerRow> {
    let result = db
        .execute(
            "SELECT embedding_space_id, produced_by, created_at FROM evidence_embedding_handles WHERE id = $id",
            &params(vec![("id", Value::Uuid(embedding_id.as_uuid()))]),
        )
        .expect("producer row");
    let row = result.rows.first()?;
    let space = match row.first() {
        Some(Value::Text(space)) => space.clone(),
        _ => return None,
    };
    let producer = match row.get(1) {
        Some(Value::Json(value)) => {
            serde_json::from_value::<EvidenceProducer>(value.clone()).ok()?
        }
        _ => return None,
    };
    let txid = match row.get(2) {
        Some(Value::TxId(txid)) => *txid,
        _ => return None,
    };
    Some(HandleProducerRow {
        space,
        producer,
        txid,
    })
}

fn handle_producer_row_by_target(
    db: &Database,
    evidence_id: EvidenceId,
) -> Option<HandleProducerRow> {
    let result = db
        .execute(
            "SELECT embedding_space_id, produced_by, created_at FROM evidence_embedding_handles WHERE target_id = $target_id",
            &params(vec![("target_id", Value::Uuid(evidence_id.as_uuid()))]),
        )
        .expect("producer row by target");
    let row = result.rows.first()?;
    let space = match row.first() {
        Some(Value::Text(space)) => space.clone(),
        _ => return None,
    };
    let producer = match row.get(1) {
        Some(Value::Json(value)) => {
            serde_json::from_value::<EvidenceProducer>(value.clone()).ok()?
        }
        _ => return None,
    };
    let txid = match row.get(2) {
        Some(Value::TxId(txid)) => *txid,
        _ => return None,
    };
    Some(HandleProducerRow {
        space,
        producer,
        txid,
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
    let mut state_delta = BTreeMap::new();
    state_delta.insert(
        format!("producer_display_{}", evidence_id.as_uuid().simple()),
        Some(serde_json::json!(source)),
    );
    let result = store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id,
        context_id,
        observation_type: "producer-diagnostic-auto".to_string(),
        source: source.to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::Text,
            source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
            mime_type: Some("text/plain".to_string()),
            retention_status: RetentionStatus::RetainedExternal,
            producer: producer("capture", "raw"),
            text_projection: Some(format!("producer diagnostic semantic text {source}")),
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta,
        properties: BTreeMap::new(),
        embeddings: Vec::new(),
    });
    (evidence_id, result)
}

fn auto_path_row(
    name: &str,
    config: StoreConfig,
    registrations: Vec<EmbedderRegistration>,
) -> (
    Option<HandleProducerRow>,
    Option<String>,
    Option<CgError>,
    String,
) {
    let store = if registrations.is_empty() {
        Store::open(config)
    } else {
        Store::open_with_embedder_registrations(config, registrations)
    };
    let store = match store {
        Ok(store) => store,
        Err(error) => {
            let debug = format!("open={error:?}");
            return (None, None, Some(error), debug);
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
            return (None, None, Some(error), debug);
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
            return (None, None, Some(error), debug);
        }
    };
    let db_path = store.db_path().to_path_buf();
    let (evidence_id, result) = record_auto_text(&store, context.id, entity.id, name);
    let output_ids = result
        .as_ref()
        .map(|output| output.embedding_ids.clone())
        .unwrap_or_default();
    let error = result.as_ref().err().cloned();
    let rendered = result
        .as_ref()
        .ok()
        .filter(|output| !output.embedding_ids.is_empty())
        .and_then(|output| output.snapshot_id)
        .and_then(|snapshot_id| {
            create_rendered_decision_for_snapshot(&store, context.id, snapshot_id, name)
        });
    drop(store);
    let db = Database::open(&db_path).expect("raw auto producer db");
    let row = handle_producer_row_by_target(&db, evidence_id);
    (
        row,
        rendered,
        error,
        format!("record={result:?}, output_embedding_ids={output_ids:?}"),
    )
}

#[test]
fn producer_metadata_query_distinguishes_model_version_within_same_embedding_space_id() {
    let fixture = fixture("producer-version");
    let (_, first_embedding) = record(&fixture, "v1", 0);
    let (_, second_embedding) = record(&fixture, "v2", 1);
    let db = raw_db(&fixture.db_path, fixture.store);
    let first_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id AND embedding_space_id = $space AND produced_by = $producer",
        &params(vec![
            ("id", Value::Uuid(first_embedding.as_uuid())),
            ("space", Value::Text(TEXT_SPACE.to_string())),
            (
                "producer",
                Value::Json(serde_json::to_value(producer("edge-memory", "v1")).expect("json")),
            ),
        ]),
    );
    let second_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id AND embedding_space_id = $space AND produced_by = $producer",
        &params(vec![
            ("id", Value::Uuid(second_embedding.as_uuid())),
            ("space", Value::Text(TEXT_SPACE.to_string())),
            (
                "producer",
                Value::Json(serde_json::to_value(producer("edge-memory", "v2")).expect("json")),
            ),
        ]),
    );

    assert!(
        first_rows == 1 && second_rows == 1,
        "same embedding space must retain distinct producer model versions, first_rows={first_rows}, second_rows={second_rows}"
    );
}

#[test]
fn producer_metadata_query_explain_uses_indexed_projection_on_system_field() {
    let fixture = fixture("producer-system-projection");
    let (_, embedding_id) = record(&fixture, "v1", 0);
    let db = raw_db(&fixture.db_path, fixture.store);
    let result = db
        .execute(
            "SELECT id FROM evidence_embedding_handles WHERE producer_system = $producer_system",
            &params(vec![(
                "producer_system",
                Value::Text("edge-memory".to_string()),
            )]),
        )
        .expect("producer projection query");
    let matching_ids = result
        .rows
        .iter()
        .filter_map(|row| match row.first() {
            Some(Value::Uuid(id)) => Some(EmbeddingId::from(*id)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let trace = result.trace;
    let pushed = trace
        .predicates_pushed
        .iter()
        .map(|predicate| predicate.as_ref())
        .collect::<Vec<_>>();
    let expected_index = "idx_evidence_embedding_handles_producer_system";
    let indexed_by_producer_system = trace.physical_plan == "IndexScan"
        && trace.index_used.as_deref() == Some(expected_index)
        && pushed.contains(&"producer_system");
    let exact_row_count = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id",
        &params(vec![("id", Value::Uuid(embedding_id.as_uuid()))]),
    );

    assert!(
        exact_row_count == 1 && matching_ids == vec![embedding_id] && indexed_by_producer_system,
        "producer diagnostic must be queryable by indexed producer.system projection, exact_row_count={exact_row_count}, matching_ids={matching_ids:?}, trace={trace:?}, pushed={pushed:?}"
    );
}

#[test]
fn producer_metadata_display_format_uniform_across_cg_bundled_consumer_trait_and_precomputed_paths()
{
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
    let (one_tensor_row, one_tensor_rendered, one_tensor_error, one_tensor_debug) = auto_path_row(
        "producer-display-one-tensor-rejected",
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
    let (stale_11_row, stale_11_rendered, stale_11_error, stale_11_debug) = auto_path_row(
        "producer-display-stale-11-tensor-rejected",
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
    let (
        projection_only_row,
        projection_only_rendered,
        projection_only_error,
        projection_only_debug,
    ) = auto_path_row(
        "producer-display-projection-only-rejected",
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
    let (wrong_identity_row, wrong_identity_rendered, wrong_identity_error, wrong_identity_debug) =
        auto_path_row(
            "producer-display-wrong-identity-17-rejected",
            wrong_identity_config,
            Vec::new(),
        );

    let bundled_dir = tempfile::tempdir().expect("bundled dir");
    let bundled_config = StoreConfig {
        db_path: bundled_dir.path().join("bundled.db"),
        ..Default::default()
    };
    let (first_party_row, first_party_rendered, _first_party_error, first_party_debug) =
        auto_path_row("producer-display-bundled", bundled_config, Vec::new());

    let consumer_dir = tempfile::tempdir().expect("consumer dir");
    let consumer_embedder = MockEmbedder::new(CONSUMER_SPACE, 768);
    let mut registered_space = consumer_embedder.embedding_space();
    registered_space.default_for_evidence_kind = Some(EvidenceKind::Text);
    let consumer_config = StoreConfig {
        db_path: consumer_dir.path().join("consumer.db"),
        extra_embedding_spaces: vec![registered_space.clone()],
        ..Default::default()
    };
    let (consumer_trait_row, consumer_trait_rendered, _consumer_error, consumer_debug) =
        auto_path_row(
            "producer-display-consumer-trait",
            consumer_config,
            vec![EmbedderRegistration {
                space: registered_space,
                embedder: Arc::new(consumer_embedder),
            }],
        );

    let fixture = fixture("producer-display-precomputed");
    let (_, precomputed_embedding, precomputed_rendered) =
        record_precomputed_basis_with_system(&fixture, "customer", "2026-05-17", 2, "precomputed");
    let db = raw_db(&fixture.db_path, fixture.store);
    let precomputed_row = handle_producer_row(&db, precomputed_embedding);
    let first_party_expected = first_party_row
        .as_ref()
        .map(|row| producer_display(&row.space, &row.producer, row.txid))
        .unwrap_or_default();
    let consumer_trait_expected = consumer_trait_row
        .as_ref()
        .map(|row| producer_display(&row.space, &row.producer, row.txid))
        .unwrap_or_default();
    let precomputed_expected = precomputed_row
        .as_ref()
        .map(|row| producer_display(&row.space, &row.producer, row.txid))
        .unwrap_or_default();
    assert!(
        one_tensor_row.is_none()
            && one_tensor_rendered.is_none()
            && corrupt_model_error(&one_tensor_error)
            && stale_11_row.is_none()
            && stale_11_rendered.is_none()
            && corrupt_model_error(&stale_11_error)
            && projection_only_row.is_none()
            && projection_only_rendered.is_none()
            && corrupt_model_error(&projection_only_error)
            && wrong_identity_row.is_none()
            && wrong_identity_rendered.is_none()
            && corrupt_model_error(&wrong_identity_error)
            && first_party_rendered
                .as_ref()
                .is_some_and(|actual| actual == &first_party_expected)
            && consumer_trait_rendered
                .as_ref()
                .is_some_and(|actual| actual == &consumer_trait_expected)
            && precomputed_rendered
                .as_ref()
                .is_some_and(|actual| actual == &precomputed_expected)
            && !first_party_expected.contains('\n')
            && !consumer_trait_expected.contains('\n')
            && !precomputed_expected.contains('\n')
            && !first_party_expected.contains("produced ")
            && !consumer_trait_expected.contains("produced ")
            && !precomputed_expected.contains("produced ")
            && first_party_expected.matches(" - ").count() == 1
            && consumer_trait_expected.matches(" - ").count() == 1
            && precomputed_expected.matches(" - ").count() == 1
            && first_party_row.as_ref().is_some_and(
                |row| row.space == TEXT_SPACE && row.producer.system == CG_EDGE_SYSTEM_TAG
            )
            && consumer_trait_row.as_ref().is_some_and(
                |row| row.space == CONSUMER_SPACE && row.producer.system == CG_EDGE_SYSTEM_TAG
            )
            && precomputed_row
                .as_ref()
                .is_some_and(|row| row.space == TEXT_SPACE && row.producer.system == "customer"),
        "producer display must reject one-tensor, stale-11-tensor, projection-only, and same-cardinality wrong-identity fake EmbeddingGemma caches with typed EmbedderModelCorrupt and production-render persisted space, audit grade, pipeline, model version, and lowercase write txid across actual bundled, consumer-trait, and precomputed ingest paths without fabricating produced-at chronology from model_version, one_tensor_row={one_tensor_row:?}, one_tensor_rendered={one_tensor_rendered:?}, one_tensor_error={one_tensor_error:?}, one_tensor_debug={one_tensor_debug}, stale_11_row={stale_11_row:?}, stale_11_rendered={stale_11_rendered:?}, stale_11_error={stale_11_error:?}, stale_11_debug={stale_11_debug}, projection_only_row={projection_only_row:?}, projection_only_rendered={projection_only_rendered:?}, projection_only_error={projection_only_error:?}, projection_only_debug={projection_only_debug}, wrong_identity_row={wrong_identity_row:?}, wrong_identity_rendered={wrong_identity_rendered:?}, wrong_identity_error={wrong_identity_error:?}, wrong_identity_debug={wrong_identity_debug}, expected_first={first_party_expected:?}, first_party_rendered={first_party_rendered:?}, first_party_debug={first_party_debug}, expected_consumer={consumer_trait_expected:?}, consumer_trait_rendered={consumer_trait_rendered:?}, consumer_debug={consumer_debug}, expected_precomputed={precomputed_expected:?}, precomputed_rendered={precomputed_rendered:?}"
    );
}

#[test]
fn decision_reasoning_chain_producer_system_predicate_query_returns_complete_answer_in_one_round_trip()
 {
    let fixture = fixture("decision-reasoning-producer");
    let (evidence_id, embedding_id) = record(&fixture, "reasoning-v1", 0);
    let db = raw_db(&fixture.db_path, fixture.store);
    let result = db
        .execute(
            "SELECT id, target_id, embedding_space_id, produced_by, created_at FROM evidence_embedding_handles WHERE id = $embedding_id AND target_id = $evidence_id AND produced_by = $producer",
            &params(vec![
                ("embedding_id", Value::Uuid(embedding_id.as_uuid())),
                ("evidence_id", Value::Uuid(evidence_id.as_uuid())),
                (
                    "producer",
                    Value::Json(
                        serde_json::to_value(producer("edge-memory", "reasoning-v1")).expect("json"),
                    ),
                ),
            ]),
        );
    let row_matches = result.as_ref().is_ok_and(|result| {
        result.rows.len() == 1
            && matches!(
                result.rows.first(),
                Some(row)
                    if matches!(row.first(), Some(Value::Uuid(id)) if *id == embedding_id.as_uuid())
                        && matches!(row.get(1), Some(Value::Uuid(id)) if *id == evidence_id.as_uuid())
                        && matches!(row.get(2), Some(Value::Text(space)) if space == TEXT_SPACE)
                        && matches!(row.get(3), Some(Value::Json(producer_json)) if producer_json == &serde_json::to_value(producer("edge-memory", "reasoning-v1")).expect("json"))
                        && matches!(row.get(4), Some(Value::TxId(_)))
            )
    });

    assert!(
        row_matches,
        "decision reasoning chain query must recover exact embedding id, target evidence, space, producer, and write txid in one stored projection, result={result:?}"
    );
}

#[test]
#[cfg(context_graph_internal_test_support)]
fn signature_match_full_chain_query_returns_centroid_producer_and_all_contributing_observation_producers_in_one_round_trip()
 {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("signature-producers.db");
    let query_counter = Arc::new(QueryCountingPlugin::default());
    let store = Store::open_with_database_plugin(
        StoreConfig {
            db_path: db_path.clone(),
            ..Default::default()
        },
        query_counter.clone(),
    )
    .expect("open");
    let context = store
        .create_context(CreateContext {
            name: format!("signature-producers-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "signature target".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    let fixture = Fixture {
        _dir: dir,
        db_path: db_path.clone(),
        store,
        context_id: context.id,
        entity_id: entity.id,
    };
    let signature_id = "signature-full-chain";
    let centroid = record_full_with_system(&fixture, "edge-memory", "centroid-v1", 0);
    let members = (0..50)
        .map(|index| {
            record_full_with_system(
                &fixture,
                if index % 2 == 0 {
                    "edge-memory"
                } else {
                    "partner-edge"
                },
                &format!("member-v{index}"),
                (index + 1) % 768,
            )
        })
        .collect::<Vec<_>>();
    let decoy = record_full_with_system(&fixture, "decoy-edge", "not-a-member", 77);
    persist_signature_aggregate(
        &fixture.store,
        aggregate(signature_id, &centroid, members.len()),
    )
    .expect("persist aggregate");
    persist_signature_producer_lineage(
        &fixture.store,
        SignatureProducerLineage {
            signature_id: signature_id.to_string(),
            centroid: contribution("centroid", &centroid),
            members: members
                .iter()
                .map(|member| contribution("member", member))
                .collect(),
        },
    )
    .expect("persist signature lineage");

    let mut expected = vec![expected_lineage_row(signature_id, "centroid", &centroid)];
    expected.extend(
        members
            .iter()
            .map(|member| expected_lineage_row(signature_id, "member", member)),
    );
    query_counter.enable_for_next_call();
    let public_result = signature_match_full_chain_producers(&fixture.store, signature_id);
    query_counter.disable();
    let public_query_count = query_counter.count();
    let public_sql = query_counter.sql();
    let public_rows_match = public_result.as_ref().is_ok_and(|rows| {
        rows.len() == expected.len()
            && expected.iter().all(|expected| {
                rows.iter()
                    .any(|actual| same_lineage_ignoring_txid(actual, expected))
            })
            && rows.iter().all(|row| row.evidence_id != decoy.evidence_id)
    });
    let public_uses_signature_lineage = public_sql.len() == 1
        && public_sql[0].contains("signatures")
        && public_sql[0].contains("signature_producer_lineage")
        && !public_sql[0].contains("WHERE id = $first OR id = $second");
    drop(fixture.store);
    let db = Database::open(&db_path).expect("raw database open");
    let raw_result = raw_signature_lineage_rows(&db, signature_id);
    let raw_rows_match = raw_result.as_ref().is_ok_and(|rows| {
        rows.len() == expected.len()
            && expected.iter().all(|expected| {
                rows.iter()
                    .any(|actual| same_lineage_ignoring_txid(actual, expected))
            })
            && rows.iter().all(|row| row.evidence_id != decoy.evidence_id)
    });

    assert!(
        public_query_count == 1
            && public_uses_signature_lineage
            && public_rows_match
            && raw_rows_match,
        "signature diagnostic chain must create a Signature lineage row, then one public query must return centroid producer plus all contributing observation producers from durable signature-linked rows, not loose evidence handles; expected_len={}, public_result={public_result:?}, public_query_count={public_query_count}, public_sql={public_sql:?}, raw_result={raw_result:?}, decoy_evidence={}",
        expected.len(),
        decoy.evidence_id
    );
}
