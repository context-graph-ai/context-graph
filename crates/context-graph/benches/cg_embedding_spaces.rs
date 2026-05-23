#[cfg(not(context_graph_internal_test_support))]
fn main() {}

#[cfg(context_graph_internal_test_support)]
use candle_core::{D, DType, Device, Tensor};
#[cfg(context_graph_internal_test_support)]
use chrono::Utc;
#[cfg(context_graph_internal_test_support)]
use context_graph::embedder::embeddinggemma::EmbeddingGemmaTextEmbedder;
#[cfg(context_graph_internal_test_support)]
use context_graph::embedder::nomic::NomicTextEmbedder;
#[cfg(context_graph_internal_test_support)]
use context_graph::embedder::{Embedder, EmbeddingInput, ModelSource};
#[cfg(context_graph_internal_test_support)]
use context_graph::{
    ContextId, CreateContext, CreateEntity, EmbedderConfig, EntityId, EntityType, EvidenceId,
    EvidenceKind, EvidenceRef, RecordObservation, RetentionStatus, Store, StoreConfig,
    test_support::embedding::{VectorSearchFilters, vector_search},
};
#[cfg(context_graph_internal_test_support)]
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(context_graph_internal_test_support)]
use safetensors::SafeTensors;
#[cfg(context_graph_internal_test_support)]
use safetensors::tensor::{Dtype, TensorView};
#[cfg(context_graph_internal_test_support)]
use std::collections::BTreeMap;
#[cfg(context_graph_internal_test_support)]
use std::hint::black_box;
#[cfg(context_graph_internal_test_support)]
use std::time::{Duration, Instant};
#[cfg(context_graph_internal_test_support)]
use tokenizers::Tokenizer;
#[cfg(context_graph_internal_test_support)]
use tokenizers::models::wordlevel::WordLevel;
#[cfg(context_graph_internal_test_support)]
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

#[cfg(context_graph_internal_test_support)]
const TEXT_SPACE: &str = "text_default_embeddinggemma_300m_768";
#[cfg(context_graph_internal_test_support)]
const COLD_START_BUDGET: Duration = Duration::from_secs(5);
#[cfg(context_graph_internal_test_support)]
const VECTOR_QUERY_BUDGET: Duration = Duration::from_millis(50);
#[cfg(context_graph_internal_test_support)]
const TOKEN_EMBEDDING_TENSOR: &str = "model.embed_tokens.weight";
#[cfg(context_graph_internal_test_support)]
const Q_PROJ_TENSOR: &str = "model.layers.0.self_attn.q_proj.weight";
#[cfg(context_graph_internal_test_support)]
const K_PROJ_TENSOR: &str = "model.layers.0.self_attn.k_proj.weight";
#[cfg(context_graph_internal_test_support)]
const V_PROJ_TENSOR: &str = "model.layers.0.self_attn.v_proj.weight";
#[cfg(context_graph_internal_test_support)]
const O_PROJ_TENSOR: &str = "model.layers.0.self_attn.o_proj.weight";
#[cfg(context_graph_internal_test_support)]
const Q_NORM_TENSOR: &str = "model.layers.0.self_attn.q_norm.weight";
#[cfg(context_graph_internal_test_support)]
const K_NORM_TENSOR: &str = "model.layers.0.self_attn.k_norm.weight";
#[cfg(context_graph_internal_test_support)]
const INPUT_LAYERNORM_TENSOR: &str = "model.layers.0.input_layernorm.weight";
#[cfg(context_graph_internal_test_support)]
const PRE_FEEDFORWARD_LAYERNORM_TENSOR: &str = "model.layers.0.pre_feedforward_layernorm.weight";
#[cfg(context_graph_internal_test_support)]
const POST_FEEDFORWARD_LAYERNORM_TENSOR: &str = "model.layers.0.post_feedforward_layernorm.weight";
#[cfg(context_graph_internal_test_support)]
const POST_ATTENTION_LAYERNORM_TENSOR: &str = "model.layers.0.post_attention_layernorm.weight";
#[cfg(context_graph_internal_test_support)]
const GATE_PROJ_TENSOR: &str = "model.layers.0.mlp.gate_proj.weight";
#[cfg(context_graph_internal_test_support)]
const UP_PROJ_TENSOR: &str = "model.layers.0.mlp.up_proj.weight";
#[cfg(context_graph_internal_test_support)]
const DOWN_PROJ_TENSOR: &str = "model.layers.0.mlp.down_proj.weight";
#[cfg(context_graph_internal_test_support)]
const FINAL_NORM_TENSOR: &str = "model.norm.weight";
#[cfg(context_graph_internal_test_support)]
const TEXT_PROJECTION_UP_TENSOR: &str = "model.text_projection.up_proj.weight";
#[cfg(context_graph_internal_test_support)]
const TEXT_PROJECTION_DOWN_TENSOR: &str = "model.text_projection.down_proj.weight";

#[cfg(context_graph_internal_test_support)]
pub fn embedder_cold_start_first_invocation_within_budget(c: &mut Criterion) {
    c.bench_function("embedder_cold_start_first_invocation_within_budget", |b| {
        b.iter(|| {
            let cache = tempfile::tempdir().expect("bench model cache");
            let (model_dir, unique_token, golden) = stage_valid_mini_embeddinggemma_cache(cache.path());
            let embedder = EmbeddingGemmaTextEmbedder::new(EmbedderConfig {
                source: ModelSource::LocalPath(model_dir),
                offline: true,
                cache_dir: Some(cache.path().to_path_buf()),
                pinned_revision: Some("embeddinggemma-300m-matryoshka-v1".to_string()),
                network_policy: "offline-cache-only".to_string(),
            });
            let started = Instant::now();
            let output = embedder
                .embed(EmbeddingInput::Text(format!("{unique_token} red sedan")))
                .expect("bench embed");
            let elapsed = started.elapsed();
            assert!(
                elapsed <= COLD_START_BUDGET,
                "cold-start bench exceeded budget {COLD_START_BUDGET:?}: {elapsed:?}"
            );
            assert!(
                vectors_close(&output.vector, &golden),
                "cold-start bench must execute the staged tokenizer+safetensors Candle path, not a synthetic vector"
            );
            black_box(output)
        })
    });
}

#[cfg(context_graph_internal_test_support)]
pub fn vector_search_per_query_latency_within_budget(c: &mut Criterion) {
    let dir = tempfile::tempdir().expect("bench tempdir");
    let db_path = dir.path().join("bench.db");
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("bench store opens");
    let context = store
        .create_context(CreateContext {
            name: format!("bench-{}", uuid::Uuid::now_v7()),
            labels: vec!["bench".to_string()],
            properties: BTreeMap::new(),
        })
        .expect("bench context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "bench entity".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("bench entity");
    for axis in 0..10_000 {
        seed_vector(&store, context.id, entity.id, axis);
    }
    let _ = db_path;
    let warm_hits = vector_search(
        &store,
        &query_vector(7),
        TEXT_SPACE,
        VectorSearchFilters {
            top_k: Some(8),
            ..Default::default()
        },
    )
    .expect("bench warm vector query");
    assert!(
        !warm_hits.is_empty(),
        "bench warm vector query must hit the populated vector index"
    );

    c.bench_function("vector_search_per_query_latency_within_budget", |b| {
        b.iter(|| {
            let started = Instant::now();
            let hits = vector_search(
                &store,
                &black_box(query_vector(7)),
                TEXT_SPACE,
                VectorSearchFilters {
                    top_k: Some(black_box(8)),
                    ..Default::default()
                },
            )
            .expect("bench vector query");
            let elapsed = started.elapsed();
            assert!(
                elapsed <= VECTOR_QUERY_BUDGET,
                "vector-search bench exceeded budget {VECTOR_QUERY_BUDGET:?}: {elapsed:?}"
            );
            black_box(hits)
        })
    });
}

#[cfg(context_graph_internal_test_support)]
pub fn bench_nomic_per_call_embed_latency_stays_under_1_2ms_cold_start_ceiling(c: &mut Criterion) {
    let embedder = NomicTextEmbedder::new(EmbedderConfig::disabled());
    c.bench_function(
        "bench_nomic_per_call_embed_latency_stays_under_1_2ms_cold_start_ceiling",
        |b| {
            b.iter(|| {
                let output = embedder
                    .embed(EmbeddingInput::Text(black_box(
                        "a small text input".to_string(),
                    )))
                    .expect("nomic bench embed");
                black_box(output.vector.len())
            })
        },
    );
}

#[cfg(context_graph_internal_test_support)]
fn seed_vector(store: &Store, context_id: ContextId, entity_id: EntityId, axis: usize) {
    let evidence_id = EvidenceId::new_v7();
    let vector = query_vector(axis);
    let result = store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id,
        context_id,
        observation_type: "bench_vector_seed".to_string(),
        source: "bench".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::Text,
            source_ref: format!("cas://sha256:{:064x}", (axis as u128) + 1),
            mime_type: Some("text/plain".to_string()),
            retention_status: RetentionStatus::RetainedExternal,
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![context_graph::PrecomputedEmbedding {
            id: context_graph::EmbeddingId::new_v7(),
            embedding_space_id: TEXT_SPACE.to_string(),
            vector,
            target: context_graph::EmbeddingTarget::Evidence(evidence_id),
            produced_by: context_graph::EvidenceProducer {
                system: "bench".to_string(),
                model_name: "embeddinggemma-300m".to_string(),
                model_version: "0".to_string(),
                pipeline_version: "bench".to_string(),
            },
        }],
    });
    assert!(result.is_ok(), "bench seed failed: {result:?}");
}

#[cfg(context_graph_internal_test_support)]
fn query_vector(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 768];
    let index = axis % vector.len();
    vector[index] = 1.0;
    vector
}

#[cfg(context_graph_internal_test_support)]
fn tensor_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[cfg(context_graph_internal_test_support)]
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
struct MiniEmbeddingGemmaWeights {
    token_embedding: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    input_layernorm: Vec<f32>,
    pre_feedforward_layernorm: Vec<f32>,
    post_feedforward_layernorm: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
    final_norm: Vec<f32>,
    text_projection_up: Vec<f32>,
    text_projection_down: Vec<f32>,
}

#[cfg(context_graph_internal_test_support)]
fn vectors_close(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(l, r)| (l - r).abs() <= 1.0e-5)
}

#[cfg(context_graph_internal_test_support)]
fn required_graph_tensors() -> Vec<&'static str> {
    vec![
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

#[cfg(context_graph_internal_test_support)]
fn tensor_row(values: &[f32], shape: (usize, usize)) -> Tensor {
    Tensor::from_slice(values, shape, &Device::Cpu).expect("candle tensor")
}

#[cfg(context_graph_internal_test_support)]
fn safetensor_f32_values(tensors: &SafeTensors<'_>, tensor_name: &str) -> Vec<f32> {
    let tensor = tensors.tensor(tensor_name).expect("fixture tensor");
    assert_eq!(tensor.dtype(), Dtype::F32, "fixture tensor dtype");
    tensor
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 bytes")))
        .collect()
}

#[cfg(context_graph_internal_test_support)]
fn mini_weights_from_safetensors(model_path: &std::path::Path) -> MiniEmbeddingGemmaWeights {
    let bytes = std::fs::read(model_path).expect("fixture model bytes");
    let tensors = SafeTensors::deserialize(&bytes).expect("fixture safetensors");

    MiniEmbeddingGemmaWeights {
        token_embedding: safetensor_f32_values(&tensors, TOKEN_EMBEDDING_TENSOR),
        q_proj: safetensor_f32_values(&tensors, Q_PROJ_TENSOR),
        k_proj: safetensor_f32_values(&tensors, K_PROJ_TENSOR),
        v_proj: safetensor_f32_values(&tensors, V_PROJ_TENSOR),
        o_proj: safetensor_f32_values(&tensors, O_PROJ_TENSOR),
        q_norm: safetensor_f32_values(&tensors, Q_NORM_TENSOR),
        k_norm: safetensor_f32_values(&tensors, K_NORM_TENSOR),
        input_layernorm: safetensor_f32_values(&tensors, INPUT_LAYERNORM_TENSOR),
        pre_feedforward_layernorm: safetensor_f32_values(
            &tensors,
            PRE_FEEDFORWARD_LAYERNORM_TENSOR,
        ),
        post_feedforward_layernorm: safetensor_f32_values(
            &tensors,
            POST_FEEDFORWARD_LAYERNORM_TENSOR,
        ),
        post_attention_layernorm: safetensor_f32_values(&tensors, POST_ATTENTION_LAYERNORM_TENSOR),
        gate_proj: safetensor_f32_values(&tensors, GATE_PROJ_TENSOR),
        up_proj: safetensor_f32_values(&tensors, UP_PROJ_TENSOR),
        down_proj: safetensor_f32_values(&tensors, DOWN_PROJ_TENSOR),
        final_norm: safetensor_f32_values(&tensors, FINAL_NORM_TENSOR),
        text_projection_up: safetensor_f32_values(&tensors, TEXT_PROJECTION_UP_TENSOR),
        text_projection_down: safetensor_f32_values(&tensors, TEXT_PROJECTION_DOWN_TENSOR),
    }
}

#[cfg(context_graph_internal_test_support)]
fn mini_rms_norm(xs: &Tensor, weight: &[f32]) -> Tensor {
    let weight = Tensor::from_slice(weight, weight.len(), &Device::Cpu).expect("rms weight");
    let hidden_size = xs.dim(D::Minus1).expect("hidden dim");
    let xs_f32 = xs.to_dtype(DType::F32).expect("rms dtype");
    let norm = (xs_f32
        .sqr()
        .expect("rms sqr")
        .sum_keepdim(D::Minus1)
        .expect("rms sum")
        / hidden_size as f64)
        .expect("rms mean");
    xs_f32
        .broadcast_div(&(norm + 1.0e-6).expect("rms eps").sqrt().expect("rms sqrt"))
        .expect("rms divide")
        .broadcast_mul(&(weight + 1.0).expect("rms weight plus one"))
        .expect("rms multiply")
}

#[cfg(context_graph_internal_test_support)]
fn mini_token_sequence(weights: &MiniEmbeddingGemmaWeights, token_ids: &[usize]) -> Tensor {
    let mut values = Vec::with_capacity(token_ids.len() * 4);
    for token_id in token_ids {
        let token_start = token_id * 4;
        values.extend_from_slice(&weights.token_embedding[token_start..token_start + 4]);
    }
    Tensor::from_slice(&values, (1, token_ids.len(), 4), &Device::Cpu).expect("token sequence")
}

#[cfg(context_graph_internal_test_support)]
fn mini_apply_rotary(q: &Tensor, k: &Tensor, seq_len: usize) -> (Tensor, Tensor) {
    let inv_freq = Tensor::from_vec(vec![1.0_f32, 0.01], (1, 2), &Device::Cpu).expect("inv freq");
    let positions = Tensor::arange(0u32, seq_len as u32, &Device::Cpu)
        .expect("positions")
        .to_dtype(DType::F32)
        .expect("position dtype")
        .reshape((seq_len, 1))
        .expect("position reshape");
    let freqs = positions.matmul(&inv_freq).expect("rotary freqs");
    let sin = freqs.sin().expect("rotary sin");
    let cos = freqs.cos().expect("rotary cos");
    (
        candle_nn::rotary_emb::rope(&q.contiguous().expect("q contiguous"), &cos, &sin)
            .expect("q rope"),
        candle_nn::rotary_emb::rope(&k.contiguous().expect("k contiguous"), &cos, &sin)
            .expect("k rope"),
    )
}

#[cfg(context_graph_internal_test_support)]
fn candle_mini_gemma_forward_parts(
    weights: &MiniEmbeddingGemmaWeights,
    token_ids: &[usize],
) -> Tensor {
    let seq_len = token_ids.len();
    let token_embedding = mini_token_sequence(weights, token_ids);
    let normalized = mini_rms_norm(&token_embedding, &weights.input_layernorm);
    let normalized_2d = normalized
        .reshape((seq_len, 4))
        .expect("normalized reshape");
    let q = normalized_2d
        .matmul(&tensor_row(&weights.q_proj, (4, 4)))
        .expect("candle q projection")
        .reshape((1, seq_len, 1, 4))
        .expect("q reshape")
        .transpose(1, 2)
        .expect("q transpose");
    let k = normalized_2d
        .matmul(&tensor_row(&weights.k_proj, (4, 4)))
        .expect("candle k projection")
        .reshape((1, seq_len, 1, 4))
        .expect("k reshape")
        .transpose(1, 2)
        .expect("k transpose");
    let v = normalized_2d
        .matmul(&tensor_row(&weights.v_proj, (4, 4)))
        .expect("candle v projection")
        .reshape((1, seq_len, 1, 4))
        .expect("v reshape")
        .transpose(1, 2)
        .expect("v transpose");
    let q = mini_rms_norm(&q, &weights.q_norm);
    let k = mini_rms_norm(&k, &weights.k_norm);
    let (q, k) = mini_apply_rotary(&q, &k, seq_len);
    let attention_scores = (q
        .matmul(&k.transpose(2, 3).expect("k transpose score"))
        .expect("attention scores")
        * 0.5)
        .expect("attention scale");
    let attention_weights =
        candle_nn::ops::softmax_last_dim(&attention_scores).expect("attention softmax");
    let attention_out = attention_weights
        .matmul(&v)
        .expect("attention values")
        .transpose(1, 2)
        .expect("attention transpose")
        .reshape((seq_len, 4))
        .expect("attention reshape")
        .matmul(&tensor_row(&weights.o_proj, (4, 4)))
        .expect("candle o projection")
        .reshape((1, seq_len, 4))
        .expect("attention output reshape");
    let attention_out = mini_rms_norm(&attention_out, &weights.post_attention_layernorm);
    let post_attention = (&attention_out + &token_embedding).expect("attention residual");
    let mlp_input = mini_rms_norm(&post_attention, &weights.pre_feedforward_layernorm);
    let mlp_input_2d = mlp_input.reshape((seq_len, 4)).expect("mlp reshape");
    let gate = mlp_input_2d
        .matmul(&tensor_row(&weights.gate_proj, (4, 8)))
        .expect("candle gate projection")
        .silu()
        .expect("candle gate activation");
    let up = mlp_input_2d
        .matmul(&tensor_row(&weights.up_proj, (4, 8)))
        .expect("candle up projection");
    let gated = (&gate * &up).expect("candle gated mlp");
    let mlp = gated
        .matmul(&tensor_row(&weights.down_proj, (8, 4)))
        .expect("candle down projection")
        .reshape((1, seq_len, 4))
        .expect("mlp output reshape");
    let mlp = mini_rms_norm(&mlp, &weights.post_feedforward_layernorm);
    let layer_out = (&post_attention + &mlp).expect("mlp residual");
    let final_sequence = mini_rms_norm(&layer_out, &weights.final_norm);
    final_sequence
        .mean_keepdim(1)
        .expect("mean pool")
        .reshape((1, 4))
        .expect("pooled reshape")
}

#[cfg(context_graph_internal_test_support)]
fn candle_mini_gemma_golden_projection(
    weights: &MiniEmbeddingGemmaWeights,
    token_ids: &[usize],
) -> Vec<f32> {
    let pooled = candle_mini_gemma_forward_parts(weights, token_ids);
    let projection_hidden = pooled
        .matmul(&tensor_row(&weights.text_projection_up, (4, 8)))
        .expect("candle projection up")
        .silu()
        .expect("candle projection activation");
    projection_hidden
        .matmul(&tensor_row(&weights.text_projection_down, (8, 768)))
        .expect("candle projection down")
        .to_vec2::<f32>()
        .expect("candle golden vector")[0]
        .clone()
}

#[cfg(context_graph_internal_test_support)]
fn normalized_768(full_projection: &[f32]) -> Vec<f32> {
    let norm = full_projection
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt();
    full_projection
        .iter()
        .map(|component| component / norm)
        .collect()
}

#[cfg(context_graph_internal_test_support)]
fn stage_valid_mini_embeddinggemma_cache(
    cache_dir: &std::path::Path,
) -> (std::path::PathBuf, String, Vec<f32>) {
    let model_dir = cache_dir.join("embeddinggemma-300m");
    std::fs::create_dir_all(&model_dir).expect("model dir");

    let unique_token = format!("edge_sentinel_{}", uuid::Uuid::now_v7().simple());
    let seed = uuid::Uuid::now_v7().as_u128();
    let weights = MiniEmbeddingGemmaWeights {
        token_embedding: seeded_values(seed, 16, 1.0),
        q_proj: seeded_values(seed.rotate_left(1), 4 * 4, 0.2),
        k_proj: seeded_values(seed.rotate_left(2), 4 * 4, 0.2),
        v_proj: seeded_values(seed.rotate_left(3), 4 * 4, 0.2),
        o_proj: seeded_values(seed.rotate_left(4), 4 * 4, 0.2),
        q_norm: seeded_values(seed.rotate_left(10), 4, 0.05),
        k_norm: seeded_values(seed.rotate_left(11), 4, 0.05),
        input_layernorm: seeded_values(seed.rotate_left(12), 4, 0.05),
        pre_feedforward_layernorm: seeded_values(seed.rotate_left(13), 4, 0.05),
        post_feedforward_layernorm: seeded_values(seed.rotate_left(14), 4, 0.05),
        post_attention_layernorm: seeded_values(seed.rotate_left(15), 4, 0.05),
        gate_proj: seeded_values(seed.rotate_left(5), 4 * 8, 0.15),
        up_proj: seeded_values(seed.rotate_left(6), 4 * 8, 0.15),
        down_proj: seeded_values(seed.rotate_left(7), 8 * 4, 0.15),
        final_norm: seeded_values(seed.rotate_left(8), 4, 0.5),
        text_projection_up: seeded_values(seed.rotate_left(9), 4 * 8, 0.2),
        text_projection_down: seeded_values(seed.reverse_bits(), 8 * 768, 0.2),
    };
    let token_embedding_bytes = tensor_bytes(&weights.token_embedding);
    let q_proj_bytes = tensor_bytes(&weights.q_proj);
    let k_proj_bytes = tensor_bytes(&weights.k_proj);
    let v_proj_bytes = tensor_bytes(&weights.v_proj);
    let o_proj_bytes = tensor_bytes(&weights.o_proj);
    let q_norm_bytes = tensor_bytes(&weights.q_norm);
    let k_norm_bytes = tensor_bytes(&weights.k_norm);
    let input_layernorm_bytes = tensor_bytes(&weights.input_layernorm);
    let pre_feedforward_layernorm_bytes = tensor_bytes(&weights.pre_feedforward_layernorm);
    let post_feedforward_layernorm_bytes = tensor_bytes(&weights.post_feedforward_layernorm);
    let post_attention_layernorm_bytes = tensor_bytes(&weights.post_attention_layernorm);
    let gate_proj_bytes = tensor_bytes(&weights.gate_proj);
    let up_proj_bytes = tensor_bytes(&weights.up_proj);
    let down_proj_bytes = tensor_bytes(&weights.down_proj);
    let final_norm_bytes = tensor_bytes(&weights.final_norm);
    let text_projection_up_bytes = tensor_bytes(&weights.text_projection_up);
    let text_projection_down_bytes = tensor_bytes(&weights.text_projection_down);
    let token_tensor =
        TensorView::new(Dtype::F32, vec![4, 4], &token_embedding_bytes).expect("token view");
    let q_tensor = TensorView::new(Dtype::F32, vec![4, 4], &q_proj_bytes).expect("q view");
    let k_tensor = TensorView::new(Dtype::F32, vec![4, 4], &k_proj_bytes).expect("k view");
    let v_tensor = TensorView::new(Dtype::F32, vec![4, 4], &v_proj_bytes).expect("v view");
    let o_tensor = TensorView::new(Dtype::F32, vec![4, 4], &o_proj_bytes).expect("o view");
    let q_norm_tensor = TensorView::new(Dtype::F32, vec![4], &q_norm_bytes).expect("q norm view");
    let k_norm_tensor = TensorView::new(Dtype::F32, vec![4], &k_norm_bytes).expect("k norm view");
    let input_layernorm_tensor =
        TensorView::new(Dtype::F32, vec![4], &input_layernorm_bytes).expect("input norm view");
    let pre_feedforward_layernorm_tensor =
        TensorView::new(Dtype::F32, vec![4], &pre_feedforward_layernorm_bytes)
            .expect("pre feedforward norm view");
    let post_feedforward_layernorm_tensor =
        TensorView::new(Dtype::F32, vec![4], &post_feedforward_layernorm_bytes)
            .expect("post feedforward norm view");
    let post_attention_layernorm_tensor =
        TensorView::new(Dtype::F32, vec![4], &post_attention_layernorm_bytes)
            .expect("post attention norm view");
    let gate_tensor = TensorView::new(Dtype::F32, vec![4, 8], &gate_proj_bytes).expect("gate view");
    let up_tensor = TensorView::new(Dtype::F32, vec![4, 8], &up_proj_bytes).expect("up view");
    let down_tensor = TensorView::new(Dtype::F32, vec![8, 4], &down_proj_bytes).expect("down view");
    let final_norm_tensor =
        TensorView::new(Dtype::F32, vec![4], &final_norm_bytes).expect("norm view");
    let text_projection_up_tensor =
        TensorView::new(Dtype::F32, vec![4, 8], &text_projection_up_bytes)
            .expect("projection up view");
    let text_projection_down_tensor =
        TensorView::new(Dtype::F32, vec![8, 768], &text_projection_down_bytes)
            .expect("projection down view");
    let model_bytes = safetensors::tensor::serialize(
        [
            (TOKEN_EMBEDDING_TENSOR, token_tensor),
            (Q_PROJ_TENSOR, q_tensor),
            (K_PROJ_TENSOR, k_tensor),
            (V_PROJ_TENSOR, v_tensor),
            (O_PROJ_TENSOR, o_tensor),
            (Q_NORM_TENSOR, q_norm_tensor),
            (K_NORM_TENSOR, k_norm_tensor),
            (INPUT_LAYERNORM_TENSOR, input_layernorm_tensor),
            (
                PRE_FEEDFORWARD_LAYERNORM_TENSOR,
                pre_feedforward_layernorm_tensor,
            ),
            (
                POST_FEEDFORWARD_LAYERNORM_TENSOR,
                post_feedforward_layernorm_tensor,
            ),
            (
                POST_ATTENTION_LAYERNORM_TENSOR,
                post_attention_layernorm_tensor,
            ),
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

    let vocab = [
        ("<unk>".to_string(), 0),
        ("red".to_string(), 1),
        ("sedan".to_string(), 2),
        (unique_token.clone(), 3),
    ]
    .into_iter()
    .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".to_string())
        .build()
        .expect("wordlevel model");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer
        .save(model_dir.join("tokenizer.json"), false)
        .expect("tokenizer");

    std::fs::write(
        model_dir.join("config.json"),
        serde_json::json!({
            "model_type": "embeddinggemma",
            "architectures": ["EmbeddingGemmaTextModel"],
            "hidden_size": 4,
            "head_dim": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "intermediate_size": 8,
            "hidden_activation": "silu",
            "rms_norm_eps": 1.0e-6,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "projection_dim": 768,
            "projection_intermediate_size": 8,
            "matryoshka_dimensions": [768, 512, 256, 128],
            "required_tensors": required_graph_tensors(),
        })
        .to_string(),
    )
    .expect("config");
    std::fs::write(
        model_dir.join("REVISION"),
        b"embeddinggemma-300m-matryoshka-v1",
    )
    .expect("revision");

    let artifact_weights = mini_weights_from_safetensors(&model_dir.join("model.safetensors"));
    let golden = normalized_768(&candle_mini_gemma_golden_projection(
        &artifact_weights,
        &[3, 1, 2],
    ));
    (model_dir, unique_token, golden)
}

#[cfg(context_graph_internal_test_support)]
criterion_group!(
    benches,
    embedder_cold_start_first_invocation_within_budget,
    vector_search_per_query_latency_within_budget,
    bench_nomic_per_call_embed_latency_stays_under_1_2ms_cold_start_ceiling
);
#[cfg(context_graph_internal_test_support)]
criterion_main!(benches);
