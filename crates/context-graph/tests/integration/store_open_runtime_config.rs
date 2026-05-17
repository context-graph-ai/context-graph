use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use candle_core::{Device, Tensor};
use chrono::Utc;
use context_graph::embedder::embeddinggemma::{
    EmbeddingGemmaForwardStage, EmbeddingGemmaLoadEvent, EmbeddingGemmaLoadObserver,
    EmbeddingGemmaTextEmbedder,
};
use context_graph::embedder::{Embedder, EmbeddingInput, ModelSource};
use context_graph::{
    CG_SCHEMA_VERSION, CgError, CreateContext, CreateEntity, EmbedderConfig, EmbedderRegistration,
    EmbeddingId, EmbeddingTarget, EntityType, EvidenceId, EvidenceKind, EvidenceProducer,
    EvidenceRef, PrecomputedEmbedding, RecordObservation, RetentionStatus, Store, StoreConfig,
    test_support::embedding::{VectorSearchFilters, vector_search},
};
use contextdb_core::Value;
use contextdb_engine::Database;
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

fn config(source: ModelSource, offline: bool, cache_dir: Option<PathBuf>) -> EmbedderConfig {
    EmbedderConfig {
        source,
        offline,
        cache_dir,
        pinned_revision: Some("rev-a".to_string()),
        network_policy: "offline".to_string(),
    }
}

const TOKEN_EMBEDDING_TENSOR: &str = "model.embed_tokens.weight";
const Q_PROJ_TENSOR: &str = "model.layers.0.self_attn.q_proj.weight";
const K_PROJ_TENSOR: &str = "model.layers.0.self_attn.k_proj.weight";
const V_PROJ_TENSOR: &str = "model.layers.0.self_attn.v_proj.weight";
const O_PROJ_TENSOR: &str = "model.layers.0.self_attn.o_proj.weight";
const GATE_PROJ_TENSOR: &str = "model.layers.0.mlp.gate_proj.weight";
const UP_PROJ_TENSOR: &str = "model.layers.0.mlp.up_proj.weight";
const DOWN_PROJ_TENSOR: &str = "model.layers.0.mlp.down_proj.weight";
const FINAL_NORM_TENSOR: &str = "model.norm.weight";
const TEXT_PROJECTION_UP_TENSOR: &str = "model.text_projection.up_proj.weight";
const TEXT_PROJECTION_DOWN_TENSOR: &str = "model.text_projection.down_proj.weight";

#[derive(Default)]
struct RecordingLoadObserver {
    events: Mutex<Vec<EmbeddingGemmaLoadEvent>>,
}

impl RecordingLoadObserver {
    fn events(&self) -> Vec<EmbeddingGemmaLoadEvent> {
        self.events.lock().expect("loader events").clone()
    }
}

impl EmbeddingGemmaLoadObserver for RecordingLoadObserver {
    fn observe(&self, event: EmbeddingGemmaLoadEvent) {
        self.events.lock().expect("loader events").push(event);
    }

    fn observe_candle_tensor(&self, stage: EmbeddingGemmaForwardStage, tensor: &Tensor) {
        self.observe(EmbeddingGemmaLoadEvent::CandleTensorDigest {
            stage,
            shape: tensor.dims().to_vec(),
            dtype: format!("{:?}", tensor.dtype()),
            device: format!("{:?}", tensor.device()),
            value_sha256: tensor_digest(tensor),
        });
    }
}

struct MiniEmbeddingGemmaWeights {
    token_embedding: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
    final_norm: Vec<f32>,
    text_projection_up: Vec<f32>,
    text_projection_down: Vec<f32>,
}

fn required_graph_tensors() -> [&'static str; 11] {
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

fn expected_tensor_shape(name: &str) -> &'static [usize] {
    match name {
        TOKEN_EMBEDDING_TENSOR | Q_PROJ_TENSOR | K_PROJ_TENSOR | V_PROJ_TENSOR | O_PROJ_TENSOR => {
            &[4, 4]
        }
        GATE_PROJ_TENSOR | UP_PROJ_TENSOR | TEXT_PROJECTION_UP_TENSOR => &[4, 8],
        DOWN_PROJ_TENSOR => &[8, 4],
        FINAL_NORM_TENSOR => &[4],
        TEXT_PROJECTION_DOWN_TENSOR => &[8, 768],
        _ => &[],
    }
}

fn sha256_fingerprint(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn sha256_f32_values(values: impl IntoIterator<Item = f32>) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn tensor_digest(tensor: &Tensor) -> String {
    let rows = tensor
        .to_vec2::<f32>()
        .expect("candle tensor digest values");
    sha256_f32_values(rows.into_iter().flatten())
}

fn observed_all_required_tensor_loads(events: &[EmbeddingGemmaLoadEvent]) -> bool {
    required_graph_tensors().iter().all(|expected_name| {
        events.iter().any(|event| {
            matches!(
                event,
                EmbeddingGemmaLoadEvent::TensorLoaded {
                    name,
                    shape,
                    dtype,
                    device,
                } if name == expected_name
                    && shape.as_slice() == expected_tensor_shape(expected_name)
                    && dtype == "F32"
                    && device == "Cpu"
            )
        })
    })
}

fn observed_forward_stage(
    events: &[EmbeddingGemmaLoadEvent],
    expected_stage: EmbeddingGemmaForwardStage,
    expected_shape: &[usize],
) -> bool {
    events.iter().any(|event| {
        matches!(
            event,
            EmbeddingGemmaLoadEvent::ForwardStage {
                stage,
                shape,
                dtype,
                device,
            } if stage == &expected_stage
                && shape.as_slice() == expected_shape
                && dtype == "F32"
                && device == "Cpu"
        )
    })
}

fn observed_all_forward_stages(events: &[EmbeddingGemmaLoadEvent]) -> bool {
    [
        (EmbeddingGemmaForwardStage::TokenEmbedding, &[1, 4][..]),
        (EmbeddingGemmaForwardStage::AttentionQkv, &[1, 4][..]),
        (EmbeddingGemmaForwardStage::AttentionOutput, &[1, 4][..]),
        (EmbeddingGemmaForwardStage::Mlp, &[1, 4][..]),
        (EmbeddingGemmaForwardStage::FinalNorm, &[1, 4][..]),
        (EmbeddingGemmaForwardStage::MeanPool, &[1, 4][..]),
        (EmbeddingGemmaForwardStage::ProjectionMlp, &[1, 768][..]),
        (EmbeddingGemmaForwardStage::NormalizedOutput, &[1, 768][..]),
    ]
    .into_iter()
    .all(|(stage, shape)| observed_forward_stage(events, stage, shape))
}

fn observed_artifact_fingerprint(
    events: &[EmbeddingGemmaLoadEvent],
    path_suffix: &str,
    expected_sha256: &str,
) -> bool {
    events.iter().any(|event| {
        matches!(
            event,
            EmbeddingGemmaLoadEvent::ArtifactFingerprinted { path_suffix: suffix, sha256 }
                if suffix.ends_with(path_suffix) && sha256 == expected_sha256
        )
    })
}

fn observed_candle_tensor_digest(
    events: &[EmbeddingGemmaLoadEvent],
    expected_stage: EmbeddingGemmaForwardStage,
    expected_shape: &[usize],
    expected_digest: &str,
) -> bool {
    events.iter().any(|event| {
        matches!(
            event,
            EmbeddingGemmaLoadEvent::CandleTensorDigest {
                stage,
                shape,
                dtype,
                device,
                value_sha256,
            } if stage == &expected_stage
                && shape.as_slice() == expected_shape
                && dtype == "F32"
                && device == "Cpu"
                && value_sha256 == expected_digest
        )
    })
}

fn observed_all_candle_tensor_digests(
    events: &[EmbeddingGemmaLoadEvent],
    expected: &[(EmbeddingGemmaForwardStage, Vec<usize>, String)],
) -> bool {
    expected.iter().all(|(stage, shape, digest)| {
        observed_candle_tensor_digest(events, stage.clone(), shape, digest)
    })
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

fn tensor_row(values: &[f32], shape: (usize, usize)) -> Tensor {
    Tensor::from_slice(values, shape, &Device::Cpu).expect("candle tensor")
}

fn safetensor_f32_values(tensors: &SafeTensors<'_>, tensor_name: &str) -> Vec<f32> {
    let tensor = tensors.tensor(tensor_name).expect("fixture tensor");
    assert_eq!(tensor.dtype(), Dtype::F32, "fixture tensor dtype");
    tensor
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 bytes")))
        .collect()
}

fn mini_weights_from_safetensors(model_path: &Path) -> MiniEmbeddingGemmaWeights {
    let bytes = std::fs::read(model_path).expect("fixture model bytes");
    let tensors = SafeTensors::deserialize(&bytes).expect("fixture safetensors");

    MiniEmbeddingGemmaWeights {
        token_embedding: safetensor_f32_values(&tensors, TOKEN_EMBEDDING_TENSOR),
        q_proj: safetensor_f32_values(&tensors, Q_PROJ_TENSOR),
        k_proj: safetensor_f32_values(&tensors, K_PROJ_TENSOR),
        v_proj: safetensor_f32_values(&tensors, V_PROJ_TENSOR),
        o_proj: safetensor_f32_values(&tensors, O_PROJ_TENSOR),
        gate_proj: safetensor_f32_values(&tensors, GATE_PROJ_TENSOR),
        up_proj: safetensor_f32_values(&tensors, UP_PROJ_TENSOR),
        down_proj: safetensor_f32_values(&tensors, DOWN_PROJ_TENSOR),
        final_norm: safetensor_f32_values(&tensors, FINAL_NORM_TENSOR),
        text_projection_up: safetensor_f32_values(&tensors, TEXT_PROJECTION_UP_TENSOR),
        text_projection_down: safetensor_f32_values(&tensors, TEXT_PROJECTION_DOWN_TENSOR),
    }
}

fn candle_mini_gemma_token_output(weights: &MiniEmbeddingGemmaWeights, token_id: usize) -> Tensor {
    let token_start = token_id * 4;
    let token = tensor_row(
        &weights.token_embedding[token_start..token_start + 4],
        (1, 4),
    );
    let q = token
        .matmul(&tensor_row(&weights.q_proj, (4, 4)))
        .expect("candle q projection");
    let k = token
        .matmul(&tensor_row(&weights.k_proj, (4, 4)))
        .expect("candle k projection");
    let v = token
        .matmul(&tensor_row(&weights.v_proj, (4, 4)))
        .expect("candle v projection");
    let attention = (((&q + &k).expect("candle q+k") + &v).expect("candle q+k+v") / 3.0)
        .expect("candle attention average");
    let attention_out = attention
        .matmul(&tensor_row(&weights.o_proj, (4, 4)))
        .expect("candle o projection");
    let gate = attention_out
        .matmul(&tensor_row(&weights.gate_proj, (4, 8)))
        .expect("candle gate projection")
        .tanh()
        .expect("candle gate activation");
    let up = attention_out
        .matmul(&tensor_row(&weights.up_proj, (4, 8)))
        .expect("candle up projection");
    let gated = (&gate * &up).expect("candle gated mlp");
    let mlp = gated
        .matmul(&tensor_row(&weights.down_proj, (8, 4)))
        .expect("candle down projection");
    let final_norm =
        Tensor::from_slice(&weights.final_norm, (1, 4), &Device::Cpu).expect("candle norm");
    ((&attention_out + &mlp).expect("candle residual") * &final_norm).expect("candle final norm")
}

fn candle_mini_gemma_stage_digests(
    weights: &MiniEmbeddingGemmaWeights,
    token_ids: &[usize],
) -> Vec<(EmbeddingGemmaForwardStage, Vec<usize>, String)> {
    let first_token = token_ids[0];
    let token_start = first_token * 4;
    let token = tensor_row(
        &weights.token_embedding[token_start..token_start + 4],
        (1, 4),
    );
    let q = token
        .matmul(&tensor_row(&weights.q_proj, (4, 4)))
        .expect("candle q projection");
    let k = token
        .matmul(&tensor_row(&weights.k_proj, (4, 4)))
        .expect("candle k projection");
    let v = token
        .matmul(&tensor_row(&weights.v_proj, (4, 4)))
        .expect("candle v projection");
    let attention = (((&q + &k).expect("candle q+k") + &v).expect("candle q+k+v") / 3.0)
        .expect("candle attention average");
    let attention_out = attention
        .matmul(&tensor_row(&weights.o_proj, (4, 4)))
        .expect("candle o projection");
    let gate = attention_out
        .matmul(&tensor_row(&weights.gate_proj, (4, 8)))
        .expect("candle gate projection")
        .tanh()
        .expect("candle gate activation");
    let up = attention_out
        .matmul(&tensor_row(&weights.up_proj, (4, 8)))
        .expect("candle up projection");
    let gated = (&gate * &up).expect("candle gated mlp");
    let mlp = gated
        .matmul(&tensor_row(&weights.down_proj, (8, 4)))
        .expect("candle down projection");
    let final_norm =
        Tensor::from_slice(&weights.final_norm, (1, 4), &Device::Cpu).expect("candle norm");
    let first_final =
        ((&attention_out + &mlp).expect("candle residual") * &final_norm).expect("candle final");
    let pooled = token_ids
        .iter()
        .map(|token_id| candle_mini_gemma_token_output(weights, *token_id))
        .reduce(|left, right| (&left + &right).expect("candle pool add"))
        .expect("at least one token");
    let pooled = (pooled / token_ids.len() as f64).expect("candle mean pool");
    let projection_hidden = pooled
        .matmul(&tensor_row(&weights.text_projection_up, (4, 8)))
        .expect("candle projection up")
        .tanh()
        .expect("candle projection activation");
    let projection = projection_hidden
        .matmul(&tensor_row(&weights.text_projection_down, (8, 768)))
        .expect("candle projection down");
    let normalized_values = normalized_768(
        &projection
            .to_vec2::<f32>()
            .expect("candle projection values")[0],
    );
    let normalized =
        Tensor::from_slice(&normalized_values, (1, 768), &Device::Cpu).expect("normalized tensor");

    vec![
        (
            EmbeddingGemmaForwardStage::TokenEmbedding,
            vec![1, 4],
            tensor_digest(&token),
        ),
        (
            EmbeddingGemmaForwardStage::AttentionQkv,
            vec![1, 4],
            tensor_digest(&attention),
        ),
        (
            EmbeddingGemmaForwardStage::AttentionOutput,
            vec![1, 4],
            tensor_digest(&attention_out),
        ),
        (
            EmbeddingGemmaForwardStage::Mlp,
            vec![1, 4],
            tensor_digest(&mlp),
        ),
        (
            EmbeddingGemmaForwardStage::FinalNorm,
            vec![1, 4],
            tensor_digest(&first_final),
        ),
        (
            EmbeddingGemmaForwardStage::MeanPool,
            vec![1, 4],
            tensor_digest(&pooled),
        ),
        (
            EmbeddingGemmaForwardStage::ProjectionMlp,
            vec![1, 768],
            tensor_digest(&projection),
        ),
        (
            EmbeddingGemmaForwardStage::NormalizedOutput,
            vec![1, 768],
            tensor_digest(&normalized),
        ),
    ]
}

fn candle_mini_gemma_golden_projection(
    weights: &MiniEmbeddingGemmaWeights,
    token_ids: &[usize],
) -> Vec<f32> {
    let pooled = token_ids
        .iter()
        .map(|token_id| candle_mini_gemma_token_output(weights, *token_id))
        .reduce(|left, right| (&left + &right).expect("candle pool add"))
        .expect("at least one token");
    let pooled = (pooled / token_ids.len() as f64).expect("candle mean pool");
    let projection_hidden = pooled
        .matmul(&tensor_row(&weights.text_projection_up, (4, 8)))
        .expect("candle projection up")
        .tanh()
        .expect("candle projection activation");
    projection_hidden
        .matmul(&tensor_row(&weights.text_projection_down, (8, 768)))
        .expect("candle projection down")
        .to_vec2::<f32>()
        .expect("candle golden vector")[0]
        .clone()
}

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

fn vectors_close(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(l, r)| (l - r).abs() <= 1.0e-5)
}

fn stage_local_embeddinggemma_cache(cache_dir: &Path) -> (PathBuf, String, Vec<f32>) {
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
        ("warm".to_string(), 1),
        ("cache".to_string(), 2),
        (unique_token.clone(), 3),
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
    std::fs::write(model_dir.join("REVISION"), b"rev-a").expect("revision");
    let artifact_weights = mini_weights_from_safetensors(&model_dir.join("model.safetensors"));
    let golden = normalized_768(&candle_mini_gemma_golden_projection(
        &artifact_weights,
        &[3, 1, 2],
    ));
    (model_dir, unique_token, golden)
}

fn stage_invalid_embeddinggemma_cache(cache_dir: &Path) -> PathBuf {
    let model_dir = cache_dir.join("embeddinggemma-300m-invalid");
    std::fs::create_dir_all(&model_dir).expect("invalid model dir");
    std::fs::write(
        model_dir.join("model.safetensors"),
        b"not-a-safetensors-file",
    )
    .expect("invalid model");
    std::fs::write(model_dir.join("tokenizer.json"), b"{\"invalid\":true}")
        .expect("invalid tokenizer");
    std::fs::write(
        model_dir.join("config.json"),
        r#"{"model_type":"embeddinggemma","projection_dim":768}"#,
    )
    .expect("invalid config");
    std::fs::write(model_dir.join("REVISION"), b"rev-a").expect("revision");
    model_dir
}

fn unit_vector_768() -> Vec<f32> {
    let mut vector = vec![0.0; 768];
    vector[0] = 1.0;
    vector
}

#[test]
fn default_text_embedder_lazy_loaded_no_cost_when_unused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache_dir = tempfile::tempdir().expect("cache");
    let (artifact_dir, _unique_token, _golden) = stage_local_embeddinggemma_cache(cache_dir.path());
    let artifact = artifact_dir.join("model.safetensors");
    EmbeddingGemmaTextEmbedder::reset_loader_entry_count();
    let embedder = EmbeddingGemmaTextEmbedder::new(config(
        ModelSource::LocalPath(artifact_dir.clone()),
        true,
        Some(cache_dir.path().to_path_buf()),
    ));
    let registration = EmbedderRegistration {
        space: embedder.embedding_space(),
        embedder: Arc::new(embedder.clone()),
    };
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: dir.path().join("lazy-open.db"),
            ..Default::default()
        },
        vec![registration],
    );

    assert!(store.is_ok());
    assert!(!embedder.lazy_handle().is_loaded());
    assert_eq!(EmbeddingGemmaTextEmbedder::loader_entry_count(), 0);
    assert!(embedder.loader_events().is_empty());
    assert_eq!(
        store.unwrap().last_query_trace().unwrap_or_default(),
        "embedding_loader=not_entered"
    );
    std::fs::remove_file(&artifact).expect("remove canary after open");
    let after_delete = embedder.embed(EmbeddingInput::Text("must load after open".to_string()));
    assert!(matches!(
        after_delete,
        Err(CgError::EmbedderModelNotAvailable { path, .. }) if path.contains("embeddinggemma-300m")
    ));
}

#[test]
fn default_text_embedder_opt_out_at_store_open_skips_load_path_entirely() {
    EmbeddingGemmaTextEmbedder::reset_loader_entry_count();
    let embedder = EmbeddingGemmaTextEmbedder::new(EmbedderConfig::disabled());
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("disabled-open.db"),
        default_text_embedder: Some(EmbedderConfig::disabled()),
        ..Default::default()
    });

    assert!(store.is_ok());
    assert_eq!(
        store.unwrap().last_query_trace().unwrap_or_default(),
        "embedding_loader=disabled"
    );
    assert_eq!(embedder.config.network_policy, "disabled");
    assert!(embedder.emit_cold_start_progress().contains("disabled"));
    assert!(matches!(
        embedder.embed(EmbeddingInput::Text("must not load".to_string())),
        Err(CgError::TextEmbedderDisabled)
    ));
    assert_eq!(EmbeddingGemmaTextEmbedder::loader_entry_count(), 0);
    assert!(embedder.loader_events().is_empty());
}

#[test]
fn embedder_offline_mode_warm_cache_validates_artifacts_before_success() {
    let cache_dir = tempfile::tempdir().expect("cache");
    let (model_dir, unique_token, expected_vector) =
        stage_local_embeddinggemma_cache(cache_dir.path());
    let expected_model_fingerprint = sha256_fingerprint(
        &std::fs::read(model_dir.join("model.safetensors")).expect("model bytes"),
    );
    let expected_stage_digests = candle_mini_gemma_stage_digests(
        &mini_weights_from_safetensors(&model_dir.join("model.safetensors")),
        &[3, 1, 2],
    );
    let observer = Arc::new(RecordingLoadObserver::default());
    let observer_trait: Arc<dyn EmbeddingGemmaLoadObserver> = observer.clone();
    let embedder = EmbeddingGemmaTextEmbedder::new_with_load_observer(
        config(
            ModelSource::LocalPath(model_dir),
            true,
            Some(cache_dir.path().to_path_buf()),
        ),
        observer_trait,
    );
    let started = Instant::now();
    let output = embedder.embed(EmbeddingInput::Text(format!("{unique_token} warm cache")));
    let elapsed = started.elapsed();

    let events = observer.events();
    assert!(matches!(
        &output,
        Ok(value)
            if value.embedding_space_id == "text_default_embeddinggemma_300m_768"
                && value.dimension == 768
                && value.producer.system == context_graph::CG_EDGE_SYSTEM_TAG
                && value.vector.len() == 768
                && vectors_close(&value.vector, &expected_vector)
    ));
    assert!(
        observed_all_required_tensor_loads(&events)
            && events.iter().any(|event| {
                matches!(
                    event,
                    EmbeddingGemmaLoadEvent::TokenIds { ids } if ids == &[3, 1, 2]
                )
            })
            && observed_all_forward_stages(&events)
            && observed_all_candle_tensor_digests(&events, &expected_stage_digests)
            && observed_artifact_fingerprint(
                &events,
                "model.safetensors",
                &expected_model_fingerprint,
            )
            && elapsed < Duration::from_secs(5),
        "warm-cache embed must prove real local Candle artifact load through all intermediate tensor digests in the structured loader observer within the cold-start budget, output={output:?}, expected_stage_digests={expected_stage_digests:?}, events={events:?}, elapsed={elapsed:?}"
    );
    assert_eq!(EmbeddingGemmaTextEmbedder::network_attempt_count(), 0);
}

#[test]
fn embedder_offline_mode_cold_cache_returns_typed_model_not_available_naming_path_and_source() {
    let cache_dir = tempfile::tempdir().expect("cache");
    let path = cache_dir.path().join("missing-hf-cache");
    let embedder = EmbeddingGemmaTextEmbedder::new(config(
        ModelSource::HuggingFace {
            repo: "google/embeddinggemma-300m".to_string(),
        },
        true,
        Some(path.clone()),
    ));
    let started = Instant::now();
    let result = embedder.embed(EmbeddingInput::Text("cold cache".to_string()));
    let elapsed = started.elapsed();

    assert!(
        matches!(
            result,
            Err(CgError::EmbedderModelNotAvailable {
                ref source,
                path: ref err_path,
            })
                if source.contains("HuggingFace") && err_path.contains("missing-hf-cache")
        ) && elapsed < Duration::from_millis(100)
            && EmbeddingGemmaTextEmbedder::network_attempt_count() == 0,
        "offline cold cache must fail fast from local cache metadata without a network attempt, result={result:?}, elapsed={elapsed:?}, network_attempts={}",
        EmbeddingGemmaTextEmbedder::network_attempt_count()
    );
}

#[test]
fn embedder_cache_dir_override_routes_load_to_explicit_path() {
    let cache_dir = tempfile::tempdir().expect("cache");
    let embedder = EmbeddingGemmaTextEmbedder::new(config(
        ModelSource::HuggingFace {
            repo: "google/embeddinggemma-300m".to_string(),
        },
        false,
        Some(cache_dir.path().join("models")),
    ));

    assert_eq!(
        embedder.config.cache_dir,
        Some(cache_dir.path().join("models"))
    );
    assert!(
        embedder
            .emit_cold_start_progress()
            .contains(cache_dir.path().to_string_lossy().as_ref())
    );
}

#[test]
fn embedder_pinned_revision_mismatch_returns_typed_revision_mismatch_error() {
    let cache_dir = tempfile::tempdir().expect("cache");
    let (model_dir, _unique_token, _golden) = stage_local_embeddinggemma_cache(cache_dir.path());
    std::fs::write(model_dir.join("REVISION"), b"rev-b").expect("mismatch revision");
    let embedder = EmbeddingGemmaTextEmbedder::new(config(
        ModelSource::LocalPath(model_dir),
        true,
        Some(cache_dir.path().to_path_buf()),
    ));
    let result = embedder.embed(EmbeddingInput::Text("revision".to_string()));

    assert!(
        matches!(
            result,
            Err(CgError::EmbedderModelRevisionMismatch {
                ref expected,
                ref actual
            })
                if expected == "rev-a" && actual == "rev-b"
        ) && EmbeddingGemmaTextEmbedder::network_attempt_count() == 0,
        "revision mismatch must read the local REVISION file and make no network attempt, result={result:?}, network_attempts={}",
        EmbeddingGemmaTextEmbedder::network_attempt_count()
    );
}

#[test]
fn embedder_partially_downloaded_cache_returns_typed_model_corrupt_no_silent_redownload() {
    let cache_dir = tempfile::tempdir().expect("cache");
    let partial_dir = cache_dir.path().join("embeddinggemma-300m");
    std::fs::create_dir_all(&partial_dir).expect("partial dir");
    std::fs::write(partial_dir.join("config.json"), b"{\"partial\":true}").expect("partial config");
    let embedder = EmbeddingGemmaTextEmbedder::new(config(
        ModelSource::HuggingFace {
            repo: "google/embeddinggemma-300m".to_string(),
        },
        true,
        Some(cache_dir.path().to_path_buf()),
    ));
    let started = Instant::now();
    let result = embedder.embed(EmbeddingInput::Text("partial".to_string()));
    let elapsed = started.elapsed();

    assert!(
        matches!(
            result,
            Err(CgError::EmbedderModelCorrupt { ref path })
                if path.contains("embeddinggemma-300m")
        ) && elapsed < Duration::from_millis(100)
            && EmbeddingGemmaTextEmbedder::network_attempt_count() == 0,
        "offline partial cache must report corruption without trying a redownload, result={result:?}, elapsed={elapsed:?}, network_attempts={}",
        EmbeddingGemmaTextEmbedder::network_attempt_count()
    );
}

#[test]
fn embedder_first_use_emits_observable_progress_signal_naming_model_version_path_revision() {
    let cache_dir = tempfile::tempdir().expect("cache");
    let embedder = EmbeddingGemmaTextEmbedder::new(config(
        ModelSource::HuggingFace {
            repo: "google/embeddinggemma-300m".to_string(),
        },
        false,
        Some(cache_dir.path().to_path_buf()),
    ));
    let progress = embedder.emit_cold_start_progress();

    assert!(progress.contains("embeddinggemma-300m"));
    assert!(progress.contains("rev-a"));
    assert!(progress.contains(cache_dir.path().to_string_lossy().as_ref()));
    assert!(
        progress.contains("model.safetensors")
            && progress.contains("tokenizer.json")
            && progress.contains("no preload at Store::open"),
        "progress signal must name lazy local artifacts and no-preload boundary, got {progress:?}"
    );
}

#[test]
fn store_open_against_mismatched_cg_schema_version_returns_legacy_schema_detected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("schema.db");
    let config = StoreConfig {
        db_path: db_path.clone(),
        expected_schema_version: CG_SCHEMA_VERSION + 1,
        ..Default::default()
    };
    let result = Store::open(config);
    let db = Database::open(&db_path).expect("raw schema db");
    let schema_version_rows = db
        .execute(
            "SELECT COUNT(*) AS n FROM cg_schema_version WHERE version = $version",
            &std::collections::HashMap::from([(
                "version".to_string(),
                Value::Int64(CG_SCHEMA_VERSION as i64),
            )]),
        )
        .expect("schema version row count");
    let schema_version_count = match schema_version_rows.rows.first().and_then(|row| row.first()) {
        Some(Value::Int64(value)) => *value,
        _ => 0,
    };
    db.execute(
        "UPDATE cg_schema_version SET version = $found_version WHERE id = $id",
        &std::collections::HashMap::from([
            ("id".to_string(), Value::Int64(1)),
            (
                "found_version".to_string(),
                Value::Int64((CG_SCHEMA_VERSION - 2) as i64),
            ),
        ]),
    )
    .expect("rewrite schema version row");
    drop(db);
    let row_backed_result = Store::open(StoreConfig {
        db_path: db_path.clone(),
        expected_schema_version: CG_SCHEMA_VERSION + 1,
        ..Default::default()
    });

    assert!(matches!(
        result,
        Err(CgError::LegacySchemaDetected {
            found_version,
            expected_version
        }) if found_version == CG_SCHEMA_VERSION && expected_version == CG_SCHEMA_VERSION + 1
    ));
    assert_eq!(
        schema_version_count, 1,
        "schema-version mismatch must be grounded in the persisted cg_schema_version row"
    );
    assert!(
        matches!(
            row_backed_result,
            Err(CgError::LegacySchemaDetected {
                found_version,
                expected_version
            }) if found_version == CG_SCHEMA_VERSION - 2
                && expected_version == CG_SCHEMA_VERSION + 1
        ),
        "schema-version mismatch must report the persisted row value, not a hardcoded current-version constant, got {row_backed_result:?}"
    );
}

#[test]
fn store_open_against_existing_v3_database_returns_legacy_schema_detected_without_embedding_space_mutation()
 {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("existing-v3.db");
    let db = Database::open(&db_path).expect("raw db");
    db.execute(
        "CREATE TABLE cg_schema_version (id INTEGER PRIMARY KEY, version INTEGER NOT NULL, installed_at TXID NOT NULL)",
        &std::collections::HashMap::new(),
    )
    .expect("version table");
    db.execute(
        "INSERT INTO cg_schema_version (id, version, installed_at) VALUES ($id, $version, $installed_at)",
        &std::collections::HashMap::from([
            ("id".to_string(), Value::Int64(1)),
            ("version".to_string(), Value::Int64((CG_SCHEMA_VERSION - 1) as i64)),
            ("installed_at".to_string(), Value::TxId(context_graph::TxId(1))),
        ]),
    )
    .expect("version row");
    drop(db);

    let result = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    });
    let reopened = Database::open(&db_path).expect("reopen raw db");

    assert!(matches!(
        result,
        Err(CgError::LegacySchemaDetected {
            found_version,
            expected_version
        }) if found_version == CG_SCHEMA_VERSION - 1 && expected_version == CG_SCHEMA_VERSION
    ));
    assert!(
        reopened.table_meta("evidence_embedding_handles").is_none()
            && reopened
                .table_meta("entity_reference_embedding_handles")
                .is_none()
            && reopened.table_meta("digest_embedding_handles").is_none(),
        "rejected legacy open must not mutate the schema into the new embedding-handle shape"
    );
}

#[test]
fn embedder_cold_start_budget_smoke_rejects_invalid_local_artifacts() {
    let cache_dir = tempfile::tempdir().expect("cache");
    let model_dir = stage_invalid_embeddinggemma_cache(cache_dir.path());
    let embedder = EmbeddingGemmaTextEmbedder::new(config(
        ModelSource::LocalPath(model_dir),
        true,
        Some(cache_dir.path().to_path_buf()),
    ));
    let started = Instant::now();
    let output = embedder.embed(EmbeddingInput::Text("budget probe".to_string()));
    let elapsed = started.elapsed();

    assert!(
        matches!(
            output,
            Err(CgError::EmbedderModelCorrupt { ref path }) if path.contains("model.safetensors") || path.contains("tokenizer.json")
        ) && EmbeddingGemmaTextEmbedder::loader_entry_count() <= 1
            && elapsed < Duration::from_millis(250),
        "cold-start path must enter the loader once and reject invalid local artifacts within the smoke budget without synthetic vectors, output={output:?}, elapsed={elapsed:?}"
    );
}

#[test]
fn vector_search_latency_budget_smoke_returns_seeded_evidence() {
    let store = Store::open(StoreConfig {
        db_path: std::env::temp_dir().join(format!("latency-budget-{}.db", uuid::Uuid::now_v7())),
        ..Default::default()
    })
    .expect("open");
    let context = store
        .create_context(CreateContext {
            name: format!("latency-context-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "latency-entity".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    let evidence_id = EvidenceId::new_v7();
    let vector = unit_vector_768();
    let recorded = store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id: entity.id,
        context_id: context.id,
        observation_type: "latency-vector-seed".to_string(),
        source: "latency-test".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::Text,
            source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
            mime_type: Some("text/plain".to_string()),
            retention_status: RetentionStatus::RetainedExternal,
            producer: EvidenceProducer {
                system: "edge-memory".to_string(),
                model_name: "embeddinggemma-300m".to_string(),
                model_version: "latency".to_string(),
                pipeline_version: "runtime-config".to_string(),
            },
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: "text_default_embeddinggemma_300m_768".to_string(),
            vector: vector.clone(),
            target: EmbeddingTarget::Evidence(evidence_id),
            produced_by: EvidenceProducer {
                system: "edge-memory".to_string(),
                model_name: "embeddinggemma-300m".to_string(),
                model_version: "latency".to_string(),
                pipeline_version: "runtime-config".to_string(),
            },
        }],
    });
    assert!(recorded.is_ok(), "latency seed failed: {recorded:?}");
    let started = Instant::now();
    let result = vector_search(
        &store,
        &vector,
        "text_default_embeddinggemma_300m_768",
        VectorSearchFilters {
            top_k: Some(1),
            ..Default::default()
        },
    );
    let elapsed = started.elapsed();

    assert!(
        result.as_ref().is_ok_and(|hits| hits
            .iter()
            .any(|hit| hit.target == EmbeddingTarget::Evidence(evidence_id)))
            && elapsed < Duration::from_millis(250),
        "expected populated vector query to return exact seeded evidence within budget, result={result:?}, elapsed={elapsed:?}"
    );
}
