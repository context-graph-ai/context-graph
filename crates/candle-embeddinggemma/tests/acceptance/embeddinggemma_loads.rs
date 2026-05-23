use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_core::{D, DType, Device, Tensor};
use candle_embeddinggemma::{
    EmbeddingGemmaForwardStage, EmbeddingGemmaLoadEvent, EmbeddingGemmaTextEmbedder,
};
use context_graph::embedder::{Embedder, EmbeddingInput};
use context_graph::{
    CgError, EmbedderConfig, EmbedderLoadEvent, EmbedderLoadObserver, ModelSource,
};
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

fn local_model_config(cache_dir: &Path) -> EmbedderConfig {
    EmbedderConfig {
        source: ModelSource::LocalPath(cache_dir.join("embeddinggemma-300m")),
        offline: true,
        cache_dir: Some(cache_dir.to_path_buf()),
        pinned_revision: Some("embeddinggemma-300m-matryoshka-v1".to_string()),
        network_policy: "offline-cache-only".to_string(),
    }
}

fn stage_invalid_embeddinggemma_cache(cache_dir: &Path) {
    let model_dir = cache_dir.join("embeddinggemma-300m");
    std::fs::create_dir_all(&model_dir).expect("model dir");
    std::fs::write(
        model_dir.join("model.safetensors"),
        b"not-a-safetensors-file",
    )
    .expect("model");
    std::fs::write(model_dir.join("tokenizer.json"), b"not-a-tokenizer").expect("tokenizer");
    std::fs::write(model_dir.join("config.json"), b"{}").expect("config");
    std::fs::write(
        model_dir.join("REVISION"),
        b"embeddinggemma-300m-matryoshka-v1",
    )
    .expect("revision");
}

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
const SHORTCUT_TEXT_PROJECTION_TENSOR: &str = "model.text_projection.weight";

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

fn expected_tensor_shape_for(
    name: &str,
    hidden_size: usize,
    intermediate_size: usize,
    projection_intermediate_size: usize,
) -> Vec<usize> {
    match name {
        TOKEN_EMBEDDING_TENSOR => vec![4, hidden_size],
        Q_PROJ_TENSOR | K_PROJ_TENSOR | V_PROJ_TENSOR | O_PROJ_TENSOR => {
            vec![hidden_size, hidden_size]
        }
        Q_NORM_TENSOR | K_NORM_TENSOR => vec![hidden_size],
        INPUT_LAYERNORM_TENSOR
        | PRE_FEEDFORWARD_LAYERNORM_TENSOR
        | POST_FEEDFORWARD_LAYERNORM_TENSOR
        | POST_ATTENTION_LAYERNORM_TENSOR => vec![hidden_size],
        GATE_PROJ_TENSOR | UP_PROJ_TENSOR => vec![hidden_size, intermediate_size],
        DOWN_PROJ_TENSOR => vec![intermediate_size, hidden_size],
        FINAL_NORM_TENSOR => vec![hidden_size],
        TEXT_PROJECTION_UP_TENSOR => vec![hidden_size, projection_intermediate_size],
        TEXT_PROJECTION_DOWN_TENSOR => vec![projection_intermediate_size, 768],
        _ => Vec::new(),
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
    let values = tensor
        .flatten_all()
        .expect("flatten candle tensor")
        .to_vec1::<f32>()
        .expect("candle tensor digest values");
    sha256_f32_values(values)
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

struct MiniEmbeddingGemmaWeights {
    hidden_size: usize,
    intermediate_size: usize,
    projection_intermediate_size: usize,
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

#[derive(Default)]
struct RecordingLoadObserver {
    events: Mutex<Vec<EmbedderLoadEvent>>,
}

impl EmbedderLoadObserver for RecordingLoadObserver {
    fn observe(&self, event: EmbedderLoadEvent) {
        self.events.lock().expect("loader events").push(event);
    }
}

fn observed_all_required_tensor_loads_for(
    events: &[EmbeddingGemmaLoadEvent],
    hidden_size: usize,
    intermediate_size: usize,
    projection_intermediate_size: usize,
) -> bool {
    required_graph_tensors().iter().all(|expected_name| {
        let expected_shape = expected_tensor_shape_for(
            expected_name,
            hidden_size,
            intermediate_size,
            projection_intermediate_size,
        );
        events.iter().any(|event| {
            matches!(
                event,
                EmbeddingGemmaLoadEvent::TensorLoaded {
                    name,
                    shape,
                    dtype,
                    device,
                } if name == expected_name
                    && shape == &expected_shape
                    && dtype == "F32"
                    && device == "Cpu"
            )
        })
    })
}

fn observed_all_required_tensor_loads(events: &[EmbeddingGemmaLoadEvent]) -> bool {
    observed_all_required_tensor_loads_for(events, 4, 8, 8)
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

fn observed_all_forward_stages_for(
    events: &[EmbeddingGemmaLoadEvent],
    hidden_size: usize,
    output_dim: usize,
) -> bool {
    let seq_len = 3;
    [
        (
            EmbeddingGemmaForwardStage::TokenEmbedding,
            vec![1, seq_len, hidden_size],
        ),
        (
            EmbeddingGemmaForwardStage::AttentionQkv,
            vec![1, 1, seq_len, seq_len],
        ),
        (
            EmbeddingGemmaForwardStage::AttentionOutput,
            vec![1, seq_len, hidden_size],
        ),
        (
            EmbeddingGemmaForwardStage::Mlp,
            vec![1, seq_len, hidden_size],
        ),
        (
            EmbeddingGemmaForwardStage::FinalNorm,
            vec![1, seq_len, hidden_size],
        ),
        (EmbeddingGemmaForwardStage::MeanPool, vec![1, hidden_size]),
        (
            EmbeddingGemmaForwardStage::ProjectionMlp,
            vec![1, output_dim],
        ),
        (
            EmbeddingGemmaForwardStage::NormalizedOutput,
            vec![1, output_dim],
        ),
    ]
    .into_iter()
    .all(|(stage, shape)| observed_forward_stage(events, stage, &shape))
}

fn observed_all_forward_stages(events: &[EmbeddingGemmaLoadEvent]) -> bool {
    observed_all_forward_stages_for(events, 4, 768)
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
        hidden_size: tensors
            .tensor(FINAL_NORM_TENSOR)
            .expect("fixture norm tensor")
            .shape()[0],
        intermediate_size: tensors
            .tensor(DOWN_PROJ_TENSOR)
            .expect("fixture down tensor")
            .shape()[0],
        projection_intermediate_size: tensors
            .tensor(TEXT_PROJECTION_DOWN_TENSOR)
            .expect("fixture projection down tensor")
            .shape()[0],
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

fn mini_token_sequence(weights: &MiniEmbeddingGemmaWeights, token_ids: &[usize]) -> Tensor {
    let hidden = weights.hidden_size;
    let mut values = Vec::with_capacity(token_ids.len() * hidden);
    for token_id in token_ids {
        let token_start = token_id * hidden;
        values.extend_from_slice(&weights.token_embedding[token_start..token_start + hidden]);
    }
    Tensor::from_slice(&values, (1, token_ids.len(), hidden), &Device::Cpu).expect("token sequence")
}

fn mini_apply_rotary(q: &Tensor, k: &Tensor, head_dim: usize, seq_len: usize) -> (Tensor, Tensor) {
    let inv_freq = (0..head_dim)
        .step_by(2)
        .map(|idx| 1f32 / 10_000f64.powf(idx as f64 / head_dim as f64) as f32)
        .collect::<Vec<_>>();
    let inv_freq = Tensor::from_vec(inv_freq, (1, head_dim / 2), &Device::Cpu).expect("inv freq");
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

fn candle_mini_gemma_forward_parts(
    weights: &MiniEmbeddingGemmaWeights,
    token_ids: &[usize],
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
    let hidden = weights.hidden_size;
    let intermediate = weights.intermediate_size;
    let seq_len = token_ids.len();
    let token_embedding = mini_token_sequence(weights, token_ids);
    let normalized = mini_rms_norm(&token_embedding, &weights.input_layernorm);
    let normalized_2d = normalized
        .reshape((seq_len, hidden))
        .expect("normalized reshape");
    let q = normalized_2d
        .matmul(&tensor_row(&weights.q_proj, (hidden, hidden)))
        .expect("candle q projection")
        .reshape((1, seq_len, 1, hidden))
        .expect("q reshape")
        .transpose(1, 2)
        .expect("q transpose");
    let k = normalized_2d
        .matmul(&tensor_row(&weights.k_proj, (hidden, hidden)))
        .expect("candle k projection")
        .reshape((1, seq_len, 1, hidden))
        .expect("k reshape")
        .transpose(1, 2)
        .expect("k transpose");
    let v = normalized_2d
        .matmul(&tensor_row(&weights.v_proj, (hidden, hidden)))
        .expect("candle v projection")
        .reshape((1, seq_len, 1, hidden))
        .expect("v reshape")
        .transpose(1, 2)
        .expect("v transpose");
    let q = mini_rms_norm(&q, &weights.q_norm);
    let k = mini_rms_norm(&k, &weights.k_norm);
    let (q, k) = mini_apply_rotary(&q, &k, hidden, seq_len);
    let attention_scores = (q
        .matmul(&k.transpose(2, 3).expect("k transpose score"))
        .expect("attention scores")
        * (1.0 / (hidden as f64).sqrt()))
    .expect("attention scale");
    let attention_weights =
        candle_nn::ops::softmax_last_dim(&attention_scores).expect("attention softmax");
    let attention_out = attention_weights
        .matmul(&v)
        .expect("attention values")
        .transpose(1, 2)
        .expect("attention transpose")
        .reshape((seq_len, hidden))
        .expect("attention reshape")
        .matmul(&tensor_row(&weights.o_proj, (hidden, hidden)))
        .expect("candle o projection")
        .reshape((1, seq_len, hidden))
        .expect("attention output reshape");
    let attention_out = mini_rms_norm(&attention_out, &weights.post_attention_layernorm);
    let post_attention = (&attention_out + &token_embedding).expect("attention residual");
    let mlp_input = mini_rms_norm(&post_attention, &weights.pre_feedforward_layernorm);
    let mlp_input_2d = mlp_input.reshape((seq_len, hidden)).expect("mlp reshape");
    let gate = mlp_input_2d
        .matmul(&tensor_row(&weights.gate_proj, (hidden, intermediate)))
        .expect("candle gate projection")
        .silu()
        .expect("candle gate activation");
    let up = mlp_input_2d
        .matmul(&tensor_row(&weights.up_proj, (hidden, intermediate)))
        .expect("candle up projection");
    let gated = (&gate * &up).expect("candle gated mlp");
    let mlp = gated
        .matmul(&tensor_row(&weights.down_proj, (intermediate, hidden)))
        .expect("candle down projection")
        .reshape((1, seq_len, hidden))
        .expect("mlp output reshape");
    let mlp = mini_rms_norm(&mlp, &weights.post_feedforward_layernorm);
    let layer_out = (&post_attention + &mlp).expect("mlp residual");
    let final_sequence = mini_rms_norm(&layer_out, &weights.final_norm);
    let pooled = final_sequence
        .mean_keepdim(1)
        .expect("mean pool")
        .reshape((1, hidden))
        .expect("pooled reshape");
    (
        token_embedding,
        attention_scores,
        attention_out,
        mlp,
        final_sequence,
        pooled,
    )
}

fn candle_mini_gemma_stage_digests(
    weights: &MiniEmbeddingGemmaWeights,
    token_ids: &[usize],
) -> Vec<(EmbeddingGemmaForwardStage, Vec<usize>, String)> {
    let hidden = weights.hidden_size;
    let projection_intermediate = weights.projection_intermediate_size;
    let (token, attention, attention_out, mlp, final_sequence, pooled) =
        candle_mini_gemma_forward_parts(weights, token_ids);
    let projection_hidden = pooled
        .matmul(&tensor_row(
            &weights.text_projection_up,
            (hidden, projection_intermediate),
        ))
        .expect("candle projection up")
        .silu()
        .expect("candle projection activation");
    let projection = projection_hidden
        .matmul(&tensor_row(
            &weights.text_projection_down,
            (projection_intermediate, 768),
        ))
        .expect("candle projection down");
    let normalized_values = matryoshka_head(
        &projection
            .to_vec2::<f32>()
            .expect("candle projection values")[0],
        768,
    );
    let normalized =
        Tensor::from_slice(&normalized_values, (1, 768), &Device::Cpu).expect("normalized tensor");

    vec![
        (
            EmbeddingGemmaForwardStage::TokenEmbedding,
            vec![1, token_ids.len(), hidden],
            tensor_digest(&token),
        ),
        (
            EmbeddingGemmaForwardStage::AttentionQkv,
            vec![1, 1, token_ids.len(), token_ids.len()],
            tensor_digest(&attention),
        ),
        (
            EmbeddingGemmaForwardStage::AttentionOutput,
            vec![1, token_ids.len(), hidden],
            tensor_digest(&attention_out),
        ),
        (
            EmbeddingGemmaForwardStage::Mlp,
            vec![1, token_ids.len(), hidden],
            tensor_digest(&mlp),
        ),
        (
            EmbeddingGemmaForwardStage::FinalNorm,
            vec![1, token_ids.len(), hidden],
            tensor_digest(&final_sequence),
        ),
        (
            EmbeddingGemmaForwardStage::MeanPool,
            vec![1, hidden],
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
    let hidden = weights.hidden_size;
    let projection_intermediate = weights.projection_intermediate_size;
    let (_, _, _, _, _, pooled) = candle_mini_gemma_forward_parts(weights, token_ids);
    let projection_hidden = pooled
        .matmul(&tensor_row(
            &weights.text_projection_up,
            (hidden, projection_intermediate),
        ))
        .expect("candle projection up")
        .silu()
        .expect("candle projection activation");
    projection_hidden
        .matmul(&tensor_row(
            &weights.text_projection_down,
            (projection_intermediate, 768),
        ))
        .expect("candle projection down")
        .to_vec2::<f32>()
        .expect("candle golden vector")[0]
        .clone()
}

fn matryoshka_head(full_projection: &[f32], dimension: usize) -> Vec<f32> {
    let mut head = full_projection[..dimension].to_vec();
    let norm = head
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt();
    for component in &mut head {
        *component /= norm;
    }
    head
}

fn vector_l2_norm(vector: &[f32]) -> f32 {
    vector
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt()
}

fn vectors_close(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(l, r)| (l - r).abs() <= 1.0e-5)
}

fn write_tokenizer_and_revision(model_dir: &Path, unique_token: &str) {
    let vocab = [
        ("<unk>".to_string(), 0),
        ("red".to_string(), 1),
        ("sedan".to_string(), 2),
        (unique_token.to_string(), 3),
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
        model_dir.join("REVISION"),
        b"embeddinggemma-300m-matryoshka-v1",
    )
    .expect("revision");
}

fn stage_projection_shortcut_embeddinggemma_cache(cache_dir: &Path) {
    let model_dir = cache_dir.join("embeddinggemma-300m");
    std::fs::create_dir_all(&model_dir).expect("model dir");

    let unique_token = format!("edge_sentinel_{}", uuid::Uuid::now_v7().simple());
    let seed = uuid::Uuid::now_v7().as_u128();
    let token_embedding = seeded_values(seed, 16, 1.0);
    let projection = seeded_values(seed.reverse_bits(), 4 * 768, 0.25);
    let token_embedding_bytes = tensor_bytes(&token_embedding);
    let projection_bytes = tensor_bytes(&projection);
    let token_tensor =
        TensorView::new(Dtype::F32, vec![4, 4], &token_embedding_bytes).expect("token view");
    let projection_tensor =
        TensorView::new(Dtype::F32, vec![4, 768], &projection_bytes).expect("projection view");
    let model_bytes = safetensors::tensor::serialize(
        [
            (TOKEN_EMBEDDING_TENSOR, token_tensor),
            (SHORTCUT_TEXT_PROJECTION_TENSOR, projection_tensor),
        ],
        None,
    )
    .expect("safetensors");
    std::fs::write(model_dir.join("model.safetensors"), model_bytes).expect("model");
    write_tokenizer_and_revision(&model_dir, &unique_token);
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::json!({
            "model_type": "embeddinggemma",
            "architectures": ["EmbeddingGemmaTextModel"],
            "hidden_size": 4,
            "projection_dim": 768,
            "required_tensors": [
                TOKEN_EMBEDDING_TENSOR,
                SHORTCUT_TEXT_PROJECTION_TENSOR,
            ],
        })
        .to_string(),
    )
    .expect("config");
}

fn stage_valid_mini_embeddinggemma_cache_with_shape(
    cache_dir: &Path,
    hidden_size: usize,
    intermediate_size: usize,
    projection_intermediate_size: usize,
) -> (PathBuf, String, Vec<f32>) {
    let model_dir = cache_dir.join("embeddinggemma-300m");
    std::fs::create_dir_all(&model_dir).expect("model dir");

    let unique_token = format!("edge_sentinel_{}", uuid::Uuid::now_v7().simple());
    let seed = uuid::Uuid::now_v7().as_u128();
    let weights = MiniEmbeddingGemmaWeights {
        hidden_size,
        intermediate_size,
        projection_intermediate_size,
        token_embedding: seeded_values(seed, 4 * hidden_size, 1.0),
        q_proj: seeded_values(seed.rotate_left(1), hidden_size * hidden_size, 0.2),
        k_proj: seeded_values(seed.rotate_left(2), hidden_size * hidden_size, 0.2),
        v_proj: seeded_values(seed.rotate_left(3), hidden_size * hidden_size, 0.2),
        o_proj: seeded_values(seed.rotate_left(4), hidden_size * hidden_size, 0.2),
        q_norm: seeded_values(seed.rotate_left(10), hidden_size, 0.05),
        k_norm: seeded_values(seed.rotate_left(11), hidden_size, 0.05),
        input_layernorm: seeded_values(seed.rotate_left(12), hidden_size, 0.05),
        pre_feedforward_layernorm: seeded_values(seed.rotate_left(13), hidden_size, 0.05),
        post_feedforward_layernorm: seeded_values(seed.rotate_left(14), hidden_size, 0.05),
        post_attention_layernorm: seeded_values(seed.rotate_left(15), hidden_size, 0.05),
        gate_proj: seeded_values(seed.rotate_left(5), hidden_size * intermediate_size, 0.15),
        up_proj: seeded_values(seed.rotate_left(6), hidden_size * intermediate_size, 0.15),
        down_proj: seeded_values(seed.rotate_left(7), intermediate_size * hidden_size, 0.15),
        final_norm: seeded_values(seed.rotate_left(8), hidden_size, 0.5),
        text_projection_up: seeded_values(
            seed.rotate_left(9),
            hidden_size * projection_intermediate_size,
            0.2,
        ),
        text_projection_down: seeded_values(
            seed.reverse_bits(),
            projection_intermediate_size * 768,
            0.2,
        ),
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
    let token_tensor = TensorView::new(Dtype::F32, vec![4, hidden_size], &token_embedding_bytes)
        .expect("token view");
    let q_tensor =
        TensorView::new(Dtype::F32, vec![hidden_size, hidden_size], &q_proj_bytes).expect("q view");
    let k_tensor =
        TensorView::new(Dtype::F32, vec![hidden_size, hidden_size], &k_proj_bytes).expect("k view");
    let v_tensor =
        TensorView::new(Dtype::F32, vec![hidden_size, hidden_size], &v_proj_bytes).expect("v view");
    let o_tensor =
        TensorView::new(Dtype::F32, vec![hidden_size, hidden_size], &o_proj_bytes).expect("o view");
    let q_norm_tensor =
        TensorView::new(Dtype::F32, vec![hidden_size], &q_norm_bytes).expect("q norm view");
    let k_norm_tensor =
        TensorView::new(Dtype::F32, vec![hidden_size], &k_norm_bytes).expect("k norm view");
    let input_layernorm_tensor =
        TensorView::new(Dtype::F32, vec![hidden_size], &input_layernorm_bytes)
            .expect("input norm view");
    let pre_feedforward_layernorm_tensor = TensorView::new(
        Dtype::F32,
        vec![hidden_size],
        &pre_feedforward_layernorm_bytes,
    )
    .expect("pre feedforward norm view");
    let post_feedforward_layernorm_tensor = TensorView::new(
        Dtype::F32,
        vec![hidden_size],
        &post_feedforward_layernorm_bytes,
    )
    .expect("post feedforward norm view");
    let post_attention_layernorm_tensor = TensorView::new(
        Dtype::F32,
        vec![hidden_size],
        &post_attention_layernorm_bytes,
    )
    .expect("post attention norm view");
    let gate_tensor = TensorView::new(
        Dtype::F32,
        vec![hidden_size, intermediate_size],
        &gate_proj_bytes,
    )
    .expect("gate view");
    let up_tensor = TensorView::new(
        Dtype::F32,
        vec![hidden_size, intermediate_size],
        &up_proj_bytes,
    )
    .expect("up view");
    let down_tensor = TensorView::new(
        Dtype::F32,
        vec![intermediate_size, hidden_size],
        &down_proj_bytes,
    )
    .expect("down view");
    let final_norm_tensor =
        TensorView::new(Dtype::F32, vec![hidden_size], &final_norm_bytes).expect("norm view");
    let text_projection_up_tensor = TensorView::new(
        Dtype::F32,
        vec![hidden_size, projection_intermediate_size],
        &text_projection_up_bytes,
    )
    .expect("projection up view");
    let text_projection_down_tensor = TensorView::new(
        Dtype::F32,
        vec![projection_intermediate_size, 768],
        &text_projection_down_bytes,
    )
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
    write_tokenizer_and_revision(&model_dir, &unique_token);

    std::fs::write(
        model_dir.join("config.json"),
        serde_json::json!({
            "model_type": "embeddinggemma",
            "architectures": ["EmbeddingGemmaTextModel"],
            "hidden_size": hidden_size,
            "head_dim": hidden_size,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "intermediate_size": intermediate_size,
            "hidden_activation": "silu",
            "rms_norm_eps": 1.0e-6,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "projection_dim": 768,
            "projection_intermediate_size": projection_intermediate_size,
            "matryoshka_dimensions": [768, 512, 256, 128],
            "required_tensors": required_graph_tensors(),
        })
        .to_string(),
    )
    .expect("config");

    let artifact_weights = mini_weights_from_safetensors(&model_dir.join("model.safetensors"));
    let golden = candle_mini_gemma_golden_projection(&artifact_weights, &[3, 1, 2]);

    (model_dir, unique_token, golden)
}

fn stage_valid_mini_embeddinggemma_cache(cache_dir: &Path) -> (PathBuf, String, Vec<f32>) {
    stage_valid_mini_embeddinggemma_cache_with_shape(cache_dir, 4, 8, 8)
}

fn embedder_with_dimension(dimension: usize, cache_dir: &Path) -> EmbeddingGemmaTextEmbedder {
    stage_invalid_embeddinggemma_cache(cache_dir);
    EmbeddingGemmaTextEmbedder::new_for_dimension(local_model_config(cache_dir), dimension)
}

fn assert_matryoshka_head_embeds_head_specific_vector(dimension: usize) {
    let cache = tempfile::tempdir().expect("model cache tempdir");
    let (_model_dir, unique_token, full_projection) =
        stage_valid_mini_embeddinggemma_cache(cache.path());
    let embedder =
        EmbeddingGemmaTextEmbedder::new_for_dimension(local_model_config(cache.path()), dimension);
    let expected_vector = matryoshka_head(&full_projection, dimension);
    let expected_space = format!("text_default_embeddinggemma_300m_{dimension}");
    let output = embedder.embed(EmbeddingInput::Text(format!("{unique_token} red sedan")));

    assert!(
        matches!(
            &output,
            Ok(value)
                if value.embedding_space_id == expected_space
                    && value.dimension == dimension
                    && value.vector.len() == dimension
                    && vectors_close(&value.vector, &expected_vector)
                    && (vector_l2_norm(&value.vector) - 1.0).abs() <= 1.0e-5
        ),
        "Matryoshka head {dimension} must run the same Candle load path, slice the 768 projection to the requested head, and L2-renormalize; output={output:?}, expected_prefix={:?}",
        &expected_vector[..expected_vector.len().min(8)]
    );
}

#[test]
fn embeddinggemma_requires_valid_candle_tokenizer_safetensors_artifacts_before_first_vector() {
    let cache = tempfile::tempdir().expect("model cache tempdir");
    stage_invalid_embeddinggemma_cache(cache.path());
    let model_dir = cache.path().join("embeddinggemma-300m");
    let model_bytes = std::fs::read(model_dir.join("model.safetensors")).expect("model bytes");
    let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"));
    let observer = Arc::new(RecordingLoadObserver::default());
    let observer_trait: Arc<dyn EmbedderLoadObserver> = observer.clone();
    let embedder = EmbeddingGemmaTextEmbedder::new_with_load_observer(
        local_model_config(cache.path()),
        observer_trait,
    );

    let device = Device::Cpu;
    assert_eq!(format!("{device:?}"), "Cpu");
    assert!(SafeTensors::deserialize(&model_bytes).is_err());
    assert!(tokenizer.is_err());
    assert!(!embedder.lazy_handle().is_loaded());
    let output = embedder.embed(EmbeddingInput::Text(
        "red sedan near loading dock".to_string(),
    ));

    assert!(
        matches!(output, Err(CgError::EmbedderModelCorrupt { ref path }) if path.contains("model.safetensors") || path.contains("tokenizer.json")),
        "invalid local artifacts must be rejected before any synthetic vector is returned, output={output:?}"
    );
}

#[test]
fn embeddinggemma_warm_cache_success_requires_real_tokenizer_safetensors_and_candle_tensor() {
    let shortcut_cache = tempfile::tempdir().expect("shortcut model cache tempdir");
    stage_projection_shortcut_embeddinggemma_cache(shortcut_cache.path());
    let shortcut_embedder =
        EmbeddingGemmaTextEmbedder::new(local_model_config(shortcut_cache.path()));
    let shortcut_output = shortcut_embedder.embed(EmbeddingInput::Text(
        "edge_sentinel_projection_only red sedan".to_string(),
    ));
    assert!(
        matches!(
            shortcut_output,
            Err(CgError::EmbedderModelCorrupt { ref path })
                if path.contains("model.safetensors") || path.contains("config.json")
        ),
        "EmbeddingGemma must reject a projection-only cache that omits the Gemma3 backbone graph tensors; output={shortcut_output:?}"
    );

    let cache = tempfile::tempdir().expect("model cache tempdir");
    let (model_dir, unique_token, golden) = stage_valid_mini_embeddinggemma_cache(cache.path());
    let model_bytes = std::fs::read(model_dir.join("model.safetensors")).expect("model bytes");
    let expected_model_fingerprint = sha256_fingerprint(&model_bytes);
    let tensors = SafeTensors::deserialize(&model_bytes).expect("valid safetensors");
    let tensor_names = tensors.names();
    let expected_stage_digests = candle_mini_gemma_stage_digests(
        &mini_weights_from_safetensors(&model_dir.join("model.safetensors")),
        &[3, 1, 2],
    );
    let config_text = std::fs::read_to_string(model_dir.join("config.json")).expect("config text");
    let tokenizer =
        Tokenizer::from_file(model_dir.join("tokenizer.json")).expect("valid tokenizer");
    let encoded = tokenizer
        .encode(unique_token.as_str(), false)
        .expect("encode sentinel");
    let observer = Arc::new(RecordingLoadObserver::default());
    let observer_trait: Arc<dyn EmbedderLoadObserver> = observer.clone();
    let embedder = EmbeddingGemmaTextEmbedder::new_with_load_observer(
        local_model_config(cache.path()),
        observer_trait,
    );

    assert!(
        required_graph_tensors()
            .iter()
            .all(|name| tensor_names.contains(name))
            && config_text.contains("\"num_hidden_layers\":1")
            && config_text.contains("\"matryoshka_dimensions\":[768,512,256,128]")
    );
    assert_eq!(encoded.get_ids(), &[3]);
    let expected_vector = matryoshka_head(&golden, 768);

    assert_eq!(expected_vector.len(), 768);
    assert!(
        expected_vector
            .iter()
            .any(|component| component.abs() > 1.0e-6)
    );
    let output = embedder.embed(EmbeddingInput::Text(format!("{unique_token} red sedan")));
    let structured_events = embedder.structured_loader_events();

    assert!(
        matches!(
            &output,
            Ok(value)
                if value.embedding_space_id == "text_default_embeddinggemma_300m_768"
                    && value.dimension == 768
                    && value.vector.len() == 768
                    && vectors_close(&value.vector, &expected_vector)
                    && (vector_l2_norm(&value.vector) - 1.0).abs() <= 1.0e-5
        ) && structured_events.iter().any(|event| {
            matches!(
                event,
                EmbeddingGemmaLoadEvent::TokenizerLoaded {
                    path_suffix,
                    vocab_size: 4,
                } if path_suffix.ends_with("tokenizer.json")
            )
        }) && observed_all_required_tensor_loads(&structured_events)
            && structured_events.iter().any(|event| {
                matches!(event, EmbeddingGemmaLoadEvent::TokenIds { ids } if ids == &[3, 1, 2])
            })
            && observed_all_forward_stages(&structured_events)
            && observed_all_candle_tensor_digests(&structured_events, &expected_stage_digests)
            && observed_artifact_fingerprint(
                &structured_events,
                "model.safetensors",
                &expected_model_fingerprint,
            ),
        "warm cache must reject projection-only shortcuts, then use tokenizer ids and the safetensors Gemma3 backbone/projection graph in a Candle forward pass matching the independently computed golden vector and every intermediate Candle tensor digest; the proof must come through the test-owned structured loader observer, output={output:?}, golden_prefix={:?}, expected_stage_digests={expected_stage_digests:?}, structured_events={structured_events:?}",
        &expected_vector[..8],
    );

    let variable_cache = tempfile::tempdir().expect("variable model cache tempdir");
    let variable_hidden = 6 + 2 * (uuid::Uuid::now_v7().as_u128() as usize % 3);
    let variable_intermediate = variable_hidden + 4;
    let variable_projection_intermediate = variable_hidden + 6;
    let (variable_model_dir, variable_token, variable_golden) =
        stage_valid_mini_embeddinggemma_cache_with_shape(
            variable_cache.path(),
            variable_hidden,
            variable_intermediate,
            variable_projection_intermediate,
        );
    let variable_expected = matryoshka_head(&variable_golden, 768);
    let variable_stage_digests = candle_mini_gemma_stage_digests(
        &mini_weights_from_safetensors(&variable_model_dir.join("model.safetensors")),
        &[3, 1, 2],
    );
    let variable_observer = Arc::new(RecordingLoadObserver::default());
    let variable_observer_trait: Arc<dyn EmbedderLoadObserver> = variable_observer.clone();
    let variable_embedder = EmbeddingGemmaTextEmbedder::new_with_load_observer(
        local_model_config(variable_cache.path()),
        variable_observer_trait,
    );
    let variable_output =
        variable_embedder.embed(EmbeddingInput::Text(format!("{variable_token} red sedan")));
    let variable_events = variable_embedder.structured_loader_events();

    assert!(
        matches!(
            &variable_output,
            Ok(value)
                if value.embedding_space_id == "text_default_embeddinggemma_300m_768"
                    && value.dimension == 768
                    && value.vector.len() == 768
                    && vectors_close(&value.vector, &variable_expected)
                    && (vector_l2_norm(&value.vector) - 1.0).abs() <= 1.0e-5
        ) && observed_all_required_tensor_loads_for(
            &variable_events,
            variable_hidden,
            variable_intermediate,
            variable_projection_intermediate,
        ) && observed_all_forward_stages_for(&variable_events, variable_hidden, 768)
            && observed_all_candle_tensor_digests(&variable_events, &variable_stage_digests),
        "warm cache loader must be config/tensor-shape driven, not hard-coded to the default 4-hidden-size mini fixture; hidden={variable_hidden}, intermediate={variable_intermediate}, projection_intermediate={variable_projection_intermediate}, output={variable_output:?}, expected_prefix={:?}, events={variable_events:?}",
        &variable_expected[..8],
    );
}

#[test]
fn embeddinggemma_progress_names_local_cache_revision_and_no_eager_load_boundary() {
    let cache = tempfile::tempdir().expect("model cache tempdir");
    stage_invalid_embeddinggemma_cache(cache.path());
    let embedder = EmbeddingGemmaTextEmbedder::new(local_model_config(cache.path()));

    assert_eq!(EmbeddingGemmaTextEmbedder::loader_entry_count(), 0);
    let progress = embedder.emit_cold_start_progress();
    assert!(
        progress.contains(cache.path().to_string_lossy().as_ref())
            && progress.contains("embeddinggemma-300m-matryoshka-v1")
            && progress.contains("offline-cache-only")
            && progress.contains("no preload at Store::open"),
        "cold-start progress must expose local cache path, pinned revision, offline policy, and no-preload boundary, got {progress:?}"
    );
}

#[test]
fn embeddinggemma_declares_768_dim_vectors_for_768_head() {
    let cache = tempfile::tempdir().expect("model cache tempdir");
    let embedder = embedder_with_dimension(768, cache.path());

    assert_eq!(
        embedder.embedding_space().embedding_space_id,
        "text_default_embeddinggemma_300m_768"
    );
    assert_eq!(embedder.embedding_space().dimension, 768);
}

#[test]
fn embeddinggemma_matryoshka_512_head_embeds_512_dim_space() {
    let cache = tempfile::tempdir().expect("model cache tempdir");
    let embedder = embedder_with_dimension(512, cache.path());

    assert_eq!(
        embedder.embedding_space().embedding_space_id,
        "text_default_embeddinggemma_300m_512"
    );
    assert_eq!(embedder.embedding_space().dimension, 512);
    assert_matryoshka_head_embeds_head_specific_vector(512);
}

#[test]
fn embeddinggemma_matryoshka_256_head_embeds_256_dim_space() {
    let cache = tempfile::tempdir().expect("model cache tempdir");
    let embedder = embedder_with_dimension(256, cache.path());

    assert_eq!(
        embedder.embedding_space().embedding_space_id,
        "text_default_embeddinggemma_300m_256"
    );
    assert_eq!(embedder.embedding_space().dimension, 256);
    assert_matryoshka_head_embeds_head_specific_vector(256);
}

#[test]
fn embeddinggemma_matryoshka_128_head_embeds_128_dim_space() {
    let cache = tempfile::tempdir().expect("model cache tempdir");
    let embedder = embedder_with_dimension(128, cache.path());

    assert_eq!(
        embedder.embedding_space().embedding_space_id,
        "text_default_embeddinggemma_300m_128"
    );
    assert_eq!(embedder.embedding_space().dimension, 128);
    assert_matryoshka_head_embeds_head_specific_vector(128);
}
