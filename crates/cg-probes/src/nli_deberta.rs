//! DeBERTa-v3 NLI cross-encoder verifier on candle (CPU): scores P(entailment) for a
//! (premise, hypothesis) pair via the sequence-classification head. The entailment class index
//! comes from the checkpoint's own id2label (exact match — see `entailment_label_index`), never
//! from an assumed label order.

use crate::verifier::{TrunkVerifier, entailment_label_index};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::{Config, DebertaV2SeqClassificationModel};
use std::path::Path;

const MAX_TOKENS: usize = 512;

pub struct DebertaNliVerifier {
    name: String,
    model: DebertaV2SeqClassificationModel,
    tokenizer: tokenizers::Tokenizer,
    entail_index: usize,
    device: Device,
}

impl DebertaNliVerifier {
    pub fn load(dir: &Path, name: &str) -> Result<Self, String> {
        let config_text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("read config.json: {e}"))?;
        let config: Config =
            serde_json::from_str(&config_text).map_err(|e| format!("parse config.json: {e}"))?;
        let id2label = config
            .id2label
            .as_ref()
            .ok_or("config.json has no id2label")?
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        let entail_index = entailment_label_index(&id2label)?;
        let mut tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| format!("load tokenizer.json: {e}"))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| format!("set truncation: {e}"))?;
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                &device,
            )
        }
        .map_err(|e| format!("load model.safetensors: {e}"))?;
        let model = DebertaV2SeqClassificationModel::load(vb, &config, None)
            .map_err(|e| format!("build DeBERTa classifier: {e}"))?;
        Ok(Self {
            name: name.to_string(),
            model,
            tokenizer,
            entail_index,
            device,
        })
    }
}

impl TrunkVerifier for DebertaNliVerifier {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn entail(&mut self, premise: &str, hypothesis: &str) -> Result<f32, String> {
        let encoding = self
            .tokenizer
            .encode((premise, hypothesis), true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let input_ids = Tensor::new(ids.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| format!("input tensor: {e}"))?;
        let logits = self
            .model
            .forward(&input_ids, None, None)
            .map_err(|e| format!("forward: {e}"))?;
        let probs = candle_nn::ops::softmax(&logits, 1).map_err(|e| format!("softmax: {e}"))?;
        let row: Vec<f32> = probs
            .squeeze(0)
            .and_then(|t| t.to_vec1())
            .map_err(|e| format!("read probs: {e}"))?;
        row.get(self.entail_index).copied().ok_or_else(|| {
            format!(
                "entail index {} out of {} classes",
                self.entail_index,
                row.len()
            )
        })
    }
}
