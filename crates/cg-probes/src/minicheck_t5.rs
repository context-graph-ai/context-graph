//! MiniCheck Flan-T5 grounding checker on candle (CPU): P(claim is supported by doc) via the
//! authors' published wire contract — input `"predict: " + doc + "</s>" + claim`, one decoder
//! step from the start token, support probability = two-way softmax over the logits at label
//! token ids [3, 209] with 209 = support.

use crate::verifier::{TrunkVerifier, minicheck_input, softmax2};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::t5::{Config, T5ForConditionalGeneration};
use std::path::Path;

const NO_SUPPORT_TOKEN: usize = 3;
const SUPPORT_TOKEN: usize = 209;
const MAX_TOKENS: usize = 2048;

pub struct MiniCheckT5Verifier {
    name: String,
    model: T5ForConditionalGeneration,
    tokenizer: tokenizers::Tokenizer,
    decoder_start: u32,
    device: Device,
}

impl MiniCheckT5Verifier {
    pub fn load(dir: &Path, name: &str) -> Result<Self, String> {
        let config_text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("read config.json: {e}"))?;
        let config: Config =
            serde_json::from_str(&config_text).map_err(|e| format!("parse config.json: {e}"))?;
        let decoder_start = config.decoder_start_token_id.unwrap_or(0) as u32;
        let mut tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| format!("load tokenizer.json: {e}"))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| format!("set truncation: {e}"))?;
        let device = Device::Cpu;
        let vb = VarBuilder::from_pth(dir.join("pytorch_model.bin"), DType::F32, &device)
            .map_err(|e| format!("load pytorch_model.bin: {e}"))?;
        let model = T5ForConditionalGeneration::load(vb, &config)
            .map_err(|e| format!("build MiniCheck T5: {e}"))?;
        Ok(Self {
            name: name.to_string(),
            model,
            tokenizer,
            decoder_start,
            device,
        })
    }
}

impl TrunkVerifier for MiniCheckT5Verifier {
    fn name(&self) -> String {
        self.name.clone()
    }

    /// premise = the grounding document, hypothesis = the claim.
    fn entail(&mut self, premise: &str, hypothesis: &str) -> Result<f32, String> {
        self.model.clear_kv_cache();
        let text = minicheck_input(premise, hypothesis);
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let input_ids = Tensor::new(ids.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| format!("input tensor: {e}"))?;
        let decoder_input = Tensor::new(&[self.decoder_start], &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| format!("decoder tensor: {e}"))?;
        let logits = self
            .model
            .forward(&input_ids, &decoder_input)
            .map_err(|e| format!("forward: {e}"))?;
        // [1, 1, vocab] → the two label-token logits.
        let flat = logits
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| format!("read logits: {e}"))?;
        let (no_support, support) = (
            *flat
                .get(NO_SUPPORT_TOKEN)
                .ok_or("vocab smaller than label token ids")?,
            *flat
                .get(SUPPORT_TOKEN)
                .ok_or("vocab smaller than label token ids")?,
        );
        let (_, p_support) = softmax2(no_support, support);
        Ok(p_support)
    }
}
