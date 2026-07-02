//! In-process, CPU-only, grammar-constrained local understanding engine.
//!
//! This crate wraps `llama-cpp-2` (which vendors and builds llama.cpp via cmake) so that cg can
//! run a bundled GGUF understanding model **in the host process** — no ollama, no server, no
//! visible runtime. It lives OUTSIDE the `context-graph` core crate (which must stay free of
//! HTTP/subprocess/native-model concerns); `cg-catalog` is the caller that maps between this
//! crate's types and the shipped `OutputContract` / `EnforcementLevel` substrate.
//!
//! ## Determinism and honesty contract
//!
//! * **Context sizing is always explicit.** [`GenerationParams`] has no `Default` that guesses a
//!   window — an unset/zero context silently truncates the prompt, a known failure mode, so a
//!   zero window is a loud error ([`LocalLlmError::ContextWindowZero`]) and a prompt that does not
//!   fit the window is a loud error ([`LocalLlmError::ContextTooSmall`]) rather than a silent cut.
//! * **Grammar constraint makes off-shape replies structurally impossible.** When the caller
//!   passes an [`LocalOutputContract::Schema`], the JSON schema is converted to a llama.cpp GBNF
//!   grammar and attached to the sampler; the reported [`LocalEnforcementLevel`] is `Native`. On
//!   any schema-conversion or grammar-attach failure the call degrades — **before generation** —
//!   to the prompt-block floor (no grammar), reported honestly as `Floor`. The level is
//!   achieved-not-configured: it reflects whether a grammar was genuinely attached, never a flag.
//! * **The recipe is not baked in here.** Temperature, context sizing and the instruction/schema
//!   prompt block are rendered by the caller's harness and handed to [`LocalUnderstandingEngine::understand`];
//!   the engine takes what it is given (the known "bare schema + default temp → grammar-valid but
//!   empty" probe lesson lives in the harness, not the engine).
//!
//! ## Threading
//!
//! llama.cpp's backend is a process-global singleton (it initializes ggml once). This crate
//! initializes it exactly once behind a [`std::sync::OnceLock`] and shares it across every engine
//! instance, so constructing more than one [`LlamaCppEngine`] in a process is safe. A single
//! engine owns its (read-only) model weights and creates a **fresh context per `understand` call**,
//! so calls do not leak KV-cache state into each other and the engine is `Send + Sync`.

mod error;

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

pub use error::LocalLlmError;

/// The output contract the caller asks the engine to honor for one call.
///
/// Mirrors the shipped `context_graph::OutputContract` shape, but is defined here so this crate
/// stays a pure llama.cpp wrapper with no `context-graph` dependency; `cg-catalog` maps between
/// the two.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum LocalOutputContract {
    /// No structural constraint — generate freely; the caller parses the reply.
    #[default]
    Unconstrained,
    /// A JSON Schema (as a string) the reply must satisfy. When it converts to a llama.cpp
    /// grammar the reply is structurally forced to match (`Native`); otherwise the call degrades
    /// to the unconstrained floor (`Floor`).
    Schema(String),
}

/// How strongly the reply's shape was actually enforced on this call — achieved, not configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalEnforcementLevel {
    /// A grammar built from the schema was genuinely attached to the sampler.
    Native,
    /// No grammar was attached (unconstrained request, or a schema that could not be represented
    /// as a grammar) — the reply rode the prompt-block floor.
    Floor,
}

impl LocalEnforcementLevel {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Native => "Native",
            Self::Floor => "Floor",
        }
    }
}

/// The distinct terminal shapes a local generation can take, mirroring `ModelCallOutcome`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalOutcome {
    /// The model produced non-empty content and stopped on an end-of-generation token.
    Captured,
    /// The model stopped on end-of-generation with no usable (non-whitespace) content — the
    /// refusal-shaped / laziest-valid-completion outcome.
    Empty,
    /// Generation hit the token budget (or the context wall) before an end-of-generation token.
    Truncation,
}

impl LocalOutcome {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Captured => "captured",
            Self::Empty => "empty",
            Self::Truncation => "truncation",
        }
    }
}

/// The per-call receipt: how the reply was enforced and what it cost, in tokens and wall-clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalReceipt {
    pub level: LocalEnforcementLevel,
    pub outcome: LocalOutcome,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub wall_clock_ms: u128,
}

/// A completed local understanding call: the reply text plus its receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalReply {
    pub text: String,
    pub receipt: LocalReceipt,
}

/// Per-call generation parameters. Deliberately has **no `Default`**: the context window must be
/// stated explicitly by the caller (the harness), because an unset window silently truncates.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationParams {
    /// The explicit context window in tokens (prompt + generation). Never a silent default.
    pub n_ctx: u32,
    /// Upper bound on generated tokens before the call reports `Truncation`.
    pub max_tokens: u32,
    /// Sampling temperature. `0.0` (the constructor default) means greedy/deterministic.
    pub temperature: f32,
    /// CPU threads for generation and prompt evaluation.
    pub n_threads: i32,
    /// Seed for the `temperature > 0` sampling path (ignored at temperature 0 / greedy).
    pub seed: u32,
}

impl GenerationParams {
    /// Construct with the required explicit context window; temperature defaults to `0.0`
    /// (greedy), threads to the machine's available parallelism, and a generous token budget.
    #[must_use]
    pub fn new(n_ctx: u32) -> Self {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        Self {
            n_ctx,
            max_tokens: 2048,
            temperature: 0.0,
            n_threads,
            seed: 0,
        }
    }

    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    #[must_use]
    pub fn with_n_threads(mut self, n_threads: i32) -> Self {
        self.n_threads = n_threads;
        self
    }

    #[must_use]
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// Build the llama.cpp context params, propagating the **explicit** context window. Errors
    /// loudly on a zero window rather than letting llama.cpp fall back to its silent default.
    fn to_context_params(&self) -> Result<LlamaContextParams, LocalLlmError> {
        let n_ctx = NonZeroU32::new(self.n_ctx).ok_or(LocalLlmError::ContextWindowZero)?;
        Ok(LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx))
            .with_n_threads(self.n_threads)
            .with_n_threads_batch(self.n_threads))
    }
}

/// A local understanding engine: load a GGUF model once, then run grammar-constrained
/// understanding calls against it in-process.
pub trait LocalUnderstandingEngine: Send + Sync {
    /// Run one understanding call. `prompt` is the fully-rendered instruction+schema+transcript
    /// text from the caller's harness; `contract` selects grammar enforcement; `params` carries
    /// the explicit context window, temperature, and budget.
    fn understand(
        &self,
        prompt: &str,
        contract: &LocalOutputContract,
        params: &GenerationParams,
    ) -> Result<LocalReply, LocalLlmError>;
}

/// Process-global llama.cpp backend. Initialized exactly once and shared by every engine.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
static BACKEND_INIT_LOCK: Mutex<()> = Mutex::new(());

fn shared_backend() -> Result<&'static LlamaBackend, LocalLlmError> {
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    // Serialize first-time init: llama.cpp's `init` errors if called twice concurrently.
    let _guard = BACKEND_INIT_LOCK
        .lock()
        .map_err(|_| LocalLlmError::Backend("backend init lock poisoned".to_string()))?;
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    let backend = LlamaBackend::init().map_err(|err| LocalLlmError::Backend(err.to_string()))?;
    let _ = BACKEND.set(backend);
    Ok(BACKEND.get().expect("backend just set"))
}

/// The llama.cpp-backed [`LocalUnderstandingEngine`]. Owns its read-only model weights; each
/// `understand` call opens a fresh context so calls never share KV-cache state.
pub struct LlamaCppEngine {
    backend: &'static LlamaBackend,
    model: LlamaModel,
    chat_template: LlamaChatTemplate,
}

impl std::fmt::Debug for LlamaCppEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaCppEngine")
            .field("n_ctx_train", &self.model.n_ctx_train())
            .finish_non_exhaustive()
    }
}

impl LlamaCppEngine {
    /// Load a GGUF model from a path, CPU-only. Reads the chat template baked into the GGUF
    /// metadata (llama.cpp carries the model's own template) so the caller never supplies one.
    pub fn load_from_gguf(path: impl AsRef<Path>) -> Result<Self, LocalLlmError> {
        let path = path.as_ref();
        let backend = shared_backend()?;
        // CPU-only: zero GPU layers. No GPU backend is compiled in (no cuda/vulkan/metal feature),
        // so this is belt-and-suspenders against a future default that offloads.
        let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
        let model = LlamaModel::load_from_file(backend, path, &model_params).map_err(|err| {
            LocalLlmError::ModelLoad {
                path: path.display().to_string(),
                reason: err.to_string(),
            }
        })?;
        let chat_template = model
            .chat_template(None)
            .map_err(|err| LocalLlmError::ChatTemplate(err.to_string()))?;
        Ok(Self {
            backend,
            model,
            chat_template,
        })
    }

    /// The model's trained context window — the ceiling a caller should size `n_ctx` against.
    #[must_use]
    pub fn n_ctx_train(&self) -> u32 {
        self.model.n_ctx_train()
    }

    /// Render the user prompt into the model's chat format with a hanging assistant turn, then
    /// tokenize with a leading BOS (llama.cpp `add_special`).
    ///
    /// Preferred path is the model's own template from GGUF metadata via
    /// `llama_chat_apply_template`. That C applier only handles the subset of jinja it recognizes,
    /// and gemma's template trips it (FFI error -1 — confirmed against this GGUF and matching the
    /// documented llama.cpp limitation). So on any apply failure we fall back to gemma's documented
    /// turn format (`<start_of_turn>user … <end_of_turn>\n<start_of_turn>model\n`) — the same shape
    /// `llama-cli` builds for gemma — which the tokenizer parses via the special-token path.
    fn render_prompt(&self, prompt: &str) -> String {
        let message = LlamaChatMessage::new("user".to_string(), prompt.to_string());
        if let Ok(message) = message
            && let Ok(rendered) =
                self.model
                    .apply_chat_template(&self.chat_template, &[message], true)
        {
            return rendered;
        }
        // gemma turn format fallback (no system role in gemma).
        format!("<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n")
    }

    fn tokenize_prompt(
        &self,
        prompt: &str,
    ) -> Result<Vec<llama_cpp_2::token::LlamaToken>, LocalLlmError> {
        let rendered = self.render_prompt(prompt);
        self.model
            .str_to_token(&rendered, AddBos::Always)
            .map_err(|err| LocalLlmError::Runtime(format!("tokenize failed: {err}")))
    }
}

impl LocalUnderstandingEngine for LlamaCppEngine {
    fn understand(
        &self,
        prompt: &str,
        contract: &LocalOutputContract,
        params: &GenerationParams,
    ) -> Result<LocalReply, LocalLlmError> {
        let started = Instant::now();
        let context_params = params.to_context_params()?;

        // Decide grammar BEFORE generation. A representable schema yields a grammar and `Native`;
        // anything else (unconstrained, or an unrepresentable schema) degrades to the `Floor`.
        let plan = grammar_plan(contract);
        // Attach the grammar to a sampler. If the grammar text is somehow rejected by llama.cpp
        // at attach time, degrade to the floor here (still before any token is generated).
        let (mut sampler, level) = match &plan.grammar {
            Some(grammar_text) => match LlamaSampler::grammar(&self.model, grammar_text, "root") {
                Ok(grammar_sampler) => (
                    build_chain(Some(grammar_sampler), params),
                    LocalEnforcementLevel::Native,
                ),
                Err(_) => (build_chain(None, params), LocalEnforcementLevel::Floor),
            },
            None => (build_chain(None, params), LocalEnforcementLevel::Floor),
        };

        let tokens = self.tokenize_prompt(prompt)?;
        let prompt_tokens = tokens.len();
        if prompt_tokens == 0 {
            return Err(LocalLlmError::Runtime(
                "prompt tokenized to zero tokens".to_string(),
            ));
        }
        // The prompt must fit the explicit window with at least one slot to generate into.
        if prompt_tokens as u32 >= params.n_ctx {
            return Err(LocalLlmError::ContextTooSmall {
                prompt_tokens,
                n_ctx: params.n_ctx,
            });
        }

        let mut context = self
            .model
            .new_context(self.backend, context_params)
            .map_err(|err| LocalLlmError::Runtime(format!("context init failed: {err}")))?;

        // Evaluate the prompt: one batch, logits only on the final token.
        let mut batch = LlamaBatch::new(params.n_ctx as usize, 1);
        let last_index = tokens.len() - 1;
        for (i, token) in tokens.iter().enumerate() {
            batch
                .add(*token, i as i32, &[0], i == last_index)
                .map_err(|err| LocalLlmError::Runtime(format!("prompt batch add failed: {err}")))?;
        }
        context
            .decode(&mut batch)
            .map_err(|err| LocalLlmError::Runtime(format!("prompt decode failed: {err}")))?;

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut text = String::new();
        let mut generated_tokens = 0usize;
        let mut position = tokens.len() as i32;
        let mut hit_budget = false;

        loop {
            if generated_tokens >= params.max_tokens as usize {
                hit_budget = true;
                break;
            }
            // Sample (and accept, so the grammar advances) from the last decoded position.
            let token = sampler.sample(&context, batch.n_tokens() - 1);
            if self.model.is_eog_token(token) {
                break;
            }
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, false, None)
                .map_err(|err| LocalLlmError::Runtime(format!("token decode failed: {err}")))?;
            text.push_str(&piece);
            generated_tokens += 1;

            if position as u32 >= params.n_ctx {
                // Context wall reached before an end-of-generation token.
                hit_budget = true;
                break;
            }
            batch.clear();
            batch
                .add(token, position, &[0], true)
                .map_err(|err| LocalLlmError::Runtime(format!("gen batch add failed: {err}")))?;
            position += 1;
            context
                .decode(&mut batch)
                .map_err(|err| LocalLlmError::Runtime(format!("gen decode failed: {err}")))?;
        }

        let outcome = finalize_outcome(&text, hit_budget);
        Ok(LocalReply {
            text,
            receipt: LocalReceipt {
                level,
                outcome,
                prompt_tokens,
                generated_tokens,
                wall_clock_ms: started.elapsed().as_millis(),
            },
        })
    }
}

/// The grammar decision for a call: the (optional) GBNF grammar and the level it implies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarPlan {
    pub grammar: Option<String>,
    pub level: LocalEnforcementLevel,
}

/// Decide, before generation, whether a grammar can enforce the contract. A schema that converts
/// to a llama.cpp grammar yields `Native`; an unconstrained request or an unrepresentable schema
/// yields the `Floor`. This is the achieved-not-configured level decision made from the request;
/// the engine downgrades further only if grammar attach then fails at runtime.
#[must_use]
pub fn grammar_plan(contract: &LocalOutputContract) -> GrammarPlan {
    match contract {
        LocalOutputContract::Unconstrained => GrammarPlan {
            grammar: None,
            level: LocalEnforcementLevel::Floor,
        },
        LocalOutputContract::Schema(schema) => match schema_to_grammar(schema) {
            Some(grammar) => GrammarPlan {
                grammar: Some(grammar),
                level: LocalEnforcementLevel::Native,
            },
            None => GrammarPlan {
                grammar: None,
                level: LocalEnforcementLevel::Floor,
            },
        },
    }
}

/// Convert a JSON Schema string to a llama.cpp GBNF grammar, returning `None` when the schema
/// cannot be represented (the degrade signal). Uses llama.cpp's own `json_schema_to_grammar`, so
/// what this crate can enforce natively matches what the runtime actually supports.
#[must_use]
pub fn schema_to_grammar(schema_json: &str) -> Option<String> {
    match llama_cpp_2::json_schema_to_grammar(schema_json) {
        Ok(grammar) if !grammar.trim().is_empty() => Some(grammar),
        _ => None,
    }
}

/// Finalize the observed outcome from the generated text and whether the budget/context wall was
/// hit. Empty (whitespace-only) content that stopped cleanly is the distinct `Empty` outcome.
#[must_use]
pub fn finalize_outcome(text: &str, hit_budget: bool) -> LocalOutcome {
    if hit_budget {
        LocalOutcome::Truncation
    } else if text.trim().is_empty() {
        LocalOutcome::Empty
    } else {
        LocalOutcome::Captured
    }
}

/// Build the sampler chain. Greedy (deterministic) at temperature 0; temperature + distribution
/// sampling above 0. A grammar sampler, when present, goes first so it masks invalid tokens
/// before the selection step.
fn build_chain(grammar: Option<LlamaSampler>, params: &GenerationParams) -> LlamaSampler {
    let mut chain: Vec<LlamaSampler> = Vec::new();
    if let Some(grammar) = grammar {
        chain.push(grammar);
    }
    if params.temperature > 0.0 {
        chain.push(LlamaSampler::temp(params.temperature));
        chain.push(LlamaSampler::dist(params.seed));
    } else {
        chain.push(LlamaSampler::greedy());
    }
    LlamaSampler::chain_simple(chain)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one canonical harvest-shaped schema converts to a grammar (happy path) and the plan
    /// reports `Native`.
    #[test]
    fn representable_schema_converts_and_plans_native() {
        let schema = r#"{
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "goals": { "type": "array", "items": { "type": "string" } },
                "count": { "type": "integer" }
            },
            "required": ["goals", "count"]
        }"#;
        let grammar = schema_to_grammar(schema).expect("representable schema converts");
        assert!(
            grammar.contains("root"),
            "grammar should define a root rule, got: {grammar}"
        );
        let plan = grammar_plan(&LocalOutputContract::Schema(schema.to_string()));
        assert_eq!(plan.level, LocalEnforcementLevel::Native);
        assert!(plan.grammar.is_some());
    }

    /// An unparseable schema cannot be represented → `schema_to_grammar` returns `None` and the
    /// plan degrades to `Floor` (before any generation).
    #[test]
    fn unrepresentable_schema_degrades_to_floor() {
        let not_a_schema = "this is not json at all";
        assert!(schema_to_grammar(not_a_schema).is_none());
        let plan = grammar_plan(&LocalOutputContract::Schema(not_a_schema.to_string()));
        assert_eq!(plan.level, LocalEnforcementLevel::Floor);
        assert!(plan.grammar.is_none());
    }

    /// An unconstrained contract never attaches a grammar and always plans `Floor`.
    #[test]
    fn unconstrained_plans_floor() {
        let plan = grammar_plan(&LocalOutputContract::Unconstrained);
        assert_eq!(plan.level, LocalEnforcementLevel::Floor);
        assert!(plan.grammar.is_none());
    }

    /// The explicit context window is propagated into the llama.cpp context params.
    #[test]
    fn context_window_is_propagated() {
        let params = GenerationParams::new(4096);
        let ctx_params = params.to_context_params().expect("non-zero window");
        assert_eq!(ctx_params.n_ctx(), NonZeroU32::new(4096));
    }

    /// A zero context window is rejected loudly rather than silently defaulting.
    #[test]
    fn zero_context_window_is_rejected() {
        let params = GenerationParams::new(0);
        assert!(matches!(
            params.to_context_params(),
            Err(LocalLlmError::ContextWindowZero)
        ));
    }

    /// Outcome is finalized from the observed reply: budget hit → truncation, empty text → empty,
    /// otherwise captured.
    #[test]
    fn outcome_is_finalized_from_reply() {
        assert_eq!(
            finalize_outcome("some content", false),
            LocalOutcome::Captured
        );
        assert_eq!(finalize_outcome("   \n  ", false), LocalOutcome::Empty);
        assert_eq!(
            finalize_outcome("some content", true),
            LocalOutcome::Truncation
        );
        assert_eq!(finalize_outcome("", false), LocalOutcome::Empty);
    }
}
