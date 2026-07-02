//! Error type for the local understanding engine. Hand-rolled to match the repo idiom
//! (no `thiserror`), leading with the user-facing failure in the message.

use std::fmt;

/// A failure while loading a local GGUF model or running a local understanding call.
#[derive(Debug)]
pub enum LocalLlmError {
    /// The llama.cpp backend could not be initialized for this process.
    Backend(String),
    /// The GGUF model file could not be loaded from disk.
    ModelLoad { path: String, reason: String },
    /// The model carries no usable chat template in its GGUF metadata.
    ChatTemplate(String),
    /// A generation call requested a zero-length context window. Context sizing must be
    /// explicit and non-zero — an unset/zero window silently truncates the prompt.
    ContextWindowZero,
    /// The rendered prompt does not fit inside the explicit context window with room to
    /// generate. Reported loudly rather than silently truncating the transcript.
    ContextTooSmall { prompt_tokens: usize, n_ctx: u32 },
    /// A llama.cpp call (tokenize / context / decode / sampler) failed.
    Runtime(String),
}

impl fmt::Display for LocalLlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(reason) => {
                write!(f, "local llm backend init failed: {reason}")
            }
            Self::ModelLoad { path, reason } => {
                write!(f, "local llm model load failed for {path}: {reason}")
            }
            Self::ChatTemplate(reason) => {
                write!(f, "local llm model has no usable chat template: {reason}")
            }
            Self::ContextWindowZero => write!(
                f,
                "local llm context window must be explicit and non-zero (a zero window truncates)"
            ),
            Self::ContextTooSmall {
                prompt_tokens,
                n_ctx,
            } => write!(
                f,
                "local llm prompt of {prompt_tokens} tokens does not fit the {n_ctx}-token context window (raise n_ctx or shorten the chunk)"
            ),
            Self::Runtime(reason) => write!(f, "local llm runtime error: {reason}"),
        }
    }
}

impl std::error::Error for LocalLlmError {}
