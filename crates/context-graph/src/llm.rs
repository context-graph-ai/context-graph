use std::path::PathBuf;

use crate::{ContextId, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeploymentSurface {
    Edge,
    Server,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnderstandingConsumer {
    ConversationCondensation,
    BlueprintNormalization,
    WebhookEntityResolution,
    SignatureSemanticJudgment,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnderstandingRequest {
    pub context_id: ContextId,
    pub consumer: UnderstandingConsumer,
    pub deployment: DeploymentSurface,
    pub input: String,
    pub paid: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInvocation {
    pub request: UnderstandingRequest,
    pub resolved_connection: Option<String>,
    pub resolved_credential: Option<String>,
    pub cache_path: Option<PathBuf>,
    pub local_artifacts: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelReply {
    pub marker: String,
    pub phrasing: String,
    pub advisory_flag: bool,
    pub malformed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnderstandingOutcome {
    pub marker: String,
    pub degraded_marker: Option<String>,
    pub phrasing: Option<String>,
    pub model_invoked: bool,
    pub resolved_connection: Option<String>,
    pub cache_path_consulted: Option<PathBuf>,
    pub stored_understanding: Option<String>,
}

#[allow(async_fn_in_trait)]
pub trait LlmClient: Send + Sync {
    async fn understand(&self, invocation: &ModelInvocation) -> Result<ModelReply>;

    #[cfg(not(target_arch = "wasm32"))]
    fn understand_native(&self, invocation: &ModelInvocation) -> Result<ModelReply>;
}
