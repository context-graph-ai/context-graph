use std::future::Future;
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
#[non_exhaustive]
pub struct ModelInvocation {
    pub request: UnderstandingRequest,
    pub resolved_connection: Option<String>,
    pub resolved_credential: Option<String>,
    pub hosted_endpoint: Option<String>,
    pub cache_path: Option<PathBuf>,
    pub local_artifacts: Vec<PathBuf>,
    pub local_catalog_authorized: bool,
}

impl ModelInvocation {
    /// Creates an invocation after a catalog/runtime consent and cache-proof check.
    ///
    /// # Safety
    ///
    /// Callers must only use this after resolving the named connection through the
    /// catalog consent chokepoint or proving the local artifact through catalog
    /// cache metadata. This constructor exists for the catalog runtime boundary;
    /// safe external code cannot forge a struct literal because the type is
    /// non-exhaustive.
    pub unsafe fn from_authorized_runtime_parts(
        request: UnderstandingRequest,
        resolved_connection: Option<String>,
        resolved_credential: Option<String>,
        hosted_endpoint: Option<String>,
        cache_path: Option<PathBuf>,
        local_artifacts: Vec<PathBuf>,
        local_catalog_authorized: bool,
    ) -> Self {
        Self {
            request,
            resolved_connection,
            resolved_credential,
            hosted_endpoint,
            cache_path,
            local_artifacts,
            local_catalog_authorized,
        }
    }
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

pub trait LlmClient: Send + Sync {
    fn understand(
        &self,
        invocation: &ModelInvocation,
    ) -> impl Future<Output = Result<ModelReply>> + Send;

    #[cfg(not(target_arch = "wasm32"))]
    fn understand_native(&self, invocation: &ModelInvocation) -> Result<ModelReply> {
        pollster::block_on(self.understand(invocation))
    }
}
