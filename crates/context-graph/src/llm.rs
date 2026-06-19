use std::future::Future;
use std::path::PathBuf;

use crate::{ContextId, ProviderUsage, Result};

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
    pub provider_config: Option<ProviderInvocationConfig>,
    pub cache_path: Option<PathBuf>,
    pub local_artifacts: Vec<PathBuf>,
    pub local_catalog_authorized: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderInvocationConfig {
    pub provider: String,
    pub provider_model: String,
    pub base_url: String,
    pub request_path: String,
    pub auth_scheme: String,
    pub dialect: String,
    pub reply_path: String,
    pub timeout_ms: u64,
    pub transport: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorizedModelInvocationParts {
    pub request: UnderstandingRequest,
    pub resolved_connection: Option<String>,
    pub resolved_credential: Option<String>,
    pub hosted_endpoint: Option<String>,
    pub provider_config: Option<ProviderInvocationConfig>,
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
    pub unsafe fn from_authorized_runtime_parts(parts: AuthorizedModelInvocationParts) -> Self {
        Self {
            request: parts.request,
            resolved_connection: parts.resolved_connection,
            resolved_credential: parts.resolved_credential,
            hosted_endpoint: parts.hosted_endpoint,
            provider_config: parts.provider_config,
            cache_path: parts.cache_path,
            local_artifacts: parts.local_artifacts,
            local_catalog_authorized: parts.local_catalog_authorized,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelReply {
    pub marker: String,
    pub phrasing: String,
    pub advisory_flag: bool,
    pub malformed: bool,
    pub usage: Option<ProviderUsage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureBucket {
    NotConnected,
    Unauthorized,
    Unreachable,
    ReplyMalformed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnderstandingOutcome {
    pub marker: String,
    pub degraded_marker: Option<String>,
    pub failure_bucket: Option<FailureBucket>,
    pub phrasing: Option<String>,
    pub model_invoked: bool,
    pub resolved_connection: Option<String>,
    pub cache_path_consulted: Option<PathBuf>,
    pub stored_understanding: Option<String>,
    pub usage: Option<ProviderUsage>,
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
