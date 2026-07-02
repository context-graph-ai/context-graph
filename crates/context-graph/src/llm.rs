use std::future::Future;
use std::path::PathBuf;

use serde_json::Value;

use crate::{CgError, ContextId, ProviderUsage, Result};

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
    SkillProposal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnderstandingRequest {
    pub context_id: ContextId,
    pub consumer: UnderstandingConsumer,
    pub deployment: DeploymentSurface,
    pub input: String,
    pub paid: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum OutputContract {
    #[default]
    Unconstrained,
    Schema(String),
}

const OUTPUT_CONTRACT_PREFIX: &str = "\x1ecg-output-contract:";
const MODEL_RECEIPT_PREFIX: &str = "\x1ecg-model-receipt:";
const MODEL_RECEIPT_MARKER_SEPARATOR: &str = ":\x1e";

impl UnderstandingRequest {
    #[must_use]
    pub fn with_output_contract(mut self, contract: OutputContract) -> Self {
        let input = self.input_without_output_contract().to_string();
        self.input = match contract {
            OutputContract::Unconstrained => input,
            OutputContract::Schema(schema) => {
                format!("{OUTPUT_CONTRACT_PREFIX}{}:{schema}{input}", schema.len())
            }
        };
        self
    }

    #[must_use]
    pub fn input_without_output_contract(&self) -> &str {
        match split_output_contract_input(&self.input) {
            Some((_, input)) => input,
            None => &self.input,
        }
    }

    #[must_use]
    pub fn output_contract(&self) -> OutputContract {
        split_output_contract_input(&self.input)
            .map(|(contract, _)| contract)
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalJsonSchema(Value);

impl CanonicalJsonSchema {
    pub fn new(value: Value) -> Result<Self> {
        if !value.is_object() {
            return Err(CgError::Engine(
                "output contract schema must be a JSON object".to_string(),
            ));
        }
        Ok(Self(canonical_json_value(value)))
    }

    #[must_use]
    pub fn as_value(&self) -> &Value {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnforcementLevel {
    Native,
    ShapeOnly,
    Floor,
}

impl EnforcementLevel {
    #[must_use]
    pub fn as_receipt_str(self) -> &'static str {
        match self {
            Self::Native => "Native",
            Self::ShapeOnly => "ShapeOnly",
            Self::Floor => "Floor",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCallOutcome {
    Captured,
    Malformed,
    Refusal,
    Truncation,
}

impl ModelCallOutcome {
    #[must_use]
    pub fn as_receipt_str(self) -> &'static str {
        match self {
            Self::Captured => "captured",
            Self::Malformed => "malformed",
            Self::Refusal => "refusal",
            Self::Truncation => "truncation",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCallReceipt {
    pub reliability_level: EnforcementLevel,
    pub outcome: ModelCallOutcome,
    pub structured_output_rejected: bool,
}

impl ModelCallReceipt {
    #[must_use]
    pub fn captured(reliability_level: EnforcementLevel) -> Self {
        Self {
            reliability_level,
            outcome: ModelCallOutcome::Captured,
            structured_output_rejected: false,
        }
    }
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
    pub structured_output: Option<String>,
    pub request_output_contract: OutputContract,
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

impl ModelReply {
    #[must_use]
    pub fn with_model_receipt(mut self, receipt: ModelCallReceipt) -> Self {
        let marker = self.marker_without_model_receipt().to_string();
        self.marker = encode_model_receipt_marker(&marker, receipt);
        self
    }

    #[must_use]
    pub fn model_call_receipt(&self) -> ModelCallReceipt {
        split_model_receipt_marker(&self.marker)
            .map(|(receipt, _)| receipt)
            .unwrap_or_else(|| {
                if self.malformed {
                    ModelCallReceipt {
                        reliability_level: EnforcementLevel::Floor,
                        outcome: ModelCallOutcome::Malformed,
                        structured_output_rejected: false,
                    }
                } else {
                    ModelCallReceipt::captured(EnforcementLevel::Floor)
                }
            })
    }

    #[must_use]
    pub fn marker_without_model_receipt(&self) -> &str {
        split_model_receipt_marker(&self.marker)
            .map(|(_, marker)| marker)
            .unwrap_or(&self.marker)
    }
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

fn split_output_contract_input(input: &str) -> Option<(OutputContract, &str)> {
    let encoded = input.strip_prefix(OUTPUT_CONTRACT_PREFIX)?;
    let (schema_len, after_len) = encoded.split_once(':')?;
    let schema_len = schema_len.parse::<usize>().ok()?;
    let schema = after_len.get(..schema_len)?;
    let input = after_len.get(schema_len..)?;
    Some((OutputContract::Schema(schema.to_string()), input))
}

fn encode_model_receipt_marker(marker: &str, receipt: ModelCallReceipt) -> String {
    format!(
        "{MODEL_RECEIPT_PREFIX}{}:{}:{}{MODEL_RECEIPT_MARKER_SEPARATOR}{marker}",
        receipt.reliability_level.as_receipt_str(),
        receipt.outcome.as_receipt_str(),
        receipt.structured_output_rejected
    )
}

fn split_model_receipt_marker(marker: &str) -> Option<(ModelCallReceipt, &str)> {
    let encoded = marker.strip_prefix(MODEL_RECEIPT_PREFIX)?;
    let (level, after_level) = encoded.split_once(':')?;
    let (outcome, after_outcome) = after_level.split_once(':')?;
    let (structured_output_rejected, marker) =
        after_outcome.split_once(MODEL_RECEIPT_MARKER_SEPARATOR)?;
    Some((
        ModelCallReceipt {
            reliability_level: enforcement_level_from_receipt_str(level)?,
            outcome: model_call_outcome_from_receipt_str(outcome)?,
            structured_output_rejected: structured_output_rejected.parse::<bool>().ok()?,
        },
        marker,
    ))
}

fn enforcement_level_from_receipt_str(level: &str) -> Option<EnforcementLevel> {
    match level {
        "Native" => Some(EnforcementLevel::Native),
        "ShapeOnly" => Some(EnforcementLevel::ShapeOnly),
        "Floor" => Some(EnforcementLevel::Floor),
        _ => None,
    }
}

fn model_call_outcome_from_receipt_str(outcome: &str) -> Option<ModelCallOutcome> {
    match outcome {
        "captured" => Some(ModelCallOutcome::Captured),
        "malformed" => Some(ModelCallOutcome::Malformed),
        "refusal" => Some(ModelCallOutcome::Refusal),
        "truncation" => Some(ModelCallOutcome::Truncation),
        _ => None,
    }
}

fn canonical_json_value(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut sorted = serde_json::Map::new();
            let mut entries = object.into_iter().collect::<Vec<_>>();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            for (key, value) in entries {
                sorted.insert(key, canonical_json_value(value));
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(canonical_json_value).collect()),
        value => value,
    }
}
