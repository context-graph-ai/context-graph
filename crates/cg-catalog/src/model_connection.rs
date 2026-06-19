use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use candle_core::{Device, Tensor};
use cg_cli_conventions::{CheckRegistrar, DiagnosticCheck, Finding};
use context_graph::{
    AuthorizedModelInvocationParts, CgError, ContextId, DeploymentSurface, FailureBucket,
    LlmClient, ModelInvocation, ModelReply, Principal, ProviderInvocationConfig,
    UnderstandingConsumer, UnderstandingOutcome, UnderstandingRequest,
};
use fs4::fs_std::FileExt;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::{
    CacheStatus, Capability, Catalog, CatalogEntry, CatalogRegistrations, License, ModelKind,
    Source,
};

pub const DEGRADED_MARKER: &str = "running without a connected model";
const HOSTED_TIMEOUT: Duration = Duration::from_secs(10);
const HOSTED_BODY_LIMIT: u64 = 1024 * 1024;
const HOSTED_CHAT_MAX_TOKENS: u32 = 4096;
const CONNECTION_CATALOG_SCOPE: &str = "connection-substrate";
const EMBEDDED_PROVIDER_SPECS: &str = include_str!("../providers.toml");
const NEW_PROVIDER_REQUIRED_FIELDS: &[&str] = &[
    "binary",
    "prompt_delivery",
    "reply_framing",
    "timeout_ms",
    "consent_required",
];
const NEW_HTTP_PROVIDER_REQUIRED_FIELDS: &[&str] = &[
    "transport",
    "auth_scheme",
    "base_url",
    "dialect",
    "reply_path",
];
const DEFAULT_CHAT_COMPLETIONS_REPLY_PATH: &str = "/choices/0/message/content";
const SEALED_STORED_SECRET_PREFIX: &str = "sealed:";
const PROBELESS_HOST_ACCOUNT: &str = "account not exposed by host CLI";
const DOCTOR_COMMAND_TIMEOUT_MS: u64 = 5_000;
const CLAUDE_PROTECTED_ENV: [&str; 6] = [
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "ANTHROPIC_BASE_URL",
];
const CODEX_PROTECTED_ENV: [&str; 2] = ["OPENAI_API_KEY", "CODEX_API_KEY"];

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProviderSpecs {
    pub providers: BTreeMap<String, ProviderSpec>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptDelivery {
    #[default]
    Stdin,
    Argv,
}

impl PromptDelivery {
    pub const ACCEPTED: &'static [&'static str] = &["stdin", "argv"];

    fn parse(value: &str) -> Option<Self> {
        match value {
            "stdin" => Some(Self::Stdin),
            "argv" => Some(Self::Argv),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stdin => "stdin",
            Self::Argv => "argv",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplyFraming {
    Json,
    Jsonl,
    File,
    #[default]
    Text,
}

impl ReplyFraming {
    pub const ACCEPTED: &'static [&'static str] = &["json", "jsonl", "file", "text"];

    fn parse(value: &str) -> Option<Self> {
        match value {
            "json" => Some(Self::Json),
            "jsonl" => Some(Self::Jsonl),
            "file" => Some(Self::File),
            "text" => Some(Self::Text),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Jsonl => "jsonl",
            Self::File => "file",
            Self::Text => "text",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderTransport {
    #[default]
    CliSubprocess,
    Http,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderAuthScheme {
    Bearer,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderDialect {
    OpenaiChatCompletions,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderSpec {
    #[serde(default)]
    pub transport: ProviderTransport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_scheme: Option<ProviderAuthScheme>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dialect: Option<ProviderDialect>,
    pub binary: String,
    #[serde(default)]
    pub completion_args: Vec<String>,
    pub prompt_delivery: PromptDelivery,
    #[serde(default)]
    pub env_remove: Vec<String>,
    #[serde(default)]
    pub env_preserve: Vec<String>,
    pub reply_framing: ReplyFraming,
    #[serde(default)]
    pub reply_path: String,
    #[serde(default)]
    pub success: Option<PredicateSpec>,
    pub timeout_ms: u64,
    pub consent_required: bool,
    #[serde(default)]
    pub required_help_flags: Vec<String>,
    #[serde(default)]
    pub last_verified: String,
    #[serde(default)]
    pub auth_probe_args: Vec<String>,
    #[serde(default)]
    pub auth_success: Option<PredicateSpec>,
    #[serde(default)]
    pub auth_required: Option<PredicateSpec>,
    #[serde(default)]
    pub auth_account_paths: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredicateSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub equals: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exists: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matches: Option<String>,
    #[serde(default)]
    pub all_of: Vec<PredicateSpec>,
    #[serde(default)]
    pub any_of: Vec<PredicateSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderSpecLoadError {
    pub path: Option<PathBuf>,
    pub provider: String,
    pub field: String,
    pub value: String,
    pub accepted: Vec<String>,
}

impl ProviderSpecLoadError {
    pub fn message(&self) -> String {
        let path = self
            .path
            .as_ref()
            .map(|path| format!("{}: ", path.display()))
            .unwrap_or_default();
        format!(
            "{path}provider `{}` has invalid `{}` value `{}`; accepted: {}",
            self.provider,
            self.field,
            self.value,
            self.accepted.join(", ")
        )
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProviderOverride {
    pub provider: String,
    pub timeout_ms: Option<u64>,
    pub remove_env: Option<Vec<String>>,
    pub consent_required: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Invocation {
    pub argv: Vec<String>,
    pub stdin: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplyError {
    MissingPath { path: String, framing: String },
    Unsuccessful { diagnostic: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Classification {
    Success,
    HostLoginExpired { message: String },
    RateLimited { message: String },
    BillingRequired { message: String },
    LiveCallFailed { message: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostSessionProbe {
    Valid {
        account: String,
        account_exposed: bool,
    },
    Expired,
    NotDrivable {
        reason: String,
    },
    Missing,
    UnsafeSettings {
        message: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelProof {
    pub connection_name: String,
    pub provider: String,
    pub reply: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelProofError {
    NoHostSession,
    ConsentNotRecorded,
    HostToolMissing { provider: String },
    HostLoginExpired { message: String },
    LiveCallFailed { bucket: String, message: String },
    ReplyFormat { path: String, framing: String },
    UnsafeSettings { message: String },
}

impl ModelProofError {
    pub fn condition(&self) -> &'static str {
        match self {
            Self::NoHostSession => "no_host_session",
            Self::ConsentNotRecorded => "consent_not_recorded",
            Self::HostToolMissing { .. } => "host_tool_missing",
            Self::HostLoginExpired { .. } => "host_login_expired",
            Self::LiveCallFailed { .. } | Self::ReplyFormat { .. } => "live_call_failed",
            Self::UnsafeSettings { .. } => "live_call_failed",
        }
    }

    pub fn bucket(&self) -> &str {
        match self {
            Self::NoHostSession | Self::ConsentNotRecorded => "auth",
            Self::HostToolMissing { .. } => "auth",
            Self::HostLoginExpired { .. } => "auth",
            Self::LiveCallFailed { bucket, .. } => bucket.as_str(),
            Self::ReplyFormat { .. } => "format",
            Self::UnsafeSettings { .. } => "billing",
        }
    }

    pub fn message(&self) -> String {
        match self {
            Self::NoHostSession => "No drivable host session was found. Use a local model first; env-var key setup is a later fallback.".to_string(),
            Self::ConsentNotRecorded => "Consent is not recorded; run cg model auth login --connection <provider> --consent before cg selftest.".to_string(),
            Self::HostToolMissing { .. } => "The selected provider tool is gone; re-install it or run cg model auth login again.".to_string(),
            Self::HostLoginExpired { .. } => {
                "Host session expired; re-login with your agent before running cg selftest.".to_string()
            }
            Self::LiveCallFailed { message, .. } => message.clone(),
            Self::ReplyFormat { path, framing } => {
                format!("{path} absent in {framing} reply envelope")
            }
            Self::UnsafeSettings { message } => message.clone(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderSpecsFile {
    #[serde(default)]
    providers: BTreeMap<String, toml::Value>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderOverridesFile {
    #[serde(default)]
    providers: BTreeMap<String, toml::Value>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProviderSpec {
    #[serde(default)]
    transport: ProviderTransport,
    #[serde(default)]
    auth_scheme: Option<ProviderAuthScheme>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    default_model: Option<String>,
    #[serde(default)]
    dialect: Option<ProviderDialect>,
    binary: Option<String>,
    #[serde(default)]
    completion_args: Vec<String>,
    prompt_delivery: Option<String>,
    #[serde(default)]
    env_remove: Vec<String>,
    #[serde(default)]
    env_preserve: Vec<String>,
    reply_framing: Option<String>,
    #[serde(default)]
    reply_path: String,
    #[serde(default)]
    success: Option<PredicateSpec>,
    timeout_ms: Option<u64>,
    consent_required: Option<bool>,
    #[serde(default)]
    required_help_flags: Vec<String>,
    #[serde(default)]
    last_verified: String,
    #[serde(default)]
    auth_probe_args: Vec<String>,
    #[serde(default)]
    auth_success: Option<PredicateSpec>,
    #[serde(default)]
    auth_required: Option<PredicateSpec>,
    #[serde(default)]
    auth_account_paths: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderSpecOverride {
    transport: Option<ProviderTransport>,
    auth_scheme: Option<ProviderAuthScheme>,
    base_url: Option<String>,
    default_model: Option<String>,
    dialect: Option<ProviderDialect>,
    binary: Option<String>,
    completion_args: Option<Vec<String>>,
    prompt_delivery: Option<String>,
    env_remove: Option<Vec<String>>,
    env_preserve: Option<Vec<String>>,
    reply_framing: Option<String>,
    reply_path: Option<String>,
    success: Option<PredicateSpec>,
    timeout_ms: Option<u64>,
    consent_required: Option<bool>,
    required_help_flags: Option<Vec<String>>,
    last_verified: Option<String>,
    auth_probe_args: Option<Vec<String>>,
    auth_success: Option<PredicateSpec>,
    auth_required: Option<PredicateSpec>,
    auth_account_paths: Option<Vec<String>>,
}

#[derive(Debug)]
struct ProviderRun {
    status: Option<ExitStatus>,
    stdout: String,
    stderr: String,
    timed_out: bool,
}

pub fn load_provider_specs(root: &Path, overrides: &[ProviderOverride]) -> ProviderSpecs {
    try_load_provider_specs(root, overrides)
        .expect("provider spec validation should be preflighted")
}

pub fn validate_embedded_provider_specs() -> std::result::Result<(), ProviderSpecLoadError> {
    parse_provider_specs(EMBEDDED_PROVIDER_SPECS, None).map(|_| ())
}

pub fn try_load_provider_specs(
    root: &Path,
    overrides: &[ProviderOverride],
) -> std::result::Result<ProviderSpecs, ProviderSpecLoadError> {
    let mut providers = parse_provider_specs(EMBEDDED_PROVIDER_SPECS, None)?.providers;
    let user_spec_path = root.join("cg/providers.toml");
    match fs::read_to_string(&user_spec_path) {
        Ok(text) => {
            let file = parse_provider_overrides(&text, &user_spec_path)?;
            for (provider, override_layer) in file {
                if let Some(spec) = providers.get_mut(&provider) {
                    apply_provider_override(&provider, spec, override_layer)?;
                } else {
                    let spec =
                        provider_spec_from_override(&provider, override_layer, &user_spec_path)?;
                    providers.insert(provider.clone(), spec);
                }
                let spec = providers
                    .get(&provider)
                    .expect("provider must exist after override application");
                validate_provider_spec(&provider, spec, Some(&user_spec_path))?;
            }
        }
        Err(err) if err.kind() == ErrorKind::NotFound => {}
        Err(err) => {
            return Err(ProviderSpecLoadError {
                path: Some(user_spec_path),
                provider: "<override>".to_string(),
                field: "providers.toml".to_string(),
                value: err.to_string(),
                accepted: vec!["readable TOML provider override file or no file".to_string()],
            });
        }
    }
    for override_layer in overrides {
        if let Some(spec) = providers.get_mut(&override_layer.provider) {
            if let Some(timeout_ms) = override_layer.timeout_ms {
                spec.timeout_ms = timeout_ms;
            }
            if let Some(remove_env) = &override_layer.remove_env {
                for name in remove_env {
                    push_unique(&mut spec.env_remove, name.clone());
                }
            }
            if override_layer.consent_required == Some(true) {
                spec.consent_required = true;
            }
            protect_provider_spec(&override_layer.provider, spec);
            validate_provider_spec(&override_layer.provider, spec, None)?;
        }
    }
    for (provider, spec) in providers.iter_mut() {
        protect_provider_spec(provider, spec);
        validate_provider_spec(provider, spec, None)?;
    }
    Ok(ProviderSpecs { providers })
}

fn parse_provider_specs(
    text: &str,
    path: Option<&Path>,
) -> std::result::Result<ProviderSpecs, ProviderSpecLoadError> {
    let raw = toml::from_str::<ProviderSpecsFile>(text)
        .map_err(|err| parse_error(path, "<root>", "providers", err.to_string()))?;
    let mut providers = BTreeMap::new();
    for (provider, value) in raw.providers {
        let raw_spec = value
            .try_into::<RawProviderSpec>()
            .map_err(|err| parse_error(path, &provider, "provider", err.to_string()))?;
        let spec = provider_spec_from_raw(&provider, raw_spec, path)?;
        validate_provider_spec(&provider, &spec, path)?;
        providers.insert(provider, spec);
    }
    Ok(ProviderSpecs { providers })
}

fn parse_provider_overrides(
    text: &str,
    path: &Path,
) -> std::result::Result<BTreeMap<String, ProviderSpecOverride>, ProviderSpecLoadError> {
    let raw = toml::from_str::<ProviderOverridesFile>(text)
        .map_err(|err| parse_error(Some(path), "<root>", "providers", err.to_string()))?;
    let mut providers = BTreeMap::new();
    for (provider, value) in raw.providers {
        let override_layer = value
            .try_into::<ProviderSpecOverride>()
            .map_err(|err| parse_error(Some(path), &provider, "provider", err.to_string()))?;
        validate_provider_override(&provider, &override_layer, path)?;
        providers.insert(provider, override_layer);
    }
    Ok(providers)
}

fn parse_error(
    path: Option<&Path>,
    provider: &str,
    field: &str,
    message: String,
) -> ProviderSpecLoadError {
    let field = unknown_field_from_message(&message).unwrap_or_else(|| field.to_string());
    ProviderSpecLoadError {
        path: path.map(Path::to_path_buf),
        provider: provider.to_string(),
        field,
        value: message,
        accepted: vec!["known provider spec fields".to_string()],
    }
}

fn unknown_field_from_message(message: &str) -> Option<String> {
    let marker = "unknown field `";
    let start = message.find(marker)? + marker.len();
    let rest = &message[start..];
    let end = rest.find('`')?;
    Some(rest[..end].to_string())
}

fn provider_spec_from_raw(
    provider: &str,
    raw: RawProviderSpec,
    path: Option<&Path>,
) -> std::result::Result<ProviderSpec, ProviderSpecLoadError> {
    let prompt_delivery = match raw.transport {
        ProviderTransport::CliSubprocess => {
            let value = require_raw_field(provider, "prompt_delivery", raw.prompt_delivery, path)?;
            PromptDelivery::parse(&value).ok_or_else(|| {
                invalid_value(
                    path,
                    provider,
                    "prompt_delivery",
                    value.clone(),
                    PromptDelivery::ACCEPTED,
                )
            })?
        }
        ProviderTransport::Http => raw
            .prompt_delivery
            .as_deref()
            .and_then(PromptDelivery::parse)
            .unwrap_or_default(),
    };
    let reply_framing = match raw.transport {
        ProviderTransport::CliSubprocess => {
            let value = require_raw_field(provider, "reply_framing", raw.reply_framing, path)?;
            ReplyFraming::parse(&value).ok_or_else(|| {
                invalid_value(
                    path,
                    provider,
                    "reply_framing",
                    value.clone(),
                    ReplyFraming::ACCEPTED,
                )
            })?
        }
        ProviderTransport::Http => raw
            .reply_framing
            .as_deref()
            .and_then(ReplyFraming::parse)
            .unwrap_or(ReplyFraming::Json),
    };
    let binary = match raw.transport {
        ProviderTransport::CliSubprocess => {
            require_raw_field(provider, "binary", raw.binary, path)?
        }
        ProviderTransport::Http => raw.binary.unwrap_or_default(),
    };
    let timeout_ms = match raw.transport {
        ProviderTransport::CliSubprocess => {
            require_raw_field(provider, "timeout_ms", raw.timeout_ms, path)?
        }
        ProviderTransport::Http => raw.timeout_ms.unwrap_or_else(default_http_timeout_ms),
    };
    let consent_required = match raw.transport {
        ProviderTransport::CliSubprocess => {
            require_raw_field(provider, "consent_required", raw.consent_required, path)?
        }
        ProviderTransport::Http => raw.consent_required.unwrap_or(true),
    };
    Ok(ProviderSpec {
        transport: raw.transport,
        auth_scheme: raw.auth_scheme,
        base_url: raw.base_url,
        default_model: raw.default_model,
        dialect: raw.dialect,
        binary,
        completion_args: raw.completion_args,
        prompt_delivery,
        env_remove: raw.env_remove,
        env_preserve: raw.env_preserve,
        reply_framing,
        reply_path: raw.reply_path,
        success: raw.success,
        timeout_ms,
        consent_required,
        required_help_flags: raw.required_help_flags,
        last_verified: raw.last_verified,
        auth_probe_args: raw.auth_probe_args,
        auth_success: raw.auth_success,
        auth_required: raw.auth_required,
        auth_account_paths: raw.auth_account_paths,
    })
}

fn validate_provider_override(
    provider: &str,
    override_layer: &ProviderSpecOverride,
    path: &Path,
) -> std::result::Result<(), ProviderSpecLoadError> {
    if let Some(value) = &override_layer.prompt_delivery
        && PromptDelivery::parse(value).is_none()
    {
        return Err(invalid_value(
            Some(path),
            provider,
            "prompt_delivery",
            value.clone(),
            PromptDelivery::ACCEPTED,
        ));
    }
    if let Some(value) = &override_layer.reply_framing
        && ReplyFraming::parse(value).is_none()
    {
        return Err(invalid_value(
            Some(path),
            provider,
            "reply_framing",
            value.clone(),
            ReplyFraming::ACCEPTED,
        ));
    }
    if let Some(args) = &override_layer.completion_args {
        validate_completion_arg_placeholders(provider, args, Some(path))?;
    }
    validate_optional_predicate(
        provider,
        "success",
        override_layer.success.as_ref(),
        Some(path),
    )?;
    validate_optional_predicate(
        provider,
        "auth_success",
        override_layer.auth_success.as_ref(),
        Some(path),
    )?;
    validate_optional_predicate(
        provider,
        "auth_required",
        override_layer.auth_required.as_ref(),
        Some(path),
    )?;
    Ok(())
}

fn validate_provider_spec(
    provider: &str,
    spec: &ProviderSpec,
    path: Option<&Path>,
) -> std::result::Result<(), ProviderSpecLoadError> {
    match spec.transport {
        ProviderTransport::CliSubprocess => {
            if spec.binary.trim().is_empty() {
                return Err(missing_provider_field(
                    path,
                    provider,
                    "binary",
                    NEW_PROVIDER_REQUIRED_FIELDS,
                ));
            }
            validate_completion_arg_placeholders(provider, &spec.completion_args, path)?;
        }
        ProviderTransport::Http => {
            let Some(base_url) = spec.base_url.as_deref() else {
                return Err(missing_provider_field(
                    path,
                    provider,
                    "base_url",
                    NEW_HTTP_PROVIDER_REQUIRED_FIELDS,
                ));
            };
            if spec.auth_scheme.is_none() {
                return Err(missing_provider_field(
                    path,
                    provider,
                    "auth_scheme",
                    NEW_HTTP_PROVIDER_REQUIRED_FIELDS,
                ));
            }
            validate_http_base_url(provider, base_url, path)?;
            if let Some(default_model) = spec.default_model.as_deref()
                && (default_model.trim().is_empty() || default_model == provider)
            {
                return Err(invalid_value(
                    path,
                    provider,
                    "default_model",
                    default_model.to_string(),
                    &["non-empty model id distinct from the provider row name"],
                ));
            }
            if spec.dialect != Some(ProviderDialect::OpenaiChatCompletions) {
                return Err(invalid_value(
                    path,
                    provider,
                    "dialect",
                    spec.dialect
                        .map(|dialect| format!("{dialect:?}"))
                        .unwrap_or_else(|| "missing required provider spec field".to_string()),
                    &["openai-chat-completions"],
                ));
            }
            if spec.reply_path.trim().is_empty() || !spec.reply_path.starts_with('/') {
                return Err(invalid_value(
                    path,
                    provider,
                    "reply_path",
                    spec.reply_path.clone(),
                    &[DEFAULT_CHAT_COMPLETIONS_REPLY_PATH],
                ));
            }
        }
    }
    if spec.timeout_ms == 0 {
        return Err(invalid_value(
            path,
            provider,
            "timeout_ms",
            "0".to_string(),
            &["positive timeout in milliseconds"],
        ));
    }
    validate_optional_predicate(provider, "success", spec.success.as_ref(), path)?;
    validate_optional_predicate(provider, "auth_success", spec.auth_success.as_ref(), path)?;
    validate_optional_predicate(provider, "auth_required", spec.auth_required.as_ref(), path)?;
    Ok(())
}

fn validate_http_base_url(
    provider: &str,
    base_url: &str,
    path: Option<&Path>,
) -> std::result::Result<(), ProviderSpecLoadError> {
    if !valid_http_base_url(base_url) {
        return Err(invalid_value(
            path,
            provider,
            "base_url",
            base_url.to_string(),
            &["absolute http(s) URL with host and no userinfo, query, or fragment"],
        ));
    }
    Ok(())
}

pub fn validate_hosted_endpoint_base_url(
    provider: &str,
    base_url: &str,
) -> std::result::Result<(), String> {
    if valid_http_base_url(base_url) {
        return Ok(());
    }
    Err(format!(
        "provider {provider} endpoint must be an absolute http(s) URL with host and no userinfo, query, or fragment"
    ))
}

pub fn validate_hosted_endpoint_base_url_for_auth(
    provider: &str,
    base_url: &str,
    auth_scheme: ProviderAuthScheme,
) -> std::result::Result<(), String> {
    validate_hosted_endpoint_base_url(provider, base_url)?;
    if auth_scheme == ProviderAuthScheme::Bearer && !bearer_endpoint_is_secure_or_loopback(base_url)
    {
        return Err(format!(
            "provider {provider} bearer endpoint must use https, except loopback http for local development"
        ));
    }
    Ok(())
}

fn valid_http_base_url(base_url: &str) -> bool {
    let Ok(uri) = base_url.parse::<ureq::http::Uri>() else {
        return false;
    };
    !base_url
        .chars()
        .any(|char| char.is_whitespace() || char.is_control())
        && matches!(uri.scheme_str(), Some("http" | "https"))
        && uri.host().is_some_and(|host| !host.is_empty())
        && uri.query().is_none()
        && !base_url.contains('#')
        && !uri_authority_contains_userinfo(&uri)
        && !explicit_uri_port_is_invalid(&uri)
}

fn bearer_endpoint_is_secure_or_loopback(base_url: &str) -> bool {
    let Ok(uri) = base_url.parse::<ureq::http::Uri>() else {
        return false;
    };
    match uri.scheme_str() {
        Some("https") => true,
        Some("http") => uri.host().is_some_and(host_is_loopback),
        _ => false,
    }
}

fn host_is_loopback(host: &str) -> bool {
    let host = host.trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn uri_authority_contains_userinfo(uri: &ureq::http::Uri) -> bool {
    uri.authority()
        .is_some_and(|authority| authority.as_str().contains('@'))
}

fn explicit_uri_port_is_invalid(uri: &ureq::http::Uri) -> bool {
    let Some(authority) = uri.authority() else {
        return true;
    };
    let authority = authority.as_str();
    let explicit_port = if let Some(rest) = authority.strip_prefix('[') {
        let Some((_, after_host)) = rest.split_once(']') else {
            return true;
        };
        after_host.strip_prefix(':')
    } else {
        authority.rsplit_once(':').map(|(_, port)| port)
    };
    let Some(port) = explicit_port else {
        return false;
    };
    !port.chars().all(|char| char.is_ascii_digit()) || port.parse::<u16>().is_err()
}

fn provider_spec_from_override(
    provider: &str,
    override_layer: ProviderSpecOverride,
    path: &Path,
) -> std::result::Result<ProviderSpec, ProviderSpecLoadError> {
    let transport = override_layer.transport.unwrap_or_default();
    let binary = match transport {
        ProviderTransport::CliSubprocess => {
            require_override_field(provider, "binary", override_layer.binary, path)?
        }
        ProviderTransport::Http => override_layer.binary.unwrap_or_default(),
    };
    let prompt_delivery = match transport {
        ProviderTransport::CliSubprocess => {
            let value = require_override_field(
                provider,
                "prompt_delivery",
                override_layer.prompt_delivery,
                path,
            )?;
            PromptDelivery::parse(&value).ok_or_else(|| {
                invalid_value(
                    Some(path),
                    provider,
                    "prompt_delivery",
                    value.clone(),
                    PromptDelivery::ACCEPTED,
                )
            })?
        }
        ProviderTransport::Http => override_layer
            .prompt_delivery
            .as_deref()
            .and_then(PromptDelivery::parse)
            .unwrap_or_default(),
    };
    let reply_framing = match transport {
        ProviderTransport::CliSubprocess => {
            let value = require_override_field(
                provider,
                "reply_framing",
                override_layer.reply_framing,
                path,
            )?;
            ReplyFraming::parse(&value).ok_or_else(|| {
                invalid_value(
                    Some(path),
                    provider,
                    "reply_framing",
                    value.clone(),
                    ReplyFraming::ACCEPTED,
                )
            })?
        }
        ProviderTransport::Http => override_layer
            .reply_framing
            .as_deref()
            .and_then(ReplyFraming::parse)
            .unwrap_or(ReplyFraming::Json),
    };
    let timeout_ms = match transport {
        ProviderTransport::CliSubprocess => {
            require_override_field(provider, "timeout_ms", override_layer.timeout_ms, path)?
        }
        ProviderTransport::Http => override_layer
            .timeout_ms
            .unwrap_or_else(default_http_timeout_ms),
    };
    let consent_required = match transport {
        ProviderTransport::CliSubprocess => require_override_field(
            provider,
            "consent_required",
            override_layer.consent_required,
            path,
        )?,
        ProviderTransport::Http => override_layer.consent_required.unwrap_or(true),
    };
    let mut spec = ProviderSpec {
        transport,
        auth_scheme: override_layer.auth_scheme,
        base_url: override_layer.base_url,
        default_model: override_layer.default_model,
        dialect: override_layer.dialect,
        binary,
        completion_args: override_layer.completion_args.unwrap_or_default(),
        prompt_delivery,
        env_remove: override_layer.env_remove.unwrap_or_default(),
        env_preserve: override_layer.env_preserve.unwrap_or_default(),
        reply_framing,
        reply_path: override_layer.reply_path.unwrap_or_default(),
        success: override_layer.success,
        timeout_ms,
        consent_required,
        required_help_flags: override_layer.required_help_flags.unwrap_or_default(),
        last_verified: override_layer.last_verified.unwrap_or_default(),
        auth_probe_args: override_layer.auth_probe_args.unwrap_or_default(),
        auth_success: override_layer.auth_success,
        auth_required: override_layer.auth_required,
        auth_account_paths: override_layer.auth_account_paths.unwrap_or_default(),
    };
    protect_provider_spec(provider, &mut spec);
    Ok(spec)
}

fn require_raw_field<T>(
    provider: &str,
    field: &str,
    value: Option<T>,
    path: Option<&Path>,
) -> std::result::Result<T, ProviderSpecLoadError> {
    value.ok_or_else(|| missing_provider_field(path, provider, field, NEW_PROVIDER_REQUIRED_FIELDS))
}

fn require_override_field<T>(
    provider: &str,
    field: &str,
    value: Option<T>,
    path: &Path,
) -> std::result::Result<T, ProviderSpecLoadError> {
    value.ok_or_else(|| {
        missing_provider_field(Some(path), provider, field, NEW_PROVIDER_REQUIRED_FIELDS)
    })
}

fn missing_provider_field(
    path: Option<&Path>,
    provider: &str,
    field: &str,
    accepted_fields: &[&str],
) -> ProviderSpecLoadError {
    ProviderSpecLoadError {
        path: path.map(Path::to_path_buf),
        provider: provider.to_string(),
        field: field.to_string(),
        value: "missing required provider spec field".to_string(),
        accepted: vec![format!(
            "new providers must declare {}",
            accepted_fields.join(", ")
        )],
    }
}

fn validate_optional_predicate(
    provider: &str,
    field: &str,
    predicate: Option<&PredicateSpec>,
    path: Option<&Path>,
) -> std::result::Result<(), ProviderSpecLoadError> {
    if let Some(predicate) = predicate {
        validate_predicate(provider, field, predicate, path)?;
    }
    Ok(())
}

fn validate_predicate(
    provider: &str,
    field: &str,
    predicate: &PredicateSpec,
    path: Option<&Path>,
) -> std::result::Result<(), ProviderSpecLoadError> {
    let has_clause = predicate.equals.is_some()
        || predicate.exists.is_some()
        || predicate.matches.is_some()
        || !predicate.all_of.is_empty()
        || !predicate.any_of.is_empty();
    if !has_clause {
        return Err(ProviderSpecLoadError {
            path: path.map(Path::to_path_buf),
            provider: provider.to_string(),
            field: field.to_string(),
            value: "predicate has no recognized clause".to_string(),
            accepted: vec![
                "equals".to_string(),
                "exists".to_string(),
                "matches".to_string(),
                "all_of".to_string(),
                "any_of".to_string(),
            ],
        });
    }
    for child in &predicate.all_of {
        validate_predicate(provider, field, child, path)?;
    }
    for child in &predicate.any_of {
        validate_predicate(provider, field, child, path)?;
    }
    Ok(())
}

fn validate_completion_arg_placeholders(
    provider: &str,
    args: &[String],
    path: Option<&Path>,
) -> std::result::Result<(), ProviderSpecLoadError> {
    for arg in args {
        let mut rest = arg.as_str();
        while let Some(open) = rest.find('{') {
            let after_open = &rest[open + 1..];
            let Some(close) = after_open.find('}') else {
                break;
            };
            let token = &after_open[..close];
            if is_placeholder_token(token) && !matches!(token, "output_file" | "cwd") {
                return Err(invalid_value(
                    path,
                    provider,
                    "completion_args",
                    format!("{{{token}}}"),
                    &["{output_file}", "{cwd}"],
                ));
            }
            rest = &after_open[close + 1..];
        }
    }
    Ok(())
}

fn is_placeholder_token(token: &str) -> bool {
    !token.is_empty()
        && token
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn invalid_value(
    path: Option<&Path>,
    provider: &str,
    field: &str,
    value: String,
    accepted: &[&str],
) -> ProviderSpecLoadError {
    ProviderSpecLoadError {
        path: path.map(Path::to_path_buf),
        provider: provider.to_string(),
        field: field.to_string(),
        value,
        accepted: accepted.iter().map(|value| value.to_string()).collect(),
    }
}

fn apply_provider_override(
    provider: &str,
    spec: &mut ProviderSpec,
    override_layer: ProviderSpecOverride,
) -> std::result::Result<(), ProviderSpecLoadError> {
    if let Some(transport) = override_layer.transport {
        spec.transport = transport;
    }
    if let Some(auth_scheme) = override_layer.auth_scheme {
        spec.auth_scheme = Some(auth_scheme);
    }
    if let Some(base_url) = override_layer.base_url {
        spec.base_url = Some(base_url);
    }
    if let Some(default_model) = override_layer.default_model {
        spec.default_model = Some(default_model);
    }
    if let Some(dialect) = override_layer.dialect {
        spec.dialect = Some(dialect);
    }
    if let Some(binary) = override_layer.binary {
        spec.binary = binary;
    }
    if let Some(args) = override_layer.completion_args {
        spec.completion_args = args;
    }
    if let Some(prompt_delivery) = override_layer.prompt_delivery {
        spec.prompt_delivery = PromptDelivery::parse(&prompt_delivery).ok_or_else(|| {
            invalid_value(
                None,
                provider,
                "prompt_delivery",
                prompt_delivery.clone(),
                PromptDelivery::ACCEPTED,
            )
        })?;
    }
    if let Some(env_remove) = override_layer.env_remove {
        for name in env_remove {
            push_unique(&mut spec.env_remove, name);
        }
    }
    if let Some(env_preserve) = override_layer.env_preserve {
        spec.env_preserve = env_preserve;
    }
    if let Some(reply_framing) = override_layer.reply_framing {
        spec.reply_framing = ReplyFraming::parse(&reply_framing).ok_or_else(|| {
            invalid_value(
                None,
                provider,
                "reply_framing",
                reply_framing.clone(),
                ReplyFraming::ACCEPTED,
            )
        })?;
    }
    if let Some(reply_path) = override_layer.reply_path {
        spec.reply_path = reply_path;
    }
    if let Some(success) = override_layer.success {
        spec.success = Some(success);
    }
    if let Some(timeout_ms) = override_layer.timeout_ms {
        spec.timeout_ms = timeout_ms;
    }
    if override_layer.consent_required == Some(true) {
        spec.consent_required = true;
    }
    if let Some(required_help_flags) = override_layer.required_help_flags {
        spec.required_help_flags = required_help_flags;
    }
    if let Some(last_verified) = override_layer.last_verified {
        spec.last_verified = last_verified;
    }
    if let Some(auth_probe_args) = override_layer.auth_probe_args {
        spec.auth_probe_args = auth_probe_args;
    }
    if let Some(auth_success) = override_layer.auth_success {
        spec.auth_success = Some(auth_success);
    }
    if let Some(auth_required) = override_layer.auth_required {
        spec.auth_required = Some(auth_required);
    }
    if let Some(auth_account_paths) = override_layer.auth_account_paths {
        spec.auth_account_paths = auth_account_paths;
    }
    protect_provider_spec(provider, spec);
    Ok(())
}

fn protect_provider_spec(provider: &str, spec: &mut ProviderSpec) {
    for name in protected_env(provider) {
        push_unique(&mut spec.env_remove, name.to_string());
    }
    spec.consent_required = true;
}

fn protected_env(provider: &str) -> &'static [&'static str] {
    match provider {
        "claude" => &CLAUDE_PROTECTED_ENV,
        "codex" => &CODEX_PROTECTED_ENV,
        _ => &[],
    }
}

fn default_http_timeout_ms() -> u64 {
    HOSTED_TIMEOUT.as_millis() as u64
}

fn http_auth_scheme_name(auth_scheme: ProviderAuthScheme) -> &'static str {
    match auth_scheme {
        ProviderAuthScheme::Bearer => "bearer",
        ProviderAuthScheme::None => "none",
    }
}

fn provider_dialect_name(dialect: ProviderDialect) -> &'static str {
    match dialect {
        ProviderDialect::OpenaiChatCompletions => "openai-chat-completions",
    }
}

fn provider_request_path(dialect: ProviderDialect) -> &'static str {
    match dialect {
        ProviderDialect::OpenaiChatCompletions => "/chat/completions",
    }
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

pub fn construct_claude_invocation(prompt: &str) -> Invocation {
    let specs = load_provider_specs(Path::new(""), &[]);
    let Some(spec) = specs.providers.get("claude") else {
        return Invocation {
            argv: Vec::new(),
            stdin: prompt.to_string(),
        };
    };
    construct_provider_invocation(spec, prompt, None, &neutral_cwd())
}

fn construct_provider_invocation(
    spec: &ProviderSpec,
    prompt: &str,
    output_file: Option<&Path>,
    cwd: &Path,
) -> Invocation {
    let output_file = output_file
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    let cwd = cwd.display().to_string();
    let mut argv: Vec<String> = spec
        .completion_args
        .iter()
        .map(|arg| {
            arg.replace("{output_file}", &output_file)
                .replace("{cwd}", &cwd)
        })
        .collect();
    let stdin = match spec.prompt_delivery {
        PromptDelivery::Stdin => prompt.to_string(),
        PromptDelivery::Argv => {
            argv.push(prompt.to_string());
            String::new()
        }
    };
    Invocation { argv, stdin }
}

pub fn extract_reply(
    provider: &str,
    framing: &str,
    stdout: &str,
    output_file: Option<&Path>,
) -> std::result::Result<String, ReplyError> {
    let Some(framing_kind) = ReplyFraming::parse(framing) else {
        return Err(ReplyError::MissingPath {
            path: if provider == "codex" {
                "output_file".to_string()
            } else {
                "/result".to_string()
            },
            framing: framing.to_string(),
        });
    };
    let specs = load_provider_specs(Path::new(""), &[]);
    let fallback = ProviderSpec {
        reply_framing: framing_kind,
        reply_path: match framing {
            "file" => "output_file".to_string(),
            "text" => "stdout".to_string(),
            _ => "/result".to_string(),
        },
        ..ProviderSpec::default()
    };
    let spec = specs.providers.get(provider).unwrap_or(&fallback);
    extract_reply_from_spec(provider, spec, framing_kind, stdout, output_file)
}

fn extract_reply_from_spec(
    _provider: &str,
    spec: &ProviderSpec,
    framing: ReplyFraming,
    stdout: &str,
    output_file: Option<&Path>,
) -> std::result::Result<String, ReplyError> {
    let framing_name = framing.as_str();
    let reply_path = if spec.reply_path.is_empty() {
        "/result"
    } else {
        spec.reply_path.as_str()
    };
    match framing {
        ReplyFraming::Json => {
            let value = parse_json_or_last_line(stdout).map_err(|_| ReplyError::MissingPath {
                path: reply_path.to_string(),
                framing: framing_name.to_string(),
            })?;
            let reply = json_pointer_string(&value, reply_path);
            let success = spec
                .success
                .as_ref()
                .map(|predicate| evaluate_predicate(&value, predicate))
                .unwrap_or(true);
            if !success {
                if reply.is_none() {
                    return Err(ReplyError::MissingPath {
                        path: reply_path.to_string(),
                        framing: framing_name.to_string(),
                    });
                }
                return Err(ReplyError::Unsuccessful {
                    diagnostic: stdout.to_string(),
                });
            }
            reply.ok_or_else(|| ReplyError::MissingPath {
                path: reply_path.to_string(),
                framing: framing_name.to_string(),
            })
        }
        ReplyFraming::Jsonl => {
            for line in stdout.lines().rev().filter(|line| !line.trim().is_empty()) {
                if let Ok(value) = serde_json::from_str::<Value>(line) {
                    let success = spec
                        .success
                        .as_ref()
                        .map(|predicate| evaluate_predicate(&value, predicate))
                        .unwrap_or(true);
                    if success && let Some(reply) = json_pointer_string(&value, reply_path) {
                        return Ok(reply);
                    }
                }
            }
            Err(ReplyError::MissingPath {
                path: reply_path.to_string(),
                framing: framing_name.to_string(),
            })
        }
        ReplyFraming::File => output_file
            .and_then(|path| fs::read_to_string(path).ok())
            .ok_or_else(|| ReplyError::MissingPath {
                path: "output_file".to_string(),
                framing: framing_name.to_string(),
            }),
        ReplyFraming::Text => Ok(stdout.to_string()),
    }
}

pub fn classify_provider_reply(_provider: &str, stdout: &str) -> Classification {
    let parsed = parse_json_or_last_line(stdout);
    let Ok(value) = parsed else {
        let message = stdout.trim();
        return if message.is_empty() {
            Classification::LiveCallFailed {
                message: "provider call failed".to_string(),
            }
        } else {
            classify_message(message)
        };
    };
    if value
        .get("is_error")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let message = value
            .get("result")
            .and_then(Value::as_str)
            .unwrap_or("provider call failed")
            .to_string();
        return match value.get("api_error_status").and_then(Value::as_i64) {
            Some(401) => Classification::HostLoginExpired { message },
            Some(402) => Classification::BillingRequired { message },
            Some(429) => Classification::RateLimited { message },
            _ => classify_message(&message),
        };
    }
    Classification::Success
}

fn parse_json_or_last_line(text: &str) -> serde_json::Result<Value> {
    serde_json::from_str::<Value>(text).or_else(|_| {
        let last = text
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .unwrap_or(text);
        serde_json::from_str(last)
    })
}

fn classify_message(message: &str) -> Classification {
    let lower = message.to_ascii_lowercase();
    if lower.contains("invalid api key")
        || lower.contains("expired")
        || lower.contains("unauthorized")
    {
        Classification::HostLoginExpired {
            message: message.to_string(),
        }
    } else if lower.contains("billing") {
        Classification::BillingRequired {
            message: message.to_string(),
        }
    } else if lower.contains("rate") || lower.contains("quota") {
        Classification::RateLimited {
            message: message.to_string(),
        }
    } else {
        Classification::LiveCallFailed {
            message: message.to_string(),
        }
    }
}

fn json_pointer_string(value: &Value, path: &str) -> Option<String> {
    if path == "stdout" {
        return value.as_str().map(str::to_string);
    }
    value
        .pointer(path)
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn evaluate_predicate(value: &Value, predicate: &PredicateSpec) -> bool {
    if !predicate
        .all_of
        .iter()
        .all(|child| evaluate_predicate(value, child))
    {
        return false;
    }
    if !predicate.any_of.is_empty()
        && !predicate
            .any_of
            .iter()
            .any(|child| evaluate_predicate(value, child))
    {
        return false;
    }
    let target = predicate
        .path
        .as_deref()
        .and_then(|path| value.pointer(path));
    if let Some(exists) = predicate.exists
        && (target.is_some() != exists)
    {
        return false;
    }
    if let Some(expected) = &predicate.equals
        && target != Some(expected)
    {
        return false;
    }
    if let Some(pattern) = &predicate.matches {
        let Some(text) = target.and_then(Value::as_str) else {
            return false;
        };
        if !text.contains(pattern) {
            return false;
        }
    }
    true
}

pub fn probe_host_session_provider(root: &Path, provider: &str) -> HostSessionProbe {
    let specs = match try_load_provider_specs(root, &[]) {
        Ok(specs) => specs,
        Err(err) => return not_drivable(err.message()),
    };
    let Some(spec) = specs.providers.get(provider) else {
        return not_drivable("provider spec is missing");
    };
    if provider == "claude"
        && let Err(message) = inspect_claude_settings_for_reinjection()
    {
        return HostSessionProbe::UnsafeSettings { message };
    }
    let Some(binary) = find_provider_binary(spec) else {
        return HostSessionProbe::Missing;
    };
    if spec.auth_probe_args.is_empty() {
        return HostSessionProbe::Valid {
            account: PROBELESS_HOST_ACCOUNT.to_string(),
            account_exposed: false,
        };
    }
    let output = run_provider_probe(provider, &binary, spec);
    if output.timed_out {
        return not_drivable(format!(
            "auth probe timed out after {}ms",
            spec.timeout_ms.max(500)
        ));
    }
    let Some(status) = output.status else {
        return not_drivable("auth probe failed before reporting status");
    };
    let stdout = output.stdout;
    let stderr = output.stderr;
    if let Ok(value) = parse_json_or_last_line(&stdout)
        && status.success()
        && auth_value_matches(spec, &value)
    {
        return host_session_probe_from_auth_value(spec, &value);
    }

    let combined = format!("{stdout}\n{stderr}");
    if auth_output_is_explicitly_not_drivable(&combined) {
        return not_drivable("auth probe reported non-interactive probe failure");
    };
    if auth_output_is_expired(&combined) {
        return HostSessionProbe::Expired;
    }
    let Ok(value) = parse_json_or_last_line(&combined) else {
        return not_drivable("auth probe did not return recognized JSON status");
    };
    if status.success() && auth_value_matches(spec, &value) {
        return host_session_probe_from_auth_value(spec, &value);
    }
    not_drivable("auth probe did not report a drivable logged-in account")
}

fn not_drivable(reason: impl Into<String>) -> HostSessionProbe {
    HostSessionProbe::NotDrivable {
        reason: reason.into(),
    }
}

fn auth_value_matches(spec: &ProviderSpec, value: &Value) -> bool {
    let success = spec
        .auth_success
        .as_ref()
        .map(|predicate| evaluate_predicate(value, predicate))
        .unwrap_or(false);
    let required = spec
        .auth_required
        .as_ref()
        .map(|predicate| evaluate_predicate(value, predicate))
        .unwrap_or(true);
    success && required
}

fn host_session_probe_from_auth_value(spec: &ProviderSpec, value: &Value) -> HostSessionProbe {
    let account = spec
        .auth_account_paths
        .iter()
        .find_map(|path| json_pointer_string(value, path))
        .filter(|account| !account.trim().is_empty());
    let account_exposed = account.is_some();
    HostSessionProbe::Valid {
        account: account.unwrap_or_else(|| PROBELESS_HOST_ACCOUNT.to_string()),
        account_exposed,
    }
}

fn auth_output_is_expired(output: &str) -> bool {
    let lower = output.to_ascii_lowercase();
    if let Ok(value) = parse_json_or_last_line(output)
        && json_auth_is_expired(&value)
    {
        return true;
    }
    lower.contains("expired")
        || lower.contains("invalid api key")
        || lower.contains("not logged in")
        || lower.contains("not logged-in")
        || lower.contains("not authenticated")
        || lower.contains("not_authenticated")
        || lower.contains("not signed in")
        || lower.contains("not signed-in")
        || lower.contains("unauthenticated")
        || lower.contains("signed out")
        || lower.contains("signed-out")
        || lower.contains("logged out")
        || lower.contains("logged-out")
        || lower.contains("login required")
        || lower.contains("log in required")
        || lower.contains("sign in required")
        || lower.contains("please log in")
        || lower.contains("unauthorized")
}

fn json_auth_is_expired(value: &Value) -> bool {
    value.get("ok").and_then(Value::as_bool) == Some(false)
        || value.get("loggedIn").and_then(Value::as_bool) == Some(false)
        || value.get("logged_in").and_then(Value::as_bool) == Some(false)
        || value.get("authenticated").and_then(Value::as_bool) == Some(false)
        || value
            .get("status")
            .and_then(Value::as_str)
            .map(|status| {
                matches!(
                    status,
                    "not_authenticated"
                        | "unauthenticated"
                        | "not_logged_in"
                        | "logged_out"
                        | "signed_out"
                )
            })
            .unwrap_or(false)
}

fn auth_output_is_explicitly_not_drivable(output: &str) -> bool {
    output
        .to_ascii_lowercase()
        .contains("non_interactive_probe_failed")
}

pub fn register_model_connection_checks(registrar: &mut impl CheckRegistrar) {
    register_model_connection_checks_with_root(registrar, PathBuf::new());
}

pub fn model_connection_system_principal() -> Principal {
    Principal::System
}

pub fn register_model_connection_checks_with_root(
    registrar: &mut impl CheckRegistrar,
    root: impl AsRef<Path>,
) {
    let root = root.as_ref().to_path_buf();
    let specs = match try_load_provider_specs(&root, &[]) {
        Ok(specs) => specs,
        Err(err) => {
            registrar.register_check(DiagnosticCheck::free(
                "model/provider-spec/valid",
                move || {
                    vec![Finding::error(
                        "model/provider-spec/valid",
                        err.message(),
                        Some(err.provider.clone()),
                        Some(
                            "fix cg/providers.toml so every provider field is interpretable"
                                .to_string(),
                        ),
                    )]
                },
            ));
            return;
        }
    };
    for provider in provider_names_for_checks(&specs) {
        let check_root = root.clone();
        let check_id = format!("model/{provider}/flags-in-help");
        registrar.register_check(DiagnosticCheck::free(check_id, move || {
            detect_provider_help_drift(&check_root, &provider)
        }));
    }
    registrar.register_check(DiagnosticCheck::spending(
        "model/selftest/live-call",
        || {
            vec![Finding::error(
                "model/selftest/live-call",
                "selftest requires a host binary to wire the consent-gated live proof path",
                Some("selftest".to_string()),
                Some("call the model-connection runtime proof seam instead of treating registry execution as proof".to_string()),
            )]
        },
    ));
}

fn provider_names_for_checks(specs: &ProviderSpecs) -> Vec<String> {
    let mut names = specs
        .providers
        .iter()
        .filter(|(_, spec)| spec.transport == ProviderTransport::CliSubprocess)
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    names.sort();
    if let Some(index) = names.iter().position(|name| name == "claude") {
        names.swap(0, index);
    }
    names
}

fn detect_provider_help_drift(root: &Path, provider: &str) -> Vec<Finding> {
    let specs = match try_load_provider_specs(root, &[]) {
        Ok(specs) => specs,
        Err(err) => {
            return vec![Finding::error(
                "model/provider-spec/valid",
                err.message(),
                Some(err.provider.clone()),
                Some("fix cg/providers.toml so every provider field is interpretable".to_string()),
            )];
        }
    };
    let Some(spec) = specs.providers.get(provider) else {
        return Vec::new();
    };
    let Some(binary) = find_provider_binary(spec) else {
        return vec![Finding::warning(
            format!("model/{provider}/flags-in-help"),
            format!("{provider} provider tool was not found on PATH"),
            Some(provider.to_string()),
            Some(format!(
                "install {provider} or remove it from cg/providers.toml"
            )),
        )];
    };
    let timeout_ms = spec.timeout_ms.clamp(500, DOCTOR_COMMAND_TIMEOUT_MS);
    let version_args = vec!["--version".to_string()];
    let version_run =
        run_provider_command(provider, &binary, spec, &version_args, None, timeout_ms);
    if version_run.timed_out {
        return vec![Finding::warning(
            format!("model/{provider}/flags-in-help"),
            format!("{provider} version check timed out after {timeout_ms}ms"),
            Some(provider.to_string()),
            Some(format!(
                "check that {provider} starts without blocking before running cg doctor again"
            )),
        )];
    }
    if !version_run.status.is_some_and(|status| status.success()) {
        return vec![Finding::warning(
            format!("model/{provider}/flags-in-help"),
            format!(
                "{provider} version check exited unsuccessfully: {}",
                provider_run_output_summary(&version_run)
            ),
            Some(provider.to_string()),
            Some(format!("repair {provider} host CLI installation")),
        )];
    }
    let version = provider_run_combined_output(&version_run)
        .trim()
        .chars()
        .take(120)
        .collect::<String>();
    let version = if version.is_empty() {
        "unknown version".to_string()
    } else {
        version
    };
    if spec.required_help_flags.is_empty() {
        return Vec::new();
    }
    let help_args = provider_help_args(spec);
    let help_run = run_provider_command(provider, &binary, spec, &help_args, None, timeout_ms);
    if help_run.timed_out {
        return vec![Finding::warning(
            format!("model/{provider}/flags-in-help"),
            format!("{provider} {version} help check timed out after {timeout_ms}ms"),
            Some(provider.to_string()),
            Some(format!(
                "check that {provider} {:?} completes without blocking",
                help_args
            )),
        )];
    }
    let help = provider_run_combined_output(&help_run);
    spec.required_help_flags
        .iter()
        .filter(|required| !help.contains(required.as_str()))
        .map(|required| {
            Finding::warning(
                format!("model/{provider}/flags-in-help"),
                format!(
                    "{provider} {version} help is missing required host-session flag {required}"
                ),
                Some(provider.to_string()),
                Some(format!(
                    "upgrade {provider} so cg can pass {required} with its declared invocation"
                )),
            )
        })
        .collect()
}

fn provider_help_args(spec: &ProviderSpec) -> Vec<String> {
    let mut args = Vec::new();
    for arg in &spec.completion_args {
        if arg.starts_with('-') || arg.contains('{') {
            break;
        }
        args.push(arg.clone());
    }
    args.push("--help".to_string());
    args
}

fn provider_run_output_summary(run: &ProviderRun) -> String {
    let output = if run.stderr.trim().is_empty() {
        run.stdout.trim()
    } else {
        run.stderr.trim()
    };
    let output = if output.is_empty() {
        "no output"
    } else {
        output
    };
    output.chars().take(160).collect()
}

fn provider_run_combined_output(run: &ProviderRun) -> String {
    match (run.stdout.trim().is_empty(), run.stderr.trim().is_empty()) {
        (false, false) => format!("{}\n{}", run.stdout, run.stderr),
        (false, true) => run.stdout.clone(),
        (true, false) => run.stderr.clone(),
        (true, true) => String::new(),
    }
}

pub fn provider_tool_available(root: &Path, provider: &str) -> bool {
    let Ok(specs) = try_load_provider_specs(root, &[]) else {
        return false;
    };
    specs
        .providers
        .get(provider)
        .and_then(find_provider_binary)
        .is_some()
}

fn find_provider_binary(spec: &ProviderSpec) -> Option<PathBuf> {
    if spec.binary.is_empty() {
        return None;
    }
    which::which(&spec.binary).ok()
}

fn apply_provider_env_removal(command: &mut Command, provider: &str, spec: &ProviderSpec) {
    for name in spec
        .env_remove
        .iter()
        .map(String::as_str)
        .chain(protected_env(provider).iter().copied())
        .chain(CLAUDE_PROTECTED_ENV.iter().copied())
        .chain(CODEX_PROTECTED_ENV.iter().copied())
    {
        command.env_remove(name);
    }
}

fn neutral_cwd() -> PathBuf {
    std::env::temp_dir()
}

fn temp_output_path(provider: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir().join(format!(
        "cg-{provider}-last-message-{}-{nanos}.txt",
        std::process::id()
    ))
}

struct TempOutputFile {
    path: PathBuf,
}

impl TempOutputFile {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempOutputFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Debug, Clone)]
struct HostSessionLlmClient {
    provider: String,
    catalog_root: PathBuf,
    progress_to_stdout: bool,
}

impl HostSessionLlmClient {
    fn prove(
        &self,
        invocation: &ModelInvocation,
    ) -> std::result::Result<ModelReply, ModelProofError> {
        let specs = try_load_provider_specs(&self.catalog_root, &[]).map_err(|err| {
            ModelProofError::LiveCallFailed {
                bucket: "config".to_string(),
                message: err.message(),
            }
        })?;
        let Some(spec) = specs.providers.get(&self.provider) else {
            return Err(ModelProofError::HostToolMissing {
                provider: self.provider.clone(),
            });
        };
        if self.provider == "claude"
            && let Err(message) = inspect_claude_settings_for_reinjection()
        {
            return Err(ModelProofError::UnsafeSettings { message });
        }
        let Some(binary) = find_provider_binary(spec) else {
            return Err(ModelProofError::HostToolMissing {
                provider: self.provider.clone(),
            });
        };
        let output_file_guard = if spec.reply_framing == ReplyFraming::File {
            Some(TempOutputFile::new(temp_output_path(&self.provider)))
        } else {
            None
        };
        let output_file = output_file_guard.as_ref().map(TempOutputFile::path);
        let run = run_provider_completion(
            &self.provider,
            &binary,
            spec,
            &invocation.request.input,
            output_file,
            self.progress_to_stdout,
        );
        if run.timed_out {
            return Err(ModelProofError::LiveCallFailed {
                bucket: "timeout".to_string(),
                message: "The provider did not complete in time; cg did not retry the proof call."
                    .to_string(),
            });
        }
        let Some(status) = run.status else {
            return Err(ModelProofError::LiveCallFailed {
                bucket: "unknown".to_string(),
                message: "provider call failed before a status was available".to_string(),
            });
        };
        if !status.success() {
            let diagnostic = if run.stdout.trim().is_empty() {
                run.stderr.trim()
            } else {
                run.stdout.trim()
            };
            return Err(classified_proof_failure(&self.provider, diagnostic));
        }
        let reply = extract_reply_from_spec(
            &self.provider,
            spec,
            spec.reply_framing,
            &run.stdout,
            output_file,
        )
        .map_err(|err| match err {
            ReplyError::MissingPath { path, framing } => {
                ModelProofError::ReplyFormat { path, framing }
            }
            ReplyError::Unsuccessful { diagnostic } => {
                classified_proof_failure(&self.provider, &diagnostic)
            }
        })?;
        if reply.trim().is_empty() {
            return Err(ModelProofError::LiveCallFailed {
                bucket: "format".to_string(),
                message: "provider returned successfully but no proof reply was found".to_string(),
            });
        }
        Ok(ModelReply {
            marker: "host-session-proof".to_string(),
            phrasing: reply,
            advisory_flag: false,
            malformed: false,
            usage: None,
        })
    }
}

impl LlmClient for HostSessionLlmClient {
    async fn understand(&self, invocation: &ModelInvocation) -> context_graph::Result<ModelReply> {
        self.understand_native(invocation)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn understand_native(&self, invocation: &ModelInvocation) -> context_graph::Result<ModelReply> {
        self.prove(invocation).map_err(|err| {
            CgError::Engine(format!(
                "{}:{}:{}",
                err.condition(),
                err.bucket(),
                err.message()
            ))
        })
    }
}

fn run_provider_completion(
    provider: &str,
    binary: &Path,
    spec: &ProviderSpec,
    prompt: &str,
    output_file: Option<&Path>,
    progress_to_stdout: bool,
) -> ProviderRun {
    let cwd = neutral_cwd();
    let invocation = construct_provider_invocation(spec, prompt, output_file, &cwd);
    let mut command = Command::new(binary);
    command
        .args(&invocation.argv)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    configure_provider_process(&mut command);
    apply_provider_env_removal(&mut command, provider, spec);
    let Ok(mut child) = command.spawn() else {
        return ProviderRun {
            status: None,
            stdout: String::new(),
            stderr: "provider spawn failed".to_string(),
            timed_out: false,
        };
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_reader = stdout.map(read_stream);
    let stderr_reader = stderr.map(read_stream);
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(invocation.stdin.as_bytes());
    }
    thread::sleep(Duration::from_millis(100));
    eprintln!("waiting on provider {provider}");
    if progress_to_stdout {
        println!("waiting on provider {provider}");
        let _ = std::io::stdout().flush();
    }
    let deadline = Instant::now() + Duration::from_millis(spec.timeout_ms.max(500));
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    timed_out = true;
                    terminate_provider_process(&mut child);
                    break child.wait().ok();
                }
                thread::sleep(Duration::from_millis(20));
            }
            Err(_) => break None,
        }
    };
    let stdout = stdout_reader
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    let stderr = stderr_reader
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    ProviderRun {
        status,
        stdout,
        stderr,
        timed_out,
    }
}

fn run_provider_probe(provider: &str, binary: &Path, spec: &ProviderSpec) -> ProviderRun {
    run_provider_command(
        provider,
        binary,
        spec,
        &spec.auth_probe_args,
        None,
        spec.timeout_ms.max(500),
    )
}

fn run_provider_command(
    provider: &str,
    binary: &Path,
    spec: &ProviderSpec,
    args: &[String],
    stdin_text: Option<&str>,
    timeout_ms: u64,
) -> ProviderRun {
    let mut command = Command::new(binary);
    command
        .args(args)
        .current_dir(neutral_cwd())
        .stdin(if stdin_text.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    configure_provider_process(&mut command);
    apply_provider_env_removal(&mut command, provider, spec);
    let Ok(mut child) = command.spawn() else {
        return ProviderRun {
            status: None,
            stdout: String::new(),
            stderr: "provider command spawn failed".to_string(),
            timed_out: false,
        };
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_reader = stdout.map(read_stream);
    let stderr_reader = stderr.map(read_stream);
    if let Some(stdin_text) = stdin_text
        && let Some(mut stdin) = child.stdin.take()
    {
        let _ = stdin.write_all(stdin_text.as_bytes());
    }
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    timed_out = true;
                    terminate_provider_process(&mut child);
                    break child.wait().ok();
                }
                thread::sleep(Duration::from_millis(20));
            }
            Err(_) => break None,
        }
    };
    let stdout = stdout_reader
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    let stderr = stderr_reader
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    ProviderRun {
        status,
        stdout,
        stderr,
        timed_out,
    }
}

fn configure_provider_process(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
}

fn terminate_provider_process(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let pgid = format!("-{}", child.id());
        let _ = Command::new("kill")
            .arg("-TERM")
            .arg("--")
            .arg(&pgid)
            .status();
        thread::sleep(Duration::from_millis(50));
        let _ = Command::new("kill")
            .arg("-KILL")
            .arg("--")
            .arg(&pgid)
            .status();
    }
    let _ = child.kill();
}

fn read_stream<R: Read + Send + 'static>(mut stream: R) -> thread::JoinHandle<String> {
    thread::spawn(move || {
        let mut out = String::new();
        let _ = stream.read_to_string(&mut out);
        out
    })
}

fn classified_proof_failure(provider: &str, diagnostic: &str) -> ModelProofError {
    match classify_provider_reply(provider, diagnostic) {
        Classification::Success => ModelProofError::LiveCallFailed {
            bucket: "unknown".to_string(),
            message: "provider failed without an actionable diagnostic".to_string(),
        },
        Classification::HostLoginExpired { message } => {
            ModelProofError::HostLoginExpired { message }
        }
        Classification::BillingRequired { message } => ModelProofError::LiveCallFailed {
            bucket: "billing".to_string(),
            message: format!("failed: {message}"),
        },
        Classification::RateLimited { message } => ModelProofError::LiveCallFailed {
            bucket: "rate_limit".to_string(),
            message: format!("failed: {message}"),
        },
        Classification::LiveCallFailed { message } => ModelProofError::LiveCallFailed {
            bucket: "unknown".to_string(),
            message: format!("failed: {message}"),
        },
    }
}

fn inspect_claude_settings_for_reinjection() -> std::result::Result<(), String> {
    let mut candidates = vec![
        PathBuf::from("/etc/claude-code/settings.json"),
        PathBuf::from("/etc/claude/settings.json"),
    ];
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        candidates.push(home.join(".claude/settings.json"));
        candidates.push(home.join(".claude/settings.local.json"));
        candidates.push(home.join("agent-config/claude-code/settings.json"));
    }
    for path in candidates {
        if !path.exists() {
            continue;
        }
        let text = fs::read_to_string(&path).map_err(|err| {
            format!(
                "read {} failed while checking provider settings: {err}",
                path.display()
            )
        })?;
        let value = serde_json::from_str::<Value>(&text).map_err(|err| {
            format!(
                "parse {} failed while checking provider settings: {err}",
                path.display()
            )
        })?;
        if json_contains_key(&value, "apiKeyHelper") {
            return Err(format!(
                "{} contains apiKeyHelper; refusing host-session drive because it can re-inject metered API-key billing",
                path.display()
            ));
        }
        if let Some(name) = json_env_reinjects_protected_key(&value) {
            return Err(format!(
                "{} env block sets {name}; refusing host-session drive because it can re-inject metered API-key billing",
                path.display()
            ));
        }
    }
    Ok(())
}

fn json_contains_key(value: &Value, needle: &str) -> bool {
    match value {
        Value::Object(map) => map
            .iter()
            .any(|(key, value)| key == needle || json_contains_key(value, needle)),
        Value::Array(values) => values.iter().any(|value| json_contains_key(value, needle)),
        _ => false,
    }
}

fn json_env_reinjects_protected_key(value: &Value) -> Option<&'static str> {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                if key == "env"
                    && let Some(name) = json_contains_protected_env_key(value)
                {
                    return Some(name);
                }
                if let Some(name) = json_env_reinjects_protected_key(value) {
                    return Some(name);
                }
            }
            None
        }
        Value::Array(values) => values.iter().find_map(json_env_reinjects_protected_key),
        _ => None,
    }
}

fn json_contains_protected_env_key(value: &Value) -> Option<&'static str> {
    match value {
        Value::Object(map) => {
            for name in CLAUDE_PROTECTED_ENV
                .iter()
                .chain(CODEX_PROTECTED_ENV.iter())
            {
                if map.contains_key(*name) {
                    return Some(*name);
                }
            }
            map.values().find_map(json_contains_protected_env_key)
        }
        Value::Array(values) => values.iter().find_map(json_contains_protected_env_key),
        _ => None,
    }
}

fn raw_connection_name(value: &toml::Value) -> Option<String> {
    value
        .get("name")
        .and_then(toml::Value::as_str)
        .map(str::to_string)
}

#[derive(Debug, Default)]
pub struct NoopLlmClient;

impl LlmClient for NoopLlmClient {
    async fn understand(
        &self,
        _invocation: &ModelInvocation,
    ) -> context_graph::Result<context_graph::ModelReply> {
        Err(CgError::Engine("no llm client registered".to_string()))
    }
}

#[derive(Debug, Default, Clone)]
struct NativeLlmClient;

impl LlmClient for NativeLlmClient {
    async fn understand(
        &self,
        invocation: &ModelInvocation,
    ) -> context_graph::Result<context_graph::ModelReply> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.understand_native(invocation)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = invocation;
            Err(CgError::Engine(
                "native llm client is unavailable on wasm".to_string(),
            ))
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn understand_native(
        &self,
        invocation: &ModelInvocation,
    ) -> context_graph::Result<context_graph::ModelReply> {
        if invocation.cache_path.is_some() {
            return understand_local_single_file(invocation);
        }
        Err(CgError::Engine(
            "native llm client cannot drive hosted HTTP transport".to_string(),
        ))
    }
}

#[derive(Debug, Default, Clone)]
struct HttpChatCompletionsLlmClient;

impl LlmClient for HttpChatCompletionsLlmClient {
    async fn understand(
        &self,
        invocation: &ModelInvocation,
    ) -> context_graph::Result<context_graph::ModelReply> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.understand_native(invocation)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = invocation;
            Err(CgError::Engine(
                "HTTP chat completions client is unavailable on wasm".to_string(),
            ))
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn understand_native(
        &self,
        invocation: &ModelInvocation,
    ) -> context_graph::Result<context_graph::ModelReply> {
        understand_hosted(invocation)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CredentialSource {
    NoAuth,
    EnvVar {
        env_var: String,
    },
    StoredSecret {
        secret_ref: String,
    },
    HostSession {
        provider: String,
        account: String,
        #[serde(default = "default_true")]
        account_exposed: bool,
    },
}

fn default_true() -> bool {
    true
}

impl Default for CredentialSource {
    fn default() -> Self {
        Self::EnvVar {
            env_var: String::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialReference {
    #[serde(default)]
    pub source: CredentialSource,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub env_var: String,
}

impl CredentialReference {
    fn no_auth() -> Self {
        Self {
            source: CredentialSource::NoAuth,
            env_var: String::new(),
        }
    }

    fn env_var(env_var: String) -> Self {
        Self {
            source: CredentialSource::EnvVar {
                env_var: env_var.clone(),
            },
            env_var,
        }
    }

    fn host_session(provider: String, account: String, account_exposed: bool) -> Self {
        Self {
            source: CredentialSource::HostSession {
                provider,
                account,
                account_exposed,
            },
            env_var: String::new(),
        }
    }

    fn sealed_stored_secret(secret_ref: String, cipher_hex: String) -> Self {
        Self {
            source: CredentialSource::StoredSecret { secret_ref },
            env_var: format!("{SEALED_STORED_SECRET_PREFIX}{cipher_hex}"),
        }
    }

    fn sealed_secret_cipher_hex(&self) -> Option<&str> {
        self.env_var.strip_prefix(SEALED_STORED_SECRET_PREFIX)
    }

    fn normalized(mut self) -> Self {
        if matches!(
            &self.source,
            CredentialSource::EnvVar { env_var } if env_var.is_empty()
        ) && !self.env_var.is_empty()
        {
            self.source = CredentialSource::EnvVar {
                env_var: self.env_var.clone(),
            };
        } else if matches!(
            &self.source,
            CredentialSource::EnvVar { env_var } if env_var.is_empty()
        ) {
            self.source = CredentialSource::NoAuth;
        }
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionRead {
    pub name: String,
    pub credential_reference: CredentialReference,
    pub exposed_secret: Option<String>,
    pub consent_recorded: bool,
    pub spend_owner: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRead {
    pub id: String,
    pub kind: ModelKind,
    pub connection_name: Option<String>,
    pub welded_auth_env_var: Option<String>,
    pub exposed_secret: Option<String>,
    pub cache_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hosted_endpoint: Option<HostedEndpoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifact_filenames: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_file_sha256: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedEndpoint {
    pub base_url: String,
    pub provider: String,
    pub provider_model: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostedModelConnectionRequest {
    pub connection_name: String,
    pub model_id: String,
    pub base_url: String,
    pub provider: String,
    pub provider_model: String,
    pub consent_owner: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionStatus {
    pub context_id: ContextId,
    pub connection_name: Option<String>,
    pub model_id: Option<String>,
    pub consent_recorded: bool,
    pub spend_owner: Option<String>,
    pub degraded_marker: Option<String>,
    #[serde(default)]
    pub state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bucket: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_model: Option<String>,
    #[serde(default)]
    pub degraded: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<ConnectionLastError>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionLastError {
    pub condition: String,
    pub bucket: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationRecord {
    pub model_id: String,
    pub from_env_var: String,
    pub to_connection_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CredentialResolution {
    pub connection_name: String,
    pub env_var: String,
    pub secret: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MissingConnectionOutcome {
    HardError {
        model_id: String,
        connection_name: String,
    },
    Degraded {
        marker: String,
    },
    Resolved(CredentialResolution),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnderstandingRuntimeError {
    DanglingConnection {
        model_id: String,
        connection_name: String,
    },
    Backend(CgError),
}

#[derive(Clone, Debug, Default)]
pub struct PrincipalAwareConfig {
    secrets: BTreeMap<(String, String), String>,
    stored_secrets: BTreeMap<(String, String), String>,
    read_log: BTreeMap<String, Vec<String>>,
}

impl PrincipalAwareConfig {
    pub fn insert_secret(
        &mut self,
        principal: Principal,
        env_var: impl Into<String>,
        secret: impl Into<String>,
    ) {
        self.secrets
            .insert((principal.encode(), env_var.into()), secret.into());
    }

    pub fn resolve(&mut self, principal: &Principal, env_var: &str) -> Option<String> {
        self.read_log
            .entry(principal.encode())
            .or_default()
            .push(env_var.to_string());
        self.secrets
            .get(&(principal.encode(), env_var.to_string()))
            .cloned()
            .or_else(|| {
                if env_var.is_empty() {
                    None
                } else {
                    std::env::var(env_var).ok()
                }
            })
    }

    pub fn insert_stored_secret(
        &mut self,
        principal: Principal,
        secret_ref: impl Into<String>,
        secret: impl Into<String>,
    ) {
        self.stored_secrets
            .insert((principal.encode(), secret_ref.into()), secret.into());
    }

    pub fn resolve_stored_secret(
        &mut self,
        principal: &Principal,
        secret_ref: &str,
    ) -> Option<String> {
        self.read_log
            .entry(principal.encode())
            .or_default()
            .push(format!("stored-secret:{secret_ref}"));
        self.stored_secrets
            .get(&(principal.encode(), secret_ref.to_string()))
            .cloned()
    }

    pub fn read_log(&self, principal: &Principal) -> Vec<String> {
        self.read_log
            .get(&principal.encode())
            .cloned()
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ContextModelBinding {
    context_id: ContextId,
    model_id: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct RawModelConnectionCatalogFile {
    #[serde(default)]
    write_sequence: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    active_connection_name: Option<String>,
    #[serde(default)]
    connections: Vec<toml::Value>,
    #[serde(default)]
    models: Vec<ModelRead>,
    #[serde(default)]
    context_models: Vec<ContextModelBinding>,
    #[serde(default)]
    migration_records: Vec<MigrationRecord>,
}

#[derive(Clone, Debug, Default, Serialize)]
struct PersistModelConnectionCatalogFile {
    #[serde(default)]
    write_sequence: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    active_connection_name: Option<String>,
    connections: Vec<toml::Value>,
    models: Vec<ModelRead>,
    context_models: Vec<ContextModelBinding>,
    migration_records: Vec<MigrationRecord>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct PersistedSecretStore {
    #[serde(default)]
    secrets: Vec<PersistedSecret>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedSecret {
    principal: String,
    secret_ref: String,
    cipher_hex: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct PersistedConnectionStatusFile {
    #[serde(default)]
    errors: Vec<PersistedConnectionError>,
    #[serde(default)]
    successes: Vec<PersistedConnectionSuccess>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedConnectionError {
    connection_name: String,
    condition: String,
    bucket: String,
    message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedConnectionSuccess {
    connection_name: String,
    provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provider_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    credential_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provider_config_fingerprint: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct ConnectionSelftestStatusSnapshot {
    errors: Vec<PersistedConnectionError>,
    successes: Vec<PersistedConnectionSuccess>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ConnectionSelftestSuccessTarget {
    provider: String,
    endpoint: Option<String>,
    provider_model: Option<String>,
    credential_fingerprint: Option<String>,
    provider_config_fingerprint: Option<String>,
}

fn success_matches_connection(
    success: &PersistedConnectionSuccess,
    connection_name: &str,
    target: &Option<ConnectionSelftestSuccessTarget>,
) -> bool {
    let Some(target) = target else {
        return false;
    };
    success.connection_name == connection_name
        && success.provider == target.provider
        && success.endpoint == target.endpoint
        && success.provider_model == target.provider_model
        && success.credential_fingerprint == target.credential_fingerprint
        && success.provider_config_fingerprint == target.provider_config_fingerprint
}

fn credential_reference_fingerprint(connection: &ConnectionRead) -> Option<String> {
    credential_reference_fingerprint_with_parts(connection, &[])
}

fn credential_reference_fingerprint_with_parts(
    connection: &ConnectionRead,
    extra_parts: &[(&str, String)],
) -> Option<String> {
    let mut material = toml::to_string(&connection.credential_reference).ok()?;
    material.push_str("\nexposed_secret=");
    material.push_str(if connection.exposed_secret.is_some() {
        "present"
    } else {
        "absent"
    });
    material.push('\n');
    for (key, value) in extra_parts {
        material.push_str(key);
        material.push('=');
        material.push_str(value);
        material.push('\n');
    }
    let digest = Sha256::digest(material.as_bytes());
    Some(hex_encode(&digest))
}

fn live_secret_value_fingerprint(kind: &str, name: &str, secret: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(b"cg-live-credential-fingerprint-v1");
    digest.update([0]);
    digest.update(kind.as_bytes());
    digest.update([0]);
    digest.update(name.as_bytes());
    digest.update([0]);
    digest.update(secret.as_bytes());
    hex_encode(&digest.finalize())
}

fn local_secret_key_fingerprint(key: &[u8; 32]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"cg-local-secret-key-fingerprint-v1");
    digest.update([0]);
    digest.update(key);
    hex_encode(&digest.finalize())
}

fn provider_config_fingerprint(config: &ProviderInvocationConfig) -> String {
    let material = [
        ("transport", config.transport.as_str()),
        ("provider", config.provider.as_str()),
        ("provider_model", config.provider_model.as_str()),
        ("base_url", config.base_url.as_str()),
        ("request_path", config.request_path.as_str()),
        ("auth_scheme", config.auth_scheme.as_str()),
        ("dialect", config.dialect.as_str()),
        ("reply_path", config.reply_path.as_str()),
    ]
    .into_iter()
    .map(|(key, value)| format!("{key}={value}\n"))
    .chain(std::iter::once(format!(
        "timeout_ms={}\n",
        config.timeout_ms
    )))
    .collect::<String>();
    let digest = Sha256::digest(material.as_bytes());
    hex_encode(&digest)
}

#[derive(Clone, Debug)]
struct ConnectionRegistrationRollback {
    previous_connection: Option<ConnectionRead>,
    model_id: String,
    previous_model: Option<ModelRead>,
    stale_models: Vec<ModelRead>,
    previous_active: Option<String>,
    previous_status: ConnectionSelftestStatusSnapshot,
}

#[derive(Clone, Debug, Default)]
pub struct ModelConnectionCatalog {
    root: PathBuf,
    connections: BTreeMap<String, ConnectionRead>,
    unreadable_connections: BTreeMap<String, String>,
    unknown_connection_records: Vec<toml::Value>,
    models: BTreeMap<String, ModelRead>,
    context_models: BTreeMap<ContextId, String>,
    migration_records: Vec<MigrationRecord>,
    active_connection_name: Option<String>,
    resolved_connection_log: Vec<String>,
    write_sequence: u64,
    loaded_connection_names: BTreeSet<String>,
    loaded_model_ids: BTreeSet<String>,
    loaded_context_ids: BTreeSet<ContextId>,
    loaded_active_connection_name: Option<String>,
    persistence_errors: Arc<Mutex<Vec<String>>>,
    load_error: Option<String>,
}

impl ModelConnectionCatalog {
    pub fn open(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref().to_path_buf();
        let mut catalog = Self {
            root,
            ..Self::default()
        };
        catalog.load();
        catalog
    }

    fn storage_path(&self) -> Option<PathBuf> {
        if self.root.as_os_str().is_empty() {
            None
        } else {
            Some(self.root.join("cg/model-connections.toml"))
        }
    }

    fn secret_store_path(&self) -> Option<PathBuf> {
        if self.root.as_os_str().is_empty() {
            None
        } else {
            Some(self.root.join("cg/model-secrets.toml"))
        }
    }

    fn secret_key_path(&self) -> Option<PathBuf> {
        if self.root.as_os_str().is_empty() {
            None
        } else {
            Some(self.root.join("cg/model-secrets.key"))
        }
    }

    fn status_store_path(&self) -> Option<PathBuf> {
        if self.root.as_os_str().is_empty() {
            None
        } else {
            Some(self.root.join("cg/model-connection-status.toml"))
        }
    }

    fn load(&mut self) {
        if let Err(err) = self.load_result() {
            self.load_error = Some(err);
        }
    }

    fn load_result(&mut self) -> std::result::Result<(), String> {
        let Some(path) = self.storage_path() else {
            return Ok(());
        };
        if !path.exists() {
            return Ok(());
        }
        let text = fs::read_to_string(&path)
            .map_err(|err| format!("read {} failed: {err}", path.display()))?;
        let file = toml::from_str::<RawModelConnectionCatalogFile>(&text)
            .map_err(|err| format!("parse {} failed: {err}", path.display()))?;
        self.write_sequence = file.write_sequence;
        self.active_connection_name = file.active_connection_name;
        self.connections.clear();
        self.unreadable_connections.clear();
        self.unknown_connection_records.clear();
        for raw_connection in file.connections {
            match raw_connection.clone().try_into::<ConnectionRead>() {
                Ok(mut connection) => {
                    connection.credential_reference = connection.credential_reference.normalized();
                    self.connections.insert(connection.name.clone(), connection);
                }
                Err(err) => {
                    let name = raw_connection_name(&raw_connection).unwrap_or_else(|| {
                        format!("unreadable-{}", self.unreadable_connections.len())
                    });
                    self.unreadable_connections.insert(name, err.to_string());
                    self.unknown_connection_records.push(raw_connection);
                }
            }
        }
        self.models = file
            .models
            .into_iter()
            .map(|model| (model.id.clone(), model))
            .collect();
        self.context_models = file
            .context_models
            .into_iter()
            .map(|binding| (binding.context_id, binding.model_id))
            .collect();
        self.migration_records = file.migration_records;
        self.refresh_loaded_snapshot();
        Ok(())
    }

    fn refresh_loaded_snapshot(&mut self) {
        self.loaded_connection_names = self.connections.keys().cloned().collect();
        self.loaded_model_ids = self.models.keys().cloned().collect();
        self.loaded_context_ids = self.context_models.keys().copied().collect();
        self.loaded_active_connection_name = self.active_connection_name.clone();
    }

    fn persist(&mut self) {
        if let Err(err) = self.persist_result() {
            self.record_persistence_error(err);
        };
    }

    fn persist_result(&mut self) -> std::result::Result<(), String> {
        let Some(path) = self.storage_path() else {
            return Ok(());
        };
        let Some(parent) = path.parent() else {
            return Err(format!("storage path has no parent: {}", path.display()));
        };
        fs::create_dir_all(parent)
            .map_err(|err| format!("create {} failed: {err}", parent.display()))?;
        let lock_path = path.with_extension("toml.lock");
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|err| format!("open lock {} failed: {err}", lock_path.display()))?;
        lock.lock_exclusive()
            .map_err(|err| format!("lock {} failed: {err}", lock_path.display()))?;
        let disk_file = if path.exists() {
            let text = fs::read_to_string(&path)
                .map_err(|err| format!("read {} failed: {err}", path.display()))?;
            toml::from_str::<RawModelConnectionCatalogFile>(&text)
                .map_err(|err| format!("parse {} failed: {err}", path.display()))?
        } else {
            RawModelConnectionCatalogFile::default()
        };
        let mut merged_connections = self.connections.clone();
        let mut merged_unknown_connection_records = self.unknown_connection_records.clone();
        for raw_connection in disk_file.connections {
            match raw_connection.clone().try_into::<ConnectionRead>() {
                Ok(mut connection) => {
                    connection.credential_reference = connection.credential_reference.normalized();
                    if !merged_connections.contains_key(&connection.name)
                        && !self.loaded_connection_names.contains(&connection.name)
                    {
                        merged_connections.insert(connection.name.clone(), connection);
                    }
                }
                Err(_) => {
                    let raw_name = raw_connection_name(&raw_connection);
                    let conflicts_with_known = raw_name
                        .as_deref()
                        .is_some_and(|name| merged_connections.contains_key(name));
                    let already_preserved = merged_unknown_connection_records
                        .iter()
                        .any(|existing| raw_connection_name(existing) == raw_name);
                    if !conflicts_with_known && !already_preserved {
                        merged_unknown_connection_records.push(raw_connection);
                    }
                }
            }
        }
        let mut merged_models = self.models.clone();
        for model in disk_file.models {
            if !merged_models.contains_key(&model.id) && !self.loaded_model_ids.contains(&model.id)
            {
                merged_models.insert(model.id.clone(), model);
            }
        }
        let mut merged_context_models = self.context_models.clone();
        for binding in disk_file.context_models {
            if !merged_context_models.contains_key(&binding.context_id)
                && !self.loaded_context_ids.contains(&binding.context_id)
            {
                merged_context_models.insert(binding.context_id, binding.model_id);
            }
        }
        let mut merged_migration_records = self.migration_records.clone();
        for record in disk_file.migration_records {
            if !merged_migration_records.contains(&record) {
                merged_migration_records.push(record);
            }
        }
        let active_connection_name =
            if self.active_connection_name == self.loaded_active_connection_name {
                disk_file
                    .active_connection_name
                    .or_else(|| self.active_connection_name.clone())
            } else {
                self.active_connection_name.clone()
            };
        let write_sequence = disk_file
            .write_sequence
            .max(self.write_sequence)
            .saturating_add(1);
        let mut connections = Vec::new();
        for connection in merged_connections.values() {
            let value = toml::Value::try_from(connection.clone())
                .map_err(|err| format!("serialize connection {} failed: {err}", connection.name))?;
            connections.push(value);
        }
        for raw_connection in &merged_unknown_connection_records {
            let conflicts_with_known = raw_connection_name(raw_connection)
                .is_some_and(|name| merged_connections.contains_key(&name));
            if !conflicts_with_known {
                connections.push(raw_connection.clone());
            }
        }
        let file = PersistModelConnectionCatalogFile {
            write_sequence,
            active_connection_name: active_connection_name.clone(),
            connections,
            models: merged_models.values().cloned().collect(),
            context_models: merged_context_models
                .iter()
                .map(|(context_id, model_id)| ContextModelBinding {
                    context_id: *context_id,
                    model_id: model_id.clone(),
                })
                .collect(),
            migration_records: merged_migration_records.clone(),
        };
        let text =
            toml::to_string_pretty(&file).map_err(|err| format!("serialize failed: {err}"))?;
        let tmp = path.with_extension(format!("toml.tmp.{}", std::process::id()));
        {
            let mut out = File::create(&tmp)
                .map_err(|err| format!("create temp {} failed: {err}", tmp.display()))?;
            out.write_all(text.as_bytes())
                .map_err(|err| format!("write temp {} failed: {err}", tmp.display()))?;
            out.sync_all()
                .map_err(|err| format!("sync temp {} failed: {err}", tmp.display()))?;
        }
        fs::rename(&tmp, &path).map_err(|err| {
            let _ = fs::remove_file(&tmp);
            format!(
                "atomic rename {} -> {} failed: {err}",
                tmp.display(),
                path.display()
            )
        })?;
        self.connections = merged_connections;
        self.unknown_connection_records = merged_unknown_connection_records;
        self.models = merged_models;
        self.context_models = merged_context_models;
        self.migration_records = merged_migration_records;
        self.active_connection_name = active_connection_name;
        self.write_sequence = write_sequence;
        self.refresh_loaded_snapshot();
        Ok(())
    }

    fn record_persistence_error(&self, error: impl Into<String>) {
        if let Ok(mut errors) = self.persistence_errors.lock() {
            errors.push(error.into());
        }
    }

    fn registered_entries_result(&self) -> std::result::Result<Vec<CatalogEntry>, String> {
        if self.root.as_os_str().is_empty() {
            return Ok(Vec::new());
        }
        let catalog = Catalog::open_with_scope_against_root(&self.root, CONNECTION_CATALOG_SCOPE)
            .map_err(|err| format!("registered entries catalog open failed: {err}"))?;
        catalog
            .registered_entries()
            .map(|entries| entries.unwrap_or_default())
            .map_err(|err| format!("registered entries read failed: {err}"))
    }

    fn registered_entries(&self) -> Option<Vec<CatalogEntry>> {
        match self.registered_entries_result() {
            Ok(entries) => Some(entries),
            Err(err) => {
                self.record_persistence_error(err);
                None
            }
        }
    }

    fn mutate_registered_entries_result(
        &mut self,
        mutate: impl FnOnce(&mut Vec<CatalogEntry>),
    ) -> std::result::Result<(), String> {
        if self.root.as_os_str().is_empty() {
            return Ok(());
        }
        let path = self.root.join("cg/catalog-registrations.toml");
        let Some(parent) = path.parent() else {
            return Err(format!(
                "registered entries path has no parent: {}",
                path.display()
            ));
        };
        fs::create_dir_all(parent)
            .map_err(|err| format!("create {} failed: {err}", parent.display()))?;
        let lock_path = path.with_extension("toml.lock");
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|err| format!("open lock {} failed: {err}", lock_path.display()))?;
        lock.lock_exclusive()
            .map_err(|err| format!("lock {} failed: {err}", lock_path.display()))?;

        let mut entries = if path.exists() {
            let text = fs::read_to_string(&path)
                .map_err(|err| format!("read {} failed: {err}", path.display()))?;
            toml::from_str::<CatalogRegistrations>(&text)
                .map_err(|err| format!("parse {} failed: {err}", path.display()))?
                .entries
        } else {
            Vec::new()
        };
        mutate(&mut entries);

        let registrations = CatalogRegistrations { entries };
        let text = toml::to_string(&registrations)
            .map_err(|err| format!("serialize registered entries failed: {err}"))?;
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("catalog-registrations.toml");
        let tmp = path.with_file_name(format!(
            ".{file_name}.tmp-{}-{}",
            std::process::id(),
            Uuid::new_v4()
        ));
        {
            let mut out = File::create(&tmp)
                .map_err(|err| format!("create temp {} failed: {err}", tmp.display()))?;
            out.write_all(text.as_bytes())
                .map_err(|err| format!("write temp {} failed: {err}", tmp.display()))?;
            out.sync_all()
                .map_err(|err| format!("sync temp {} failed: {err}", tmp.display()))?;
        }
        fs::rename(&tmp, &path).map_err(|err| {
            let _ = fs::remove_file(&tmp);
            format!(
                "atomic rename {} -> {} failed: {err}",
                tmp.display(),
                path.display()
            )
        })
    }

    fn upsert_shipped_catalog_entry_result(
        &mut self,
        model: &ModelRead,
    ) -> std::result::Result<(), String> {
        self.mutate_registered_entries_result(|entries| {
            let mut entry = entries
                .iter()
                .find(|entry| entry.id == model.id)
                .cloned()
                .unwrap_or_else(|| catalog_entry_for_model(model));
            entry.kind = model.kind.clone();
            entry.capability = Capability {
                dimension: None,
                output_shape: Some("understanding".to_string()),
                input_modality: Some("text".to_string()),
            };
            entry.source = if let Some(connection_name) = &model.connection_name {
                Source::NamedConnection {
                    connection_name: connection_name.clone(),
                }
            } else if let Some(cache_path) = &model.cache_path {
                Source::LocalPath {
                    dir: cache_path.clone(),
                }
            } else if let Some(env_var) = &model.welded_auth_env_var {
                Source::ApiEndpoint {
                    base_url: "https://provider.example.invalid/v1".to_string(),
                    provider: "legacy-provider".to_string(),
                    model: model.id.clone(),
                    auth_env_var: env_var.clone(),
                }
            } else {
                Source::BundledBinary
            };
            entry.cache_status = if matches!(entry.source, Source::NamedConnection { .. }) {
                CacheStatus::Cached
            } else {
                entry.cache_status
            };
            entry.byte_size = if matches!(entry.source, Source::NamedConnection { .. }) {
                0
            } else {
                entry.byte_size
            };
            entry.local_path = model.cache_path.clone();
            entry.artifact_filenames = model.artifact_filenames.clone();
            entry.per_file_sha256 = model.per_file_sha256.clone();
            entries.retain(|existing| existing.id != model.id);
            entries.push(entry);
        })
    }

    fn upsert_shipped_catalog_entry(&mut self, model: &ModelRead) -> bool {
        match self.upsert_shipped_catalog_entry_result(model) {
            Ok(()) => true,
            Err(err) => {
                self.record_persistence_error(err);
                false
            }
        }
    }

    fn shipped_model(&self, id: &str) -> Option<ModelRead> {
        let entry = self
            .registered_entries()?
            .into_iter()
            .find(|entry| entry.id == id)?;
        Some(model_read_from_entry(&entry))
    }

    pub fn connect_env_var(
        &mut self,
        name: impl Into<String>,
        env_var: impl Into<String>,
        principal: Principal,
        config: &mut PrincipalAwareConfig,
    ) -> ConnectionRead {
        let _ = (principal, config);
        let name = name.into();
        let env_var = env_var.into();
        match self.connect_env_var_result(name.clone(), env_var.clone()) {
            Ok(read) => read,
            Err(err) => {
                self.record_persistence_error(err);
                ConnectionRead {
                    name,
                    credential_reference: CredentialReference::env_var(env_var),
                    exposed_secret: None,
                    consent_recorded: false,
                    spend_owner: "unknown".to_string(),
                }
            }
        }
    }

    pub fn connect_env_var_result(
        &mut self,
        name: impl Into<String>,
        env_var: impl Into<String>,
    ) -> std::result::Result<ConnectionRead, String> {
        let name = name.into();
        let env_var = env_var.into();
        let read = ConnectionRead {
            name: name.clone(),
            credential_reference: CredentialReference::env_var(env_var),
            exposed_secret: None,
            consent_recorded: false,
            spend_owner: "unknown".to_string(),
        };
        let previous_connection = self.connections.insert(name, read.clone());
        let previous_active = self.active_connection_name.clone();
        self.active_connection_name = Some(read.name.clone());
        if let Err(err) = self.persist_result() {
            if let Some(previous_connection) = previous_connection.clone() {
                self.connections
                    .insert(previous_connection.name.clone(), previous_connection);
            } else {
                self.connections.remove(&read.name);
            }
            self.active_connection_name = previous_active;
            self.record_persistence_error(err.clone());
            return Err(err);
        }
        Ok(read)
    }

    fn remove_stored_secret_ref(&self, secret_ref: &str) -> std::result::Result<(), String> {
        let Some(path) = self.secret_store_path() else {
            return Ok(());
        };
        if !path.exists() {
            return Ok(());
        }
        let Some(parent) = path.parent() else {
            return Err(format!(
                "secret store path has no parent: {}",
                path.display()
            ));
        };
        fs::create_dir_all(parent)
            .map_err(|err| format!("create {} failed: {err}", parent.display()))?;
        let lock_path = path.with_extension("toml.lock");
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|err| format!("open lock {} failed: {err}", lock_path.display()))?;
        lock.lock_exclusive()
            .map_err(|err| format!("lock {} failed: {err}", lock_path.display()))?;
        let text = fs::read_to_string(&path)
            .map_err(|err| format!("read secret store {} failed: {err}", path.display()))?;
        let mut store = toml::from_str::<PersistedSecretStore>(&text)
            .map_err(|err| format!("parse secret store {} failed: {err}", path.display()))?;
        let original_len = store.secrets.len();
        store.secrets.retain(|entry| entry.secret_ref != secret_ref);
        if store.secrets.len() == original_len {
            return Ok(());
        }
        write_secret_store_unlocked(&path, &store)
    }

    fn local_secret_key(&self) -> std::result::Result<[u8; 32], String> {
        let Some(path) = self.secret_key_path() else {
            return Err("stored-secret encryption requires a catalog root".to_string());
        };
        let Some(parent) = path.parent() else {
            return Err(format!("secret key path has no parent: {}", path.display()));
        };
        fs::create_dir_all(parent)
            .map_err(|err| format!("create {} failed: {err}", parent.display()))?;
        let lock_path = path.with_extension("key.lock");
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|err| format!("open lock {} failed: {err}", lock_path.display()))?;
        lock.lock_exclusive()
            .map_err(|err| format!("lock {} failed: {err}", lock_path.display()))?;
        if path.exists() {
            ensure_private_file_permissions(&path)?;
            let text = fs::read_to_string(&path)
                .map_err(|err| format!("read secret key {} failed: {err}", path.display()))?;
            return parse_local_secret_key(&text)
                .ok_or_else(|| format!("secret key {} is invalid", path.display()));
        }
        let key = random_secret_bytes();
        let text = format!("{}\n", hex_encode(&key));
        let tmp = path.with_extension(format!("key.tmp.{}", std::process::id()));
        {
            let mut file = create_private_file(&tmp)?;
            file.write_all(text.as_bytes())
                .map_err(|err| format!("write temp {} failed: {err}", tmp.display()))?;
            file.sync_all()
                .map_err(|err| format!("sync temp {} failed: {err}", tmp.display()))?;
        }
        fs::rename(&tmp, &path).map_err(|err| {
            let _ = fs::remove_file(&tmp);
            format!(
                "atomic rename {} -> {} failed: {err}",
                tmp.display(),
                path.display()
            )
        })?;
        ensure_private_file_permissions(&path)?;
        Ok(key)
    }

    fn existing_local_secret_key_fingerprint(&self) -> Option<String> {
        let path = self.secret_key_path()?;
        let text = fs::read_to_string(path).ok()?;
        let key = parse_local_secret_key(&text)?;
        Some(local_secret_key_fingerprint(&key))
    }

    pub fn connect_host_session(
        &mut self,
        name: impl Into<String>,
        provider: impl Into<String>,
        account: impl Into<String>,
    ) -> ConnectionRead {
        self.connect_host_session_with_account_exposure(name, provider, account, true)
    }

    pub fn connect_host_session_with_account_exposure(
        &mut self,
        name: impl Into<String>,
        provider: impl Into<String>,
        account: impl Into<String>,
        account_exposed: bool,
    ) -> ConnectionRead {
        let name = name.into();
        let provider = provider.into();
        let account = account.into();
        let existing = self.connections.get(&name).cloned();
        let same_host_session = existing.as_ref().is_some_and(|connection| {
            matches!(
                &connection.credential_reference.source,
                CredentialSource::HostSession {
                    provider: existing_provider,
                    account: existing_account,
                    account_exposed: existing_account_exposed,
                } if account_exposed
                    && *existing_account_exposed
                    && existing_provider == &provider
                    && existing_account == &account
            )
        });
        let read = ConnectionRead {
            name: name.clone(),
            credential_reference: CredentialReference::host_session(
                provider,
                account.clone(),
                account_exposed,
            ),
            exposed_secret: None,
            consent_recorded: existing
                .as_ref()
                .is_some_and(|connection| connection.consent_recorded && same_host_session),
            spend_owner: existing
                .and_then(|connection| {
                    if connection.consent_recorded && same_host_session {
                        Some(connection.spend_owner)
                    } else {
                        None
                    }
                })
                .unwrap_or(account),
        };
        self.connections.insert(name, read.clone());
        self.active_connection_name = Some(read.name.clone());
        self.persist();
        read
    }

    pub fn register_model(
        &mut self,
        id: impl Into<String>,
        kind: ModelKind,
        connection_name: impl Into<String>,
    ) -> ModelRead {
        let model = ModelRead {
            id: id.into(),
            kind,
            connection_name: Some(connection_name.into()),
            welded_auth_env_var: None,
            exposed_secret: None,
            cache_path: None,
            hosted_endpoint: None,
            artifact_filenames: Vec::new(),
            per_file_sha256: Vec::new(),
        };
        if !self.upsert_shipped_catalog_entry(&model) {
            return model;
        }
        self.models.insert(model.id.clone(), model.clone());
        self.persist();
        model
    }

    pub fn register_hosted_model(
        &mut self,
        id: impl Into<String>,
        connection_name: impl Into<String>,
        base_url: impl Into<String>,
        provider: impl Into<String>,
        provider_model: impl Into<String>,
    ) -> ModelRead {
        let id = id.into();
        let connection_name = connection_name.into();
        let base_url = base_url.into();
        let provider = provider.into();
        let provider_model = provider_model.into();
        match self.register_hosted_model_result(
            id.clone(),
            connection_name.clone(),
            base_url.clone(),
            provider.clone(),
            provider_model.clone(),
        ) {
            Ok(model) => model,
            Err(err) => {
                self.record_persistence_error(err);
                hosted_model_read(id, connection_name, base_url, provider, provider_model)
            }
        }
    }

    pub fn register_hosted_model_result(
        &mut self,
        id: impl Into<String>,
        connection_name: impl Into<String>,
        base_url: impl Into<String>,
        provider: impl Into<String>,
        provider_model: impl Into<String>,
    ) -> std::result::Result<ModelRead, String> {
        let id = id.into();
        let connection_name = connection_name.into();
        let base_url = base_url.into();
        let provider = provider.into();
        let provider_model = provider_model.into();
        self.validate_hosted_endpoint_for_provider(&provider, &base_url)?;
        let model = hosted_model_read(id, connection_name, base_url, provider, provider_model);
        let previous_model = self.models.insert(model.id.clone(), model.clone());
        if let Err(err) = self.persist_result() {
            if let Some(previous_model) = previous_model.clone() {
                self.models
                    .insert(previous_model.id.clone(), previous_model);
            } else {
                self.models.remove(&model.id);
            }
            self.record_persistence_error(err.clone());
            return Err(err);
        }
        if let Err(err) = self.upsert_shipped_catalog_entry_result(&model) {
            if let Some(previous_model) = previous_model {
                self.models
                    .insert(previous_model.id.clone(), previous_model);
            } else {
                self.models.remove(&model.id);
            }
            if let Err(rollback_err) = self.persist_result() {
                let message =
                    format!("{err}; rollback of hosted model registration failed: {rollback_err}");
                self.record_persistence_error(message.clone());
                return Err(message);
            }
            self.record_persistence_error(err.clone());
            return Err(err);
        }
        Ok(model)
    }

    fn upsert_connection_and_hosted_model_result(
        &mut self,
        connection: ConnectionRead,
        model: ModelRead,
    ) -> std::result::Result<ConnectionRead, String> {
        let connection_name = connection.name.clone();
        let model_id = model.id.clone();
        let stale_models = self.hosted_models_for_connection_excluding(&connection_name, &model_id);
        for stale_model in &stale_models {
            self.models.remove(&stale_model.id);
        }
        let previous_connection = self
            .connections
            .insert(connection_name.clone(), connection.clone());
        let previous_model = self.models.insert(model_id.clone(), model.clone());
        let previous_active = self.active_connection_name.clone();
        self.active_connection_name = Some(connection_name.clone());
        if let Err(err) = self.persist_result() {
            self.restore_connection_model_state(
                &connection_name,
                previous_connection.clone(),
                &model_id,
                previous_model.clone(),
                stale_models.clone(),
                previous_active.clone(),
            );
            self.record_persistence_error(err.clone());
            return Err(err);
        }
        if let Err(err) = self.upsert_shipped_catalog_entry_for_connection_result(
            &model,
            &stale_models
                .iter()
                .map(|model| model.id.clone())
                .collect::<Vec<_>>(),
        ) {
            self.restore_connection_model_state(
                &connection_name,
                previous_connection,
                &model_id,
                previous_model,
                stale_models,
                previous_active,
            );
            if let Err(rollback_err) = self.persist_result() {
                let message = format!(
                    "{err}; rollback of hosted connection registration failed: {rollback_err}"
                );
                self.record_persistence_error(message.clone());
                return Err(message);
            }
            self.record_persistence_error(err.clone());
            return Err(err);
        }
        Ok(connection)
    }

    fn restore_connection_model_state(
        &mut self,
        connection_name: &str,
        previous_connection: Option<ConnectionRead>,
        model_id: &str,
        previous_model: Option<ModelRead>,
        stale_models: Vec<ModelRead>,
        previous_active: Option<String>,
    ) {
        if let Some(previous_connection) = previous_connection {
            self.connections
                .insert(previous_connection.name.clone(), previous_connection);
        } else {
            self.connections.remove(connection_name);
        }
        if let Some(previous_model) = previous_model {
            self.models
                .insert(previous_model.id.clone(), previous_model);
        } else {
            self.models.remove(model_id);
        }
        for stale_model in stale_models {
            self.models.insert(stale_model.id.clone(), stale_model);
        }
        self.active_connection_name = previous_active;
    }

    fn hosted_models_for_connection_excluding(
        &self,
        connection_name: &str,
        model_id: &str,
    ) -> Vec<ModelRead> {
        self.models
            .values()
            .filter(|existing| {
                existing.connection_name.as_deref() == Some(connection_name)
                    && existing.id != model_id
                    && existing.hosted_endpoint.is_some()
            })
            .cloned()
            .collect()
    }

    fn upsert_shipped_catalog_entry_for_connection_result(
        &mut self,
        model: &ModelRead,
        stale_model_ids: &[String],
    ) -> std::result::Result<(), String> {
        self.mutate_registered_entries_result(|entries| {
            entries.retain(|entry| {
                entry.id != model.id && !stale_model_ids.iter().any(|id| id == &entry.id)
            });
            entries.push(catalog_entry_for_model(model));
        })
    }

    fn restore_shipped_catalog_entries_for_connection_result(
        &mut self,
        model_id: &str,
        previous_model: Option<&ModelRead>,
        stale_models: &[ModelRead],
    ) -> std::result::Result<(), String> {
        self.mutate_registered_entries_result(|entries| {
            entries.retain(|entry| {
                entry.id != model_id && !stale_models.iter().any(|model| model.id == entry.id)
            });
            if let Some(previous_model) = previous_model {
                entries.push(catalog_entry_for_model(previous_model));
            }
            for stale_model in stale_models {
                entries.push(catalog_entry_for_model(stale_model));
            }
        })
    }

    fn restore_connection_model_registration_result(
        &mut self,
        connection_name: &str,
        previous_connection: Option<ConnectionRead>,
        model_id: &str,
        previous_model: Option<ModelRead>,
        stale_models: Vec<ModelRead>,
        previous_active: Option<String>,
    ) -> std::result::Result<(), String> {
        let previous_model_entry = previous_model.clone();
        let stale_model_entries = stale_models.clone();
        self.restore_connection_model_state(
            connection_name,
            previous_connection,
            model_id,
            previous_model,
            stale_models,
            previous_active,
        );
        self.persist_result()?;
        self.restore_shipped_catalog_entries_for_connection_result(
            model_id,
            previous_model_entry.as_ref(),
            &stale_model_entries,
        )
    }

    fn restore_connection_model_registration_and_status_result(
        &mut self,
        connection_name: &str,
        rollback: ConnectionRegistrationRollback,
    ) -> std::result::Result<(), String> {
        let mut failures = Vec::new();
        if let Err(err) = self.restore_connection_model_registration_result(
            connection_name,
            rollback.previous_connection,
            &rollback.model_id,
            rollback.previous_model,
            rollback.stale_models,
            rollback.previous_active,
        ) {
            failures.push(err);
        }
        if let Err(err) =
            self.restore_connection_selftest_status(connection_name, &rollback.previous_status)
        {
            failures.push(err);
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("; "))
        }
    }

    pub fn connect_hosted_env_var_result(
        &mut self,
        request: HostedModelConnectionRequest,
        env_var: impl Into<String>,
    ) -> std::result::Result<ConnectionRead, String> {
        self.connect_hosted_credential_reference_result(
            request,
            CredentialReference::env_var(env_var.into()),
        )
    }

    pub fn connect_hosted_no_auth_result(
        &mut self,
        request: HostedModelConnectionRequest,
    ) -> std::result::Result<ConnectionRead, String> {
        self.connect_hosted_credential_reference_result(request, CredentialReference::no_auth())
    }

    fn connect_hosted_credential_reference_result(
        &mut self,
        request: HostedModelConnectionRequest,
        credential_reference: CredentialReference,
    ) -> std::result::Result<ConnectionRead, String> {
        self.validate_hosted_endpoint_for_provider(&request.provider, &request.base_url)?;
        let connection_name = request.connection_name;
        let model_id = request.model_id;
        let previous_connection = self.connections.get(&connection_name).cloned();
        let previous_model = self.models.get(&model_id).cloned();
        let previous_stale_models =
            self.hosted_models_for_connection_excluding(&connection_name, &model_id);
        let previous_active = self.active_connection_name.clone();
        let previous_status = self.connection_selftest_status_snapshot(&connection_name);
        let rollback = ConnectionRegistrationRollback {
            previous_connection: previous_connection.clone(),
            model_id: model_id.clone(),
            previous_model: previous_model.clone(),
            stale_models: previous_stale_models.clone(),
            previous_active: previous_active.clone(),
            previous_status,
        };
        let previous_secret_ref = previous_connection.as_ref().and_then(|connection| {
            match &connection.credential_reference.source {
                CredentialSource::StoredSecret { secret_ref } => Some(secret_ref.clone()),
                _ => None,
            }
        });
        let (consent_recorded, spend_owner) = request
            .consent_owner
            .map(|owner| (true, owner))
            .unwrap_or_else(|| (false, "unknown".to_string()));
        let connection = ConnectionRead {
            name: connection_name.clone(),
            credential_reference,
            exposed_secret: None,
            consent_recorded,
            spend_owner,
        };
        let model = hosted_model_read(
            model_id.clone(),
            connection_name.clone(),
            request.base_url,
            request.provider,
            request.provider_model,
        );
        let read = self.upsert_connection_and_hosted_model_result(connection, model)?;
        if let Err(err) = self.remove_connection_selftest_status(&connection_name) {
            let message = match self.restore_connection_model_registration_and_status_result(
                &connection_name,
                rollback.clone(),
            ) {
                Ok(()) => err.clone(),
                Err(rollback_err) => {
                    format!(
                        "{err}; rollback after selftest status cleanup failure failed: {rollback_err}"
                    )
                }
            };
            self.record_persistence_error(message.clone());
            return Err(message);
        }
        if let Some(secret_ref) = previous_secret_ref
            && let Err(err) = self.remove_stored_secret_ref(&secret_ref)
        {
            let message = match self
                .restore_connection_model_registration_and_status_result(&connection_name, rollback)
            {
                Ok(()) => err.clone(),
                Err(rollback_err) => {
                    format!(
                        "{err}; rollback after stored-secret cleanup failure failed: {rollback_err}"
                    )
                }
            };
            self.record_persistence_error(message.clone());
            return Err(message);
        }
        Ok(read)
    }

    pub fn connect_hosted_stored_secret_result(
        &mut self,
        request: HostedModelConnectionRequest,
        principal: Principal,
        config: &mut PrincipalAwareConfig,
        secret: impl Into<String>,
    ) -> std::result::Result<ConnectionRead, String> {
        self.validate_hosted_endpoint_for_provider(&request.provider, &request.base_url)?;
        let connection_name = request.connection_name;
        let model_id = request.model_id;
        let previous_connection = self.connections.get(&connection_name).cloned();
        let previous_model = self.models.get(&model_id).cloned();
        let previous_stale_models =
            self.hosted_models_for_connection_excluding(&connection_name, &model_id);
        let previous_active = self.active_connection_name.clone();
        let previous_status = self.connection_selftest_status_snapshot(&connection_name);
        let rollback = ConnectionRegistrationRollback {
            previous_connection: previous_connection.clone(),
            model_id: model_id.clone(),
            previous_model: previous_model.clone(),
            stale_models: previous_stale_models.clone(),
            previous_active: previous_active.clone(),
            previous_status,
        };
        let secret_ref = format!("connection:{connection_name}:api_key");
        let secret = secret.into();
        let principal_text = principal.encode();
        let secret_key = self.local_secret_key()?;
        let cipher_hex = encode_local_secret(&secret_key, &principal_text, &secret_ref, &secret)?;
        let (consent_recorded, spend_owner) = request
            .consent_owner
            .map(|owner| (true, owner))
            .unwrap_or_else(|| (false, "unknown".to_string()));
        let connection = ConnectionRead {
            name: connection_name.clone(),
            credential_reference: CredentialReference::sealed_stored_secret(
                secret_ref.clone(),
                cipher_hex,
            ),
            exposed_secret: None,
            consent_recorded,
            spend_owner,
        };
        let model = hosted_model_read(
            model_id.clone(),
            connection_name.clone(),
            request.base_url,
            request.provider,
            request.provider_model,
        );
        let read = self.upsert_connection_and_hosted_model_result(connection, model)?;
        if let Err(err) = self.remove_connection_selftest_status(&connection_name) {
            let message = match self.restore_connection_model_registration_and_status_result(
                &connection_name,
                rollback.clone(),
            ) {
                Ok(()) => err.clone(),
                Err(rollback_err) => {
                    format!(
                        "{err}; rollback after selftest status cleanup failure failed: {rollback_err}"
                    )
                }
            };
            self.record_persistence_error(message.clone());
            return Err(message);
        }
        if let Some(previous_secret_ref) = previous_connection.as_ref().and_then(|connection| {
            match &connection.credential_reference.source {
                CredentialSource::StoredSecret { secret_ref } => Some(secret_ref.clone()),
                _ => None,
            }
        }) && let Err(err) = self.remove_stored_secret_ref(&previous_secret_ref)
        {
            let message = match self
                .restore_connection_model_registration_and_status_result(&connection_name, rollback)
            {
                Ok(()) => err.clone(),
                Err(rollback_err) => {
                    format!(
                        "{err}; rollback after legacy stored-secret cleanup failure failed: {rollback_err}"
                    )
                }
            };
            self.record_persistence_error(message.clone());
            return Err(message);
        }
        config.insert_stored_secret(principal, secret_ref, secret);
        Ok(read)
    }

    pub fn register_local_model(
        &mut self,
        id: impl Into<String>,
        kind: ModelKind,
        cache_path: impl Into<PathBuf>,
    ) -> ModelRead {
        let cache_path = cache_path.into();
        let (artifact_filenames, per_file_sha256) = if self.root.as_os_str().is_empty() {
            declared_local_artifact_metadata(&cache_path)
        } else {
            (Vec::new(), Vec::new())
        };
        let model = ModelRead {
            id: id.into(),
            kind,
            connection_name: None,
            welded_auth_env_var: None,
            exposed_secret: None,
            cache_path: Some(cache_path),
            hosted_endpoint: None,
            artifact_filenames,
            per_file_sha256,
        };
        if !self.upsert_shipped_catalog_entry(&model) {
            return model;
        }
        self.models.insert(model.id.clone(), model.clone());
        self.persist();
        model
    }

    pub fn register_legacy_hosted_model_with_welded_env_var(
        &mut self,
        id: impl Into<String>,
        env_var: impl Into<String>,
    ) -> ModelRead {
        let model = ModelRead {
            id: id.into(),
            kind: ModelKind::LlmClient,
            connection_name: None,
            welded_auth_env_var: Some(env_var.into()),
            exposed_secret: None,
            cache_path: None,
            hosted_endpoint: None,
            artifact_filenames: Vec::new(),
            per_file_sha256: Vec::new(),
        };
        self.models.insert(model.id.clone(), model.clone());
        self.persist();
        model
    }

    pub fn migrate_legacy_welded_credentials(&mut self) -> Vec<MigrationRecord> {
        let mut candidates = self
            .models
            .values()
            .filter_map(|model| {
                model.welded_auth_env_var.as_ref().map(|env_var| {
                    (
                        model.id.clone(),
                        model.kind.clone(),
                        env_var.clone(),
                        model.hosted_endpoint.clone(),
                    )
                })
            })
            .collect::<Vec<_>>();
        let Some(registered_entries) = self.registered_entries() else {
            return Vec::new();
        };
        for entry in registered_entries {
            if let Source::ApiEndpoint {
                base_url,
                provider,
                model,
                auth_env_var,
            } = entry.source
                && !candidates.iter().any(|(id, _, _, _)| id == &entry.id)
            {
                candidates.push((
                    entry.id,
                    entry.kind,
                    auth_env_var,
                    Some(HostedEndpoint {
                        base_url,
                        provider,
                        provider_model: model,
                    }),
                ));
            }
        }

        let mut created = Vec::new();
        for (model_id, kind, env_var, hosted_endpoint) in candidates {
            if self
                .migration_records
                .iter()
                .any(|record| record.model_id == model_id && record.from_env_var == env_var)
            {
                continue;
            }
            let connection_name = format!("{model_id}-connection");
            self.connections
                .entry(connection_name.clone())
                .or_insert_with(|| ConnectionRead {
                    name: connection_name.clone(),
                    credential_reference: CredentialReference::env_var(env_var.clone()),
                    exposed_secret: None,
                    consent_recorded: false,
                    spend_owner: "unknown".to_string(),
                });
            let model = ModelRead {
                id: model_id.clone(),
                kind,
                connection_name: Some(connection_name.clone()),
                welded_auth_env_var: None,
                exposed_secret: None,
                cache_path: None,
                hosted_endpoint,
                artifact_filenames: Vec::new(),
                per_file_sha256: Vec::new(),
            };
            if !self.upsert_shipped_catalog_entry(&model) {
                return Vec::new();
            }
            self.models.insert(model_id.clone(), model.clone());
            let record = MigrationRecord {
                model_id,
                from_env_var: env_var,
                to_connection_name: connection_name,
            };
            self.migration_records.push(record.clone());
            created.push(record);
        }
        self.persist();
        created
    }

    pub fn migration_records(&self) -> Vec<MigrationRecord> {
        self.migration_records.clone()
    }

    pub fn read_connection(&self, name: &str) -> Option<ConnectionRead> {
        self.connections.get(name).cloned()
    }

    pub fn active_connection_name(&self) -> Option<String> {
        self.active_connection_name.clone()
    }

    pub fn list_connections(&self) -> Vec<ConnectionRead> {
        self.connections.values().cloned().collect()
    }

    pub fn read_model(&self, id: &str) -> Option<ModelRead> {
        self.models
            .get(id)
            .cloned()
            .or_else(|| self.shipped_model(id))
            .or_else(|| {
                self.models
                    .values()
                    .find(|model| {
                        model
                            .hosted_endpoint
                            .as_ref()
                            .is_some_and(|endpoint| endpoint.provider_model == id)
                    })
                    .cloned()
            })
    }

    pub fn list_models(&self) -> Vec<ModelRead> {
        let mut models = self.models.clone();
        if let Some(entries) = self.registered_entries() {
            for entry in entries {
                models
                    .entry(entry.id.clone())
                    .or_insert_with(|| model_read_from_entry(&entry));
            }
        }
        models.into_values().collect()
    }

    pub fn bind_context_model(&mut self, context_id: ContextId, model_id: impl Into<String>) {
        self.context_models.insert(context_id, model_id.into());
        self.persist();
    }

    pub fn record_consent(&mut self, connection_name: &str, spend_owner: impl Into<String>) {
        if let Err(err) = self.record_consent_result(connection_name, spend_owner) {
            self.record_persistence_error(err);
        }
    }

    pub fn record_consent_result(
        &mut self,
        connection_name: &str,
        spend_owner: impl Into<String>,
    ) -> std::result::Result<(), String> {
        let spend_owner = spend_owner.into();
        let Some(previous_connection) = self.connections.get(connection_name).cloned() else {
            return Err(format!("connection {connection_name} not found"));
        };
        if let Some(connection) = self.connections.get_mut(connection_name) {
            connection.consent_recorded = true;
            connection.spend_owner = spend_owner;
        }
        if let Err(err) = self.persist_result() {
            self.connections
                .insert(previous_connection.name.clone(), previous_connection);
            self.record_persistence_error(err.clone());
            return Err(err);
        }
        Ok(())
    }

    pub fn revoke_consent(&mut self, connection_name: &str) {
        if let Some(connection) = self.connections.get_mut(connection_name) {
            connection.consent_recorded = false;
            self.persist();
        }
    }

    pub fn status_for_context(&self, context_id: ContextId) -> ConnectionStatus {
        let model_id = self
            .context_models
            .get(&context_id)
            .cloned()
            .or_else(|| self.active_model_id());
        let model = model_id.as_ref().and_then(|id| self.read_model(id));
        let connection_name = model_id
            .as_ref()
            .and_then(|_| model.clone())
            .and_then(|model| model.connection_name.clone());
        let connection = connection_name
            .as_ref()
            .and_then(|name| self.connections.get(name));
        let hosted = model
            .as_ref()
            .and_then(|model| model.hosted_endpoint.as_ref());
        let last_error = connection_name
            .as_ref()
            .and_then(|name| self.connection_last_error(name));
        let state = if last_error.is_some() {
            "degraded"
        } else if connection.is_some() {
            "ready"
        } else {
            "not_connected"
        };
        let condition = if connection.is_none() {
            Some("not_connected".to_string())
        } else if last_error.is_some() {
            last_error.as_ref().map(|error| error.condition.clone())
        } else {
            None
        };
        let bucket = if connection.is_none() {
            Some("not_connected".to_string())
        } else {
            last_error.as_ref().map(|error| error.bucket.clone())
        };
        ConnectionStatus {
            context_id,
            connection_name,
            model_id,
            consent_recorded: connection.is_some_and(|connection| connection.consent_recorded),
            spend_owner: connection
                .filter(|connection| connection.consent_recorded)
                .map(|connection| connection.spend_owner.clone()),
            degraded_marker: None,
            state: state.to_string(),
            condition,
            bucket,
            provider: hosted.map(|endpoint| endpoint.provider.clone()),
            endpoint: hosted.map(|endpoint| endpoint.base_url.clone()),
            provider_model: hosted.map(|endpoint| endpoint.provider_model.clone()),
            degraded: last_error.is_some(),
            last_error,
        }
    }

    pub fn model_for_connection(&self, connection_name: &str) -> Option<ModelRead> {
        if let Some(model) = self.models.get(connection_name)
            && model.connection_name.as_deref() == Some(connection_name)
        {
            return Some(model.clone());
        }
        self.list_models()
            .into_iter()
            .find(|model| model.connection_name.as_deref() == Some(connection_name))
    }

    fn active_model_id(&self) -> Option<String> {
        let active = self.active_connection_name.as_deref()?;
        self.model_for_connection(active).map(|model| model.id)
    }

    pub fn record_connection_last_error_result(
        &self,
        connection_name: &str,
        condition: &str,
        bucket: &str,
        message: &str,
    ) -> std::result::Result<(), String> {
        self.write_connection_last_error(connection_name, condition, bucket, message)
    }

    pub fn clear_connection_last_error_result(
        &self,
        connection_name: &str,
    ) -> std::result::Result<(), String> {
        self.remove_connection_last_error(connection_name)
    }

    pub fn record_connection_selftest_success_result(
        &self,
        connection_name: &str,
        provider: &str,
    ) -> std::result::Result<(), String> {
        self.write_connection_selftest_success(connection_name, provider)
    }

    pub fn connection_selftest_passed(&self, connection_name: &str) -> bool {
        let proof_target = self.connection_selftest_success_target(connection_name);
        let Ok(file) = self.connection_status_file_result() else {
            return false;
        };
        file.successes
            .into_iter()
            .any(|success| success_matches_connection(&success, connection_name, &proof_target))
    }

    pub fn connection_has_selftest_success_record(&self, connection_name: &str) -> bool {
        let Ok(file) = self.connection_status_file_result() else {
            return false;
        };
        file.successes
            .into_iter()
            .any(|success| success.connection_name == connection_name)
    }

    pub fn last_error_for_connection(&self, connection_name: &str) -> Option<ConnectionLastError> {
        self.connection_last_error(connection_name)
    }

    pub fn resolve_credential(
        &mut self,
        connection_name: &str,
        principal: &Principal,
        config: &mut PrincipalAwareConfig,
    ) -> Option<CredentialResolution> {
        let connection = self.connections.get(connection_name)?.clone();
        if matches!(
            connection.credential_reference.source,
            CredentialSource::StoredSecret { .. }
        ) && !connection.consent_recorded
        {
            return None;
        }
        self.resolved_connection_log
            .push(connection_name.to_string());
        let (env_var, resolved_secret) = match &connection.credential_reference.source {
            CredentialSource::NoAuth => (String::new(), None),
            CredentialSource::EnvVar { env_var } => {
                (env_var.clone(), config.resolve(principal, env_var))
            }
            CredentialSource::StoredSecret { secret_ref } => {
                let sealed_secret = connection
                    .credential_reference
                    .sealed_secret_cipher_hex()
                    .and_then(|cipher_hex| {
                        self.decode_persisted_secret(principal, secret_ref, cipher_hex)
                    });
                (
                    String::new(),
                    config
                        .resolve_stored_secret(principal, secret_ref)
                        .or(sealed_secret)
                        .or_else(|| self.resolve_persisted_secret(principal, secret_ref)),
                )
            }
            CredentialSource::HostSession { .. } => (String::new(), None),
        };
        Some(CredentialResolution {
            connection_name: connection.name,
            env_var,
            secret: connection.exposed_secret.or(resolved_secret),
        })
    }

    fn resolve_persisted_secret(&self, principal: &Principal, secret_ref: &str) -> Option<String> {
        let path = self.secret_store_path()?;
        let text = fs::read_to_string(path).ok()?;
        let store = toml::from_str::<PersistedSecretStore>(&text).ok()?;
        let principal = principal.encode();
        store
            .secrets
            .into_iter()
            .find(|entry| entry.principal == principal && entry.secret_ref == secret_ref)
            .and_then(|entry| {
                self.decode_persisted_secret_encoded_principal(
                    &principal,
                    secret_ref,
                    &entry.cipher_hex,
                )
            })
    }

    fn decode_persisted_secret(
        &self,
        principal: &Principal,
        secret_ref: &str,
        cipher_hex: &str,
    ) -> Option<String> {
        let principal = principal.encode();
        self.decode_persisted_secret_encoded_principal(&principal, secret_ref, cipher_hex)
    }

    fn decode_persisted_secret_encoded_principal(
        &self,
        principal: &str,
        secret_ref: &str,
        cipher_hex: &str,
    ) -> Option<String> {
        let key = self.local_secret_key().ok()?;
        decode_local_secret(&key, principal, secret_ref, cipher_hex)
    }

    fn connection_status_file_result(
        &self,
    ) -> std::result::Result<PersistedConnectionStatusFile, String> {
        let Some(path) = self.status_store_path() else {
            return Ok(PersistedConnectionStatusFile::default());
        };
        if !path.exists() {
            return Ok(PersistedConnectionStatusFile::default());
        }
        let text = fs::read_to_string(&path)
            .map_err(|err| format!("read status store {} failed: {err}", path.display()))?;
        toml::from_str::<PersistedConnectionStatusFile>(&text)
            .map_err(|err| format!("parse status store {} failed: {err}", path.display()))
    }

    fn connection_status_file_or_record_error(&self) -> PersistedConnectionStatusFile {
        match self.connection_status_file_result() {
            Ok(file) => file,
            Err(err) => {
                self.record_persistence_error(err);
                PersistedConnectionStatusFile::default()
            }
        }
    }

    fn connection_selftest_status_snapshot(
        &self,
        connection_name: &str,
    ) -> ConnectionSelftestStatusSnapshot {
        let file = self.connection_status_file_or_record_error();
        ConnectionSelftestStatusSnapshot {
            errors: file
                .errors
                .into_iter()
                .filter(|error| error.connection_name == connection_name)
                .collect(),
            successes: file
                .successes
                .into_iter()
                .filter(|success| success.connection_name == connection_name)
                .collect(),
        }
    }

    fn connection_last_error(&self, connection_name: &str) -> Option<ConnectionLastError> {
        match self.connection_status_file_result() {
            Ok(file) => file
                .errors
                .into_iter()
                .find(|error| error.connection_name == connection_name)
                .map(|error| ConnectionLastError {
                    condition: error.condition,
                    bucket: error.bucket,
                    message: error.message,
                }),
            Err(err) => Some(ConnectionLastError {
                condition: "status_store_unreadable".to_string(),
                bucket: "config".to_string(),
                message: err,
            }),
        }
    }

    fn write_connection_last_error(
        &self,
        connection_name: &str,
        condition: &str,
        bucket: &str,
        message: &str,
    ) -> std::result::Result<(), String> {
        let Some(path) = self.status_store_path() else {
            return Ok(());
        };
        mutate_status_store(&path, |file| {
            file.errors
                .retain(|error| error.connection_name != connection_name);
            file.successes
                .retain(|success| success.connection_name != connection_name);
            file.errors.push(PersistedConnectionError {
                connection_name: connection_name.to_string(),
                condition: condition.to_string(),
                bucket: bucket.to_string(),
                message: message.to_string(),
            });
            true
        })
    }

    fn remove_connection_last_error(
        &self,
        connection_name: &str,
    ) -> std::result::Result<(), String> {
        let Some(path) = self.status_store_path() else {
            return Ok(());
        };
        if !path.exists() {
            return Ok(());
        }
        mutate_status_store(&path, |file| {
            let original_len = file.errors.len();
            file.errors
                .retain(|error| error.connection_name != connection_name);
            file.errors.len() != original_len
        })
    }

    fn write_connection_selftest_success(
        &self,
        connection_name: &str,
        provider: &str,
    ) -> std::result::Result<(), String> {
        let Some(path) = self.status_store_path() else {
            return Ok(());
        };
        let proof_target = self.connection_selftest_success_target(connection_name);
        mutate_status_store(&path, |file| {
            file.errors
                .retain(|error| error.connection_name != connection_name);
            file.successes
                .retain(|success| success.connection_name != connection_name);
            file.successes.push(PersistedConnectionSuccess {
                connection_name: connection_name.to_string(),
                provider: provider.to_string(),
                endpoint: proof_target
                    .as_ref()
                    .and_then(|target| target.endpoint.clone()),
                provider_model: proof_target
                    .as_ref()
                    .and_then(|target| target.provider_model.clone()),
                credential_fingerprint: proof_target
                    .as_ref()
                    .and_then(|target| target.credential_fingerprint.clone()),
                provider_config_fingerprint: proof_target
                    .as_ref()
                    .and_then(|target| target.provider_config_fingerprint.clone()),
            });
            true
        })
    }

    fn connection_selftest_success_target(
        &self,
        connection_name: &str,
    ) -> Option<ConnectionSelftestSuccessTarget> {
        let connection = self.connections.get(connection_name)?;
        match &connection.credential_reference.source {
            CredentialSource::HostSession { provider, .. } => {
                Some(ConnectionSelftestSuccessTarget {
                    provider: provider.clone(),
                    endpoint: None,
                    provider_model: None,
                    credential_fingerprint: credential_reference_fingerprint(connection),
                    provider_config_fingerprint: None,
                })
            }
            CredentialSource::NoAuth
            | CredentialSource::EnvVar { .. }
            | CredentialSource::StoredSecret { .. } => {
                let model = self.model_for_connection(connection_name)?;
                let provider_config = self.provider_invocation_config(&model)?;
                let credential_fingerprint =
                    self.connection_credential_fingerprint(connection, &provider_config)?;
                Some(ConnectionSelftestSuccessTarget {
                    provider: provider_config.provider.clone(),
                    endpoint: Some(provider_config.base_url.clone()),
                    provider_model: Some(provider_config.provider_model.clone()),
                    credential_fingerprint: Some(credential_fingerprint),
                    provider_config_fingerprint: Some(provider_config_fingerprint(
                        &provider_config,
                    )),
                })
            }
        }
    }

    fn connection_credential_fingerprint(
        &self,
        connection: &ConnectionRead,
        provider_config: &ProviderInvocationConfig,
    ) -> Option<String> {
        match &connection.credential_reference.source {
            CredentialSource::NoAuth => credential_reference_fingerprint(connection),
            CredentialSource::EnvVar { env_var } => {
                if provider_config.auth_scheme == "none" {
                    return credential_reference_fingerprint(connection);
                }
                let secret = std::env::var(env_var).ok()?;
                let secret = secret.trim();
                if secret.is_empty() {
                    return None;
                }
                let secret_fingerprint = live_secret_value_fingerprint("env", env_var, secret);
                credential_reference_fingerprint_with_parts(
                    connection,
                    &[("live_secret", secret_fingerprint)],
                )
            }
            CredentialSource::StoredSecret { .. } => {
                let Some(cipher_hex) = connection.credential_reference.sealed_secret_cipher_hex()
                else {
                    return credential_reference_fingerprint(connection);
                };
                let key_fingerprint = self.existing_local_secret_key_fingerprint()?;
                credential_reference_fingerprint_with_parts(
                    connection,
                    &[
                        ("sealed_cipher", cipher_hex.to_string()),
                        ("local_key", key_fingerprint),
                    ],
                )
            }
            CredentialSource::HostSession { .. } => credential_reference_fingerprint(connection),
        }
    }

    fn remove_connection_selftest_status(
        &self,
        connection_name: &str,
    ) -> std::result::Result<(), String> {
        let Some(path) = self.status_store_path() else {
            return Ok(());
        };
        if !path.exists() {
            return Ok(());
        }
        mutate_status_store(&path, |file| {
            let original_error_len = file.errors.len();
            let original_success_len = file.successes.len();
            file.errors
                .retain(|error| error.connection_name != connection_name);
            file.successes
                .retain(|success| success.connection_name != connection_name);
            file.errors.len() != original_error_len || file.successes.len() != original_success_len
        })
    }

    fn restore_connection_selftest_status(
        &self,
        connection_name: &str,
        snapshot: &ConnectionSelftestStatusSnapshot,
    ) -> std::result::Result<(), String> {
        let Some(path) = self.status_store_path() else {
            return Ok(());
        };
        if !path.exists() && snapshot.errors.is_empty() && snapshot.successes.is_empty() {
            return Ok(());
        }
        mutate_status_store(&path, |file| {
            let original_error_len = file.errors.len();
            let original_success_len = file.successes.len();
            file.errors
                .retain(|error| error.connection_name != connection_name);
            file.successes
                .retain(|success| success.connection_name != connection_name);
            let removed_existing = file.errors.len() != original_error_len
                || file.successes.len() != original_success_len;
            file.errors.extend(snapshot.errors.clone());
            file.successes.extend(snapshot.successes.clone());
            removed_existing || !snapshot.errors.is_empty() || !snapshot.successes.is_empty()
        })
    }

    fn resolve_credential_for_paid_call(
        &mut self,
        connection_name: &str,
        principal: &Principal,
        config: &mut PrincipalAwareConfig,
    ) -> Option<CredentialResolution> {
        let connection = self.connections.get(connection_name)?;
        if !connection.consent_recorded {
            return None;
        }
        self.resolve_credential(connection_name, principal, config)
    }

    pub fn resolve_model_connection(
        &mut self,
        model_id: &str,
        principal: &Principal,
        config: &mut PrincipalAwareConfig,
    ) -> MissingConnectionOutcome {
        let Some(model) = self.read_model(model_id) else {
            return MissingConnectionOutcome::Degraded {
                marker: DEGRADED_MARKER.to_string(),
            };
        };
        if let Some(connection_name) = model.connection_name {
            return self
                .resolve_credential(&connection_name, principal, config)
                .map(MissingConnectionOutcome::Resolved)
                .unwrap_or_else(|| MissingConnectionOutcome::HardError {
                    model_id: model_id.to_string(),
                    connection_name,
                });
        }
        MissingConnectionOutcome::Degraded {
            marker: DEGRADED_MARKER.to_string(),
        }
    }

    pub fn model_cache_path(&self, model_id: &str) -> Option<PathBuf> {
        self.read_model(model_id)
            .and_then(|model| model.cache_path.clone())
    }

    pub fn resolved_connection_log(&self) -> &[String] {
        &self.resolved_connection_log
    }

    pub fn persistence_errors(&self) -> Vec<String> {
        self.persistence_errors
            .lock()
            .map(|errors| errors.clone())
            .unwrap_or_default()
    }

    pub fn load_error(&self) -> Option<&str> {
        self.load_error.as_deref()
    }

    pub fn unreadable_connection_error(&self, connection_name: &str) -> Option<&str> {
        self.unreadable_connections
            .get(connection_name)
            .map(String::as_str)
    }

    pub fn write_sequence(&self) -> u64 {
        self.write_sequence
    }

    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    fn declared_local_model(
        &self,
        model_id: &str,
        model: &ModelRead,
    ) -> std::result::Result<Option<PreparedLocalModel>, String> {
        let Some(model_cache_path) = model.cache_path.as_ref() else {
            return Ok(None);
        };
        let local = self.authorized_local_model(model_id, model, model_cache_path)?;
        if local.artifact_filenames.len() != 1 {
            return Err(format!(
                "local llm model {model_id} must declare exactly one artifact in v1, found {}",
                local.artifact_filenames.len()
            ));
        }
        if local.per_file_sha256.len() != 1 {
            return Err(format!(
                "local llm model {model_id} must declare exactly one artifact sha in v1, found {}",
                local.per_file_sha256.len()
            ));
        }
        let artifact_name = &local.artifact_filenames[0];
        let artifact_path = local.cache_path.join(artifact_name);
        if !artifact_path.is_file() {
            return Err(format!(
                "declared local llm artifact missing: {}",
                artifact_path.display()
            ));
        }
        let Some((_, expected_sha)) = local
            .per_file_sha256
            .iter()
            .find(|(file, _)| file == artifact_name)
        else {
            return Err(format!(
                "declared local llm artifact {artifact_name} has no matching sha"
            ));
        };
        let actual_sha = sha256_file_hex(&artifact_path)?;
        if actual_sha != *expected_sha {
            return Err(format!(
                "declared local llm artifact sha mismatch for {artifact_name}: expected {expected_sha}, actual {actual_sha}"
            ));
        }
        Ok(Some(PreparedLocalModel {
            cache_path: local.cache_path,
            artifact_paths: vec![artifact_path],
            catalog_authorized: local.catalog_authorized,
        }))
    }

    fn authorized_local_model(
        &self,
        model_id: &str,
        model: &ModelRead,
        model_cache_path: &Path,
    ) -> std::result::Result<AuthorizedLocalModel, String> {
        if self.root.as_os_str().is_empty() {
            return Ok(AuthorizedLocalModel {
                cache_path: model_cache_path.to_path_buf(),
                artifact_filenames: model.artifact_filenames.clone(),
                per_file_sha256: model.per_file_sha256.clone(),
                catalog_authorized: false,
            });
        }
        let catalog = Catalog::open_with_scope_against_root(&self.root, CONNECTION_CATALOG_SCOPE)
            .map_err(|err| format!("local llm catalog open failed: {err}"))?;
        let entry = catalog
            .show(model_id)
            .map_err(|err| format!("local llm catalog metadata read failed: {err}"))?;
        let Source::LocalPath { .. } = &entry.source else {
            return Err(format!(
                "local llm catalog entry {model_id} is not a local-path model"
            ));
        };
        if entry.cache_status != CacheStatus::Cached {
            return Err(format!(
                "local llm catalog entry {model_id} is not cached: {:?}",
                entry.cache_status
            ));
        }
        let expected_cache_path = self.root.join(model_id);
        let Some(cache_path) = entry.local_path.clone() else {
            return Err(format!(
                "local llm catalog entry {model_id} has no authorized cache path"
            ));
        };
        if !same_existing_path(&cache_path, &expected_cache_path) {
            return Err(format!(
                "local llm catalog entry {model_id} cache path {} is outside expected catalog cache {}",
                cache_path.display(),
                expected_cache_path.display()
            ));
        }
        verify_local_manifest_identity(model_id, &cache_path, &entry.artifact_filenames)?;
        Ok(AuthorizedLocalModel {
            cache_path,
            artifact_filenames: entry.artifact_filenames,
            per_file_sha256: entry.per_file_sha256,
            catalog_authorized: true,
        })
    }

    fn provider_invocation_config(&self, model: &ModelRead) -> Option<ProviderInvocationConfig> {
        let endpoint = model.hosted_endpoint.as_ref()?;
        let specs = match try_load_provider_specs(&self.root, &[]) {
            Ok(specs) => specs,
            Err(err) => {
                self.record_persistence_error(err.message());
                return None;
            }
        };
        let Some(spec) = specs.providers.get(&endpoint.provider) else {
            self.record_persistence_error(format!(
                "provider {} has no provider spec row",
                endpoint.provider
            ));
            return None;
        };
        if spec.transport != ProviderTransport::Http {
            return None;
        }
        let Some(auth_scheme) = spec.auth_scheme else {
            self.record_persistence_error(format!(
                "provider {} missing HTTP auth scheme",
                endpoint.provider
            ));
            return None;
        };
        let Some(dialect) = spec.dialect else {
            self.record_persistence_error(format!(
                "provider {} missing HTTP dialect",
                endpoint.provider
            ));
            return None;
        };
        if spec.reply_path.trim().is_empty() {
            self.record_persistence_error(format!(
                "provider {} missing HTTP reply path",
                endpoint.provider
            ));
            return None;
        }
        if spec.timeout_ms == 0 {
            self.record_persistence_error(format!(
                "provider {} has invalid HTTP timeout",
                endpoint.provider
            ));
            return None;
        }
        let base_url = endpoint.base_url.clone();
        if let Err(err) =
            validate_hosted_endpoint_base_url_for_auth(&endpoint.provider, &base_url, auth_scheme)
        {
            self.record_persistence_error(err);
            return None;
        }
        let reply_path = spec.reply_path.clone();
        let timeout_ms = spec.timeout_ms;
        Some(ProviderInvocationConfig {
            provider: endpoint.provider.clone(),
            provider_model: endpoint.provider_model.clone(),
            base_url,
            request_path: provider_request_path(dialect).to_string(),
            auth_scheme: http_auth_scheme_name(auth_scheme).to_string(),
            dialect: provider_dialect_name(dialect).to_string(),
            reply_path,
            timeout_ms,
            transport: "http".to_string(),
        })
    }

    fn validate_hosted_endpoint_for_provider(
        &self,
        provider: &str,
        base_url: &str,
    ) -> std::result::Result<(), String> {
        let specs = try_load_provider_specs(&self.root, &[]).map_err(|err| err.message())?;
        let spec = specs
            .providers
            .get(provider)
            .ok_or_else(|| format!("provider {provider} has no provider spec row"))?;
        if spec.transport != ProviderTransport::Http {
            return Err(format!("provider {provider} is not an HTTP transport row"));
        }
        let Some(auth_scheme) = spec.auth_scheme else {
            return Err(format!("provider {provider} missing HTTP auth scheme"));
        };
        if spec.dialect.is_none() {
            return Err(format!("provider {provider} missing HTTP dialect"));
        }
        if spec.reply_path.trim().is_empty() {
            return Err(format!("provider {provider} missing HTTP reply path"));
        }
        validate_hosted_endpoint_base_url_for_auth(provider, base_url, auth_scheme)
    }
}

#[derive(Debug)]
struct AuthorizedLocalModel {
    cache_path: PathBuf,
    artifact_filenames: Vec<String>,
    per_file_sha256: Vec<(String, String)>,
    catalog_authorized: bool,
}

#[derive(Debug)]
struct PreparedLocalModel {
    cache_path: PathBuf,
    artifact_paths: Vec<PathBuf>,
    catalog_authorized: bool,
}

pub struct UnderstandingRuntime<C = NoopLlmClient> {
    catalog: ModelConnectionCatalog,
    principal: Principal,
    config: PrincipalAwareConfig,
    clients: BTreeMap<String, Arc<C>>,
    native_fallback: Option<Arc<NativeLlmClient>>,
    default_model_id: Option<String>,
}

impl<C> std::fmt::Debug for UnderstandingRuntime<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("UnderstandingRuntime")
            .field("catalog", &self.catalog)
            .field("principal", &self.principal)
            .field("config", &self.config)
            .field("client_count", &self.clients.len())
            .field("native_fallback", &self.native_fallback.is_some())
            .field("default_model_id", &self.default_model_id)
            .finish()
    }
}

impl<C> UnderstandingRuntime<C> {
    pub fn new(
        catalog: ModelConnectionCatalog,
        principal: Principal,
        config: PrincipalAwareConfig,
    ) -> Self {
        Self {
            catalog,
            principal,
            config,
            clients: BTreeMap::new(),
            native_fallback: Some(Arc::new(NativeLlmClient)),
            default_model_id: None,
        }
    }
}

impl<C: LlmClient> UnderstandingRuntime<C> {
    pub fn with_client(mut self, model_id: impl Into<String>, client: Arc<C>) -> Self {
        let model_id = model_id.into();
        self.default_model_id = Some(model_id.clone());
        self.clients.insert(model_id, client);
        self
    }

    pub fn catalog(&self) -> &ModelConnectionCatalog {
        &self.catalog
    }

    pub fn catalog_mut(&mut self) -> &mut ModelConnectionCatalog {
        &mut self.catalog
    }

    pub fn config(&self) -> &PrincipalAwareConfig {
        &self.config
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn prove_host_session_connection(
        &mut self,
        connection_name: &str,
        prompt: &str,
        progress_to_stdout: bool,
    ) -> std::result::Result<ModelProof, ModelProofError> {
        let Some(connection) = self.catalog.connections.get(connection_name).cloned() else {
            return Err(ModelProofError::NoHostSession);
        };
        let resolution = self.catalog.resolve_credential_for_paid_call(
            connection_name,
            &self.principal,
            &mut self.config,
        );
        let Some(resolution) = resolution else {
            return if connection.consent_recorded {
                Err(ModelProofError::NoHostSession)
            } else {
                Err(ModelProofError::ConsentNotRecorded)
            };
        };
        let request = UnderstandingRequest {
            context_id: ContextId::new_v7(),
            consumer: UnderstandingConsumer::ConversationCondensation,
            deployment: DeploymentSurface::Edge,
            input: prompt.to_string(),
            paid: true,
        };
        if let CredentialSource::HostSession { provider, .. } =
            connection.credential_reference.source.clone()
        {
            let invocation = unsafe {
                ModelInvocation::from_authorized_runtime_parts(AuthorizedModelInvocationParts {
                    request,
                    resolved_connection: Some(resolution.connection_name.clone()),
                    resolved_credential: resolution.secret,
                    hosted_endpoint: None,
                    provider_config: None,
                    cache_path: None,
                    local_artifacts: Vec::new(),
                    local_catalog_authorized: false,
                })
            };
            let client = HostSessionLlmClient {
                provider: provider.clone(),
                catalog_root: self.catalog.root.clone(),
                progress_to_stdout,
            };
            let reply = client.prove(&invocation)?;
            return Ok(ModelProof {
                connection_name: resolution.connection_name,
                provider,
                reply: reply.phrasing,
            });
        }
        let Some(model) = self.catalog.model_for_connection(connection_name) else {
            return Err(ModelProofError::NoHostSession);
        };
        let provider_config = self
            .catalog
            .provider_invocation_config(&model)
            .ok_or(ModelProofError::NoHostSession)?;
        let hosted_endpoint = model
            .hosted_endpoint
            .as_ref()
            .map(|endpoint| endpoint.base_url.clone());
        let invocation = unsafe {
            ModelInvocation::from_authorized_runtime_parts(AuthorizedModelInvocationParts {
                request,
                resolved_connection: Some(resolution.connection_name.clone()),
                resolved_credential: resolution.secret,
                hosted_endpoint,
                provider_config: Some(provider_config.clone()),
                cache_path: None,
                local_artifacts: Vec::new(),
                local_catalog_authorized: false,
            })
        };
        let reply = HttpChatCompletionsLlmClient
            .understand_native(&invocation)
            .map_err(|err| ModelProofError::LiveCallFailed {
                bucket: proof_bucket_from_failure(failure_bucket_from_error(&err)).to_string(),
                message: format!("{err}"),
            })?;
        if reply.malformed || reply.phrasing.trim().is_empty() {
            return Err(ModelProofError::ReplyFormat {
                path: provider_config.reply_path,
                framing: "json".to_string(),
            });
        }
        Ok(ModelProof {
            connection_name: resolution.connection_name,
            provider: provider_config.provider,
            reply: reply.phrasing,
        })
    }

    pub async fn understand_async(
        &mut self,
        request: UnderstandingRequest,
    ) -> std::result::Result<UnderstandingOutcome, UnderstandingRuntimeError> {
        match self.prepare_understanding(&request)? {
            PreparedUnderstanding::Invoke {
                model_id,
                invocation,
            } => {
                let reply = if invocation_uses_http_transport(&invocation) {
                    HttpChatCompletionsLlmClient.understand(&invocation).await
                } else if let Some(client) = self.clients.get(&model_id).cloned() {
                    client.understand(&invocation).await
                } else if let Some(native) = self.native_fallback.clone() {
                    native.understand(&invocation).await
                } else {
                    return Ok(degraded_outcome(&request, FailureBucket::NotConnected));
                };
                match reply {
                    Ok(reply) => {
                        Ok(self.outcome_after_reply(&request, &model_id, &invocation, reply))
                    }
                    Err(err) => {
                        self.record_provider_attempt_failure(&model_id, &invocation, &err);
                        Ok(degraded_outcome(&request, failure_bucket_from_error(&err)))
                    }
                }
            }
            PreparedUnderstanding::Degraded(outcome) => Ok(outcome),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn understand(
        &mut self,
        request: UnderstandingRequest,
    ) -> std::result::Result<UnderstandingOutcome, UnderstandingRuntimeError> {
        match self.prepare_understanding(&request)? {
            PreparedUnderstanding::Invoke {
                model_id,
                invocation,
            } => {
                let reply = if invocation_uses_http_transport(&invocation) {
                    HttpChatCompletionsLlmClient.understand_native(&invocation)
                } else if let Some(client) = self.clients.get(&model_id).cloned() {
                    client.understand_native(&invocation)
                } else if let Some(native) = self.native_fallback.clone() {
                    native.understand_native(&invocation)
                } else {
                    return Ok(degraded_outcome(&request, FailureBucket::NotConnected));
                };
                match reply {
                    Ok(reply) => {
                        Ok(self.outcome_after_reply(&request, &model_id, &invocation, reply))
                    }
                    Err(err) => {
                        self.record_provider_attempt_failure(&model_id, &invocation, &err);
                        Ok(degraded_outcome(&request, failure_bucket_from_error(&err)))
                    }
                }
            }
            PreparedUnderstanding::Degraded(outcome) => Ok(outcome),
        }
    }

    fn prepare_understanding(
        &mut self,
        request: &UnderstandingRequest,
    ) -> std::result::Result<PreparedUnderstanding, UnderstandingRuntimeError> {
        let model_id = self
            .catalog
            .context_models
            .get(&request.context_id)
            .cloned()
            .or_else(|| self.catalog.active_model_id())
            .or_else(|| self.default_model_id.clone());
        let Some(model_id) = model_id else {
            return Ok(PreparedUnderstanding::Degraded(degraded_outcome(
                request,
                FailureBucket::NotConnected,
            )));
        };
        let mut resolved_connection = None;
        let mut resolved_credential = None;
        let mut local_artifacts = Vec::new();
        let mut local_catalog_authorized = false;
        let Some(model) = self.catalog.read_model(&model_id) else {
            return Ok(PreparedUnderstanding::Degraded(degraded_outcome(
                request,
                FailureBucket::NotConnected,
            )));
        };
        if model.kind != ModelKind::LlmClient {
            return Ok(PreparedUnderstanding::Degraded(degraded_outcome(
                request,
                FailureBucket::NotConnected,
            )));
        }
        let mut cache_path = model.cache_path.clone();
        let hosted_endpoint = model
            .hosted_endpoint
            .as_ref()
            .map(|endpoint| endpoint.base_url.clone());
        let provider_config = self.catalog.provider_invocation_config(&model);
        if hosted_endpoint.is_some() && provider_config.is_none() {
            return Ok(PreparedUnderstanding::Degraded(degraded_outcome(
                request,
                FailureBucket::Unreachable,
            )));
        }
        if cache_path.is_some() {
            match self.catalog.declared_local_model(&model_id, &model) {
                Ok(Some(local)) => {
                    cache_path = Some(local.cache_path);
                    local_artifacts = local.artifact_paths;
                    local_catalog_authorized = local.catalog_authorized;
                }
                Ok(None) => {}
                Err(err) => {
                    self.catalog.record_persistence_error(err);
                    return Ok(PreparedUnderstanding::Degraded(degraded_outcome(
                        request,
                        FailureBucket::Unreachable,
                    )));
                }
            };
            if local_artifacts.is_empty() {
                return Ok(PreparedUnderstanding::Degraded(degraded_outcome(
                    request,
                    FailureBucket::NotConnected,
                )));
            }
        } else {
            if let Some(connection_name) = model.connection_name.as_deref() {
                let Some(connection) = self.catalog.connections.get(connection_name) else {
                    return Err(UnderstandingRuntimeError::DanglingConnection {
                        model_id,
                        connection_name: connection_name.to_string(),
                    });
                };
                if !connection.consent_recorded {
                    return Ok(PreparedUnderstanding::Degraded(degraded_outcome(
                        request,
                        FailureBucket::Unauthorized,
                    )));
                }
            }
            if let Some(connection_name) = model.connection_name.as_deref() {
                match self.catalog.resolve_credential_for_paid_call(
                    connection_name,
                    &self.principal,
                    &mut self.config,
                ) {
                    Some(resolution) => {
                        resolved_connection = Some(resolution.connection_name);
                        resolved_credential = resolution.secret;
                    }
                    None => {
                        return Err(UnderstandingRuntimeError::DanglingConnection {
                            model_id,
                            connection_name: connection_name.to_string(),
                        });
                    }
                }
            } else {
                match self.catalog.resolve_model_connection(
                    &model_id,
                    &self.principal,
                    &mut self.config,
                ) {
                    MissingConnectionOutcome::Resolved(resolution) => {
                        resolved_connection = Some(resolution.connection_name);
                        resolved_credential = resolution.secret;
                    }
                    MissingConnectionOutcome::HardError {
                        model_id,
                        connection_name,
                    } => {
                        return Err(UnderstandingRuntimeError::DanglingConnection {
                            model_id,
                            connection_name,
                        });
                    }
                    MissingConnectionOutcome::Degraded { .. } => {
                        return Ok(PreparedUnderstanding::Degraded(degraded_outcome(
                            request,
                            FailureBucket::NotConnected,
                        )));
                    }
                }
            }
        }

        Ok(PreparedUnderstanding::Invoke {
            model_id,
            // SAFETY: this path has either passed the named-connection consent
            // check above or the local catalog/cache proof branch.
            invocation: unsafe {
                Box::new(ModelInvocation::from_authorized_runtime_parts(
                    AuthorizedModelInvocationParts {
                        request: request.clone(),
                        resolved_connection,
                        resolved_credential,
                        hosted_endpoint,
                        provider_config,
                        cache_path,
                        local_artifacts,
                        local_catalog_authorized,
                    },
                ))
            },
        })
    }

    fn outcome_after_reply(
        &mut self,
        request: &UnderstandingRequest,
        model_id: &str,
        invocation: &ModelInvocation,
        reply: context_graph::ModelReply,
    ) -> UnderstandingOutcome {
        if reply.malformed {
            self.record_provider_diagnostic(model_id, invocation, "malformed provider reply");
            degraded_outcome(request, FailureBucket::ReplyMalformed)
        } else {
            success_outcome(request, invocation, reply)
        }
    }

    fn record_provider_attempt_failure(
        &self,
        model_id: &str,
        invocation: &ModelInvocation,
        err: &CgError,
    ) {
        self.record_provider_diagnostic(model_id, invocation, &format!("provider failed: {err}"));
    }

    fn record_provider_diagnostic(
        &self,
        model_id: &str,
        invocation: &ModelInvocation,
        reason: &str,
    ) {
        let route = invocation
            .resolved_connection
            .as_deref()
            .map(|connection| format!("connection={connection}"))
            .or_else(|| {
                invocation
                    .cache_path
                    .as_ref()
                    .map(|path| format!("cache_path={}", path.display()))
            })
            .unwrap_or_else(|| "route=unresolved".to_string());
        self.catalog.record_persistence_error(format!(
            "llm provider attempt model_id={model_id} {route}: {reason}"
        ));
    }
}

enum PreparedUnderstanding {
    Invoke {
        model_id: String,
        invocation: Box<ModelInvocation>,
    },
    Degraded(UnderstandingOutcome),
}

fn invocation_uses_http_transport(invocation: &ModelInvocation) -> bool {
    invocation
        .provider_config
        .as_ref()
        .is_some_and(|config| config.transport == "http")
}

fn success_outcome(
    request: &UnderstandingRequest,
    invocation: &ModelInvocation,
    reply: context_graph::ModelReply,
) -> UnderstandingOutcome {
    UnderstandingOutcome {
        marker: public_marker(request, &reply),
        degraded_marker: None,
        failure_bucket: None,
        phrasing: Some(reply.phrasing.clone()),
        model_invoked: true,
        resolved_connection: invocation.resolved_connection.clone(),
        cache_path_consulted: invocation.cache_path.clone(),
        stored_understanding: None,
        usage: reply.usage,
    }
}

fn understand_local_single_file(
    invocation: &ModelInvocation,
) -> context_graph::Result<context_graph::ModelReply> {
    if !invocation.local_catalog_authorized {
        return Err(CgError::Engine(
            "local llm artifact was not authorized by catalog cache metadata".to_string(),
        ));
    }
    if invocation.local_artifacts.len() != 1 {
        return Err(CgError::Engine(
            "local llm client supports exactly one single-file safetensors artifact in v1"
                .to_string(),
        ));
    }
    let artifact_path = &invocation.local_artifacts[0];
    let proof = local_candle_artifact_proof(artifact_path)?;
    let artifact = artifact_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("local-artifact");
    Ok(context_graph::ModelReply {
        marker: understanding_marker(&invocation.request),
        phrasing: format!(
            "local-candle:{artifact}:{proof}:{}",
            invocation.request.input
        ),
        advisory_flag: false,
        malformed: false,
        usage: None,
    })
}

fn local_candle_artifact_proof(path: &Path) -> context_graph::Result<String> {
    let bytes = fs::read(path)
        .map_err(|err| CgError::Engine(format!("local llm artifact read failed: {err}")))?;
    let tensors = SafeTensors::deserialize(&bytes).map_err(|err| {
        CgError::Engine(format!(
            "local llm artifact is not a single-file safetensors model: {err}"
        ))
    })?;
    let mut names = tensors.names();
    names.sort_unstable();
    let tensor_name = names
        .first()
        .copied()
        .ok_or_else(|| CgError::Engine("local llm safetensors file has no tensors".to_string()))?;
    let tensor = tensors
        .tensor(tensor_name)
        .map_err(|err| CgError::Engine(format!("local llm tensor lookup failed: {err}")))?;
    if tensor.shape().is_empty() || tensor.shape().contains(&0) {
        return Err(CgError::Engine(
            "local llm tensor has an empty shape".to_string(),
        ));
    }
    let digest = Sha256::digest(tensor.data());
    let scalar = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]) as f32;
    let loaded = Tensor::from_slice(&[scalar], (1,), &Device::Cpu)
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|err| CgError::Engine(format!("local llm candle tensor load failed: {err}")))?;
    Ok(format!(
        "{}:{:?}:{:?}:{:08x}",
        tensor_name,
        tensor.dtype(),
        tensor.shape(),
        loaded[0].to_bits()
    ))
}

fn same_existing_path(left: &Path, right: &Path) -> bool {
    match (fs::canonicalize(left), fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn verify_local_manifest_identity(
    model_id: &str,
    cache_path: &Path,
    entry_artifacts: &[String],
) -> std::result::Result<(), String> {
    let manifest_path = cache_path.join("catalog.toml");
    let text = fs::read_to_string(&manifest_path).map_err(|err| {
        format!(
            "read local llm manifest {} failed: {err}",
            manifest_path.display()
        )
    })?;
    let manifest = toml::from_str::<toml::Value>(&text).map_err(|err| {
        format!(
            "parse local llm manifest {} failed: {err}",
            manifest_path.display()
        )
    })?;
    let manifest_model_id = manifest
        .get("model_id")
        .and_then(toml::Value::as_str)
        .ok_or_else(|| {
            format!(
                "local llm manifest {} missing model_id",
                manifest_path.display()
            )
        })?;
    if manifest_model_id != model_id {
        return Err(format!(
            "local llm manifest model_id mismatch: expected {model_id}, found {manifest_model_id}"
        ));
    }
    let cache_status = manifest
        .get("cache_status")
        .and_then(toml::Value::as_str)
        .ok_or_else(|| {
            format!(
                "local llm manifest {} missing cache_status",
                manifest_path.display()
            )
        })?;
    if !cache_status.eq_ignore_ascii_case("cached") {
        return Err(format!(
            "local llm manifest {model_id} is not cached: {cache_status}"
        ));
    }
    let manifest_artifacts = manifest
        .get("artifact_filenames")
        .and_then(toml::Value::as_array)
        .ok_or_else(|| {
            format!(
                "local llm manifest {} missing artifact_filenames",
                manifest_path.display()
            )
        })?
        .iter()
        .filter_map(toml::Value::as_str)
        .filter(|file| *file != "catalog.toml")
        .map(str::to_string)
        .collect::<Vec<_>>();
    if manifest_artifacts != entry_artifacts {
        return Err(format!(
            "local llm manifest artifacts for {model_id} do not match catalog metadata"
        ));
    }
    Ok(())
}

fn understand_hosted(
    invocation: &ModelInvocation,
) -> context_graph::Result<context_graph::ModelReply> {
    if invocation.resolved_connection.is_none() {
        return Err(CgError::Engine("no hosted connection resolved".to_string()));
    }
    let provider_config = invocation.provider_config.as_ref().ok_or_else(|| {
        provider_failure(
            FailureBucket::NotConnected,
            "hosted llm endpoint is not configured for this model",
        )
    })?;
    if provider_config.transport != "http" || provider_config.dialect != "openai-chat-completions" {
        return Err(provider_failure(
            FailureBucket::NotConnected,
            "hosted llm provider transport is not supported by the native client",
        ));
    }
    let authorization = match provider_config.auth_scheme.as_str() {
        "bearer" => {
            let credential = invocation
                .resolved_credential
                .as_deref()
                .unwrap_or_default()
                .trim();
            if credential.is_empty() {
                return Err(provider_failure(
                    FailureBucket::Unauthorized,
                    "hosted llm bearer credential did not resolve for this principal",
                ));
            }
            Some(format!("Bearer {credential}"))
        }
        "none" => None,
        _ => {
            return Err(provider_failure(
                FailureBucket::Unauthorized,
                "hosted llm auth scheme is not supported",
            ));
        }
    };
    let endpoint = provider_request_url(&provider_config.base_url, &provider_config.request_path);
    let agent_config = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_millis(provider_config.timeout_ms)))
        .max_redirects(0)
        .http_status_as_error(false)
        .build();
    let agent: ureq::Agent = agent_config.into();
    let body = serde_json::json!({
        "model": provider_config.provider_model.as_str(),
        "max_tokens": HOSTED_CHAT_MAX_TOKENS,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": invocation.request.input.as_str(),
        }],
    });
    let body = serde_json::to_string(&body).map_err(|err| {
        CgError::Engine(format!("hosted llm request serialization failed: {err}"))
    })?;
    let mut request = agent
        .post(&endpoint)
        .header("Content-Type", "application/json");
    if let Some(authorization) = authorization {
        request = request.header("Authorization", authorization);
    }
    let mut response = request
        .send(body)
        .map_err(|err| provider_failure(ureq_failure_bucket(&err), &format!("{err}")))?;
    let status = response.status().as_u16();
    if (400..500).contains(&status) {
        return Err(provider_failure(
            FailureBucket::Unauthorized,
            &format!("hosted llm request returned HTTP {status}"),
        ));
    }
    if status >= 300 {
        return Err(provider_failure(
            FailureBucket::Unreachable,
            &format!("hosted llm request returned HTTP {status}"),
        ));
    }
    let reader = response
        .body_mut()
        .with_config()
        .limit(HOSTED_BODY_LIMIT)
        .reader();
    let mut payload = String::new();
    let mut reader = reader;
    reader.read_to_string(&mut payload).map_err(|err| {
        provider_failure(
            FailureBucket::Unreachable,
            &format!("hosted llm response read failed: {err}"),
        )
    })?;
    let spec = ProviderSpec {
        reply_framing: ReplyFraming::Json,
        reply_path: invocation
            .provider_config
            .as_ref()
            .map(|config| config.reply_path.clone())
            .unwrap_or_else(|| DEFAULT_CHAT_COMPLETIONS_REPLY_PATH.to_string()),
        ..ProviderSpec::default()
    };
    let phrasing = match extract_reply_from_spec(
        invocation
            .provider_config
            .as_ref()
            .map(|config| config.provider.as_str())
            .unwrap_or("openai-compatible"),
        &spec,
        ReplyFraming::Json,
        &payload,
        None,
    ) {
        Ok(reply) if !reply.trim().is_empty() => reply,
        Ok(_) | Err(_) => {
            return Ok(context_graph::ModelReply {
                marker: understanding_marker(&invocation.request),
                phrasing: String::new(),
                advisory_flag: false,
                malformed: true,
                usage: None,
            });
        }
    };
    Ok(context_graph::ModelReply {
        marker: understanding_marker(&invocation.request),
        phrasing,
        advisory_flag: false,
        malformed: false,
        usage: None,
    })
}

const PROVIDER_FAILURE_PREFIX: &str = "llm_provider_failure";

fn provider_failure(bucket: FailureBucket, message: &str) -> CgError {
    CgError::Engine(format!(
        "{PROVIDER_FAILURE_PREFIX}:{}:{message}",
        failure_bucket_name(bucket)
    ))
}

fn failure_bucket_from_error(err: &CgError) -> FailureBucket {
    let CgError::Engine(message) = err else {
        return FailureBucket::Unreachable;
    };
    if let Some(rest) = message.strip_prefix(PROVIDER_FAILURE_PREFIX)
        && let Some(rest) = rest.strip_prefix(':')
        && let Some((bucket, _)) = rest.split_once(':')
    {
        return failure_bucket_from_name(bucket).unwrap_or(FailureBucket::Unreachable);
    }
    FailureBucket::Unreachable
}

fn failure_bucket_name(bucket: FailureBucket) -> &'static str {
    match bucket {
        FailureBucket::NotConnected => "not_connected",
        FailureBucket::Unauthorized => "unauthorized",
        FailureBucket::Unreachable => "unreachable",
        FailureBucket::ReplyMalformed => "reply_malformed",
    }
}

fn failure_bucket_from_name(name: &str) -> Option<FailureBucket> {
    match name {
        "not_connected" => Some(FailureBucket::NotConnected),
        "unauthorized" => Some(FailureBucket::Unauthorized),
        "unreachable" => Some(FailureBucket::Unreachable),
        "reply_malformed" => Some(FailureBucket::ReplyMalformed),
        _ => None,
    }
}

fn proof_bucket_from_failure(bucket: FailureBucket) -> &'static str {
    match bucket {
        FailureBucket::NotConnected => "not_connected",
        FailureBucket::Unauthorized => "auth",
        FailureBucket::Unreachable => "unreachable",
        FailureBucket::ReplyMalformed => "format",
    }
}

fn ureq_failure_bucket(err: &ureq::Error) -> FailureBucket {
    match err {
        ureq::Error::StatusCode(status) if (400..500).contains(status) => {
            FailureBucket::Unauthorized
        }
        _ => FailureBucket::Unreachable,
    }
}

fn provider_request_url(base_url: &str, request_path: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        request_path.trim_start_matches('/')
    )
}

fn declared_local_artifact_metadata(path: &Path) -> (Vec<String>, Vec<(String, String)>) {
    let files = local_regular_files(path);
    let mut shas = Vec::new();
    for file in &files {
        let artifact = path.join(file);
        if let Ok(sha) = sha256_file_hex(&artifact) {
            shas.push((file.clone(), sha));
        }
    }
    (files, shas)
}

fn local_regular_files(path: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(path) else {
        return Vec::new();
    };
    let mut files = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if path.is_file() {
                entry.file_name().to_str().map(str::to_string)
            } else {
                None
            }
        })
        .filter(|file| file != "catalog.toml" && file != ".lock")
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn sha256_file_hex(path: &Path) -> std::result::Result<String, String> {
    let bytes = fs::read(path).map_err(|err| format!("read {} failed: {err}", path.display()))?;
    let digest = Sha256::digest(&bytes);
    Ok(format!("{digest:x}"))
}

fn mutate_status_store(
    path: &Path,
    mutate: impl FnOnce(&mut PersistedConnectionStatusFile) -> bool,
) -> std::result::Result<(), String> {
    let Some(parent) = path.parent() else {
        return Err(format!(
            "status store path has no parent: {}",
            path.display()
        ));
    };
    fs::create_dir_all(parent)
        .map_err(|err| format!("create {} failed: {err}", parent.display()))?;
    let lock_path = path.with_extension("toml.lock");
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|err| format!("open lock {} failed: {err}", lock_path.display()))?;
    lock.lock_exclusive()
        .map_err(|err| format!("lock {} failed: {err}", lock_path.display()))?;
    let mut file = if path.exists() {
        let text = fs::read_to_string(path)
            .map_err(|err| format!("read status store {} failed: {err}", path.display()))?;
        toml::from_str::<PersistedConnectionStatusFile>(&text)
            .map_err(|err| format!("parse status store {} failed: {err}", path.display()))?
    } else {
        PersistedConnectionStatusFile::default()
    };
    if !mutate(&mut file) {
        return Ok(());
    }
    write_status_store_unlocked(path, &file)
}

fn write_status_store_unlocked(
    path: &Path,
    file: &PersistedConnectionStatusFile,
) -> std::result::Result<(), String> {
    let text = toml::to_string_pretty(file)
        .map_err(|err| format!("serialize status store failed: {err}"))?;
    let tmp = path.with_extension(format!("toml.tmp.{}", std::process::id()));
    {
        let mut out = File::create(&tmp)
            .map_err(|err| format!("create temp {} failed: {err}", tmp.display()))?;
        out.write_all(text.as_bytes())
            .map_err(|err| format!("write temp {} failed: {err}", tmp.display()))?;
        out.sync_all()
            .map_err(|err| format!("sync temp {} failed: {err}", tmp.display()))?;
    }
    fs::rename(&tmp, path).map_err(|err| {
        let _ = fs::remove_file(&tmp);
        format!(
            "atomic rename {} -> {} failed: {err}",
            tmp.display(),
            path.display()
        )
    })?;
    Ok(())
}

fn create_private_file(path: &Path) -> std::result::Result<File, String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(path)
            .map_err(|err| format!("create private file {} failed: {err}", path.display()))
    }
    #[cfg(not(unix))]
    {
        File::create(path)
            .map_err(|err| format!("create private file {} failed: {err}", path.display()))
    }
}

fn ensure_private_file_permissions(path: &Path) -> std::result::Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let metadata = fs::metadata(path)
            .map_err(|err| format!("read permissions for {} failed: {err}", path.display()))?;
        let mode = metadata.permissions().mode();
        if mode & 0o077 != 0 {
            fs::set_permissions(path, fs::Permissions::from_mode(0o600)).map_err(|err| {
                format!("repair permissions for {} failed: {err}", path.display())
            })?;
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

fn write_secret_store_unlocked(
    path: &Path,
    store: &PersistedSecretStore,
) -> std::result::Result<(), String> {
    let text = toml::to_string_pretty(store)
        .map_err(|err| format!("serialize secret store failed: {err}"))?;
    let tmp = path.with_extension(format!("toml.tmp.{}", std::process::id()));
    {
        let mut file = File::create(&tmp)
            .map_err(|err| format!("create temp {} failed: {err}", tmp.display()))?;
        file.write_all(text.as_bytes())
            .map_err(|err| format!("write temp {} failed: {err}", tmp.display()))?;
        file.sync_all()
            .map_err(|err| format!("sync temp {} failed: {err}", tmp.display()))?;
    }
    fs::rename(&tmp, path).map_err(|err| {
        let _ = fs::remove_file(&tmp);
        format!(
            "atomic rename {} -> {} failed: {err}",
            tmp.display(),
            path.display()
        )
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

fn parse_local_secret_key(text: &str) -> Option<[u8; 32]> {
    let bytes = hex_decode(text.trim())?;
    bytes.try_into().ok()
}

fn encode_local_secret(
    key: &[u8; 32],
    principal: &str,
    secret_ref: &str,
    secret: &str,
) -> std::result::Result<String, String> {
    let nonce_bytes = random_secret_nonce();
    let aad = local_secret_aad(principal, secret_ref);
    let ciphertext =
        xor_with_local_secret_stream(key, &nonce_bytes, aad.as_bytes(), secret.as_bytes());
    let tag = local_secret_tag(key, &nonce_bytes, aad.as_bytes(), &ciphertext);
    let mut payload = nonce_bytes.to_vec();
    payload.extend(ciphertext);
    payload.extend(tag);
    Ok(hex_encode(&payload))
}

fn decode_local_secret(
    key: &[u8; 32],
    principal: &str,
    secret_ref: &str,
    cipher_hex: &str,
) -> Option<String> {
    let payload = hex_decode(cipher_hex)?;
    if payload.len() < 44 {
        return None;
    }
    let (nonce, sealed) = payload.split_at(12);
    let (ciphertext, tag) = sealed.split_at(sealed.len().checked_sub(32)?);
    let aad = local_secret_aad(principal, secret_ref);
    let expected_tag = local_secret_tag(key, nonce, aad.as_bytes(), ciphertext);
    if !constant_time_eq(tag, &expected_tag) {
        return None;
    }
    let plaintext = xor_with_local_secret_stream(key, nonce, aad.as_bytes(), ciphertext);
    String::from_utf8(plaintext).ok()
}

fn local_secret_aad(principal: &str, secret_ref: &str) -> String {
    format!("cg-local-secret-v2\0{principal}\0{secret_ref}")
}

fn random_secret_bytes() -> [u8; 32] {
    let mut key = [0_u8; 32];
    key[..16].copy_from_slice(Uuid::new_v4().as_bytes());
    key[16..].copy_from_slice(Uuid::new_v4().as_bytes());
    key
}

fn random_secret_nonce() -> [u8; 12] {
    let mut nonce = [0_u8; 12];
    nonce.copy_from_slice(&Uuid::new_v4().as_bytes()[..12]);
    nonce
}

fn xor_with_local_secret_stream(key: &[u8; 32], nonce: &[u8], aad: &[u8], input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len());
    for (block_index, chunk) in input.chunks(32).enumerate() {
        let mut digest = Sha256::new();
        digest.update(b"cg-local-secret-stream-v1");
        digest.update(key);
        digest.update(nonce);
        digest.update(aad);
        digest.update((block_index as u64).to_le_bytes());
        let block = digest.finalize();
        output.extend(
            chunk
                .iter()
                .zip(block.iter())
                .map(|(byte, mask)| byte ^ mask),
        );
    }
    output
}

fn local_secret_tag(key: &[u8; 32], nonce: &[u8], aad: &[u8], ciphertext: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"cg-local-secret-tag-v1");
    digest.update(key);
    digest.update(nonce);
    digest.update(aad);
    digest.update(ciphertext);
    digest.finalize().into()
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0_u8, |diff, (left, right)| diff | (left ^ right))
        == 0
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn hex_decode(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(text.len() / 2);
    for pair in text.as_bytes().chunks_exact(2) {
        let high = hex_nibble(pair[0])?;
        let low = hex_nibble(pair[1])?;
        out.push((high << 4) | low);
    }
    Some(out)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn public_marker(request: &UnderstandingRequest, reply: &context_graph::ModelReply) -> String {
    let deterministic = understanding_marker(request);
    let scaffold_marker = format!("understood:{}", request.input);
    if reply.marker == deterministic || reply.marker == scaffold_marker {
        reply.marker.clone()
    } else {
        deterministic
    }
}

fn understanding_marker(request: &UnderstandingRequest) -> String {
    match request.consumer {
        context_graph::UnderstandingConsumer::ConversationCondensation => {
            format!("condensed:{}", request.input)
        }
        context_graph::UnderstandingConsumer::BlueprintNormalization => {
            format!("blueprint:{}", request.input)
        }
        context_graph::UnderstandingConsumer::WebhookEntityResolution => {
            format!("entity:{}", request.input)
        }
        context_graph::UnderstandingConsumer::SignatureSemanticJudgment => {
            format!("signature-judgment:{}", request.input)
        }
    }
}

fn catalog_entry_for_model(model: &ModelRead) -> CatalogEntry {
    let source = if let Some(connection_name) = &model.connection_name {
        Source::NamedConnection {
            connection_name: connection_name.clone(),
        }
    } else if let Some(cache_path) = &model.cache_path {
        Source::LocalPath {
            dir: cache_path.clone(),
        }
    } else if let Some(env_var) = &model.welded_auth_env_var {
        Source::ApiEndpoint {
            base_url: "https://provider.example.invalid/v1".to_string(),
            provider: "legacy-provider".to_string(),
            model: model.id.clone(),
            auth_env_var: env_var.clone(),
        }
    } else {
        Source::BundledBinary
    };
    let named_connection = matches!(source, Source::NamedConnection { .. });
    CatalogEntry {
        id: model.id.clone(),
        kind: model.kind.clone(),
        capability: Capability {
            dimension: None,
            output_shape: Some("understanding".to_string()),
            input_modality: Some("text".to_string()),
        },
        license: License {
            spdx: "Proprietary".to_string(),
            url: "https://provider.example.invalid/license".to_string(),
            requires_manual_acceptance: false,
        },
        source,
        byte_size: 0,
        cache_status: if named_connection {
            CacheStatus::Cached
        } else {
            CacheStatus::NotCached
        },
        local_path: model.cache_path.clone(),
        install_instructions: None,
        audit_footnote: None,
        acceptance_recorded: false,
        manual_install_url: None,
        artifact_filenames: model.artifact_filenames.clone(),
        per_file_sha256: model.per_file_sha256.clone(),
    }
}

fn hosted_model_read(
    id: String,
    connection_name: String,
    base_url: String,
    provider: String,
    provider_model: String,
) -> ModelRead {
    ModelRead {
        id,
        kind: ModelKind::LlmClient,
        connection_name: Some(connection_name),
        welded_auth_env_var: None,
        exposed_secret: None,
        cache_path: None,
        hosted_endpoint: Some(HostedEndpoint {
            base_url,
            provider,
            provider_model,
        }),
        artifact_filenames: Vec::new(),
        per_file_sha256: Vec::new(),
    }
}

fn model_read_from_entry(entry: &CatalogEntry) -> ModelRead {
    match &entry.source {
        Source::NamedConnection { connection_name } => ModelRead {
            id: entry.id.clone(),
            kind: entry.kind.clone(),
            connection_name: Some(connection_name.clone()),
            welded_auth_env_var: None,
            exposed_secret: None,
            cache_path: None,
            hosted_endpoint: None,
            artifact_filenames: entry.artifact_filenames.clone(),
            per_file_sha256: entry.per_file_sha256.clone(),
        },
        Source::ApiEndpoint {
            base_url,
            provider,
            model,
            auth_env_var,
        } => ModelRead {
            id: entry.id.clone(),
            kind: entry.kind.clone(),
            connection_name: None,
            welded_auth_env_var: Some(auth_env_var.clone()),
            exposed_secret: None,
            cache_path: None,
            hosted_endpoint: Some(HostedEndpoint {
                base_url: base_url.clone(),
                provider: provider.clone(),
                provider_model: model.clone(),
            }),
            artifact_filenames: entry.artifact_filenames.clone(),
            per_file_sha256: entry.per_file_sha256.clone(),
        },
        Source::LocalPath { dir } => ModelRead {
            id: entry.id.clone(),
            kind: entry.kind.clone(),
            connection_name: None,
            welded_auth_env_var: None,
            exposed_secret: None,
            cache_path: entry.local_path.clone().or_else(|| Some(dir.clone())),
            hosted_endpoint: None,
            artifact_filenames: entry.artifact_filenames.clone(),
            per_file_sha256: entry.per_file_sha256.clone(),
        },
        _ => ModelRead {
            id: entry.id.clone(),
            kind: entry.kind.clone(),
            connection_name: None,
            welded_auth_env_var: None,
            exposed_secret: None,
            cache_path: None,
            hosted_endpoint: None,
            artifact_filenames: entry.artifact_filenames.clone(),
            per_file_sha256: entry.per_file_sha256.clone(),
        },
    }
}

fn degraded_outcome(
    request: &UnderstandingRequest,
    failure_bucket: FailureBucket,
) -> UnderstandingOutcome {
    UnderstandingOutcome {
        marker: format!("declarative:{}", request.input),
        degraded_marker: Some(DEGRADED_MARKER.to_string()),
        failure_bucket: Some(failure_bucket),
        phrasing: None,
        model_invoked: false,
        resolved_connection: None,
        cache_path_consulted: None,
        stored_understanding: Some(request.input.clone()),
        usage: None,
    }
}
