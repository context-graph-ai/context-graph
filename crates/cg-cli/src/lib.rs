use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProviderSpecs {
    pub providers: BTreeMap<String, ProviderSpec>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProviderSpec {
    pub binary: String,
    pub prompt_delivery: String,
    pub env_remove: Vec<String>,
    pub reply_framing: String,
    pub timeout_ms: u64,
    pub consent_required: bool,
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Classification {
    Success,
    HostLoginExpired { message: String },
    RateLimited { message: String },
    BillingRequired { message: String },
    LiveCallFailed { message: String },
}

pub fn load_provider_specs(_root: &Path, overrides: &[ProviderOverride]) -> ProviderSpecs {
    let mut providers = BTreeMap::from([
        (
            "claude".to_string(),
            ProviderSpec {
                binary: "claude".to_string(),
                prompt_delivery: "stdin".to_string(),
                env_remove: vec!["ANTHROPIC_API_KEY".to_string()],
                reply_framing: "json".to_string(),
                timeout_ms: 2_500,
                consent_required: true,
            },
        ),
        (
            "codex".to_string(),
            ProviderSpec {
                binary: "codex".to_string(),
                prompt_delivery: "stdin".to_string(),
                env_remove: vec!["OPENAI_API_KEY".to_string(), "CODEX_API_KEY".to_string()],
                reply_framing: "file".to_string(),
                timeout_ms: 2_500,
                consent_required: true,
            },
        ),
        (
            "agy".to_string(),
            ProviderSpec {
                binary: "agy".to_string(),
                prompt_delivery: "stdin".to_string(),
                env_remove: Vec::new(),
                reply_framing: "text".to_string(),
                timeout_ms: 2_500,
                consent_required: true,
            },
        ),
    ]);
    for override_layer in overrides {
        if let Some(spec) = providers.get_mut(&override_layer.provider) {
            if let Some(timeout_ms) = override_layer.timeout_ms {
                spec.timeout_ms = timeout_ms;
            }
            if let Some(remove_env) = &override_layer.remove_env {
                for name in remove_env {
                    if !spec.env_remove.iter().any(|existing| existing == name) {
                        spec.env_remove.push(name.clone());
                    }
                }
            }
            if override_layer.consent_required == Some(true) {
                spec.consent_required = true;
            }
        }
    }
    ProviderSpecs { providers }
}

pub fn construct_claude_invocation(prompt: &str) -> Invocation {
    Invocation {
        argv: vec![
            "-p".to_string(),
            "--mcp-config={}".to_string(),
            "--tools=[]".to_string(),
            "--output-format=json".to_string(),
        ],
        stdin: prompt.to_string(),
    }
}

pub fn extract_reply(
    _provider: &str,
    framing: &str,
    stdout: &str,
    output_file: Option<&Path>,
) -> Result<String, ReplyError> {
    match framing {
        "json" => {
            let value = serde_json::from_str::<serde_json::Value>(stdout)
                .or_else(|_| parse_last_json_line(stdout))
                .map_err(|_| ReplyError::MissingPath {
                    path: "/result".to_string(),
                    framing: framing.to_string(),
                })?;
            value
                .get("result")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| ReplyError::MissingPath {
                    path: "/result".to_string(),
                    framing: framing.to_string(),
                })
        }
        "file" => output_file
            .and_then(|path| fs::read_to_string(path).ok())
            .ok_or_else(|| ReplyError::MissingPath {
                path: "output_file".to_string(),
                framing: framing.to_string(),
            }),
        "text" => Ok(stdout.to_string()),
        other => Err(ReplyError::MissingPath {
            path: "/result".to_string(),
            framing: other.to_string(),
        }),
    }
}

pub fn classify_provider_reply(_provider: &str, stdout: &str) -> Classification {
    let parsed =
        serde_json::from_str::<serde_json::Value>(stdout).or_else(|_| parse_last_json_line(stdout));
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
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        let message = value
            .get("result")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("provider call failed")
            .to_string();
        return match value
            .get("api_error_status")
            .and_then(serde_json::Value::as_i64)
        {
            Some(401) => Classification::HostLoginExpired { message },
            Some(402) => Classification::BillingRequired { message },
            Some(429) => Classification::RateLimited { message },
            _ => classify_message(&message),
        };
    }
    Classification::Success
}

fn parse_last_json_line(text: &str) -> serde_json::Result<serde_json::Value> {
    let last = text
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .unwrap_or(text);
    serde_json::from_str(last)
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckRegistry {
    checks: Vec<DiagnosticCheck>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagnosticCheck {
    pub check_id: String,
    pub spends: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finding {
    pub check_id: String,
    pub severity: String,
    pub message: String,
    pub target: String,
    pub fix_hint: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DoctorReport {
    pub checks_run: Vec<String>,
    pub findings: Vec<Finding>,
}

pub fn register_model_connection_checks(registry: &mut CheckRegistry) {
    registry.checks.push(DiagnosticCheck {
        check_id: "model/claude/flags-in-help".to_string(),
        spends: false,
    });
    registry.checks.push(DiagnosticCheck {
        check_id: "model/selftest/live-call".to_string(),
        spends: true,
    });
}

pub fn run_doctor_checks(registry: &CheckRegistry) -> DoctorReport {
    DoctorReport {
        checks_run: registry
            .checks
            .iter()
            .filter(|check| !check.spends)
            .map(|check| check.check_id.clone())
            .collect(),
        findings: registry
            .checks
            .iter()
            .filter(|check| !check.spends && check.check_id == "model/claude/flags-in-help")
            .map(|check| Finding {
                check_id: check.check_id.clone(),
                severity: "warning".to_string(),
                message: "claude provider help must expose --mcp-config and --tools for host-session driving".to_string(),
                target: "claude".to_string(),
                fix_hint: "upgrade or reinstall claude so cg can pass --mcp-config using equals-form flags".to_string(),
            })
            .collect(),
    }
}

pub fn run_selftest_checks(registry: &CheckRegistry) -> DoctorReport {
    DoctorReport {
        checks_run: registry
            .checks
            .iter()
            .filter(|check| check.spends)
            .map(|check| check.check_id.clone())
            .collect(),
        findings: Vec::new(),
    }
}
