use std::collections::BTreeMap;
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

pub fn load_provider_specs(_root: &Path, _overrides: &[ProviderOverride]) -> ProviderSpecs {
    ProviderSpecs::default()
}

pub fn construct_claude_invocation(_prompt: &str) -> Invocation {
    Invocation::default()
}

pub fn extract_reply(
    _provider: &str,
    _framing: &str,
    _stdout: &str,
    _output_file: Option<&Path>,
) -> Result<String, ReplyError> {
    Ok("No provider reply was extracted.".to_string())
}

pub fn classify_provider_reply(_provider: &str, _stdout: &str) -> Classification {
    Classification::Success
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

pub fn register_model_connection_checks(_registry: &mut CheckRegistry) {}

pub fn run_doctor_checks(registry: &CheckRegistry) -> DoctorReport {
    DoctorReport {
        checks_run: registry
            .checks
            .iter()
            .filter(|check| !check.spends)
            .map(|check| check.check_id.clone())
            .collect(),
        findings: Vec::new(),
    }
}

pub fn run_selftest_checks(_registry: &CheckRegistry) -> DoctorReport {
    DoctorReport::default()
}
