use serde::Serialize;
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Warning,
    Error,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Finding {
    pub check_id: String,
    pub severity: Severity,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix_hint: Option<String>,
}

impl Finding {
    pub fn warning(
        check_id: impl Into<String>,
        message: impl Into<String>,
        target: Option<String>,
        fix_hint: Option<String>,
    ) -> Self {
        Self {
            check_id: check_id.into(),
            severity: Severity::Warning,
            message: message.into(),
            target,
            fix_hint,
        }
    }

    pub fn error(
        check_id: impl Into<String>,
        message: impl Into<String>,
        target: Option<String>,
        fix_hint: Option<String>,
    ) -> Self {
        Self {
            check_id: check_id.into(),
            severity: Severity::Error,
            message: message.into(),
            target,
            fix_hint,
        }
    }
}

type Detector = Arc<dyn Fn() -> Vec<Finding> + Send + Sync + 'static>;

#[derive(Clone)]
pub struct DiagnosticCheck {
    pub check_id: String,
    pub spending: bool,
    detector: Detector,
}

impl DiagnosticCheck {
    pub fn free(
        check_id: impl Into<String>,
        detector: impl Fn() -> Vec<Finding> + Send + Sync + 'static,
    ) -> Self {
        Self {
            check_id: check_id.into(),
            spending: false,
            detector: Arc::new(detector),
        }
    }

    pub fn spending(
        check_id: impl Into<String>,
        detector: impl Fn() -> Vec<Finding> + Send + Sync + 'static,
    ) -> Self {
        Self {
            check_id: check_id.into(),
            spending: true,
            detector: Arc::new(detector),
        }
    }

    pub fn detect(&self) -> Vec<Finding> {
        (self.detector)()
    }
}

impl fmt::Debug for DiagnosticCheck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiagnosticCheck")
            .field("check_id", &self.check_id)
            .field("spending", &self.spending)
            .finish_non_exhaustive()
    }
}

pub trait CheckRegistrar {
    fn register_check(&mut self, check: DiagnosticCheck);
}

#[derive(Default, Debug)]
pub struct CheckRegistry {
    checks: Vec<DiagnosticCheck>,
}

impl CheckRegistry {
    pub fn register(&mut self, check: DiagnosticCheck) {
        self.register_check(check);
    }

    pub fn checks(&self) -> &[DiagnosticCheck] {
        &self.checks
    }
}

impl CheckRegistrar for CheckRegistry {
    fn register_check(&mut self, check: DiagnosticCheck) {
        self.checks.push(check);
    }
}

#[derive(Debug, Serialize)]
pub struct DoctorReport {
    pub ok: bool,
    pub checks_run: Vec<String>,
    pub findings: Vec<Finding>,
}

pub fn run_doctor_checks(registry: &CheckRegistry) -> DoctorReport {
    run_checks(registry, false)
}

pub fn run_selftest_checks(registry: &CheckRegistry) -> DoctorReport {
    run_checks(registry, true)
}

fn run_checks(registry: &CheckRegistry, spending: bool) -> DoctorReport {
    let mut findings = Vec::new();
    let mut checks_run = Vec::new();
    for check in registry
        .checks()
        .iter()
        .filter(|check| check.spending == spending)
    {
        checks_run.push(check.check_id.clone());
        findings.extend(check.detect());
    }
    let ok = findings.is_empty();
    DoctorReport {
        ok,
        checks_run,
        findings,
    }
}
