//! Distributed-acceleration fabric — the consumer surface a cg node uses to offload session
//! *understanding* onto another enrolled machine in its own tenant fabric, then apply the result
//! back onto the original capture.
//!
//! SCAFFOLD STATE: every function here is a deliberately-wrong-but-honest stub. Bodies return
//! empty/default/open values so the failing acceptance tests fail on an assertion (never a panic).
//! A later effort implements the real submission / worker / apply / movement / retention behavior
//! over the shipped contextdb work ledger. No coordination logic is authored here yet.

use std::path::Path;

use serde_json::Value;

use contextdb_engine::work_ledger;
use contextdb_server::SyncClient;

use crate::ingest::ConversationExtractor;
use crate::tier0::{Tier0CaptureReport, Tier0CaptureRequest, capture_tier0_inner};
use crate::{ContextId, DigestId, Result, Store};

/// The default per-node fabric config file, next to the memory db under `<data_root>/cg/`.
pub const FABRIC_CONFIG_FILE: &str = "fabric.toml";

/// The `[fabric]` deployment config for one node. Loaded from `<data_root>/cg/fabric.toml`; a node
/// with no such file is a plain single-node install and never contacts a hub.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FabricConfig {
    /// The pasted hub enrollment ticket / endpoint spec (transport-neutral).
    pub hub_endpoint: Option<String>,
    /// The tenant the node enrolls under.
    pub tenant: Option<String>,
    /// Input-sharing switch. cg default is ON — an ephemeral copy of understanding input may move
    /// to the user's own same-tenant machines.
    pub sharing: bool,
    /// Grace period, in milliseconds, before an ephemeral input copy is swept after result-lands.
    pub grace_ms: i64,
    /// Per-source opt-out list: a source named here keeps its input local (a `local_path` ref).
    pub per_source_opt_out: Vec<String>,
}

impl Default for FabricConfig {
    fn default() -> Self {
        Self {
            hub_endpoint: None,
            tenant: None,
            // Sharing is ON by default — proven by input material actually moving, never a silent
            // constant. The one visible off knob is `sharing = false` in fabric.toml.
            sharing: true,
            // A day, on the order the ratified retention contract specifies.
            grace_ms: 86_400_000,
            per_source_opt_out: Vec::new(),
        }
    }
}

impl FabricConfig {
    /// Parse a `[fabric]` config from TOML text. An omitted `sharing` key defaults ON.
    pub fn from_toml_str(text: &str) -> std::result::Result<Self, String> {
        let value: toml::Value =
            toml::from_str(text).map_err(|err| format!("fabric.toml: {err}"))?;
        let section = value.get("fabric").unwrap_or(&value);
        let mut config = FabricConfig::default();
        if let Some(endpoint) = section.get("hub_endpoint").and_then(|v| v.as_str()) {
            config.hub_endpoint = Some(endpoint.to_string());
        }
        if let Some(tenant) = section.get("tenant").and_then(|v| v.as_str()) {
            config.tenant = Some(tenant.to_string());
        }
        if let Some(sharing) = section.get("sharing").and_then(|v| v.as_bool()) {
            config.sharing = sharing;
        }
        if let Some(grace) = section.get("grace_ms").and_then(|v| v.as_integer()) {
            config.grace_ms = grace;
        }
        if let Some(list) = section.get("per_source_opt_out").and_then(|v| v.as_array()) {
            config.per_source_opt_out = list
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
        }
        Ok(config)
    }

    /// Load `<data_root>/cg/fabric.toml`. Returns `Ok(None)` when the file is absent (single-node).
    pub fn load(data_root: &Path) -> Result<Option<Self>> {
        let path = data_root.join("cg").join(FABRIC_CONFIG_FILE);
        match std::fs::read_to_string(&path) {
            Ok(text) => Self::from_toml_str(&text)
                .map(Some)
                .map_err(crate::CgError::Engine),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(crate::CgError::Engine(format!(
                "reading {}: {err}",
                path.display()
            ))),
        }
    }
}

/// A held per-data-root worker lock. Dropping it releases the lock.
#[derive(Debug)]
pub struct WorkerLock {
    _file: std::fs::File,
}

/// Acquire the per-data-root worker lock so a single node runs one worker at a time. Returns `None`
/// when another worker already holds it — the second worker is then a no-op. This is the same
/// advisory file lock the CLI worker used, lifted to the library surface.
pub fn acquire_worker_lock(data_root: &Path) -> Option<WorkerLock> {
    let path = data_root.join("understand-worker.lock");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
        .ok()?;
    use fs4::fs_std::FileExt;
    match file.try_lock_exclusive() {
        Ok(true) => Some(WorkerLock { _file: file }),
        _ => None,
    }
}

/// A capture that submits its per-chunk understanding onto the fabric ledger rather than the old
/// private single-machine queue.
pub struct FabricCapture<'a> {
    /// The tier-0 capture request (verbatim digest, local embedding, kept raw — unchanged).
    pub request: Tier0CaptureRequest,
    /// The node's fabric config, or `None` for a single-node install.
    pub config: Option<&'a FabricConfig>,
    /// Per-source opt-out flag from `cg import`/`cg record` — keeps this source's input local.
    pub source_opt_out: bool,
    /// A test-only fault seam so the abort-mid-capture rollback is provable without a real crash.
    pub fault: Option<&'a dyn CaptureFault>,
}

impl<'a> FabricCapture<'a> {
    /// A plain fabric capture with no config, no opt-out, no fault injection.
    pub fn new(request: Tier0CaptureRequest) -> Self {
        Self {
            request,
            config: None,
            source_opt_out: false,
            fault: None,
        }
    }
}

/// A capture fault-injection seam, honored by the submission path between the digest write and the
/// job/input writes so a test can force the whole capture transaction to roll back. Production
/// passes no fault; the double lives in test support.
pub trait CaptureFault {
    /// Called after the digest write, before the job/input writes. Returning `Err` aborts capture.
    fn fail_after_digest(&self) -> Result<()>;
}

/// Capture a conversation onto the fabric: tier-0 verbatim digest + local embedding + kept raw
/// chunk, and — in the SAME atomic $0 transaction — a ledger understanding-job row plus its copied
/// input rows. A source named in the node's `per_source_opt_out` list, or flagged `source_opt_out`
/// on this capture, keeps its understanding input local (an `InputRef::local_path`, no copied
/// `work_inputs` row). Submission is model-free: the executor runs the model later, elsewhere.
pub fn capture(store: &Store, capture: FabricCapture<'_>) -> Result<Tier0CaptureReport> {
    let opt_out = capture.source_opt_out
        || capture
            .config
            .map(|config| {
                config
                    .per_source_opt_out
                    .iter()
                    .any(|source| source == &capture.request.source)
            })
            .unwrap_or(false);
    capture_tier0_inner(store, capture.request, opt_out, capture.fault)
}

/// Derive the ledger job id for a tier-0 digest — a deterministic 1:1 function of the `DigestId`.
/// Re-importing identical content yields the same digest and thus the same job id, so the ledger's
/// insert-if-not-exists dedups the resubmit and apply maps the job id back to its digest. This is the
/// same derivation the submission path uses, kept single-sourced in `tier0`.
pub fn job_id_for_digest(digest: DigestId) -> String {
    crate::tier0::understanding_job_id(digest)
}

/// A job's user-visible state, mapped from the ledger's computed state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FabricJobStatus {
    /// The ledger job id (derived 1:1 from the anchor digest).
    pub job_id: String,
    /// The computed state as a plain word: `pending` | `leased` | `done` | `failed` | `cancelled`.
    pub state: String,
    /// The node that executed the job — distinct from the submitter for a remote execution.
    pub executor_node_id: Option<String>,
    /// The token/time receipt carried back from the executor, verbatim.
    pub receipt: Option<Value>,
}

/// Read the fabric job statuses for one context, as `cg status` renders them. The clock is injected
/// via `now_ms` so a lease past its TTL renders as reclaimable/pending, never leased/running — status
/// honesty consults the clock rather than rendering a stale stored lease.
///
/// SCAFFOLD STUB: returns no statuses.
pub fn job_statuses(
    store: &Store,
    context: ContextId,
    now_ms: i64,
) -> Result<Vec<FabricJobStatus>> {
    let _ = (store, context, now_ms);
    Ok(Vec::new())
}

/// The outcome of applying a completed remote result onto its anchor digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplyOutcome {
    /// Understanding was applied onto the named digest.
    Applied(DigestId),
    /// The anchor digest was absent locally (e.g. `cg forget`-purged) — nothing written.
    AnchorMissing,
    /// The result was already applied (write-once completion marker held) — a no-op.
    AlreadyApplied,
}

/// Apply a completed `work_results` row onto the original digest, anchor-checked and idempotent.
///
/// SCAFFOLD STUB: reports the anchor missing and writes nothing.
pub fn apply_result(
    store: &Store,
    job_id: &str,
    extractor: &dyn ConversationExtractor,
) -> Result<ApplyOutcome> {
    let _ = (store, job_id, extractor);
    Ok(ApplyOutcome::AnchorMissing)
}

/// The movement decision for one input reference under the current sharing posture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MovementDecision {
    /// The input copy may move to same-tenant machines.
    MayMove,
    /// The input stays on this machine (opted-out source, or sharing off).
    LocalOnly,
    /// The reference kind is unrecognized — refused (default-deny).
    Refused,
}

/// Decide whether an input reference of a given kind may move, mirroring the shipped ledger's
/// `MovementPolicy::permits_input_read` as a three-way outcome. The submitter's own material may
/// always move; for anyone else a movable kind (`ledger_input` / `blob_ref`) moves only when sharing
/// is on. A `local_path` ref never leaves its node. An unrecognized/future kind is refused
/// (default-deny) — never admitted because it "isn't recognized as local."
pub fn movement_decision(ref_kind: &str, sharing_on: bool, is_submitter: bool) -> MovementDecision {
    match ref_kind {
        work_ledger::REF_KIND_LEDGER_INPUT | work_ledger::REF_KIND_BLOB_REF => {
            if sharing_on || is_submitter {
                MovementDecision::MayMove
            } else {
                MovementDecision::LocalOnly
            }
        }
        work_ledger::REF_KIND_LOCAL_PATH => MovementDecision::LocalOnly,
        _ => MovementDecision::Refused,
    }
}

/// Count the input copies that would enter this node's outbound push set under `config`. With sharing
/// off, the OFF switch dominates — nothing leaves the machine — so the count is zero regardless of
/// any copied material. Otherwise it is the number of ephemeral `work_inputs` copies (opted-out
/// sources never copied a row, so they never count).
pub fn input_material_in_push_set(store: &Store, config: &FabricConfig) -> Result<usize> {
    if !config.sharing {
        return Ok(0);
    }
    let db = store.sync_database();
    let rows = db
        .scan("work_inputs", db.snapshot())
        .map_err(crate::tier0::map_engine)?;
    Ok(rows.len())
}

/// Drain and locally execute pending understanding jobs for this node, applying each result onto
/// its digest. Returns the number of jobs carried to `done`.
///
/// SCAFFOLD STUB: executes nothing.
pub fn drain_local_worker(
    store: &Store,
    extractor: &dyn ConversationExtractor,
    node_id: &str,
    budget: Option<usize>,
) -> Result<usize> {
    let _ = (store, extractor, node_id, budget);
    Ok(0)
}

/// A fabric worker's runtime configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FabricWorkerConfig {
    /// This node's stable id.
    pub node_id: String,
    /// The capability tags this node advertises (text understanding, native-schema output).
    pub advertised_tags: Vec<String>,
    /// Lease duration in milliseconds — a few minutes covers a slow local chunk, expires promptly
    /// for a second attempt after a crash.
    pub lease_ms: i64,
    /// Whether this node accepts moved input copies.
    pub sharing: bool,
}

impl FabricWorkerConfig {
    /// A default worker config for a node id.
    pub fn for_node(node_id: &str) -> Self {
        Self {
            node_id: node_id.to_string(),
            advertised_tags: vec!["class:cg.understanding".to_string()],
            lease_ms: 180_000,
            sharing: true,
        }
    }
}

/// The outcome of one worker poll against the hub.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FabricPollOutcome {
    /// No matching pending work this poll.
    NoWork,
    /// This node executed a job.
    Executed { job_id: String },
    /// This node lost the claim race for a job; it executed nothing.
    LostClaim { holder: Option<String> },
    /// This node applied a completed result onto its digest.
    Applied { digest: DigestId },
}

/// Poll the hub once through `client`: claim a matching job (consuming the shipped ledger's
/// exclusivity), materialize inputs under the movement policy, run the model, record the result, and
/// — on the digest-owning node — apply it. The clock is injected via `now_ms` (no wall-clock reads).
///
/// SCAFFOLD STUB: finds no work.
pub fn run_worker_once(
    client: &SyncClient,
    store: &Store,
    extractor: &dyn ConversationExtractor,
    config: &FabricWorkerConfig,
    now_ms: i64,
) -> Result<FabricPollOutcome> {
    let _ = (client, store, extractor, config, now_ms);
    Ok(FabricPollOutcome::NoWork)
}

/// Sweep this node's own ephemeral `work_inputs` copies for terminal jobs past `grace_ms`, driven by
/// an injected `now_ms`. Returns the number of input rows deleted.
///
/// SCAFFOLD STUB: deletes nothing.
pub fn run_input_retention(store: &Store, now_ms: i64, grace_ms: i64) -> Result<usize> {
    let _ = (store, now_ms, grace_ms);
    Ok(0)
}

/// The remote-execution receipt: the executor node id, tokens, and wall-clock a job's worker
/// produced. It lands on the operational `observations.properties` lane and is structurally barred
/// from a decision-basis input.
///
/// It carries NO `From<_> for serde_json::Value` (mirroring `CapturedUsage`): the operational
/// receipt is structurally barred from flowing into a decision-basis lane. Render it explicitly via
/// [`FabricExecutionReceipt::to_properties_value`] onto the `observations.properties` lane instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FabricExecutionReceipt {
    /// The node that executed the job.
    pub executor_node_id: String,
    /// Tokens the executor's model reported.
    pub total_tokens: u64,
    /// Wall-clock milliseconds the execution took on the executor.
    pub wall_clock_ms: u64,
}

impl FabricExecutionReceipt {
    /// Render the receipt onto the operational properties lane.
    pub fn to_properties_value(&self) -> Value {
        serde_json::json!({
            "executor_node_id": self.executor_node_id,
            "total_tokens": self.total_tokens,
            "wall_clock_ms": self.wall_clock_ms,
        })
    }
}
