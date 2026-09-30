//! Distributed-acceleration fabric — the consumer seam a cg node uses to offload session
//! *understanding* onto another enrolled machine in its own tenant fabric, then apply the result
//! back onto the original capture.
//!
//! A node plays two roles that collapse onto one machine in the single-node case: a SUBMITTER (owns
//! the tier-0 digest and the permanent raw store) and a WORKER (any enrolled capable node that runs
//! the model). Submission, the movement knob, apply, retention, and status are always-compiled
//! library surfaces over the shipped contextdb work ledger; the CLI renders over them. Only the
//! `SyncClient`-consuming worker poll ([`run_worker_once`]) is gated behind `fabric-sync`, so the
//! embeddable core stays transport- and tokio-free. cg authors no coordination logic of its own —
//! it consumes the shipped ledger's claim exclusivity, lease/abandonment, movement policy, and
//! per-node input retention unchanged.

use std::collections::HashMap;
use std::path::Path;

use contextdb_core::Value as DbValue;
use serde_json::Value;

use contextdb_engine::work_ledger;

use crate::capture_sync_status::{ModelCallUsage, UsageReportState};
use crate::harness::run_capture_into_digest_with_completion;
use crate::ingest::{CaptureRequest, ConversationExtractor};
use crate::tier0::{
    Tier0CaptureReport, Tier0CaptureRequest, capture_tier0_inner, completion_exists_local,
    context_digests, job_exists, map_engine, understanding_job_id,
};
use crate::{ContextId, DigestId, Result, Store};

// The SyncClient-consuming worker seam is the only part of the fabric that pulls
// contextdb-server (and thus tokio); it lives behind the `fabric-sync` feature so the default
// embeddable-core build stays tokio-free. Everything else here — submission, apply, movement,
// retention, and status — is always compiled and needs no transport.
#[cfg(feature = "fabric-sync")]
use contextdb_server::SyncClient;

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
    /// The on-disk read lives in `cg-node-fs`, one layer out, so the embeddable core stays free of
    /// filesystem access.
    pub fn load(data_root: &Path) -> Result<Option<Self>> {
        let path = data_root.join("cg").join(FABRIC_CONFIG_FILE);
        match cg_node_fs::load_text_if_present(&path) {
            Ok(Some(text)) => Self::from_toml_str(&text)
                .map(Some)
                .map_err(crate::CgError::Engine),
            Ok(None) => Ok(None),
            Err(err) => Err(crate::CgError::Engine(format!(
                "reading {}: {err}",
                path.display()
            ))),
        }
    }
}

/// A held per-data-root worker lock. Dropping it releases the lock. The lock primitive lives in
/// `cg-node-fs` so the embeddable core carries no filesystem access.
pub use cg_node_fs::WorkerLock;

/// Acquire the per-data-root worker lock so a single node runs one worker at a time. Returns `None`
/// when another worker already holds it — the second worker is then a no-op. The advisory lock
/// itself lives in `cg-node-fs`, one layer out from the embeddable core.
pub fn acquire_worker_lock(data_root: &Path) -> Option<WorkerLock> {
    cg_node_fs::acquire_worker_lock(&data_root.join("understand-worker.lock")).ok()
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
    // Global sharing OFF, a per-source opt-out match, or the per-invocation `--no-share` flag all
    // resolve to the SAME mechanism: the understanding input is submitted as an `InputRef::local_path`
    // and NO `work_inputs` copy is written, so the input copy that could leak never exists. Sharing-off
    // is not a display constant filtered downstream — it is the absence of the copy at the source.
    let sharing_off = capture
        .config
        .map(|config| !config.sharing)
        .unwrap_or(false);
    let per_source_opt_out = capture
        .config
        .map(|config| {
            config
                .per_source_opt_out
                .iter()
                .any(|source| source == &capture.request.source)
        })
        .unwrap_or(false);
    let opt_out = capture.source_opt_out || sharing_off || per_source_opt_out;
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
/// State and receipt are COMPUTED from the ledger rows, never a column cg flips: a landed
/// `work_results` row is `done` and carries the executor's node id and receipt verbatim (the receipt
/// originates on the executor and rides sync unmangled); otherwise the engine's `job_state` maps the
/// computed pending/leased/failed/cancelled — an expired lease falls back to pending under `now_ms`.
pub fn job_statuses(
    store: &Store,
    context: ContextId,
    now_ms: i64,
) -> Result<Vec<FabricJobStatus>> {
    let db = store.internal_db();
    let mut statuses = Vec::new();
    for (digest_id, _order) in context_digests(store, context)? {
        let job_id = understanding_job_id(digest_id);
        if !job_exists(store, &job_id)? {
            continue;
        }
        // A landed result is `done` regardless of the clock, and carries the executor's node id and
        // receipt straight from the synced `work_results` row — never reconstructed on the submitter.
        if let Some(result) = work_ledger::job_result(db, &job_id).map_err(map_engine)? {
            statuses.push(FabricJobStatus {
                job_id,
                state: "done".to_string(),
                executor_node_id: Some(result.executor_node_id),
                receipt: Some(result.receipt),
            });
            continue;
        }
        let state = match work_ledger::job_state(db, &job_id, now_ms).map_err(map_engine)? {
            work_ledger::JobState::Pending => "pending",
            work_ledger::JobState::Leased { .. } => "leased",
            work_ledger::JobState::Done => "done",
            work_ledger::JobState::Failed => "failed",
            work_ledger::JobState::Cancelled => "cancelled",
        };
        statuses.push(FabricJobStatus {
            job_id,
            state: state.to_string(),
            executor_node_id: None,
            receipt: None,
        });
    }
    Ok(statuses)
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

/// Apply a completed result onto the original digest, anchor-checked and idempotent.
///
/// The job id maps back to its anchor digest (a job id is derived 1:1 from the tier-0 `DigestId`).
/// Apply is anchor-checked: if the anchor digest is absent locally — never captured, or `cg
/// forget`-purged — nothing is written, no digest is resurrected, and the outcome is observable at
/// the library level (`AnchorMissing`). The retained write-once completion marker makes a second
/// apply of the same result a no-op (`AlreadyApplied`), so a late remote result for a job already
/// applied locally adds no duplicate memory. Understanding is anchored onto the existing digest via
/// `run_capture_into_digest`, so a chunk yields exactly one digest that transitions verbatim →
/// structured.
pub fn apply_result(
    store: &Store,
    job_id: &str,
    extractor: &dyn ConversationExtractor,
) -> Result<ApplyOutcome> {
    apply_result_inner(store, job_id, extractor, None)
}

/// Apply a completed result while injecting a capture fault at the completion-marker boundary — the
/// test-only seam that proves the graph mutation and the completion marker share ONE transaction: a
/// fault fired after the graph rows are staged but before commit rolls BOTH back, so neither the
/// structured graph nor the apply marker survives. Production uses [`apply_result`] (no fault).
pub fn apply_result_with_fault(
    store: &Store,
    job_id: &str,
    extractor: &dyn ConversationExtractor,
    fault: &dyn CaptureFault,
) -> Result<ApplyOutcome> {
    apply_result_inner(store, job_id, extractor, Some(fault))
}

fn apply_result_inner(
    store: &Store,
    job_id: &str,
    extractor: &dyn ConversationExtractor,
    fault: Option<&dyn CaptureFault>,
) -> Result<ApplyOutcome> {
    let Some(digest_id) = resolve_anchor_digest(store, job_id)? else {
        return Ok(ApplyOutcome::AnchorMissing);
    };
    // Honest fast path: apply the WORKER's recorded structured output onto the anchor, model-free. The
    // distributed promise is that a worker's understanding becomes the submitter's memory WITHOUT
    // re-running a model on the submitter, so whenever a `work_results` payload exists we deserialize
    // and commit THAT — never re-extract.
    if let Some(outcome) = apply_recorded_result(store, digest_id, fault)? {
        return Ok(outcome);
    }
    // Fallback: no worker result was recorded for this job — a single-node or out-of-band apply that
    // names the work directly, before any executor ran. Understand locally over the kept chunk through
    // the extractor. The write-once apply marker (LOCAL, never synced) commits in the SAME transaction
    // as the structured graph write, so a crash can never leave the graph applied but the marker missing.
    if store.get_digest(digest_id)?.is_none() {
        return Ok(ApplyOutcome::AnchorMissing);
    }
    let digest_text = digest_id.to_string();
    if completion_exists_local(store, &digest_text)? {
        return Ok(ApplyOutcome::AlreadyApplied);
    }
    let Some((context_id, context_label)) = digest_context(store, digest_id)? else {
        return Ok(ApplyOutcome::AnchorMissing);
    };
    let Some(chunk) = chunk_text_for_digest(store, &digest_text)? else {
        // The raw understanding substrate is gone (purged) — nothing to apply onto the anchor.
        return Ok(ApplyOutcome::AnchorMissing);
    };
    run_capture_into_digest_with_completion(
        store,
        CaptureRequest {
            context_label,
            transcript: chunk,
            model_available: true,
        },
        extractor,
        &crate::harness::CaptureHarnessConfig::default(),
        Some(digest_id),
        Some(crate::ingest::CaptureCompletionMarker {
            context_id,
            digest_id,
            model: "fabric-apply",
            fault,
        }),
    )?;
    Ok(ApplyOutcome::Applied(digest_id))
}

/// Commit a job's RECORDED worker result (the structured harvest the executor produced and pushed on
/// the `work_results` row) onto its anchor digest, MODEL-FREE — the honest distributed apply. Returns:
///   * `Ok(None)` when the job has no deserializable worker result yet, so the caller may fall back to
///     local extraction. (The single-node drain sentinel output is not a harvest, but a locally-drained
///     job already carries its completion marker and short-circuits to `AlreadyApplied` first.)
///   * `Ok(Some(AnchorMissing))` when the anchor digest is absent — never captured, or `cg forget`-purged.
///   * `Ok(Some(AlreadyApplied))` when the write-once completion marker already holds.
///   * `Ok(Some(Applied(_)))` when the worker's payload was committed onto the anchor.
fn apply_recorded_result(
    store: &Store,
    digest_id: DigestId,
    fault: Option<&dyn CaptureFault>,
) -> Result<Option<ApplyOutcome>> {
    if store.get_digest(digest_id)?.is_none() {
        return Ok(Some(ApplyOutcome::AnchorMissing));
    }
    let digest_text = digest_id.to_string();
    if completion_exists_local(store, &digest_text)? {
        return Ok(Some(ApplyOutcome::AlreadyApplied));
    }
    let Some(harvest) = worker_result_harvest(store, &understanding_job_id(digest_id))? else {
        return Ok(None);
    };
    let Some((context_id, context_label)) = digest_context(store, digest_id)? else {
        return Ok(Some(ApplyOutcome::AnchorMissing));
    };
    let Some(chunk) = chunk_text_for_digest(store, &digest_text)? else {
        return Ok(Some(ApplyOutcome::AnchorMissing));
    };
    crate::harness::apply_harvest_into_digest_with_completion(
        store,
        CaptureRequest {
            context_label,
            transcript: chunk,
            model_available: true,
        },
        harvest,
        &crate::harness::CaptureHarnessConfig::default(),
        Some(digest_id),
        Some(crate::ingest::CaptureCompletionMarker {
            context_id,
            digest_id,
            model: "fabric-apply",
            fault,
        }),
    )?;
    Ok(Some(ApplyOutcome::Applied(digest_id)))
}

/// Deserialize a completed job's recorded structured output into a harvest, or `None` when the job has
/// no result row yet or its output is not a structured harvest (e.g. the single-node drain sentinel,
/// whose memory was already applied inline and is guarded by its completion marker).
fn worker_result_harvest(
    store: &Store,
    job_id: &str,
) -> Result<Option<crate::ingest::StructuredHarvest>> {
    let Some(result) = work_ledger::job_result(store.internal_db(), job_id).map_err(map_engine)?
    else {
        return Ok(None);
    };
    Ok(serde_json::from_slice::<crate::ingest::StructuredHarvest>(&result.output).ok())
}

/// After a `cg sync pull`, materialize every remote worker result that has arrived. For each local
/// anchor digest whose understanding job now has a landed `work_results` row (`done`) but whose
/// completion marker is absent, deserialize the worker's structured output and commit it onto the
/// digest — model-free, idempotent under the write-once marker, and dropped when the anchor digest was
/// `cg forget`-purged. This is the seam the CLI's `cg sync pull` calls so a pulled result actually
/// BECOMES memory rather than only flipping status to `done`. Returns the digests newly materialized.
pub fn apply_pulled_results(store: &Store) -> Result<Vec<DigestId>> {
    let mut applied = Vec::new();
    for context in store.list_contexts()? {
        for (digest_id, _order) in context_digests(store, context.id)? {
            let job_id = understanding_job_id(digest_id);
            if !job_exists(store, &job_id)? {
                continue;
            }
            if let Some(ApplyOutcome::Applied(id)) = apply_recorded_result(store, digest_id, None)?
            {
                applied.push(id);
            }
        }
    }
    Ok(applied)
}

/// Map a ledger job id back to its anchor digest. A real understanding job id encodes the digest
/// uuid (`cg.understanding.<uuid>`); an out-of-band job id (a caller that names the work directly)
/// anchors onto the single digest that carries an understanding job. Returns `None` when no anchor
/// can be resolved.
fn resolve_anchor_digest(store: &Store, job_id: &str) -> Result<Option<DigestId>> {
    if let Some(uuid) = job_id
        .strip_prefix("cg.understanding.")
        .and_then(|rest| uuid::Uuid::parse_str(rest).ok())
    {
        return Ok(Some(DigestId::from(uuid)));
    }
    let mut best: Option<(i128, DigestId)> = None;
    for context in store.list_contexts()? {
        for (digest_id, order) in context_digests(store, context.id)? {
            if job_exists(store, &understanding_job_id(digest_id))?
                && best.is_none_or(|(seen, _)| order > seen)
            {
                best = Some((order, digest_id));
            }
        }
    }
    Ok(best.map(|(_, digest)| digest))
}

/// The context (id + label) a digest was captured under, or `None` if the digest is absent.
fn digest_context(store: &Store, digest_id: DigestId) -> Result<Option<(ContextId, String)>> {
    let result = store.read_query(
        "SELECT context_id FROM digests WHERE id = $id",
        &HashMap::from([("id".to_string(), DbValue::Uuid(digest_id.as_uuid()))]),
    )?;
    let Some(DbValue::Uuid(context_uuid)) = result.rows.first().and_then(|row| row.first()) else {
        return Ok(None);
    };
    let context_id = ContextId::from(*context_uuid);
    let label = store
        .list_contexts()?
        .into_iter()
        .find(|context| context.id == context_id)
        .map(|context| context.name)
        .unwrap_or_default();
    Ok(Some((context_id, label)))
}

/// The kept raw chunk text for a digest (the understanding substrate), or `None` if purged.
fn chunk_text_for_digest(store: &Store, digest_text: &str) -> Result<Option<String>> {
    let result = store.read_query(
        "SELECT content FROM cg_source_chunks WHERE digest_id = $digest_id",
        &HashMap::from([(
            "digest_id".to_string(),
            DbValue::Text(digest_text.to_string()),
        )]),
    )?;
    Ok(result.rows.first().and_then(|row| match row.first() {
        Some(DbValue::Text(content)) => Some(content.clone()),
        _ => None,
    }))
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

/// Count the ephemeral understanding-input copies that actually sit in this node's outbound push set.
/// The answer is computed from the REAL rows present, never from re-reading the sharing flag: an input
/// copy only exists as a `work_inputs` row when submission chose to make one (sharing on, not opted
/// out), and every `work_inputs` row has the engine-owned two-way declaration, so
/// its presence IS its push-set membership. With sharing off (or an opted-out source) the copy was
/// never written, so the count is naturally zero — the OFF switch dominates at the source, not by a
/// downstream filter. `config` is accepted for API symmetry with the status renderer; the count does
/// not branch on it.
pub fn input_material_in_push_set(store: &Store, config: &FabricConfig) -> Result<usize> {
    let _ = config;
    input_material_count(store.as_graph_rows())
}

/// [`input_material_in_push_set`] through the read-only reader; the reader must declare
/// `work_inputs` (as [`crate::event_history_reader_declaration`] does).
pub fn input_material_in_push_set_read_only(
    reader: &crate::ConsumerReader,
    config: &FabricConfig,
) -> Result<usize> {
    let _ = config;
    input_material_count(&reader.graph())
}

/// The one push-set count, authored once over the shared row seam: every `work_inputs` row is an
/// input copy in the push set.
fn input_material_count(rows: &dyn crate::store::graph_rows::GraphRows) -> Result<usize> {
    let answer = rows.query_rows("SELECT COUNT(*) AS n FROM work_inputs", &HashMap::new())?;
    match answer.rows.first().and_then(|row| row.first()) {
        Some(DbValue::Int64(count)) => usize::try_from(*count)
            .map_err(|_| crate::CgError::Engine("push-set count is negative".into())),
        other => Err(crate::CgError::Engine(format!(
            "push-set count returned {other:?}"
        ))),
    }
}

/// Drain and locally execute pending understanding jobs for this node, applying each result onto its
/// digest. This is the single-node degrade: submitter, worker, and digest owner are one machine, so
/// the hub round-trip collapses to a local claim + execute + record + apply. Jobs drain in PRIORITY
/// order (most-retrieved first) and stop at `budget`; each carried job records a `work_results` row
/// naming this node as executor (so `cg status` shows a done job with its executor and receipt) and a
/// write-once completion marker. Returns the number of jobs carried to `done`.
pub fn drain_local_worker(
    store: &Store,
    extractor: &dyn ConversationExtractor,
    node_id: &str,
    budget: Option<usize>,
) -> Result<usize> {
    // Gather the drainable backlog across every context in PRIORITY order (touch DESC), then honor
    // the budget ceiling globally. `list_pending_digests` already orders each context's backlog.
    let mut pending: Vec<(ContextId, String, crate::understanding::PendingDigest)> = Vec::new();
    for context in store.list_contexts()? {
        let label = context.name.clone();
        for candidate in crate::understanding::list_pending_digests(store, context.id, None)? {
            pending.push((context.id, label.clone(), candidate));
        }
    }
    pending.sort_by_key(|candidate| std::cmp::Reverse(candidate.2.touch_count));

    let mut drained = 0usize;
    for (context_id, context_label, candidate) in pending {
        if budget.is_some_and(|cap| drained >= cap) {
            break;
        }
        let digest_id = candidate.digest_id;
        let job_id = understanding_job_id(digest_id);
        let db = store.internal_db();
        // Claim locally (the single-machine claim takes an unexpiring lease). A concurrent second
        // in-process worker is already excluded by the per-data-root lock; the ledger claim is the
        // belt to that suspenders.
        let attempt = work_ledger::next_attempt(db, &job_id).map_err(map_engine)?;
        match work_ledger::insert_claim(db, &job_id, attempt, node_id, i64::MAX, 0)
            .map_err(map_engine)?
        {
            work_ledger::ClaimInsert::Inserted => {}
            work_ledger::ClaimInsert::AlreadyHeld { .. } => continue,
        }
        // Execute: run the model over the kept chunk and anchor the structured graph onto the
        // ORIGINAL digest — the executor's structuring and the owner's apply collapse on one node.
        // The write-once apply marker joins the SAME transaction as the graph write (a crash can
        // never leave the graph applied but the marker missing).
        let (report, _harness) = run_capture_into_digest_with_completion(
            store,
            CaptureRequest {
                context_label,
                transcript: candidate.chunk_text.clone(),
                model_available: true,
            },
            extractor,
            &crate::harness::CaptureHarnessConfig::default(),
            Some(digest_id),
            Some(crate::ingest::CaptureCompletionMarker {
                context_id,
                digest_id,
                model: "fabric-local",
                fault: None,
            }),
        )?;
        // Record the exactly-once result (executor = this node). This is the worker-plane receipt,
        // deliberately a separate append-only ledger write from the owner-plane apply marker above.
        // Its tokens and wall-clock are the model's REAL reported usage for this understanding (read
        // from the capture report's usage), never the input byte length or a zero clock.
        let receipt = FabricExecutionReceipt {
            executor_node_id: node_id.to_string(),
            total_tokens: total_usage_tokens(&report.model_usage),
            wall_clock_ms: total_usage_elapsed_ms(&report.model_usage),
        };
        work_ledger::record_result(
            store.internal_db(),
            &job_id,
            attempt,
            node_id,
            b"structured-locally",
            receipt.to_properties_value(),
            0,
        )
        .map_err(map_engine)?;
        drained += 1;
    }
    Ok(drained)
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

/// Poll the hub once through `client`: list the jobs this node may claim, claim one by push (consuming
/// the shipped ledger's exclusivity — the push reply is the arbitration verdict), materialize its
/// input under the movement policy, run the model through the existing extractor path, and record the
/// exactly-once result. The executor produces the structured RESULT and receipt; it never writes the
/// graph (apply happens on the digest owner). The clock is injected via `now_ms`.
///
/// This poll consumes the shipped `claim_job` and ledger primitives unchanged — cg authors only the
/// executor body and the poll ordering. It deliberately does NOT pull before claiming so a genuine
/// claim race is observable: a loser sees the winner's live claim through its own push reply and
/// reports `LostClaim`; a claim whose competitor's lease has expired is reclaimed at a higher attempt.
///
/// This is the INNER claim/execute/record primitive: it does no hub I/O at the poll boundary, so it
/// scans only rows already present in `client.db()`. A real remote worker never calls it directly —
/// it drives [`run_worker_poll`], which pulls the tenant's ledger BEFORE this poll (so peers' jobs
/// are visible) and pushes the recorded result/failure AFTER it (so the outcome reaches the digest
/// owner without waiting for the next claim's push to piggyback it). Calling `run_worker_once`
/// directly is the claim-race harness only: it is the seam where a second in-process worker that has
/// NOT pulled the winner's completed result can still be observed racing the claim.
#[cfg(feature = "fabric-sync")]
pub fn run_worker_once<E: ConversationExtractor + Sync>(
    client: &SyncClient,
    store: &Store,
    extractor: &E,
    config: &FabricWorkerConfig,
    now_ms: i64,
) -> Result<FabricPollOutcome> {
    let _ = store;
    let executor = CgWorkExecutor {
        extractor,
        node_id: config.node_id.clone(),
    };
    let policy = work_ledger::MovementPolicy {
        auto_propagate: config.sharing,
    };
    // The async claim/push path must run on a Tokio runtime. run_worker_once is a synchronous
    // surface (its callers — the CLI worker, the tests — do not thread a runtime through it), and it
    // may be called from inside another runtime (the test hub runs on the caller's runtime). So the
    // poll runs on a dedicated thread with its own runtime; the transport handlers the client dials
    // are driven inline there, independent of the caller's runtime.
    let outcome = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let runtime = contextdb_server::tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|err| {
                        crate::CgError::Engine(format!("fabric worker runtime: {err}"))
                    })?;
                runtime
                    .block_on(cg_poll_once(client, &executor, config, policy, now_ms))
                    .map_err(map_engine)
            })
            .join()
            .map_err(|_| crate::CgError::Engine("fabric worker poll thread panicked".to_string()))?
    })?;
    Ok(outcome)
}

/// The self-sufficient worker poll a real remote node runs: pull the tenant's ledger from the hub,
/// run the inner [`run_worker_once`] claim/execute/record poll, then push the recorded result/failure
/// back to the hub. This is what makes a genuine remote worker functional.
///
/// Without the leading pull the worker scans only its own local db and never sees a job another node
/// submitted — the two-root rehearsal defect where a worker polled for minutes while a remote job
/// stayed `pending`. Without the trailing push a worker that executed its last job then idled would
/// strand that result locally: the outcome would otherwise only reach the hub piggybacked on the NEXT
/// claim's push, which never happens once there is no more work to claim.
///
/// Both hub round-trips are best-effort — an unreachable hub degrades to the offline single-node path
/// (the local db is the arbiter; a local claim/result rides the next successful push), keeping the
/// ledger's `Won { synced: false }` offline convention. The pull uses the same durable table
/// declarations as `cg sync pull`.
#[cfg(feature = "fabric-sync")]
pub fn run_worker_poll<E: ConversationExtractor + Sync>(
    client: &SyncClient,
    store: &Store,
    extractor: &E,
    config: &FabricWorkerConfig,
    now_ms: i64,
) -> Result<FabricPollOutcome> {
    // Pull the tenant's ledger BEFORE the poll so peers' jobs become visible in the local db the
    // inner poll scans. Best-effort: an unreachable hub just leaves the local db as-is (offline).
    hub_best_effort(client, HubDirection::Pull)?;
    let outcome = run_worker_once(client, store, extractor, config, now_ms)?;
    // Push AFTER the poll so a recorded result/failure reaches the hub — and thus the digest owner's
    // next pull — immediately. Harmless when the poll wrote nothing (an empty push is a status
    // exchange), best-effort on the same offline convention.
    hub_best_effort(client, HubDirection::Push)?;
    Ok(outcome)
}

/// The direction of a best-effort hub round-trip.
#[cfg(feature = "fabric-sync")]
enum HubDirection {
    Pull,
    Push,
}

/// A best-effort hub drain run on a dedicated thread with its own runtime, mirroring
/// [`run_worker_once`]'s scaffolding: the poll surface is synchronous and may be called from inside
/// another runtime (the test hub runs on the caller's runtime), so the transport is driven on a fresh
/// thread. A hub round-trip that errors is the offline single-node path, never a worker failure, so
/// pull/push errors are deliberately swallowed (the loop simply stops); only a runtime-build or
/// thread panic surfaces.
///
/// Pull uses the durable table declarations exactly as `cg sync pull` does, and both directions
/// drain in a bounded loop until they make no further progress (the same convergence the CLI's
/// `drain_sync_pull`/`drain_sync_push` provide).
#[cfg(feature = "fabric-sync")]
fn hub_best_effort(client: &SyncClient, direction: HubDirection) -> Result<()> {
    const MAX_ROUNDS: usize = 32;
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let runtime = contextdb_server::tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|err| {
                        crate::CgError::Engine(format!("fabric worker runtime: {err}"))
                    })?;
                runtime.block_on(async {
                    match direction {
                        HubDirection::Pull => {
                            for _ in 0..MAX_ROUNDS {
                                let before = client.pull_watermark();
                                if client.pull_default().await.is_err() {
                                    break;
                                }
                                if client.pull_watermark() == before {
                                    break;
                                }
                            }
                        }
                        HubDirection::Push => {
                            for _ in 0..MAX_ROUNDS {
                                match client.has_pending_push_changes() {
                                    Ok(true) => {}
                                    _ => break,
                                }
                                if client.push().await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                });
                Ok(())
            })
            .join()
            .map_err(|_| crate::CgError::Engine("fabric worker hub thread panicked".to_string()))?
    })
}

/// cg's executor over the existing `ConversationExtractor` path. It materializes its job's input
/// bytes, runs the model, and returns the structured RESULT plus a receipt carrying THIS node's id
/// (so the owner can tell which machine did the work) — it never owns the digest or writes the graph.
#[cfg(feature = "fabric-sync")]
struct CgWorkExecutor<'a, E: ConversationExtractor + Sync> {
    extractor: &'a E,
    node_id: String,
}

#[cfg(feature = "fabric-sync")]
impl<E: ConversationExtractor + Sync> contextdb_server::work_ledger::WorkExecutor
    for CgWorkExecutor<'_, E>
{
    fn execute(
        &self,
        _job: &work_ledger::JobSnapshot,
        inputs: &work_ledger::ExecutionInputs,
        _should_abandon: &dyn Fn() -> bool,
    ) -> contextdb_server::work_ledger::ExecutionVerdict {
        use contextdb_server::work_ledger::{ExecutionOutput, ExecutionVerdict};
        // The understanding input is the copied chunk text, in seq order.
        let transcript = inputs
            .iter()
            .map(|(_seq, bytes)| String::from_utf8_lossy(bytes).into_owned())
            .collect::<Vec<_>>()
            .join("\n");
        // The executor produces the structured RESULT only; it never anchors onto a context, so the
        // label is immaterial to the extraction (the owner's apply binds the context). The ledger
        // JobSnapshot carries no context label.
        let harvest = match self.extractor.extract(crate::ingest::ExtractionRequest {
            context_label: "fabric",
            transcript: &transcript,
            prior_decisions: &[],
        }) {
            Ok(harvest) => harvest,
            Err(err) => return ExecutionVerdict::Failed(err.to_string()),
        };
        // The worker's REAL spend for this job: the model's own reported token usage and the measured
        // per-call wall-clock, read from the SAME `drain_model_usage` source the worker's stderr line
        // reports — never the input byte length, never a zero clock. The owner reads this receipt
        // verbatim off `work_results`; it is never reconstructed on the submitter.
        let usage = self.extractor.drain_model_usage();
        let receipt = FabricExecutionReceipt {
            executor_node_id: self.node_id.clone(),
            total_tokens: total_usage_tokens(&usage),
            wall_clock_ms: total_usage_elapsed_ms(&usage),
        };
        // The output is the worker's full STRUCTURED HARVEST — the owner applies THIS payload onto the
        // anchor digest without re-running a model. Serialized verbatim; the owner deserializes it.
        let output = serde_json::to_vec(&harvest).unwrap_or_default();
        ExecutionVerdict::Completed(ExecutionOutput {
            output,
            receipt: receipt.to_properties_value(),
        })
    }
}

/// One cg worker poll: claim a matching job by push (arbitrated by the hub), execute, and record the
/// result. Consumes the shipped `claim_job`/`materialize_inputs`/`record_result`/`should_abandon`
/// unchanged; authors only the ordering (no pre-claim pull; reclaim a job whose competitor's lease
/// expired at a higher attempt).
#[cfg(feature = "fabric-sync")]
async fn cg_poll_once<E: ConversationExtractor + Sync>(
    client: &SyncClient,
    executor: &CgWorkExecutor<'_, E>,
    config: &FabricWorkerConfig,
    policy: work_ledger::MovementPolicy,
    now_ms: i64,
) -> std::result::Result<FabricPollOutcome, contextdb_core::Error> {
    use contextdb_server::work_ledger::{
        self as server_ledger, ClaimOutcome, ExecutionVerdict, WorkExecutor,
    };

    let db = client.db();
    let jobs = work_ledger::claimable_jobs(
        db,
        &config.node_id,
        &config.advertised_tags,
        &policy,
        now_ms,
    )?;
    if jobs.is_empty() {
        return Ok(FabricPollOutcome::NoWork);
    }

    let mut lost_holder: Option<String> = None;
    let mut lost_any = false;
    for job in jobs {
        // Reclaim guard: a lost claim whose competitor's lease has since expired is retried at a
        // higher attempt; a live competitor stops the retry. Bounded by the job's attempt ceiling.
        let mut guard = 0i64;
        loop {
            guard += 1;
            if guard > job.max_attempts.max(1) + 2 {
                lost_any = true;
                break;
            }
            let attempt = work_ledger::next_attempt(db, &job.job_id)?;
            let lease_deadline_ms = now_ms + config.lease_ms;
            match server_ledger::claim_job(
                client,
                &job.job_id,
                attempt,
                &config.node_id,
                lease_deadline_ms,
                now_ms,
            )
            .await?
            {
                ClaimOutcome::Won { .. } => {
                    // A claim is a licence to TRY: honor the shipped abandonment predicate.
                    if work_ledger::should_abandon(
                        db,
                        &job.job_id,
                        attempt,
                        &config.node_id,
                        now_ms,
                    )? {
                        return Ok(FabricPollOutcome::NoWork);
                    }
                    let inputs = match work_ledger::materialize_inputs(
                        db,
                        &job.job_id,
                        &config.node_id,
                        &policy,
                    ) {
                        Ok(inputs) => inputs,
                        Err(err) => {
                            work_ledger::record_failure(
                                db,
                                &job.job_id,
                                attempt,
                                &config.node_id,
                                &format!("input materialization failed: {err}"),
                                now_ms,
                            )?;
                            return Ok(FabricPollOutcome::NoWork);
                        }
                    };
                    let never_abandon = || false;
                    return match executor.execute(&job, &inputs, &never_abandon) {
                        ExecutionVerdict::Completed(output) => {
                            work_ledger::record_result(
                                db,
                                &job.job_id,
                                attempt,
                                &config.node_id,
                                &output.output,
                                output.receipt,
                                now_ms,
                            )?;
                            Ok(FabricPollOutcome::Executed {
                                job_id: job.job_id.clone(),
                            })
                        }
                        ExecutionVerdict::Failed(err) => {
                            work_ledger::record_failure(
                                db,
                                &job.job_id,
                                attempt,
                                &config.node_id,
                                &err,
                                now_ms,
                            )?;
                            Ok(FabricPollOutcome::NoWork)
                        }
                        ExecutionVerdict::Abandoned => Ok(FabricPollOutcome::NoWork),
                    };
                }
                ClaimOutcome::Lost { holder } => {
                    // claim_job pulled + reconciled to the hub's winner. If the job is still pending
                    // (the competitor's lease has expired), retry at a fresh attempt; otherwise the
                    // winner holds it — report the lost claim without executing.
                    match work_ledger::job_state(db, &job.job_id, now_ms)? {
                        work_ledger::JobState::Pending => continue,
                        _ => {
                            lost_holder = holder;
                            lost_any = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    if lost_any {
        Ok(FabricPollOutcome::LostClaim {
            holder: lost_holder,
        })
    } else {
        Ok(FabricPollOutcome::NoWork)
    }
}

/// Sweep this node's own ephemeral `work_inputs` copies for terminal jobs past `grace_ms`, driven by
/// an injected `now_ms`. This is the ledger's ONLY delete and is everywhere-local — the executor edge
/// sweeps its own copy on the same result-lands + grace boundary, not just the submitter. The state
/// computation never reads `work_inputs`, so deleting inputs provably cannot change a job's computed
/// state (it stays `done`, executor and receipt intact). Returns the number of input rows deleted.
pub fn run_input_retention(store: &Store, now_ms: i64, grace_ms: i64) -> Result<usize> {
    work_ledger::run_input_retention(store.internal_db(), now_ms, grace_ms).map_err(map_engine)
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

    /// Parse a receipt back from the operational-lane value a worker recorded — the inverse of
    /// [`FabricExecutionReceipt::to_properties_value`]. `None` when the value is missing a field.
    pub fn from_properties_value(value: &Value) -> Option<Self> {
        Some(Self {
            executor_node_id: value.get("executor_node_id")?.as_str()?.to_string(),
            total_tokens: value.get("total_tokens")?.as_u64()?,
            wall_clock_ms: value.get("wall_clock_ms")?.as_u64()?,
        })
    }
}

/// The recorded execution receipt for a job, or `None` when no result has landed yet. Read straight
/// from the `work_results` row — the executor's own numbers, never reconstructed. The `cg worker` loop
/// reads it for its stderr/budget so the log line and the stored receipt report the SAME real tokens
/// and wall-clock (one source of truth).
pub fn recorded_receipt(store: &Store, job_id: &str) -> Result<Option<FabricExecutionReceipt>> {
    let Some(result) = work_ledger::job_result(store.internal_db(), job_id).map_err(map_engine)?
    else {
        return Ok(None);
    };
    Ok(FabricExecutionReceipt::from_properties_value(
        &result.receipt,
    ))
}

/// Sum the model-reported total tokens across a run's usage records — the real provider token count,
/// zero when a run reported no usage. Never the input byte length.
fn total_usage_tokens(usage: &[ModelCallUsage]) -> u64 {
    usage
        .iter()
        .map(|call| match &call.usage {
            UsageReportState::Tokens(provider) => provider.total_tokens,
            _ => 0,
        })
        .sum()
}

/// Sum the measured per-call wall-clock across a run's usage records, in milliseconds — the real time
/// the model calls took, the same figure the worker's stderr line reports.
fn total_usage_elapsed_ms(usage: &[ModelCallUsage]) -> u64 {
    usage
        .iter()
        .map(|call| call.elapsed.as_millis() as u64)
        .sum()
}
