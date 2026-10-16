//! Public released-interpretation contracts: candidate admission, value-free coverage-gap and
//! status reads, bounded generation builds, and atomic interpretation/search head activation.

use std::collections::BTreeSet;
use std::fmt;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    AdapterCoverageGapId, CgError, ContextId, ConversationEvent, EventCoverage, EventId,
    EventRenderRole, EventSearchGenerationId, InterpretationGenerationId, Result, Store,
    StoreConfig, TxId,
};

pub type ConnectorId = String;

/// One explicit, canonical set of Contexts authorized for an operation.
///
/// Construction sorts and deduplicates IDs and refuses an empty input. There is no implicit
/// all-Context value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ContextSelector {
    context_ids: Vec<ContextId>,
}

impl<'de> Deserialize<'de> for ContextSelector {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let context_ids = Vec::<ContextId>::deserialize(deserializer)?;
        Self::new(context_ids).map_err(serde::de::Error::custom)
    }
}

impl ContextSelector {
    pub fn new(context_ids: impl IntoIterator<Item = ContextId>) -> Result<Self> {
        let context_ids = context_ids.into_iter().collect::<BTreeSet<_>>();
        if context_ids.is_empty() {
            return Err(CgError::EmptyContextSelector);
        }
        Ok(Self {
            context_ids: context_ids.into_iter().collect(),
        })
    }

    pub fn one(context_id: ContextId) -> Self {
        Self {
            context_ids: vec![context_id],
        }
    }

    pub fn context_ids(&self) -> &[ContextId] {
        &self.context_ids
    }

    pub fn into_context_ids(self) -> Vec<ContextId> {
        self.context_ids
    }
}

impl TryFrom<Vec<ContextId>> for ContextSelector {
    type Error = CgError;

    fn try_from(value: Vec<ContextId>) -> Result<Self> {
        Self::new(value)
    }
}

/// One captured adapter/anchor declaration accepted by an interpretation release.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventDeclarationVersion {
    pub adapter_version: String,
    pub anchor_declaration_version: String,
}

/// The immutable release coordinates supplied independently by both installed objects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleasedInterpretationDeclaration {
    pub connector_id: ConnectorId,
    pub target_interpretation_version: String,
    pub target_adapter_version: String,
    pub target_anchor_declaration_version: String,
    pub accepted_source_declarations: Vec<EventDeclarationVersion>,
    pub mapping_contract_id: String,
    pub mapping_contract_sha256: String,
}

/// One complete captured anchor row supplied to the independent preservation validator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapturedEventAnchor {
    pub context_id: ContextId,
    pub event_id: EventId,
    pub declaration_version: String,
    pub connector_id: ConnectorId,
    pub source_instance_ref: String,
    pub session_ref: String,
    pub render_role: EventRenderRole,
    pub interaction_ref: Option<String>,
    pub key: String,
    pub canonical_value: String,
    pub identity_value: String,
    pub original_value: Value,
}

/// Immutable captured event plus its complete normalized anchor rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapturedEventAndAnchors {
    pub event: ConversationEvent,
    pub anchors: Vec<CapturedEventAnchor>,
}

/// The only nine fields a released interpretation may replace.
///
/// Captured identity, sequence, hashes, captured fields, and `source_remainder` cannot be carried
/// by this type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveEventFields {
    pub occurred_at: Option<DateTime<Utc>>,
    pub author: String,
    pub kind: String,
    pub render_role: EventRenderRole,
    pub interaction_ref: Option<String>,
    pub text: Option<String>,
    pub payload: Value,
    pub anchors: Value,
    pub coverage: EventCoverage,
}

pub trait ReleasedConnectorInterpretation: Send + Sync {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration;

    fn interpret_event(&self, captured: &ConversationEvent) -> Result<EffectiveEventFields>;
}

pub trait InstalledInterpretationContract: Send + Sync {
    fn declaration(&self) -> &ReleasedInterpretationDeclaration;

    fn validate(
        &self,
        captured: &CapturedEventAndAnchors,
        candidate: &EffectiveEventFields,
    ) -> Result<()>;
}

/// A candidate admitted by core validation and the independently installed contract.
///
/// It has no public fields or constructor. [`admit_interpretation`] is the sole construction path.
#[allow(dead_code)]
pub struct AdmittedEffectiveEvent {
    pub(crate) fields: EffectiveEventFields,
    pub(crate) effective_sha256: String,
}

pub fn admit_interpretation(
    contract: &dyn InstalledInterpretationContract,
    captured: &CapturedEventAndAnchors,
    candidate: EffectiveEventFields,
) -> Result<AdmittedEffectiveEvent> {
    let (fields, effective_sha256) =
        crate::store::event_interpretation::admit_candidate(contract, captured, candidate)?;
    Ok(AdmittedEffectiveEvent {
        fields,
        effective_sha256,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterCoverageGapState {
    Uncovered,
    Covered,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterCoverageGap {
    pub id: AdapterCoverageGapId,
    pub context_id: ContextId,
    pub connector_id: ConnectorId,
    pub captured_adapter_version: String,
    pub fingerprint_sha256: String,
    pub affected_event_ids: Vec<EventId>,
    pub occurrence_count: u64,
    pub state: AdapterCoverageGapState,
    pub first_seen_at: TxId,
    pub last_seen_at: TxId,
}

impl fmt::Debug for AdapterCoverageGap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdapterCoverageGap")
            .field("id", &self.id)
            .field("context_id", &self.context_id)
            .field("connector_id", &self.connector_id)
            .field("captured_adapter_version", &self.captured_adapter_version)
            .field("fingerprint_sha256", &self.fingerprint_sha256)
            .field("affected_event_ids", &self.affected_event_ids)
            .field("occurrence_count", &self.occurrence_count)
            .field("state", &self.state)
            .field(
                "seen_transactions",
                &(self.first_seen_at, self.last_seen_at),
            )
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Page {
    pub after: Option<AdapterCoverageGapId>,
    pub limit: NonZeroU32,
    pub affected_event_handle_limit: NonZeroU32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterCoverageGapPage {
    pub gaps: Vec<AdapterCoverageGap>,
    pub next_after: Option<AdapterCoverageGapId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventWatermark {
    pub event_count: u64,
    pub ordered_source_set_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterpretationGenerationState {
    Capturing,
    Building,
    Ready,
    Failed,
    Retired,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterpretationContextStatus {
    pub context_id: ContextId,
    pub connector_id: ConnectorId,
    pub active_generation_id: Option<InterpretationGenerationId>,
    pub active_target_interpretation_version: Option<String>,
    pub source_watermark: Option<EventWatermark>,
    pub interpreted_event_count: u64,
    pub uncovered_gap_count: u64,
    pub state: Option<InterpretationGenerationState>,
    pub sanitized_failure: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterpretationStatusSet {
    pub selected_context_ids: Vec<ContextId>,
    pub contexts: Vec<InterpretationContextStatus>,
}

#[derive(Clone)]
pub struct InterpretationBuildRequest {
    pub store: StoreConfig,
    pub context_id: ContextId,
    pub connector_id: ConnectorId,
    pub target_interpretation_version: String,
    pub mapper: Arc<dyn ReleasedConnectorInterpretation>,
    pub contract: Arc<dyn InstalledInterpretationContract>,
    pub max_events_per_batch: NonZeroU32,
    pub max_bytes_per_batch: NonZeroUsize,
    pub lease_duration_ms: NonZeroU64,
}

impl fmt::Debug for InterpretationBuildRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InterpretationBuildRequest")
            .field("store", &self.store)
            .field("context_id", &self.context_id)
            .field("connector_id", &self.connector_id)
            .field(
                "target_interpretation_version",
                &self.target_interpretation_version,
            )
            .field("mapper_declaration", self.mapper.declaration())
            .field("contract_declaration", self.contract.declaration())
            .field("max_events_per_batch", &self.max_events_per_batch)
            .field("max_bytes_per_batch", &self.max_bytes_per_batch)
            .field("lease_duration_ms", &self.lease_duration_ms)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterpretationBuildReceipt {
    pub context_id: ContextId,
    pub connector_id: ConnectorId,
    pub generation_id: InterpretationGenerationId,
    pub accepted_source_declarations: Vec<EventDeclarationVersion>,
    pub target_interpretation_version: String,
    pub target_adapter_version: String,
    pub target_anchor_declaration_version: String,
    pub mapping_contract_id: String,
    pub mapping_contract_sha256: String,
    pub source_watermark: EventWatermark,
    pub source_content_sha256: String,
    pub effective_content_sha256: String,
    pub eligible_event_count: u64,
    pub interpreted_event_count: u64,
    pub failed_event_count: u64,
    pub state: InterpretationGenerationState,
    pub completed_at: Option<TxId>,
    pub sanitized_failure: Option<String>,
}

#[derive(Clone)]
pub struct InterpretationActivationRequest {
    pub store: StoreConfig,
    pub context_id: ContextId,
    pub connector_id: ConnectorId,
    pub target_interpretation_version: String,
    pub mapper: Arc<dyn ReleasedConnectorInterpretation>,
    pub contract: Arc<dyn InstalledInterpretationContract>,
    pub interpretation_generation_id: InterpretationGenerationId,
    pub replacement_search_generation_id: EventSearchGenerationId,
}

impl fmt::Debug for InterpretationActivationRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InterpretationActivationRequest")
            .field("store", &self.store)
            .field("context_id", &self.context_id)
            .field("connector_id", &self.connector_id)
            .field(
                "target_interpretation_version",
                &self.target_interpretation_version,
            )
            .field("mapper_declaration", self.mapper.declaration())
            .field("contract_declaration", self.contract.declaration())
            .field(
                "interpretation_generation_id",
                &self.interpretation_generation_id,
            )
            .field(
                "replacement_search_generation_id",
                &self.replacement_search_generation_id,
            )
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterpretationActivationReceipt {
    pub context_id: ContextId,
    pub connector_id: ConnectorId,
    pub target_interpretation_version: String,
    pub source_watermark: EventWatermark,
    pub previous_interpretation_generation_id: Option<InterpretationGenerationId>,
    pub interpretation_generation_id: InterpretationGenerationId,
    pub previous_search_generation_id: Option<EventSearchGenerationId>,
    pub search_generation_id: EventSearchGenerationId,
    pub source_content_sha256: String,
    pub effective_content_sha256: String,
    pub mapping_contract_id: String,
    pub mapping_contract_sha256: String,
    pub interpreted_event_count: u64,
    pub passage_count: u64,
    pub transaction_id: TxId,
    pub changed: bool,
    pub sanitized_failure: Option<String>,
}

#[derive(Debug, Default)]
pub struct ConnectorInterpretationRunner;

impl ConnectorInterpretationRunner {
    pub fn build(request: InterpretationBuildRequest) -> Result<InterpretationBuildReceipt> {
        crate::store::event_interpretation::validate_installed_release(
            request.context_id,
            &request.connector_id,
            &request.target_interpretation_version,
            request.mapper.as_ref(),
            request.contract.as_ref(),
        )?;
        crate::store::interpretation_build::build(request)
    }

    pub fn activate(
        request: InterpretationActivationRequest,
    ) -> Result<InterpretationActivationReceipt> {
        crate::store::event_interpretation::validate_installed_release(
            request.context_id,
            &request.connector_id,
            &request.target_interpretation_version,
            request.mapper.as_ref(),
            request.contract.as_ref(),
        )?;
        crate::store::event_interpretation::activate(&request)
    }
}

impl Store {
    pub fn list_adapter_coverage_gaps(
        &self,
        selector: ContextSelector,
        connector_id: ConnectorId,
        page: Page,
    ) -> Result<AdapterCoverageGapPage> {
        crate::store::event_interpretation::list_coverage_gaps(self, selector, connector_id, page)
    }

    pub fn interpretation_status(
        &self,
        selector: ContextSelector,
        connector_id: ConnectorId,
    ) -> Result<InterpretationStatusSet> {
        crate::store::event_interpretation::interpretation_status(self, selector, connector_id)
    }
}
