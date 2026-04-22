use std::fmt;

use crate::{DecisionStatus, IntentionId, IntentionStatus, TxId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CgError {
    NotImplemented {
        symbol: &'static str,
    },
    InvalidIdentity {
        context: &'static str,
    },
    UniqueViolation {
        entity: &'static str,
        id: Option<String>,
    },
    MissingIntention,
    IntentionNotFound {
        id: IntentionId,
    },
    NotFound {
        id: Option<String>,
        missing: Option<&'static str>,
    },
    ImmutableField {
        field: String,
    },
    ImmutableEntity {
        kind: &'static str,
    },
    IllegalTransition {
        from: DecisionStatus,
        to: DecisionStatus,
    },
    IntentionIllegalTransition {
        from: IntentionStatus,
        to: IntentionStatus,
    },
    ActiveDecisionsReference {
        intention_id: IntentionId,
        decision_ids: Vec<String>,
    },
    CrossContextRelationship,
    CrossContextObservation,
    CrossContextViolation,
    ActiveIntentionProtection,
    CyclicCitation,
    InvalidDepth {
        requested: u32,
        cap: u8,
    },
    VectorDimensionMismatch {
        expected: usize,
        got: usize,
    },
    EngineStateMachineError(String),
    ForeignKeyViolation {
        table: &'static str,
        column: &'static str,
        ref_table: &'static str,
    },
    Rollback {
        txid: TxId,
    },
    Engine(String),
}

impl fmt::Display for CgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CgError {}
