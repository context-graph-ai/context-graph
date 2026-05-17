use serde::{Deserialize, Serialize};

use crate::{EntityId, InvalidationId, Result, Store};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MismatchKind {
    FaceReferenceNoMatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverrideFailureReason {
    EntityAwareNoMatch {
        reference_space: String,
        live_space: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageBody {
    EntityMismatch {
        entity_id: EntityId,
        entity_name: String,
        mismatch_kind: MismatchKind,
        override_failure_reason: OverrideFailureReason,
    },
    ClassOnlyDetection {
        class_name: String,
    },
}

impl Store {
    pub fn render_school_gate_message_body(
        &self,
        _invalidation_id: InvalidationId,
    ) -> Result<MessageBody> {
        Ok(MessageBody::ClassOnlyDetection {
            class_name: "person".to_string(),
        })
    }
}
