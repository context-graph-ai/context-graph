use serde::{Deserialize, Serialize};

use crate::{CgError, EntityId, InvalidationId, Result, Store};

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
    pub fn render_entity_alert_message_body(
        &self,
        invalidation_id: InvalidationId,
    ) -> Result<MessageBody> {
        let invalidation =
            self.invalidation_by_id(invalidation_id)?
                .ok_or_else(|| CgError::NotFound {
                    id: Some(invalidation_id.as_uuid().to_string()),
                    missing: Some("invalidation"),
                })?;
        let mut entity_id = None;
        let mut entity_name = None;
        let mut override_failure_reason = None;
        for diff in invalidation
            .basis_diff
            .divergences
            .iter()
            .flat_map(|divergence| &divergence.value_diffs)
        {
            match diff.field.as_str() {
                "entity_id" => {
                    entity_id = diff.current_value.as_ref().and_then(entity_id_from_json);
                }
                "entity_name" => {
                    entity_name = diff
                        .current_value
                        .as_ref()
                        .and_then(|value| value.as_str().map(str::to_string));
                }
                "override_failure_reason" => {
                    override_failure_reason = diff.current_value.as_ref().and_then(|value| {
                        serde_json::from_value::<OverrideFailureReason>(value.clone()).ok()
                    });
                }
                _ => {}
            }
        }
        match (entity_id, entity_name, override_failure_reason) {
            (Some(entity_id), Some(entity_name), Some(override_failure_reason)) => {
                Ok(MessageBody::EntityMismatch {
                    entity_id,
                    entity_name,
                    mismatch_kind: MismatchKind::FaceReferenceNoMatch,
                    override_failure_reason,
                })
            }
            _ => Ok(MessageBody::ClassOnlyDetection {
                class_name: "person".to_string(),
            }),
        }
    }
}

fn entity_id_from_json(value: &serde_json::Value) -> Option<EntityId> {
    serde_json::from_value::<EntityId>(value.clone())
        .ok()
        .or_else(|| {
            uuid::Uuid::parse_str(value.as_str()?)
                .ok()
                .map(EntityId::from)
        })
}
