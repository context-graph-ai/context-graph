use std::fmt;

use context_graph::TxId;

use crate::SwapOption;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactValidationMismatch {
    pub file: String,
    pub expected_sha: String,
    pub actual_sha: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CgCatalogError {
    NotFound,
    ManualAcceptanceRequired {
        model_id: String,
        canonical_message: String,
    },
    ManualAcceptanceRequiredAtReceiver {
        model_id: String,
        canonical_message: String,
    },
    DimensionChangeRequiresConfirmation {
        current_dim: u32,
        proposed_dim: u32,
        options: Vec<SwapOption>,
    },
    ArtifactValidationFailed {
        file: String,
        expected_sha: String,
        actual_sha: String,
        mismatches: Vec<ArtifactValidationMismatch>,
    },
    ArtifactShaMismatch {
        expected: String,
        actual: String,
    },
    NonMonotonicAcceptanceTimestamp {
        previous: TxId,
        proposed: TxId,
    },
    CacheIncomplete {
        missing_files: Vec<String>,
    },
    SourceUnavailable {
        model_id: String,
        message: String,
    },
    UnknownModelId {
        id: String,
    },
}

impl fmt::Display for CgCatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "model entry was not found"),
            Self::ManualAcceptanceRequired { model_id, .. } => {
                write!(f, "manual license acceptance required for {model_id}")
            }
            Self::ManualAcceptanceRequiredAtReceiver { model_id, .. } => {
                write!(
                    f,
                    "manual license acceptance required before receiving {model_id}"
                )
            }
            Self::DimensionChangeRequiresConfirmation {
                current_dim,
                proposed_dim,
                ..
            } => write!(
                f,
                "dimension change from {current_dim} to {proposed_dim} requires confirmation"
            ),
            Self::ArtifactValidationFailed { file, .. } => {
                write!(f, "artifact validation failed for {file}")
            }
            Self::ArtifactShaMismatch { .. } => write!(f, "artifact SHA did not match"),
            Self::NonMonotonicAcceptanceTimestamp { .. } => {
                write!(f, "acceptance timestamps must be strictly increasing")
            }
            Self::CacheIncomplete { missing_files } => {
                write!(f, "cache is incomplete: {}", missing_files.join(", "))
            }
            Self::SourceUnavailable { model_id, message } => {
                write!(f, "source unavailable for {model_id}: {message}")
            }
            Self::UnknownModelId { id } => write!(f, "unknown model id {id}"),
        }
    }
}

impl std::error::Error for CgCatalogError {}
