use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaudeConnectorError {
    MalformedRecord {
        reason: String,
    },
    Io {
        operation: &'static str,
        path: PathBuf,
        reason: String,
    },
}

impl fmt::Display for ClaudeConnectorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedRecord { reason } => {
                write!(formatter, "Claude source record is malformed: {reason}")
            }
            Self::Io {
                operation,
                path,
                reason,
            } => write!(
                formatter,
                "Claude connector could not {operation} at {}: {reason}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for ClaudeConnectorError {}
