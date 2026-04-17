pub mod error;
pub mod schema;
pub mod store;
pub mod types;

pub use contextdb_core::TxId;
pub use error::CgError;
pub use store::Store;
pub use types::*;

pub type Result<T> = std::result::Result<T, CgError>;
pub type AgentId = String;
