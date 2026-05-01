//! Group E — wrapper plans (CLI, MCP server, connectors, import).
//!
//! Plans: 17 cg-cli, 18 cg-mcp-server, 19 cg-connector-claude-code,
//! 20 cg-connector-gemini, 21 cg-connector-codex, 22 cg-import-chat-history.
//!
//! Wrapper-class plans add ONLY wrapper-specific assumption probes
//! (process-exit semantics, stdout/stderr framing, line-delimited JSON
//! framing, hook-script idempotency, file-rename atomicity for hook config
//! writes, concurrent CLI subprocess opens of the same store). They do NOT
//! duplicate engine probes from underlying primitive-owner plans.
//!
//! Most wrapper assumptions are exercisable only against shipped wrapper
//! binaries (cg-cli, cg-mcp-server, connector hook scripts). Plan 17 is
//! unshipped; the binaries do not exist today. Those assumptions are
//! catalogued as BLOCKED in the per-plan result files. The runnable subset
//! that touches the released contextdb engine alone:
//!   * file-rename atomicity for hook config writes (OS-level, exercised in
//!     the connector plan files)
//!   * engine-level in-process two-handle store-open semantics, used as a
//!     proxy for "two CLI subprocesses opening the same store" — same engine
//!     surface, different process boundary
//!
//! All probe modules ship even when their runnable surface is small, so the
//! per-plan catalogue files always cite a real source path.

pub mod plan_17;
pub mod plan_18;
pub mod plan_19;
pub mod plan_20;
pub mod plan_21;
pub mod plan_22;
