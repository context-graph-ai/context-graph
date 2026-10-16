use std::env;
use std::path::PathBuf;

pub(super) const CONNECTION_SCOPE: &str = "connection-substrate";

/// Where cg keeps this machine's data.
///
/// One of the two LOCATIONS a `cg` install is configured with. `CG_DATA_ROOT` says where the data
/// lives; contextdb's own `CONTEXTDB_OWNER_READ_RUNTIME_DIR` says which directory the owner channel
/// lives in (see `context_graph::owner_control::owner_read_config`) — the one variable a container or a
/// packaged service provisions so both sides of the owner plane meet in the same place. Those two
/// are this binary's supported configuration surface: neither decides whether anything runs or
/// what a read may spend, and `HOME` is read only to find `CG_DATA_ROOT`'s default and, with
/// `CLAUDE_CONFIG_DIR` (Claude Code's own configuration directory), where Claude Code keeps the local
/// conversation history `cg import` reads.
///
/// This binary reads six further variables, and they are development affordances rather than
/// configuration, named here rather than left for a reader to find: `CG_NO_UNDERSTAND_KICK`
/// suppresses the background understanding worker (:870), `CG_USE_TEST_LEXICAL_EMBEDDER` swaps in
/// the lexical mock embedder (:4249, :4281), and `CG_FAIL_STATUS_METRIC_READ` refuses one named
/// `cg status` read. `CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER` pauses the event-search worker after
/// catch-up and before it releases the handoff, while
/// `CG_EVENT_SEARCH_TEST_PRE_CATCH_UP_BARRIER` pauses it before catch-up.
/// `CG_FORGET_TEST_BEFORE_PURGE_SOCKET` pauses forget after its writable open and before erasure.
/// These barriers and the status fault injection are compiled out of ordinary builds. The libraries
/// below this binary take environment inputs of their own — the model cache root and the read-probe
/// pair among them — so this list speaks for `main.rs` and claims nothing beyond it.
pub(super) fn data_root() -> PathBuf {
    env::var_os("CG_DATA_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".cg"))
}

pub(super) fn home_dir() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}
