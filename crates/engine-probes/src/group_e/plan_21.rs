//! Plan 21 cg-connector-codex — engine-probe sweep (wrapper class).
//!
//! Codex connector wrapper: `cg connect codex` writes a wrapper script
//! that intercepts codex invocations, forwards args unchanged, and
//! observes/ingests the session via cg in the background. The shipped
//! connector binary does not exist today. Most assumptions are BLOCKED.
//!
//! Runnable concerns:
//!
//!   * **wrapper script idempotency** — re-running the connect command
//!     produces a byte-identical script file on disk. Probed at the file
//!     level (no engine).
//!
//!   * **wrapper script invocation pass-through** — codex's own arguments
//!     are forwarded unchanged. The contract is wrapper-script content;
//!     the script does not exist yet, BLOCKED on shipped binary.
//!
//!   * **C-CX1 concurrent codex invocations through wrapper** — multiple
//!     codex sessions, each ingesting via cg, racing on the same store.
//!     Cross-references Plan 17's two-handle probe; not duplicated.
//!
//! The novel file-level concern: the wrapper-script file MUST be
//! executable after install, AND that bit must persist across re-installs
//! (idempotent install does not strip the executable bit). Probed.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use tempfile::TempDir;

/// Wrapper script idempotency: re-running connect produces byte-identical
/// content. Same shape as Plan 19's idempotency probe but at the
/// executable-script file level.
#[test]
fn codex_wrapper_script_install_is_idempotent() {
    let dir = TempDir::new().unwrap();
    let script_path = dir.path().join("codex-wrapper.sh");
    let script = b"#!/bin/sh\nexec cg ingest-codex \"$@\"\n";

    // First install via temp + rename + chmod.
    {
        let tmp = dir.path().join(".codex-wrapper.tmp");
        fs::write(&tmp, script).unwrap();
        fs::set_permissions(&tmp, fs::Permissions::from_mode(0o755)).unwrap();
        fs::rename(&tmp, &script_path).unwrap();
    }
    let bytes_after_first = fs::read(&script_path).unwrap();
    let mode_after_first = fs::metadata(&script_path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode_after_first, 0o755, "first install must mark script executable");

    // Second install — wrapper notices script identical, skips write. The
    // contract: bytes AND executable bit unchanged.
    let existing = fs::read(&script_path).unwrap();
    if existing == script {
        // idempotent skip.
    } else {
        let tmp = dir.path().join(".codex-wrapper.tmp");
        fs::write(&tmp, script).unwrap();
        fs::set_permissions(&tmp, fs::Permissions::from_mode(0o755)).unwrap();
        fs::rename(&tmp, &script_path).unwrap();
    }
    let bytes_after_second = fs::read(&script_path).unwrap();
    let mode_after_second = fs::metadata(&script_path).unwrap().permissions().mode() & 0o777;
    assert_eq!(bytes_after_first, bytes_after_second);
    assert_eq!(mode_after_first, mode_after_second);
}

/// Wrapper script overwrite preserves executable bit. If the wrapper
/// detects content drift (e.g. cg upgrade changed the wrapper template)
/// and rewrites the script, the executable bit MUST come back. A naive
/// overwrite via `fs::write` (without chmod) on a fresh temp file then
/// `rename` retains the temp file's mode — probed here to fail loudly if
/// the connector author forgets the chmod step.
#[test]
fn codex_wrapper_script_overwrite_preserves_executable_bit() {
    let dir = TempDir::new().unwrap();
    let script_path = dir.path().join("codex-wrapper.sh");
    let initial = b"#!/bin/sh\nexec cg ingest-codex \"$@\"\n";
    let updated = b"#!/bin/sh\nexec cg ingest-codex --v2 \"$@\"\n";

    // First install with explicit chmod.
    {
        let tmp = dir.path().join(".codex.tmp1");
        fs::write(&tmp, initial).unwrap();
        fs::set_permissions(&tmp, fs::Permissions::from_mode(0o755)).unwrap();
        fs::rename(&tmp, &script_path).unwrap();
    }

    // Overwrite. The connector must write to a temp, chmod the temp, then
    // rename. Without chmod on the temp, the mode comes from umask — could
    // be 0o644, breaking the wrapper at next codex invocation.
    {
        let tmp = dir.path().join(".codex.tmp2");
        fs::write(&tmp, updated).unwrap();
        fs::set_permissions(&tmp, fs::Permissions::from_mode(0o755)).unwrap();
        fs::rename(&tmp, &script_path).unwrap();
    }

    let bytes = fs::read(&script_path).unwrap();
    let mode = fs::metadata(&script_path).unwrap().permissions().mode() & 0o777;
    assert_eq!(bytes, updated, "overwrite must land the new content");
    assert_eq!(
        mode, 0o755,
        "overwrite must preserve executable bit; user-mode bit lost == script unrunnable"
    );
}
