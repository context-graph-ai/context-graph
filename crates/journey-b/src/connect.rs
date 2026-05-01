//! Plan 17 cg-cli + Plan 19 cg-connector-claude-code stubs.
//!
//! Per Track 2 stub policy:
//! - Plan 17 cg-cli is "wrapper plans without binaries": the spike calls
//!   the underlying Store APIs directly; the wrapper-process-boundary
//!   aspect (exit codes, JSON framing) is NOT exercised. Marked in findings.
//! - Plan 19 cg-connector-claude-code is engine-blocked / no impl: the
//!   stub writes hook config to a tempdir.
//!
//! Acceptance contracts mirrored:
//! - `cli_init_scaffolds_on_disk_config` — `init` writes `.cg/config.toml`
//!   in the cwd-equivalent and creates `.cg/data/`.
//! - `connect_claude_code_writes_hook_idempotent` — re-running `connect_claude_code`
//!   on the same `$HOME` is a byte-equal no-op on `.claude/settings.json`.
//! - `cli_standalone_no_server` — operates without a running server.

use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

pub struct CgInit {
    pub config_path: PathBuf,
    pub data_dir: PathBuf,
}

/// `cg init` mirror — creates `.cg/config.toml` + `.cg/data/` under the
/// supplied root (which the journey passes a TempDir for).
///
/// This is plain filesystem work; the cg crate itself is the authority on
/// the on-disk format. The findings entry flags this stub as a re-walk
/// target when Plan 17 ships.
pub fn cg_init(root: &Path, context_name: &str) -> CgInit {
    let cg_dir = root.join(".cg");
    let data_dir = cg_dir.join("data");
    fs::create_dir_all(&data_dir).unwrap();
    let config_path = cg_dir.join("config.toml");
    let body = format!(
        "default_context = \"{context_name}\"\n\
         data_dir = \"{}\"\n",
        data_dir.display()
    );
    fs::write(&config_path, body).unwrap();
    CgInit {
        config_path,
        data_dir,
    }
}

/// `cg connect claude-code` mirror — writes a `UserPromptSubmit` hook into
/// `$HOME/.claude/settings.json`. Idempotent: re-running with the same
/// hook command is a byte-equal no-op.
///
/// Per `connect_claude_code_writes_hook_idempotent`:
/// - hook command contains `cg ingest --agent claude-code`
/// - `timeout` and `statusMessage` fields present
/// - second invocation produces a byte-equal file
pub fn connect_claude_code(home: &Path) -> PathBuf {
    let claude_dir = home.join(".claude");
    fs::create_dir_all(&claude_dir).unwrap();
    let settings_path = claude_dir.join("settings.json");

    let mut settings: Value = if settings_path.exists() {
        let body = fs::read_to_string(&settings_path).unwrap();
        if body.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(&body).unwrap_or_else(|_| json!({}))
        }
    } else {
        json!({})
    };

    let hook_cmd = "cg ingest --agent claude-code";
    let hooks = settings
        .as_object_mut()
        .unwrap()
        .entry("hooks")
        .or_insert_with(|| json!({}));
    let prompt_submit = hooks
        .as_object_mut()
        .unwrap()
        .entry("UserPromptSubmit")
        .or_insert_with(|| json!([]));
    let arr = prompt_submit.as_array_mut().unwrap();
    let already_present = arr.iter().any(|entry| {
        entry
            .get("command")
            .and_then(|c| c.as_str())
            .map(|s| s == hook_cmd)
            .unwrap_or(false)
    });
    if !already_present {
        arr.push(json!({
            "command": hook_cmd,
            "timeout": 30,
            "statusMessage": "cg-injecting-context"
        }));
    }

    // Stable serialization — writing pretty JSON guarantees byte-equal idempotency.
    let body = serde_json::to_string_pretty(&settings).unwrap();
    fs::write(&settings_path, body).unwrap();
    settings_path
}
