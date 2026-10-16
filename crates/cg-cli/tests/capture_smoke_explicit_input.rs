//! A live capture smoke requires an explicitly selected input before starting work.

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

/// The capture live smoke refuses to start without an explicit input file.
#[cfg(unix)]
#[test]
fn the_capture_live_smoke_refuses_to_start_without_an_explicit_input_file() {
    use std::fs;
    use std::path::Path;
    use std::process::{Command, Stdio};

    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap();
    let fixture = tempfile::tempdir().expect("isolated smoke environment");
    let home = fixture.path().join("home");
    let decoy = home.join(".claude/projects/x/big.jsonl");
    fs::create_dir_all(decoy.parent().unwrap()).expect("decoy history directory");
    fs::write(
        &decoy,
        "private conversation must never become an implicit input\n",
    )
    .expect("decoy history");
    fs::set_permissions(&decoy, fs::Permissions::from_mode(0o000)).expect("unreadable decoy");
    let invoked = fixture.path().join("binary-invoked");
    let guard = fixture.path().join("guard-binary");
    fs::write(
        &guard,
        "#!/bin/sh\nprintf invoked > \"$CG_SMOKE_PROBE\"\nexit 91\n",
    )
    .expect("binary invocation guard");
    fs::set_permissions(&guard, fs::Permissions::from_mode(0o700)).expect("executable guard");
    let log = fixture.path().join("smoke.log");
    let output = Command::new("bash")
        .arg(root.join("scripts/cg-capture-live-smoke.sh"))
        .current_dir(fixture.path())
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", &home)
        .env("CG_WORKTREE", fixture.path())
        .env("CG_DATA_ROOT", fixture.path().join("data-root"))
        .env("CG_SMOKE_LOG", &log)
        .env("CG_SMOKE_SKIP_BUILD", "1")
        .env("CG", &guard)
        .env("CONTEXTDB", &guard)
        .env("CG_SMOKE_PROBE", &invoked)
        .stdin(Stdio::null())
        .output()
        .expect("run smoke with neither input nor credentials");
    // Restore permissions so the fixture can be removed on every normal assertion path.
    fs::set_permissions(&decoy, fs::Permissions::from_mode(0o600)).expect("restore decoy");
    let diagnostic = format!(
        "{}\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
        fs::read_to_string(&log).unwrap_or_default()
    );
    assert!(
        !output.status.success(),
        "missing input must refuse the smoke"
    );
    assert!(
        diagnostic.contains("CG_SMOKE_REAL_JSONL"),
        "refusal names the required explicit input, before requiring credentials: {diagnostic}"
    );
    assert!(
        !diagnostic.contains("big.jsonl"),
        "decoy history is never selected: {diagnostic}"
    );
    assert!(
        !diagnostic.to_lowercase().contains("permission denied"),
        "the unreadable decoy is never opened: {diagnostic}"
    );
    assert!(
        !invoked.exists(),
        "missing input refuses before any binary invocation"
    );
}
