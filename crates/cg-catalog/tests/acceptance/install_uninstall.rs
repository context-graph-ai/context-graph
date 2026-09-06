use cg_catalog::{
    CG_MODEL_CACHE_ROOT_ENV, DEFAULT_CACHE_ROOT_LINUX_MACOS, DEFAULT_CACHE_ROOT_WINDOWS,
};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

/// The environment variable the canonical model-cache resolver consults before `$HOME/.cache`.
/// Every harness call below removes it, so a run on a machine that sets it exercises the same
/// fallback the assertions describe rather than silently taking a different root.
const XDG_CACHE_HOME_ENV: &str = "XDG_CACHE_HOME";

const CG_HOOK_MARKER: &str = "# BEGIN cg-injected-hook-";
const CLAUDE_AGENT: &str = "claude-code";
const CURSOR_AGENT: &str = "cursor";
const GEMINI_AGENT: &str = "gemini-cli";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate lives under repo/crates/cg-catalog")
        .to_path_buf()
}

fn install_script() -> PathBuf {
    repo_root().join("scripts/install.sh")
}

fn run_install(home: &Path, prefix: &Path, extra_args: &[&str]) -> Output {
    assert!(install_script().exists(), "repo-root install script exists");
    let mut command = Command::new("bash");
    command
        .arg(install_script())
        .arg("--prefix")
        .arg(prefix)
        .args(extra_args)
        .env("HOME", home)
        .env_remove(CG_MODEL_CACHE_ROOT_ENV)
        .env_remove(XDG_CACHE_HOME_ENV)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command.output().expect("install script spawns")
}

fn run_uninstall(home: &Path, prefix: &Path, extra_args: &[&str]) -> Output {
    let uninstall = prefix.join("cg-uninstall.sh");
    assert!(uninstall.exists(), "installed uninstall script exists");
    Command::new(uninstall)
        .args(extra_args)
        .env("HOME", home)
        .env_remove(CG_MODEL_CACHE_ROOT_ENV)
        .env_remove(XDG_CACHE_HOME_ENV)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("uninstall script spawns")
}

fn run_uninstall_with_stdin(
    home: &Path,
    prefix: &Path,
    extra_args: &[&str],
    input: &[u8],
) -> Output {
    let uninstall = prefix.join("cg-uninstall.sh");
    assert!(uninstall.exists(), "installed uninstall script exists");
    let mut child = Command::new(uninstall)
        .args(extra_args)
        .env("HOME", home)
        .env_remove(CG_MODEL_CACHE_ROOT_ENV)
        .env_remove(XDG_CACHE_HOME_ENV)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("uninstall script spawns");
    child
        .stdin
        .as_mut()
        .expect("stdin is piped")
        .write_all(input)
        .expect("confirmation input writes");
    child.wait_with_output().expect("uninstall script exits")
}

fn run_cg(home: &Path, prefix: &Path, args: &[&str]) -> Output {
    let cg = prefix.join("bin/cg");
    assert!(cg.exists(), "installed cg command exists");
    Command::new(cg)
        .args(args)
        .env("HOME", home)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("cg command spawns")
}

fn assert_success(output: &Output, label: &str) {
    assert!(
        output.status.success(),
        "{label} failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn count_occurrences(haystack: &str, needle: &str) -> usize {
    haystack.match_indices(needle).count()
}

fn write_agent_config(path: &Path, agent: &str, cg_hook_count: usize) {
    fs::create_dir_all(path.parent().expect("agent config has parent"))
        .expect("agent config parent created");
    let mut hooks = vec![format!("user-custom-hook:{agent}:echo hello")];
    for index in 0..cg_hook_count {
        hooks.push(format!("{CG_HOOK_MARKER}{agent}-{index}"));
        hooks.push(format!("cg observe --agent {agent}"));
        hooks.push(format!("# END cg-injected-hook-{agent}-{index}"));
    }
    let json_hooks = hooks
        .into_iter()
        .map(|hook| format!("    {hook:?}"))
        .collect::<Vec<_>>()
        .join(",\n");
    fs::write(path, format!("{{\n  \"hooks\": [\n{json_hooks}\n  ]\n}}\n"))
        .expect("agent config written");
}

fn claude_config(home: &Path) -> PathBuf {
    home.join("agent-config/claude-code/settings.json")
}

fn cursor_config(home: &Path) -> PathBuf {
    home.join(".cursor/hooks.json")
}

fn gemini_config(home: &Path) -> PathBuf {
    home.join(".gemini/settings.json")
}

/// The REAL Claude Code settings path the `cg` binary installs injection hooks into (structured
/// JSON), replacing the old placeholder `agent-config/claude-code/settings.json`.
fn real_claude_settings(home: &Path) -> PathBuf {
    home.join(".claude/settings.json")
}

fn create_dot_cg_projects(home: &Path) -> Vec<PathBuf> {
    ["a", "b", "c"]
        .into_iter()
        .map(|name| {
            let dir = home.join("projects").join(name).join(".cg");
            fs::create_dir_all(&dir).expect(".cg dir created");
            fs::write(dir.join("sentinel.txt"), format!("sentinel-{name}\n"))
                .expect(".cg sentinel written");
            dir
        })
        .collect()
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;

    fs::metadata(path)
        .expect("metadata readable")
        .permissions()
        .mode()
        & 0o111
        != 0
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.exists()
}

#[test]
fn install_script_idempotent_re_running_does_not_duplicate_hook_injections() {
    let temp = tempfile::tempdir().expect("tempdir");
    let home = temp.path().join("home");
    let prefix = temp.path().join("prefix");
    fs::create_dir_all(&home).expect("home created");
    write_agent_config(&claude_config(&home), CLAUDE_AGENT, 0);

    let first = run_install(&home, &prefix, &[]);
    assert_success(&first, "first install");
    let second = run_install(&home, &prefix, &[]);
    assert_success(&second, "second install");

    let settings = fs::read_to_string(claude_config(&home)).expect("claude settings readable");
    assert_eq!(
        count_occurrences(&settings, CG_HOOK_MARKER),
        1,
        "install should inject exactly one cg hook marker across repeated runs"
    );
    assert_eq!(
        count_occurrences(&settings, "user-custom-hook:claude-code:echo hello"),
        1,
        "install must not duplicate or rewrite unrelated user hooks"
    );
}

#[test]
fn uninstall_script_default_removes_binary_cache_user_config_and_surgically_removes_agent_config_hooks()
 {
    let temp = tempfile::tempdir().expect("tempdir");
    let home = temp.path().join("home");
    let prefix = temp.path().join("prefix");
    let cache_root = home.join(".cache/cg/models");
    fs::create_dir_all(&home).expect("home created");
    fs::create_dir_all(&cache_root).expect("cache root created");
    write_agent_config(&claude_config(&home), CLAUDE_AGENT, 1);

    let install = run_install(&home, &prefix, &[]);
    assert_success(&install, "install");
    fs::write(home.join(".config/cg/config.toml"), "installed = true\n")
        .expect("cg user config written");
    let before = fs::read_to_string(claude_config(&home)).expect("claude settings readable");
    assert!(before.contains(CG_HOOK_MARKER));

    let uninstall = run_uninstall(&home, &prefix, &[]);
    assert_success(&uninstall, "uninstall");

    assert!(
        !prefix.join("bin/cg").exists(),
        "default uninstall should remove the installed cg binary"
    );
    assert!(
        !cache_root.exists(),
        "default uninstall should remove the visible model cache root"
    );
    assert!(
        !home.join(".config/cg").exists(),
        "default uninstall should remove cg user config"
    );
    let after = fs::read_to_string(claude_config(&home)).expect("claude settings remains");
    assert!(
        !after.contains(CG_HOOK_MARKER),
        "default uninstall should remove only cg-injected hook blocks"
    );
    assert!(
        after.contains("user-custom-hook:claude-code:echo hello"),
        "default uninstall must preserve unrelated user hook bytes"
    );
    serde_json::from_str::<serde_json::Value>(&after).expect("agent config remains valid JSON");
}

#[test]
fn uninstall_script_default_preserves_per_project_dot_cg_directories_and_prints_find_one_liner() {
    let temp = tempfile::tempdir().expect("tempdir");
    let home = temp.path().join("home");
    let prefix = temp.path().join("prefix");
    fs::create_dir_all(&home).expect("home created");
    let dot_cg_dirs = create_dot_cg_projects(&home);
    let install = run_install(&home, &prefix, &[]);
    assert_success(&install, "install");

    let uninstall = run_uninstall(&home, &prefix, &[]);
    assert_success(&uninstall, "uninstall");

    for dir in &dot_cg_dirs {
        assert!(
            dir.join("sentinel.txt").exists(),
            "default uninstall must preserve project-local .cg data at {}",
            dir.display()
        );
    }
    let stdout = String::from_utf8_lossy(&uninstall.stdout);
    assert!(
        stdout.contains(&format!("find {} -type d -name .cg", home.display()))
            || dot_cg_dirs
                .iter()
                .all(|dir| stdout.contains(&dir.display().to_string())),
        "default uninstall should print a find one-liner or list every preserved .cg directory"
    );
}

#[test]
fn uninstall_script_purge_flag_prompts_for_confirmation_and_removes_dot_cg_only_after_confirmation()
{
    let temp = tempfile::tempdir().expect("tempdir");
    let home = temp.path().join("home");
    let prefix = temp.path().join("prefix");
    fs::create_dir_all(&home).expect("home created");
    let dot_cg_dirs = create_dot_cg_projects(&home);
    let install = run_install(&home, &prefix, &[]);
    assert_success(&install, "install");

    let no_output = run_uninstall_with_stdin(&home, &prefix, &["--purge"], b"no\n");
    assert_success(&no_output, "uninstall --purge no");
    let no_stdout = String::from_utf8_lossy(&no_output.stdout);
    assert!(
        no_stdout.contains("Remove all .cg directories"),
        "--purge should visibly prompt before destructive project-memory removal"
    );
    for dir in &dot_cg_dirs {
        assert!(
            dir.join("sentinel.txt").exists(),
            "answering no must preserve {}",
            dir.display()
        );
    }

    let yes_output = run_uninstall_with_stdin(&home, &prefix, &["--purge"], b"yes\n");
    assert_success(&yes_output, "uninstall --purge yes");
    for dir in &dot_cg_dirs {
        assert!(
            !dir.exists(),
            "answering yes should remove {}",
            dir.display()
        );
    }
}

#[test]
fn disconnect_command_removes_single_agent_hook_other_connections_untouched() {
    let temp = tempfile::tempdir().expect("tempdir");
    let home = temp.path().join("home");
    let prefix = temp.path().join("prefix");
    fs::create_dir_all(&home).expect("home created");
    write_agent_config(&claude_config(&home), CLAUDE_AGENT, 1);
    write_agent_config(&cursor_config(&home), CURSOR_AGENT, 1);
    write_agent_config(&gemini_config(&home), GEMINI_AGENT, 1);
    let install = run_install(&home, &prefix, &[]);
    assert_success(&install, "install");

    let cursor_before = fs::read(cursor_config(&home)).expect("cursor config readable");
    let gemini_before = fs::read(gemini_config(&home)).expect("gemini config readable");
    let disconnect = run_cg(&home, &prefix, &["disconnect", CLAUDE_AGENT]);
    assert_success(&disconnect, "cg disconnect claude-code");

    let claude_after = fs::read_to_string(claude_config(&home)).expect("claude config readable");
    assert!(
        !claude_after.contains("cg observe --agent claude-code"),
        "disconnect claude-code should remove only claude-code cg hook lines"
    );
    assert!(
        claude_after.contains("user-custom-hook:claude-code:echo hello"),
        "disconnect should preserve claude-code user hook"
    );
    assert_eq!(
        fs::read(cursor_config(&home)).expect("cursor config readable"),
        cursor_before,
        "disconnect claude-code must not rewrite cursor config bytes"
    );
    assert_eq!(
        fs::read(gemini_config(&home)).expect("gemini config readable"),
        gemini_before,
        "disconnect claude-code must not rewrite gemini config bytes"
    );
}

#[test]
fn connect_remove_all_removes_cg_injection_hooks_from_real_settings_without_touching_other_agents()
{
    let temp = tempfile::tempdir().expect("tempdir");
    let home = temp.path().join("home");
    let prefix = temp.path().join("prefix");
    fs::create_dir_all(&home).expect("home created");
    write_agent_config(&cursor_config(&home), CURSOR_AGENT, 1);
    let cursor_before = fs::read(cursor_config(&home)).expect("cursor config readable");
    let install = run_install(&home, &prefix, &[]);
    assert_success(&install, "install");

    // cg installs its injection hooks into the REAL ~/.claude/settings.json (structured JSON).
    let connect = run_cg(&home, &prefix, &["connect", CLAUDE_AGENT]);
    assert_success(&connect, "cg connect claude-code");
    let installed =
        fs::read_to_string(real_claude_settings(&home)).expect("claude settings written");
    assert!(
        installed.contains("cg hook claude-code"),
        "connect installs cg injection hooks in ~/.claude/settings.json: {installed}"
    );

    let remove_all = run_cg(&home, &prefix, &["connect", CLAUDE_AGENT, "--remove-all"]);
    assert_success(&remove_all, "cg connect claude-code --remove-all");
    let after = fs::read_to_string(real_claude_settings(&home)).expect("claude settings readable");
    assert!(
        !after.contains("cg hook claude-code"),
        "--remove-all clears cg injection hooks from the real settings: {after}"
    );
    assert_eq!(
        fs::read(cursor_config(&home)).expect("cursor config readable"),
        cursor_before,
        "--remove-all for claude-code must not rewrite cursor config bytes"
    );
}

#[test]
fn uninstall_lattice_three_levels_compose_disconnect_then_remove_all_then_uninstall_is_legal_sequence()
 {
    let temp = tempfile::tempdir().expect("tempdir");
    let home = temp.path().join("home");
    let prefix = temp.path().join("prefix");
    fs::create_dir_all(&home).expect("home created");
    let dot_cg_dirs = create_dot_cg_projects(&home);
    write_agent_config(&cursor_config(&home), CURSOR_AGENT, 1);
    let install = run_install(&home, &prefix, &[]);
    assert_success(&install, "install");

    let connect = run_cg(&home, &prefix, &["connect", CLAUDE_AGENT]);
    assert_success(&connect, "cg connect claude-code");

    // Level 1: disconnect removes cg's injection hooks from the real settings file.
    let disconnect = run_cg(&home, &prefix, &["disconnect", CLAUDE_AGENT]);
    assert_success(&disconnect, "cg disconnect claude-code");
    let claude_after_disconnect =
        fs::read_to_string(real_claude_settings(&home)).expect("claude settings readable");
    assert!(
        !claude_after_disconnect.contains("cg hook claude-code"),
        "level 1 disconnect should remove cg's injection hooks"
    );

    // Level 2: remove-all is idempotent after disconnect (nothing left to remove) and still legal.
    let remove_all = run_cg(&home, &prefix, &["connect", CLAUDE_AGENT, "--remove-all"]);
    assert_success(&remove_all, "cg connect claude-code --remove-all");

    // Level 3: uninstall removes the binary and preserves project .cg directories.
    let uninstall = run_uninstall(&home, &prefix, &[]);
    assert_success(&uninstall, "uninstall");
    assert!(
        !prefix.join("bin/cg").exists(),
        "level 3 uninstall should remove binary after lower lattice levels ran"
    );
    for dir in &dot_cg_dirs {
        assert!(
            dir.join("sentinel.txt").exists(),
            "lattice sequence must preserve project memory at {}",
            dir.display()
        );
    }
}

#[test]
fn install_script_runs_on_linux_and_macos_and_emits_correct_cache_root_per_platform() {
    let temp = tempfile::tempdir().expect("tempdir");
    let home = temp.path().join("home");
    let prefix = temp.path().join("prefix");
    fs::create_dir_all(&home).expect("home created");

    let install = run_install(&home, &prefix, &[]);
    assert_success(&install, "install");
    let uninstall_content =
        fs::read_to_string(prefix.join("cg-uninstall.sh")).expect("uninstall script readable");

    assert_eq!(DEFAULT_CACHE_ROOT_LINUX_MACOS, "~/.cache/cg/models");
    assert_eq!(DEFAULT_CACHE_ROOT_WINDOWS, "%LOCALAPPDATA%\\cg\\models");
    let expected_cache_root = if cfg!(target_os = "windows") {
        DEFAULT_CACHE_ROOT_WINDOWS.to_string()
    } else {
        home.join(DEFAULT_CACHE_ROOT_LINUX_MACOS.trim_start_matches("~/"))
            .display()
            .to_string()
    };
    assert!(
        uninstall_content.contains(&expected_cache_root),
        "uninstall script should render platform-correct cache root {expected_cache_root}"
    );
}

/// What the installer seeds is what `cg` loads, on a machine that relocates its cache.
///
/// `XDG_CACHE_HOME` is set by default on Nix, on systemd-homed setups, and in containers that put
/// caches on a mounted volume. The canonical resolver honours it; a script that hardcodes
/// `$HOME/.cache` instead downloads 547 MB into a directory nothing reads, and then reports it
/// removed the model cache while leaving the one in use on disk. The expectation here is the
/// resolver's own answer under the same environment, never a literal, so the script and the
/// library rule cannot drift apart again.
#[test]
fn the_installer_seeds_the_model_cache_root_the_library_reads_when_the_cache_home_is_relocated() {
    let temp = tempfile::tempdir().expect("tempdir");
    let home = temp.path().join("home");
    let xdg_cache = temp.path().join("xdg-cache");
    let prefix = temp.path().join("prefix");
    fs::create_dir_all(&home).expect("home created");
    fs::create_dir_all(&xdg_cache).expect("xdg cache created");

    let install = Command::new("bash")
        .arg(install_script())
        .arg("--prefix")
        .arg(&prefix)
        .env("HOME", &home)
        .env("XDG_CACHE_HOME", &xdg_cache)
        .env_remove(CG_MODEL_CACHE_ROOT_ENV)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("install script spawns");
    assert_success(&install, "install with a relocated cache home");

    let expected_root = cache_root_resolved_by_the_library(&home, Some(&xdg_cache));
    assert_eq!(
        expected_root,
        xdg_cache.join("cg/models"),
        "the library's own resolver must put the model cache under XDG_CACHE_HOME when it is set; \
         if this changed, the installer expectation below has to change with it"
    );

    let uninstall_content =
        fs::read_to_string(prefix.join("cg-uninstall.sh")).expect("uninstall script readable");
    let expected_line = format!("CACHE_ROOT=\"{}\"", expected_root.display());
    assert!(
        uninstall_content.contains(&expected_line),
        "the installer must seed, and the uninstaller it writes must remove, the exact directory \
         the library reads ({}); otherwise a prefetch downloads the model twice and an uninstall \
         leaves the copy in use behind. Uninstaller says:\n{uninstall_content}",
        expected_root.display()
    );
}

/// The model-cache root `cg_catalog::default_cache_root()` resolves under a given environment,
/// read by calling it with that environment in place rather than by restating its rule.
fn cache_root_resolved_by_the_library(home: &Path, xdg_cache_home: Option<&Path>) -> PathBuf {
    static ENVIRONMENT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = ENVIRONMENT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let saved_home = std::env::var_os("HOME");
    let saved_xdg = std::env::var_os(XDG_CACHE_HOME_ENV);
    let saved_override = std::env::var_os(CG_MODEL_CACHE_ROOT_ENV);
    // SAFETY: serialized by ENVIRONMENT_LOCK above and restored before returning.
    unsafe {
        std::env::set_var("HOME", home);
        std::env::remove_var(CG_MODEL_CACHE_ROOT_ENV);
        match xdg_cache_home {
            Some(path) => std::env::set_var(XDG_CACHE_HOME_ENV, path),
            None => std::env::remove_var(XDG_CACHE_HOME_ENV),
        }
    }
    let resolved = cg_catalog::default_cache_root();
    // SAFETY: same lock, restoring the exact prior state including "was unset".
    unsafe {
        match saved_home {
            Some(value) => std::env::set_var("HOME", value),
            None => std::env::remove_var("HOME"),
        }
        match saved_xdg {
            Some(value) => std::env::set_var(XDG_CACHE_HOME_ENV, value),
            None => std::env::remove_var(XDG_CACHE_HOME_ENV),
        }
        match saved_override {
            Some(value) => std::env::set_var(CG_MODEL_CACHE_ROOT_ENV, value),
            None => std::env::remove_var(CG_MODEL_CACHE_ROOT_ENV),
        }
    }
    resolved
}

#[test]
fn install_script_writes_uninstall_with_executable_permission_and_shebang() {
    let temp = tempfile::tempdir().expect("tempdir");
    let home = temp.path().join("home");
    let prefix = temp.path().join("prefix");
    fs::create_dir_all(&home).expect("home created");

    let install = run_install(&home, &prefix, &[]);
    assert_success(&install, "install");
    let uninstall = prefix.join("cg-uninstall.sh");
    let content = fs::read_to_string(&uninstall).expect("uninstall script readable");
    assert!(
        content.starts_with("#!"),
        "uninstall script should have a shebang"
    );
    assert!(
        is_executable(&uninstall),
        "uninstall script should be directly executable"
    );

    let help = run_uninstall(&home, &prefix, &["--help"]);
    assert_success(&help, "uninstall --help");
    let stdout = String::from_utf8_lossy(&help.stdout);
    assert!(
        stdout.contains("--purge") && stdout.contains("preserve"),
        "uninstall help should document purge and default .cg preservation"
    );
}
