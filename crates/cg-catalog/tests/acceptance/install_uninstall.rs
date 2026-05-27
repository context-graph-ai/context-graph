use cg_catalog::{
    CG_MODEL_CACHE_ROOT_ENV, DEFAULT_CACHE_ROOT_LINUX_MACOS, DEFAULT_CACHE_ROOT_WINDOWS,
};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

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
fn connect_remove_all_command_removes_all_hooks_for_one_agent_kind_across_every_agent_config() {
    let temp = tempfile::tempdir().expect("tempdir");
    let home = temp.path().join("home");
    let prefix = temp.path().join("prefix");
    fs::create_dir_all(&home).expect("home created");
    let claude_paths = [
        home.join("agent-config/claude-code/settings.json"),
        home.join("work/one/agent-config/claude-code/settings.json"),
        home.join("work/two/agent-config/claude-code/settings.json"),
    ];
    for path in &claude_paths {
        write_agent_config(path, CLAUDE_AGENT, 1);
    }
    write_agent_config(&cursor_config(&home), CURSOR_AGENT, 1);
    let cursor_before = fs::read(cursor_config(&home)).expect("cursor config readable");
    let install = run_install(&home, &prefix, &[]);
    assert_success(&install, "install");

    let remove_all = run_cg(&home, &prefix, &["connect", CLAUDE_AGENT, "--remove-all"]);
    assert_success(&remove_all, "cg connect claude-code --remove-all");

    for path in &claude_paths {
        let content = fs::read_to_string(path).expect("claude config readable");
        assert!(
            !content.contains("cg observe --agent claude-code"),
            "--remove-all should remove every claude-code cg hook in {}",
            path.display()
        );
    }
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
    write_agent_config(&claude_config(&home), CLAUDE_AGENT, 1);
    write_agent_config(
        &home.join("work/other/agent-config/claude-code/settings.json"),
        CLAUDE_AGENT,
        1,
    );
    write_agent_config(&cursor_config(&home), CURSOR_AGENT, 1);
    let install = run_install(&home, &prefix, &[]);
    assert_success(&install, "install");

    let disconnect = run_cg(&home, &prefix, &["disconnect", CLAUDE_AGENT]);
    assert_success(&disconnect, "cg disconnect claude-code");
    let claude_after_disconnect =
        fs::read_to_string(claude_config(&home)).expect("claude config readable");
    assert!(
        !claude_after_disconnect.contains("cg observe --agent claude-code"),
        "level 1 disconnect should remove the selected agent hook"
    );

    let remove_all = run_cg(&home, &prefix, &["connect", CLAUDE_AGENT, "--remove-all"]);
    assert_success(&remove_all, "cg connect claude-code --remove-all");
    let other_claude =
        fs::read_to_string(home.join("work/other/agent-config/claude-code/settings.json"))
            .expect("config readable");
    assert!(
        !other_claude.contains("cg observe --agent claude-code"),
        "level 2 remove-all should clear remaining claude-code hooks"
    );

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
