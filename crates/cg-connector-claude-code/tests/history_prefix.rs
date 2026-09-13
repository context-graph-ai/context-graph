use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use cg_connector_claude_code::ClaudeHistoryLocator;
use serde_json::{Value, json};
use tempfile::tempdir;

fn claude_project_key(project: &Path) -> String {
    project
        .to_str()
        .expect("synthetic project path is Unicode")
        .encode_utf16()
        .map(|unit| match unit {
            0x41..=0x5a | 0x61..=0x7a | 0x30..=0x39 => {
                char::from_u32(u32::from(unit)).expect("ASCII UTF-16 unit")
            }
            _ => '-',
        })
        .collect()
}

fn write_records(path: &Path, records: &[Value]) {
    fs::create_dir_all(path.parent().expect("history parent")).expect("create history parent");
    let mut file = fs::File::create(path).expect("create synthetic history");
    for record in records {
        writeln!(
            file,
            "{}",
            serde_json::to_string(record).expect("serialize record")
        )
        .expect("write synthetic record");
    }
}

fn introductory_records() -> Vec<Value> {
    vec![
        json!({"type": "custom-title", "customTitle": "Synthetic queued session"}),
        json!({"type": "queue-operation", "operation": "enqueue"}),
    ]
}

fn candidate_store(history_root: &Path, project: &Path) -> PathBuf {
    history_root.join(claude_project_key(project))
}

#[test]
fn history_locator_finds_the_first_cwd_after_introductory_records() {
    let sandbox = tempdir().expect("temporary history root");
    let project = sandbox.path().join("project with spaces");
    fs::create_dir_all(&project).expect("create project");
    let canonical_project = project.canonicalize().expect("canonical project");
    let history_root = sandbox.path().join("claude").join("projects");
    let source = candidate_store(&history_root, &canonical_project).join("session.jsonl");

    let mut records = introductory_records();
    records.push(json!({"type": "user", "cwd": canonical_project}));
    write_records(&source, &records);

    let history = ClaudeHistoryLocator::new(history_root)
        .resolve_project(&project)
        .expect("resolve synthetic history");
    assert_eq!(history.canonical_project, canonical_project);
    assert_eq!(
        history.sources.len(),
        1,
        "the first cwd identifies the project"
    );
    assert_eq!(
        history.sources[0].path,
        source.canonicalize().expect("canonical selected source")
    );
}

#[test]
fn history_locator_rejects_another_projects_first_cwd_and_files_without_cwd() {
    let sandbox = tempdir().expect("temporary history root");
    let project = sandbox.path().join("project");
    let other_project = sandbox.path().join("other-project");
    fs::create_dir_all(&project).expect("create project");
    fs::create_dir_all(&other_project).expect("create other project");
    let canonical_project = project.canonicalize().expect("canonical project");
    let canonical_other = other_project
        .canonicalize()
        .expect("canonical other project");
    let history_root = sandbox.path().join("claude").join("projects");
    let store = candidate_store(&history_root, &canonical_project);

    let mut another_project = introductory_records();
    another_project.push(json!({"type": "user", "cwd": canonical_other}));
    write_records(&store.join("another-project.jsonl"), &another_project);

    let mut no_cwd = introductory_records();
    no_cwd.push(json!({"type": "summary", "summary": "synthetic structural record"}));
    write_records(&store.join("no-cwd.jsonl"), &no_cwd);

    let history = ClaudeHistoryLocator::new(history_root)
        .resolve_project(&project)
        .expect("resolve synthetic history");
    assert!(
        history.sources.is_empty(),
        "only the first cwd can identify a source"
    );
}

#[test]
fn history_locator_keeps_encoded_candidate_and_symlink_alias_selection() {
    let sandbox = tempdir().expect("temporary history root");
    let project = sandbox.path().join("project with spaces");
    let alias = sandbox.path().join("project-alias");
    fs::create_dir_all(&project).expect("create project");
    std::os::unix::fs::symlink(&project, &alias).expect("create project alias");
    let canonical_project = project.canonicalize().expect("canonical project");
    let history_root = sandbox.path().join("claude").join("projects");
    let source = candidate_store(&history_root, &canonical_project).join("session.jsonl");
    write_records(
        &source,
        &[json!({"type": "user", "cwd": canonical_project})],
    );

    let direct = ClaudeHistoryLocator::new(history_root.clone())
        .resolve_project(&project)
        .expect("resolve direct project");
    let through_alias = ClaudeHistoryLocator::new(history_root)
        .resolve_project(&alias)
        .expect("resolve project alias");
    assert_eq!(direct.canonical_project, canonical_project);
    assert_eq!(through_alias.canonical_project, canonical_project);
    assert_eq!(
        through_alias
            .sources
            .iter()
            .map(|source| source.path.clone())
            .collect::<Vec<_>>(),
        direct
            .sources
            .iter()
            .map(|source| source.path.clone())
            .collect::<Vec<_>>()
    );
}
