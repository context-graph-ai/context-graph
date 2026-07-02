mod support;

use serde_json::Value;

use context_graph::skill_store::{
    LearnedSkill, SkillApplyStrategy, SkillProvenance, persist_skill,
};
use context_graph::{Store, StoreConfig};

use support::{CliHarness, parse_single_json_object, write_file};

#[test]
fn cg_skills_add_file_persists_human_authored_skill_and_export_round_trips_skill_md() {
    let harness = CliHarness::new();
    let skill_path = harness.data_root.path().join("repository-retype.SKILL.md");
    write_file(
        &skill_path,
        "# description: retype context-graph file captures as repository\n\
         ## rule\n\
         when an entity named context-graph is typed File -> Repository\n",
    );

    let add = harness.run(&[
        "skills",
        "add",
        skill_path.to_str().expect("utf8 skill path"),
    ]);
    assert_eq!(
        add.code, 0,
        "`cg skills add <file>` must accept a well-formed SKILL.md as a human-authored skill; \
         stdout={:?} stderr={:?}",
        add.stdout, add.stderr
    );

    let list = harness.run(&["skills"]);
    assert_eq!(
        list.code, 0,
        "`cg skills` must list the human-authored skill after add; stderr={:?}",
        list.stderr
    );
    let listed = parse_single_json_object(&list.stdout).expect("skills list must be JSON");
    let skills = listed
        .get("skills")
        .and_then(Value::as_array)
        .expect("skills list must carry a skills array");
    let added = skills
        .iter()
        .find(|skill| {
            skill
                .get("description")
                .and_then(Value::as_str)
                .is_some_and(|description| description.contains("context-graph"))
        })
        .unwrap_or_else(|| panic!("added skill must appear in `cg skills`; listed={skills:?}"));
    let skill_id = added
        .get("id")
        .and_then(Value::as_str)
        .expect("listed skill must carry id");
    assert_eq!(
        added.get("provenance").and_then(Value::as_str),
        Some("human-authored"),
        "skills added from a file must carry human-authored provenance"
    );

    let export = harness.run(&["skills", "export", skill_id]);
    assert_eq!(
        export.code, 0,
        "`cg skills export <id>` must print the skill as SKILL.md text; stdout={:?} stderr={:?}",
        export.stdout, export.stderr
    );
    assert!(
        export.stdout.contains("# description:")
            && export.stdout.contains("## rule")
            && export.stdout.contains("context-graph"),
        "export must round-trip inspectable SKILL.md text, got stdout={:?}",
        export.stdout
    );
}

/// An `apply_strategy` string this build does not implement — exactly what an older edge sees
/// after syncing a newer edge's row — must round-trip byte-for-byte through `cg skills` JSON,
/// never coerced to an implemented strategy. `cg skills add`/`export` only ever produce the
/// SKILL.md text shape (no apply_strategy field), so this seeds the row directly at the store
/// level, the same way a sync pull would land it.
#[test]
fn cg_skills_json_preserves_unknown_apply_strategy_verbatim() {
    let harness = CliHarness::new();

    // Any capture initializes the on-disk store and its default context.
    let record = harness.run(&[
        "record",
        "--context",
        "strategy-fold",
        "hello world, nothing decided yet",
    ]);
    assert_eq!(
        record.code, 0,
        "seed record must succeed; stdout={:?} stderr={:?}",
        record.stdout, record.stderr
    );

    let db_path = harness.data_root.path().join("context-graph.db");
    let store = Store::open(StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("open the seeded store directly, the same file `cg` just wrote to");
    assert!(
        persist_skill(
            &store,
            &LearnedSkill {
                id: "sk-future-strategy".to_string(),
                description: "future strategy fold-seam skill".to_string(),
                rule: "when an entity named context-graph is typed File → Repository".to_string(),
                effect: None,
                apply_strategy: SkillApplyStrategy::Unknown("future-strategy".to_string()),
                provenance: SkillProvenance::HumanAuthored,
                active: true,
                held_for_review: false,
                fire_right: 0,
                fire_wrong: 0,
            },
        ),
        "persisting a row with an unsupported apply_strategy must succeed (never rejected)"
    );
    drop(store);

    let list = harness.run(&["skills", "--context", "strategy-fold"]);
    assert_eq!(
        list.code, 0,
        "`cg skills` must succeed even with an unsupported-strategy row present; stderr={:?}",
        list.stderr
    );
    let listed = parse_single_json_object(&list.stdout).expect("skills list must be JSON");
    let skills = listed
        .get("skills")
        .and_then(Value::as_array)
        .expect("skills list must carry a skills array");
    let found = skills
        .iter()
        .find(|skill| skill.get("id").and_then(Value::as_str) == Some("sk-future-strategy"))
        .unwrap_or_else(|| {
            panic!("unsupported-strategy skill must still be listed; listed={skills:?}")
        });
    assert_eq!(
        found.get("apply_strategy").and_then(Value::as_str),
        Some("future-strategy"),
        "an unsupported apply_strategy string must round-trip byte-for-byte through `cg skills` \
         JSON, never coerced to an implemented strategy; got {found:?}"
    );
}
