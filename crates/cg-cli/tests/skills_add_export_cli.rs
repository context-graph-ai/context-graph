mod support;

use serde_json::Value;

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
