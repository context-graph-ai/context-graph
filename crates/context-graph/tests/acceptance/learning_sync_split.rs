// Sync split acceptance tests (6 tests).
//
// The store-state journeys use the same authenticated, ticketed Iroh path as production.
// The declaration proof reads the installed table metadata, which is the one sync authority.

use std::collections::HashMap;

use context_graph::skill_store::{
    FIRING_TRACE_TABLE, REVIEW_CURSOR_TABLE, SKILL_TABLE, SkillApplyStrategy, SkillEffectEnvelope,
    TEACHING_EVENT_TABLE,
};
use context_graph::{Store, TxId};
use contextdb_core::Value as DbValue;

use super::capture_sync_common::{TestStore, bidirectional_sync, push_pull, real_sync_fixture};

fn declared_sync_policy(store: &Store, table: &str) -> (Option<String>, Option<String>) {
    let meta = store
        .sync_database()
        .table_meta(table)
        .unwrap_or_else(|| panic!("{table} must be installed"));
    (
        meta.sync_direction
            .map(|direction| direction.sql().to_string()),
        meta.conflict_policy
            .and_then(|policy| policy.declared_clause())
            .map(str::to_string),
    )
}

fn insert_skill(store: &Store, id: &str, description: &str, rule: &str, provenance: &str) {
    let db = store.sync_database();
    db.execute(
        "INSERT INTO cg_learned_skills (id, description, rule, provenance, active, held_for_review, fire_right, fire_wrong, created_at) VALUES ($id, $description, $rule, $provenance, $active, $held_for_review, $fire_right, $fire_wrong, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Text(id.to_string())),
            ("description".to_string(), DbValue::Text(description.to_string())),
            ("rule".to_string(), DbValue::Text(rule.to_string())),
            ("provenance".to_string(), DbValue::Text(provenance.to_string())),
            ("active".to_string(), DbValue::Bool(true)),
            ("held_for_review".to_string(), DbValue::Bool(false)),
            ("fire_right".to_string(), DbValue::Int64(0)),
            ("fire_wrong".to_string(), DbValue::Int64(0)),
            ("created_at".to_string(), DbValue::TxId(TxId(1))),
        ]),
    )
    .expect("insert skill for sync test");
}

fn insert_structured_retype_skill(
    store: &Store,
    id: &str,
    description: &str,
    rule: &str,
    provenance: &str,
    effect_json: &str,
    apply_strategy: &str,
) {
    let db = store.sync_database();
    db.execute(
        "INSERT INTO cg_learned_skills (id, description, rule, effect, apply_strategy, provenance, active, held_for_review, fire_right, fire_wrong, created_at) VALUES ($id, $description, $rule, $effect, $apply_strategy, $provenance, $active, $held_for_review, $fire_right, $fire_wrong, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Text(id.to_string())),
            ("description".to_string(), DbValue::Text(description.to_string())),
            ("rule".to_string(), DbValue::Text(rule.to_string())),
            ("effect".to_string(), DbValue::Text(effect_json.to_string())),
            (
                "apply_strategy".to_string(),
                DbValue::Text(apply_strategy.to_string()),
            ),
            ("provenance".to_string(), DbValue::Text(provenance.to_string())),
            ("active".to_string(), DbValue::Bool(true)),
            ("held_for_review".to_string(), DbValue::Bool(false)),
            ("fire_right".to_string(), DbValue::Int64(0)),
            ("fire_wrong".to_string(), DbValue::Int64(0)),
            ("created_at".to_string(), DbValue::TxId(TxId(1))),
        ]),
    )
    .expect("insert structured skill for sync test");
}

fn insert_teaching_event(
    store: &Store,
    id: &str,
    skill_id: &str,
    item_id: &str,
    kind: &str,
    content: &str,
) {
    let db = store.sync_database();
    db.execute(
        "INSERT INTO cg_teaching_events (id, skill_id, corrected_item_id, kind, content, created_at) VALUES ($id, $skill_id, $corrected_item_id, $kind, $content, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Text(id.to_string())),
            ("skill_id".to_string(), DbValue::Text(skill_id.to_string())),
            ("corrected_item_id".to_string(), DbValue::Text(item_id.to_string())),
            ("kind".to_string(), DbValue::Text(kind.to_string())),
            ("content".to_string(), DbValue::Text(content.to_string())),
            ("created_at".to_string(), DbValue::TxId(TxId(1))),
        ]),
    )
    .expect("insert teaching event");
}

fn insert_firing_trace(store: &Store, id: &str, skill_id: &str, capture_id: &str) {
    let db = store.sync_database();
    db.execute(
        "INSERT INTO cg_skill_firing_trace (id, skill_id, capture_id, correct, created_at) VALUES ($id, $skill_id, $capture_id, $correct, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Text(id.to_string())),
            ("skill_id".to_string(), DbValue::Text(skill_id.to_string())),
            ("capture_id".to_string(), DbValue::Text(capture_id.to_string())),
            ("correct".to_string(), DbValue::Null),
            ("created_at".to_string(), DbValue::TxId(TxId(1))),
        ]),
    )
    .expect("insert firing trace");
}

fn count_table_rows(store: &Store, table: &str) -> usize {
    let db = store.sync_database();
    let snapshot = db.snapshot();
    db.scan(table, snapshot).expect("scan table").len()
}

fn skill_row_by_id(store: &Store, skill_id: &str) -> Option<HashMap<String, DbValue>> {
    let db = store.sync_database();
    let snapshot = db.snapshot();
    db.point_lookup(
        "cg_learned_skills",
        "id",
        &DbValue::Text(skill_id.to_string()),
        snapshot,
    )
    .expect("skill point lookup")
    .map(|row| row.values)
}

#[tokio::test(flavor = "multi_thread")]
async fn skills_sync_tenant_wide_hub_arbitrated() {
    let fixture = real_sync_fixture("skills_sync_tenant_wide_hub_arbitrated").await;

    let skill_desc = "retype context-graph file entity to repository";
    let skill_rule = "when an entity named context-graph is typed File → Repository";
    let skill_effect = r#"{"type":"retype_entity","entity_name":"context-graph","from_type":"File","to_type":"Repository"}"#;
    let skill_strategy = "post-harvest-retype";
    let skill_prov = "llm-authored";
    insert_structured_retype_skill(
        &fixture.edge1.store,
        "sk-sync-hub",
        skill_desc,
        skill_rule,
        skill_prov,
        skill_effect,
        skill_strategy,
    );

    push_pull(&fixture.edge1_client, &fixture.edge2_client).await;

    // Verify the raw row arrived on edge2 (sync engine carries it regardless of policy).
    let row = skill_row_by_id(&fixture.edge2.store, "sk-sync-hub");
    assert!(
        row.is_some(),
        "skill row must appear on edge2 after push/pull"
    );
    let row = row.unwrap();

    let synced_desc = match row.get("description") {
        Some(DbValue::Text(t)) => t.clone(),
        _ => panic!("description must be a Text value on synced row"),
    };
    assert_eq!(
        synced_desc, skill_desc,
        "synced skill description must be identical to edge1"
    );

    let synced_rule = match row.get("rule") {
        Some(DbValue::Text(t)) => t.clone(),
        _ => panic!("rule must be a Text value on synced row"),
    };
    assert_eq!(
        synced_rule, skill_rule,
        "synced skill rule must be identical to edge1"
    );

    let synced_effect = match row.get("effect") {
        Some(DbValue::Text(t)) => t.clone(),
        _ => panic!("effect must be a Text value on synced row"),
    };
    assert_eq!(
        synced_effect, skill_effect,
        "synced skill effect envelope must be identical to edge1"
    );

    let synced_strategy = match row.get("apply_strategy") {
        Some(DbValue::Text(t)) => t.clone(),
        _ => panic!("apply_strategy must be a Text value on synced row"),
    };
    assert_eq!(
        synced_strategy, skill_strategy,
        "synced skill apply_strategy tag must be identical to edge1"
    );

    let synced_prov = match row.get("provenance") {
        Some(DbValue::Text(t)) => t.clone(),
        _ => panic!("provenance must be a Text value on synced row"),
    };
    assert_eq!(
        synced_prov, skill_prov,
        "synced skill provenance must be identical to edge1"
    );

    let listed = context_graph::skill_store::list_skills_in_store(&fixture.edge2.store)
        .expect("read succeeds");
    assert_eq!(
        listed.len(),
        1,
        "list_skills_in_store must surface the synced skill; got {}",
        listed.len()
    );
    let listed_skill = &listed[0];
    assert_eq!(
        listed_skill.id, "sk-sync-hub",
        "listed skill id must match the synced row"
    );
    assert_eq!(
        listed_skill.effect,
        Some(SkillEffectEnvelope::RetypeEntity {
            entity_name: "context-graph".to_string(),
            from_type: "File".to_string(),
            to_type: "Repository".to_string(),
        }),
        "list_skills_in_store must preserve the synced structured effect envelope"
    );
    assert_eq!(
        listed_skill.apply_strategy,
        SkillApplyStrategy::PostHarvestRetype,
        "list_skills_in_store must preserve the synced strategy tag"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn teaching_events_sync_tenant_wide_order_free_set_union() {
    let fixture = real_sync_fixture("teaching_events_sync_tenant_wide_order_free_set_union").await;

    insert_teaching_event(
        &fixture.edge1.store,
        "ev-edge1-a",
        "sk-union",
        "item-1",
        "correction",
        "technology_choice",
    );
    insert_teaching_event(
        &fixture.edge1.store,
        "ev-edge1-b",
        "sk-union",
        "item-2",
        "correction",
        "code_reference",
    );
    insert_teaching_event(
        &fixture.edge2.store,
        "ev-edge2-a",
        "sk-union",
        "item-3",
        "correction",
        "technology_choice",
    );
    insert_teaching_event(
        &fixture.edge2.store,
        "ev-edge2-b",
        "sk-union",
        "item-4",
        "reversal",
        "",
    );

    bidirectional_sync(&fixture.edge1_client, &fixture.edge2_client).await;

    // Verify the raw rows converged (sync engine carries them).
    let edge1_raw = count_table_rows(&fixture.edge1.store, "cg_teaching_events");
    let edge2_raw = count_table_rows(&fixture.edge2.store, "cg_teaching_events");
    assert_eq!(
        edge1_raw, 4,
        "edge1 must hold all 4 teaching events in the table after bidirectional sync; got {edge1_raw}"
    );
    assert_eq!(
        edge2_raw, 4,
        "edge2 must hold all 4 teaching events in the table after bidirectional sync; got {edge2_raw}"
    );

    let listed = context_graph::skill_store::list_teaching_events_in_store(&fixture.edge2.store)
        .expect("read succeeds");
    assert_eq!(
        listed.len(),
        4,
        "list_teaching_events_in_store must return all 4 synced teaching events; got {}",
        listed.len()
    );
}

// Test 3: firing trace must remain on its originating device.
#[tokio::test(flavor = "multi_thread")]
async fn firing_trace_stays_local_per_device_not_synced() {
    let fixture = real_sync_fixture("firing_trace_stays_local_per_device_not_synced").await;

    assert_eq!(
        declared_sync_policy(&fixture.edge1.store, FIRING_TRACE_TABLE),
        (Some("SYNC OFF".to_string()), None),
        "the local firing trace must declare its local-only behavior in table metadata"
    );

    for i in 0..3usize {
        insert_firing_trace(
            &fixture.edge1.store,
            &format!("ft-local-{i}"),
            "sk-ft",
            &format!("cap-{i}"),
        );
    }

    push_pull(&fixture.edge1_client, &fixture.edge2_client).await;

    let edge2_count = count_table_rows(&fixture.edge2.store, "cg_skill_firing_trace");
    assert_eq!(
        edge2_count, 0,
        "firing-trace rows must NOT sync to edge2; edge2 has {edge2_count} rows"
    );

    let edge1_count = count_table_rows(&fixture.edge1.store, "cg_skill_firing_trace");
    assert_eq!(
        edge1_count, 3,
        "edge1 must retain its 3 local firing-trace rows"
    );
}

// Test 4: after one A→B sync, B holds the skill identity AND
// zero A-origin firing-trace rows.
#[tokio::test(flavor = "multi_thread")]
async fn firing_trace_stays_local_while_skill_crosses_in_one_sync() {
    let fixture =
        real_sync_fixture("firing_trace_stays_local_while_skill_crosses_in_one_sync").await;

    let desc = "crosses but trace stays local";
    let rule = "rule: cross-split";
    let prov = "human-correction";
    insert_skill(&fixture.edge1.store, "sk-cross", desc, rule, prov);
    for i in 0..4usize {
        insert_firing_trace(
            &fixture.edge1.store,
            &format!("ft-cross-{i}"),
            "sk-cross",
            &format!("cap-cross-{i}"),
        );
    }

    push_pull(&fixture.edge1_client, &fixture.edge2_client).await;

    let skill_on_b = skill_row_by_id(&fixture.edge2.store, "sk-cross");
    assert!(
        skill_on_b.is_some(),
        "skill must appear on edge2 after one sync"
    );
    let skill_row = skill_on_b.unwrap();
    let b_desc = match skill_row.get("description") {
        Some(DbValue::Text(t)) => t.clone(),
        _ => panic!("description must be Text"),
    };
    assert_eq!(b_desc, desc, "skill description on edge2 must equal edge1");

    let b_trace_count = count_table_rows(&fixture.edge2.store, "cg_skill_firing_trace");
    assert_eq!(
        b_trace_count, 0,
        "edge2 must have zero firing-trace rows after one sync (trace stays local); got {b_trace_count}"
    );
}

// Test 5: a reversal/undo on edge A propagates on the next sync. The skills table's declared
// KEEP LATEST policy converges the hub-accepted later mutation.
#[tokio::test(flavor = "multi_thread")]
async fn reversal_undo_propagates_on_next_sync_order_tolerant_and_global() {
    let fixture =
        real_sync_fixture("reversal_undo_propagates_on_next_sync_order_tolerant_and_global").await;

    insert_skill(
        &fixture.edge1.store,
        "sk-reversal",
        "reversal test skill",
        "reversal rule",
        "human-correction",
    );
    push_pull(&fixture.edge1_client, &fixture.edge2_client).await;

    assert!(
        skill_row_by_id(&fixture.edge2.store, "sk-reversal").is_some(),
        "skill must be on edge2 before reversal test"
    );

    // Insert a reversal teaching event on edge1 (simulates `cg undo`).
    insert_teaching_event(
        &fixture.edge1.store,
        "ev-reversal",
        "sk-reversal",
        "item-rev",
        "reversal",
        "",
    );

    // Mark the skill as inactive on edge1 (undo effect).
    {
        let db = fixture.edge1.store.sync_database();
        db.execute(
            "UPDATE cg_learned_skills SET active = $active WHERE id = $id",
            &HashMap::from([
                ("id".to_string(), DbValue::Text("sk-reversal".to_string())),
                ("active".to_string(), DbValue::Bool(false)),
            ]),
        )
        .expect("mark skill inactive on reversal");
    }

    push_pull(&fixture.edge1_client, &fixture.edge2_client).await;

    let edge2_teaching_count = count_table_rows(&fixture.edge2.store, "cg_teaching_events");
    assert_eq!(
        edge2_teaching_count, 1,
        "reversal teaching event must propagate to edge2; got {edge2_teaching_count}"
    );

    // Edge2's skill must reflect the hub-accepted later inactive mutation.
    let skill_on_b = skill_row_by_id(&fixture.edge2.store, "sk-reversal");
    assert!(
        skill_on_b.is_some(),
        "skill row must still exist on edge2 after reversal sync"
    );
    let active_val = skill_on_b.unwrap().get("active").cloned();
    assert_eq!(
        active_val,
        Some(DbValue::Bool(false)),
        "skill must be inactive on edge2 after the hub-accepted KEEP LATEST reversal"
    );
}

// Test 6: the installed metadata is the complete declaration authority for learning state.
#[test]
fn learned_skill_sync_table_policies_match_the_declared_contract() {
    let store = TestStore::new("learned_skill_sync_table_policies_match_the_declared_contract");

    assert_eq!(
        declared_sync_policy(&store.store, SKILL_TABLE),
        (
            Some("SYNC TWO WAY".to_string()),
            Some("SYNC CONFLICT KEEP LATEST".to_string())
        ),
        "learned skill state must converge by the declared latest mutation"
    );

    assert_eq!(
        declared_sync_policy(&store.store, TEACHING_EVENT_TABLE),
        (
            Some("SYNC TWO WAY".to_string()),
            Some("SYNC CONFLICT KEEP FIRST".to_string())
        ),
        "teaching events are independently identified additive history"
    );

    assert_eq!(
        declared_sync_policy(&store.store, FIRING_TRACE_TABLE),
        (Some("SYNC OFF".to_string()), None),
        "firing trace is local operational state"
    );

    assert_eq!(
        declared_sync_policy(&store.store, REVIEW_CURSOR_TABLE),
        (Some("SYNC OFF".to_string()), None),
        "review cursor is local operational state"
    );
}
