// throwaway inspector
fn main() {
    let root = std::env::args().nth(1).expect("root");
    let store = context_graph::Store::open(context_graph::StoreConfig {
        db_path: std::path::PathBuf::from(&root).join("context-graph.db"),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("open");
    for c in store.list_contexts().expect("contexts") {
        let ints = store
            .list_intentions_in_context(c.id)
            .map(|v| v.len())
            .unwrap_or(999);
        let ents = store
            .list_entities(context_graph::ListEntityFilter {
                context_id: Some(c.id),
                ..Default::default()
            })
            .map(|v| v.len())
            .unwrap_or(999);
        let unserved = store
            .decisions_without_serves_in_context(c.id)
            .map(|v| v.len())
            .unwrap_or(999);
        println!(
            "context {:?}: intentions={ints} entities={ents} unserved_decisions={unserved}",
            c.name
        );
    }
}
