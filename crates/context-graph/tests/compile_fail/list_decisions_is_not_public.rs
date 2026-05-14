use context_graph::{ListDecisionFilter, Store, StoreConfig};

fn main() {
    let db_path = std::env::temp_dir().join(format!(
        "context-graph-list-decisions-{}.db",
        std::process::id()
    ));
    let store = Store::open(StoreConfig {
        db_path,
        ..Default::default()
    })
    .expect("open");
    let _ = store.list_decisions(ListDecisionFilter::default());
}
