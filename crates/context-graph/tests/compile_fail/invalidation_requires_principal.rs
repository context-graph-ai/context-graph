use context_graph::{InvalidationId, Store, StoreConfig};

fn main() {
    let store = Store::open(StoreConfig::default()).expect("store");
    let id = InvalidationId::new_v7();
    let _ = store.acknowledge_invalidation(id);
}
