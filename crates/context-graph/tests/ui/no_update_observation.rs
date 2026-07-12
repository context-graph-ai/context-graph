// Observations are immutable recorded facts; a public Store::update_observation would let well-intentioned cleanup code rewrite history the Decisions rest on.
use context_graph::{ObservationId, Store, StoreConfig};

fn main() {
    let store = Store::open(StoreConfig::default()).expect("open");
    let _ = store.update_observation(ObservationId::new_v7());
}
