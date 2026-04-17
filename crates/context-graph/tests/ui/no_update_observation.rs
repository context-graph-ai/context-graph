// Observations are immutable recorded facts (Invariant 4); a public Store::update_observation would let well-intentioned cleanup code rewrite history the Decisions rest on.
use context_graph::{ObservationId, Store};

fn main() {
    let store = Store::open("tmp");
    let _ = store.update_observation(ObservationId::new_v7());
}
