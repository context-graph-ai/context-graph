use context_graph::{ObservationId, Store};

fn main() {
    let store = Store::open("tmp");
    let _ = store.update_observation(ObservationId::new_v7());
}
