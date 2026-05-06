// Digest is immutable (Invariant 16); exposing Store::update_digest would let callers rewrite the provenance source of every extracted Intention, Decision, and Observation.
use context_graph::{Store, StoreConfig};

fn main() {
    let store = Store::open(StoreConfig::default()).expect("open");
    let _ = store.update_digest();
}
