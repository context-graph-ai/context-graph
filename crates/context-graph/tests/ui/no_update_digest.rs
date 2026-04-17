// Digest is immutable (Invariant 16); exposing Store::update_digest would let callers rewrite the provenance source of every extracted Intention, Decision, and Observation.
use context_graph::Store;

fn main() {
    let store = Store::open("tmp");
    let _ = store.update_digest();
}
