use context_graph::Store;

fn main() {
    let store = Store::open("tmp");
    let _ = store.update_digest();
}
