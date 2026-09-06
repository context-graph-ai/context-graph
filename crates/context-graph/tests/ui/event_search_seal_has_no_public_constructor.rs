// Nobody outside this crate can mint an `EventSearchSeal`, so nobody outside this crate can forge
// a sealed event-search value under someone else's encoding profile or claim canonical
// event-history authority. The only route to a seal is this library's own checked
// `prepare_document`/`prepare_query` entrypoints, and only for the length of one call into an
// embedder's hook.
//
// The committed `.stderr` snapshot beside this file is what makes the guard real: it pins the
// error to `new` being private. Making the constructor `pub` changes that output and turns this
// case red, which a bare "some compile error happened" check would not.
use context_graph::embedder::EventSearchSeal;

fn main() {
    let _seal = EventSearchSeal::new("attacker_profile".to_string(), true);
}
