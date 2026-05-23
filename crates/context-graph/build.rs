fn main() {
    println!("cargo:rustc-check-cfg=cfg(context_graph_internal_test_support)");
    println!("cargo:rustc-check-cfg=cfg(trybuild)");
}
