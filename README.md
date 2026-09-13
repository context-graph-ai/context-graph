# context-graph

Durable, queryable memory for an agent or an application: the intentions it was
serving, the decisions it made, the observations it recorded, and the digests
that tie them together — stored in a local contextdb database and searched by
meaning as well as by structure.<!-- enforced by: `find_via_digests_uses_contextdb_vector_index_and_graph_traversal_not_local_scan`, `intention_create_persists_across_reopen` -->

This repository ships two things. `context-graph` is the embeddable memory core:
link it as a library and you get the ontology, the store and its domain
behavior, embedder registration, bounded status reads, the administrative owner
plane, and a read-only view for a process that must not change the store. `cg`
is the command-line binary built on top of it — one consumer of the core, not a
privileged one.<!-- enforced by: `one_handle_serves_owner_frames_and_embeds_interleaved`, `a_statement_that_would_change_the_store_is_refused_by_the_read_only_surface` -->

The core is embedded rather than served. There is no daemon and no
caller-configured socket address: one process holds the store's writable handle,
everything else is a typed call, and a question asked of a *running*
context-graph travels contextdb's local owner channel, addressed by the store
path itself.<!-- enforced by: `the_crate_carries_no_socket_transport_and_no_path_taking_control_api`, `a_live_owner_answers_a_control_frame_addressed_only_by_the_store_path` -->

## Build it

You need:

- Rust **1.94.1**. `rust-toolchain.toml` pins it, so `rustup` installs the right
  toolchain on first build.
- Network access to <https://github.com/context-graph-ai/contextdb> on the first
  build. The workspace depends on contextdb by git revision
  `46e99e2eefa20853b109ae17a60627d4d7202f5d` (`Cargo.toml`, the `contextdb-*`
  entries, locked in `Cargo.lock`); Cargo fetches it, so no separate contextdb
  checkout is needed. CI prints that locked revision next to the Context Graph
  commit it built (`.github/workflows/ci.yml`).<!-- enforced by: `manifests_lockfile_and_readme_name_the_same_exact_contextdb_revision`, `ci_workflow_validates_and_verifies_the_exact_contextdb_revision`, `readme_states_the_rust_version_rust_toolchain_toml_pins` -->

Then, from this repository's root:

```sh
cargo build --release -p context-graph-cli
```

That produces the `cg` binary at `target/release/cg`. Put it on your `PATH`, or
call it by path.<!-- enforced by: `every_readme_documented_command_is_named_in_cg_help` -->

## First run

Import a project's Claude Code conversation history, then search it:

```
$ cd ~/projects/my-app
$ cg import . --format text
No context is linked to this project.
Context name [my-app]:
context: my-app (project binding, new)
sources: 18 files / 17 sessions
history captured: 12,481 new events (0 already present)
semantic search: scheduled in the background — 12,481 events still need passage embeddings; `cg status` reports when it is ready
$ cg status --format text
$ cg find "why did we pick Iroh instead of NATS?" --format text
$ cg show <passage-id> --format text
```

The Context question is asked once, on the terminal; later commands run from the project or any
sub-directory use the same Context with no flag. Capture needs no model. Making history searchable
uses the local Nomic model — install it once with `cg model fetch --defaults`; `cg import` never
downloads anything. The store lives under `~/.cg` unless `CG_DATA_ROOT` says otherwise.<!-- enforced by: `first_interactive_project_import_accepts_suggested_name_on_enter_descendant_reads_reuse_binding_and_noninteractive_import_returns_retry_without_writes`, `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight`, `the_binary_reads_exactly_the_environment_it_declares`, `with_no_data_root_override_the_store_lives_at_dot_cg_under_home` -->

`cg status` reads the store without changing it, and a machine with no store is told so
(`store_not_found`) instead of being given one. `cg forget --session <ref>` or
`cg forget --context <name>` deletes imported history and everything derived from it. Model
connections are a separate surface: `cg model auth login --connection <provider> --consent`, then
`cg model status`.<!-- enforced by: `cg_status_is_read_only_uses_project_binding_or_explicit_context_set_and_returns_store_not_found_without_creating_either`, `forgetting_a_session_erases_its_events_and_everything_derived_while_keeping_the_context`, `forgetting_a_context_erases_all_of_its_history_and_the_next_capture_lands_in_the_same_context`, `model_auth_login_non_interactive_consent_reaches_same_enforced_record` -->

`cg --help` lists the full command surface.

## Where to read next

| You are | Read |
| --- | --- |
| Running `cg`, or embedding the crate | [`crates/context-graph/README.md`](crates/context-graph/README.md) — the `cg import`, `cg find`, `cg show`, `cg status` and `cg forget` contracts in full, choosing an embedder, reading a store you must not change, and tenant fabric configuration. It is also the crate's rendered API documentation. |
| Changing this repository | [`AGENTS.md`](AGENTS.md) — which crate a change belongs in, and the invariants a change may never break, each with the file that enforces it. |

## License

Apache-2.0. See [`LICENSE`](LICENSE).
