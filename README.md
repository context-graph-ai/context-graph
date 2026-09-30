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

You need Rust **1.94.1** via `rustup`. The repository's `rust-toolchain.toml` selects the
toolchain, and Cargo fetches the pinned ContextDB git dependency described in
[Build details](#build-details).<!-- enforced by: `readme_states_the_rust_version_rust_toolchain_toml_pins`, `manifests_lockfile_and_readme_name_the_same_exact_contextdb_revision` -->

Build from this repository's root:

```sh
cargo build --release -p context-graph-cli
```

This produces the `cg` binary at `target/release/cg`. Put it on your `PATH`, or
call it by path.<!-- enforced by: `every_readme_documented_command_is_named_in_cg_help` -->

## First run

With `cg` installed, start with a project that has Claude Code conversation history. Install the local Nomic search
model once, then import that project's history. Replace `~/projects/my-app` with your project
path; `--search wait` keeps this first import in the foreground until its search work finishes.<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets`, `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

```sh
cg model fetch --defaults
cd ~/projects/my-app
cg import . --search wait --format text
cg status --format text
cg find "why did we pick Iroh instead of NATS?" --format text
```

Use a question from your project's history. To inspect a returned passage, replace `<passage-id>`
with the handle printed by find and run `cg show <passage-id> --format text`.<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets` -->

For a project without a binding, the first interactive import prompts
`Context name [<project-basename>]:`; press Enter to accept the suggested name. Once the project
is bound, later commands from it or a sub-directory use that Context without a flag. Capture needs no model. Making history searchable
uses the local Nomic model — install it once with `cg model fetch --defaults`; `cg import` never
downloads anything. The store lives under `~/.cg` unless `CG_DATA_ROOT` says otherwise.<!-- enforced by: `first_interactive_project_import_accepts_suggested_name_on_enter_descendant_reads_reuse_binding_and_noninteractive_import_returns_retry_without_writes`, `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight`, `the_binary_reads_exactly_the_environment_it_declares`, `with_no_data_root_override_the_store_lives_at_dot_cg_under_home` -->

If status reports `blocked — local Nomic artifact is not present`, capture is retained but search
needs the model. Run `cg model fetch --defaults`, then repeat the original import with
`--search wait`. A missing-model import reports `captured_search_behind` and exits 1; `--search
defer` can capture without the model.<!-- enforced by: `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

Passage embedding uses available CPU parallelism by default. Use
`cg import <project-directory> --search wait --embedding-workers N` for a positive per-import
worker count and `[search].embedding_workers` for the node configuration; the command-line value
wins over configuration, and configuration wins over the CPU default. `--embedding-workers 1`
makes passage embedding calls serial, although it does not assign CPU affinity and Candle may
still use its own internal threads. The control changes execution only: it does not rebuild a
search generation or re-embed unchanged history. There is no `--fast` or `--slow` mode. The full
import and worker delivery contract is in the crate README.<!-- enforced by: `omitted_worker_policy_resolves_to_rust_available_parallelism`, `configured_worker_policy_is_used_when_the_cli_is_silent`, `explicit_cli_worker_policy_wins_over_configuration_in_both_directions`, `embedding_worker_cap_one_keeps_the_second_call_out_until_the_first_returns` -->

Search embedding starts with the 10 most recent sessions, newest first, while capture still keeps
every session. Use `--search-sessions N|all` to change the build scope and
`--search-order newest|oldest` to change processing order; `[search].sessions` and
`[search].order` set the data-root defaults, with command-line values taking precedence. `oldest`
selects the oldest sessions and builds them oldest first; `newest` selects and builds the newest.
Previously indexed sessions remain
searchable when the scope narrows, and widening reuses their vectors. An `all` first build can take
much longer. The outside-scope count always means captured events outside the current build
selection; because narrowing retains indexed results, JSON and text output separately report how
many of those outside-scope events are already indexed. Command-line choices apply only to that
invocation; `[search]` choices persist for the data root.<!-- enforced by: `recent_session_scope_builds_the_ten_most_recent_newest_first_and_widens_without_reembedding`, `oldest_order_builds_the_ten_oldest_sessions_oldest_first`, `search_work_policy_defaults_to_newest_ten_and_cli_overrides_config` -->

When an import prints a retry command, its `--search-sessions` and `--search-order` repeat the
resolved invocation policy. An explicit `--embedding-workers` value is repeated too; an omitted
worker value continues to come from persistent `[search]` configuration or the CPU default.<!-- enforced by: `interrupted_all_scope_retry_resumes_all_sessions_in_the_same_oldest_order`, `missing_model_retry_keeps_the_resolved_scope_without_a_hosted_or_downloaded_fallback` -->

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
| Capturing Claude Code history and preparing search | [cg-import](skills/cg-import/SKILL.md) — import project history and choose when to build search coverage. |
| Finding stored conversation passages | [cg-find](skills/cg-find/SKILL.md) — search imported history within selected Contexts. |
| Reading the source behind a search result | [cg-show](skills/cg-show/SKILL.md) — expand a passage to its stored events. |
| Inspecting capture and search readiness | [cg-status](skills/cg-status/SKILL.md) — interpret coverage and configure read memory. |
| Deleting imported history | [cg-forget](skills/cg-forget/SKILL.md) — erase a selected session or Context's history. |
| Preparing the default search model | [cg-model-fetch](skills/cg-model-fetch/SKILL.md) — fetch Nomic or adopt a complete artifact-only cache. |
| Changing this repository | [`AGENTS.md`](AGENTS.md) — which crate a change belongs in, and the invariants a change may never break, each with the file that enforces it. |

<!-- enforced by: `event_history_journey_with_real_nomic_imports_finds_shows_and_forgets`, `cg_status_uses_the_default_and_a_valid_custom_read_budget_then_refuses_an_actual_low_budget`, `forgetting_a_context_erases_all_of_its_history_and_the_next_capture_lands_in_the_same_context`, `explicit_fetch_makes_a_complete_artifact_only_cache_usable` -->

## Build details

The manifests declare [context-graph-ai/contextdb](https://github.com/context-graph-ai/contextdb)
as a git dependency and pin `5ecedbeb785800fcea8dd4ddc96b263ca8ba29e9` in `Cargo.toml` and
`Cargo.lock`. Cargo fetches that exact revision when building. The command recipes assume an
installed `cg`. CI records the locked revision beside the Context Graph commit it builds
(`.github/workflows/ci.yml`).<!-- enforced by: `manifests_lockfile_and_readme_name_the_same_exact_contextdb_revision`, `ci_workflow_validates_and_verifies_the_exact_contextdb_revision` -->

## License

Apache-2.0. See [`LICENSE`](LICENSE).
