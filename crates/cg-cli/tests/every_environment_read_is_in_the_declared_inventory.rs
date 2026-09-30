//! The `cg` binary's environment surface is what the binary DECLARES it is, checked mechanically.
//!
//! ## The promise
//!
//! An operator packaging `cg` — into a container, a service unit, a Home Assistant add-on — has to
//! know the complete list of what the binary reads from the environment, because anything on that
//! list is something their deployment either has to provision or has to make sure is absent. A
//! second promise sits under it: what a shipped `cg` reads is only ever a LOCATION. Nothing in the
//! environment decides whether something runs, or what a read may spend.
//!
//! A prose declaration alone rots the first time somebody adds an `env::var` and forgets it, and
//! the operator finds out from a deployment that behaves differently on two machines. So the
//! declaration is checked here: the set of names the source actually reads must equal the declared
//! inventory exactly, and every name in it must be named in the binary's own inventory comment.
//! Adding an environment read fails this test until the comment is updated — which is the point.
//!
//! ## The inventory
//!
//! Two LOCATIONS configure an install: `CG_DATA_ROOT` (where the data lives, read here) and
//! `CONTEXTDB_OWNER_READ_RUNTIME_DIR` (which directory the owner channel lives in, read one layer
//! down in `context_graph::owner_control` under the substrate's own name — cg invents no name of
//! its own for it and does not re-read it). `HOME` is read only to find `CG_DATA_ROOT`'s default.
//! Six development switches sit beside them; the two event-search barriers, forget barrier and status fault
//! injection are not in a shipped binary at all.
//!
//! ## What this test does not claim
//!
//! It reads source text, not a compiled binary, so it is a declaration check rather than a
//! link-time proof. It ignores whole-line comments (which is how the inventory prose itself is
//! skipped) and would miss an environment read assembled from a runtime-built name — a thing this
//! codebase does not do, and which the reviewer of any such change would have to argue for.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The complete set of environment variables the `cg` sources may read, and what each one is for.
/// A name reaching the source without reaching this list is the failure this file exists to cause.
const DECLARED_INVENTORY: [(&str, &str); 9] = [
    (
        "CG_DATA_ROOT",
        "location: where this machine's cg data lives",
    ),
    (
        "CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER",
        "development switch: test-only barrier after event-search catch-up and before handoff",
    ),
    (
        "CG_EVENT_SEARCH_TEST_PRE_CATCH_UP_BARRIER",
        "development switch: test-only barrier before event-search catch-up",
    ),
    (
        "CG_FORGET_TEST_BEFORE_PURGE_SOCKET",
        "development switch: test-only barrier after forget opens and before erasure",
    ),
    (
        "CLAUDE_CONFIG_DIR",
        "location: Claude Code's configuration directory, where its local history lives",
    ),
    (
        "HOME",
        "default source for CG_DATA_ROOT, read for nothing else",
    ),
    (
        "CG_NO_UNDERSTAND_KICK",
        "development switch: suppress the background understanding worker",
    ),
    (
        "CG_USE_TEST_LEXICAL_EMBEDDER",
        "development switch: swap in the lexical mock embedder",
    ),
    (
        "CG_FAIL_STATUS_METRIC_READ",
        "development switch: refuse one named cg status read; test builds only",
    ),
];

/// The second LOCATION input. It is deliberately NOT read by this crate: one directory has to serve
/// both halves of the owner plane, so it is read once, under the substrate's own name, in the layer
/// that owns the owner plane.
const OWNER_PLANE_LOCATION: &str = "CONTEXTDB_OWNER_READ_RUNTIME_DIR";

/// The feature that compiles the fault-injection door, and the read it guards.
const FAULT_FEATURE: &str = "status-fault-injection";
const FAULT_ENV: &str = "CG_FAIL_STATUS_METRIC_READ";

fn cg_cli_manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_sources_below(directory: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![directory.to_path_buf()];
    while let Some(next) = pending.pop() {
        let listing = std::fs::read_dir(&next)
            .unwrap_or_else(|error| panic!("read {}: {error}", next.display()));
        for entry in listing {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                found.push(path);
            }
        }
    }
    assert!(
        !found.is_empty(),
        "no Rust sources under {} — this test would pass by reading nothing",
        directory.display()
    );
    found.sort();
    found
}

/// A source line that is entirely a comment. The inventory prose names these variables in words,
/// and prose is not a read.
fn is_comment(line: &str) -> bool {
    line.trim_start().starts_with("//")
}

/// The `const NAME: &str = "VALUE";` bindings in `text`, so a read written through a named constant
/// is seen as the read it is. A parser that only understood inline literals would report a source
/// reading `env::var_os(SOME_CONST)` as reading nothing at all — which is the one way an
/// environment read could slip past this file unnoticed.
fn string_constants(text: &str) -> BTreeMap<String, String> {
    let mut found = BTreeMap::new();
    for line in text.lines() {
        if is_comment(line) {
            continue;
        }
        let trimmed = line.trim_start();
        let Some(rest) = trimmed
            .strip_prefix("const ")
            .or_else(|| trimmed.strip_prefix("pub const "))
        else {
            continue;
        };
        let Some((name, value)) = rest.split_once(": &str = ") else {
            continue;
        };
        let value = value.trim();
        let Some(literal) = value.strip_prefix('"') else {
            continue;
        };
        let Some(end) = literal.find('"') else {
            continue;
        };
        found.insert(name.trim().to_string(), literal[..end].to_string());
    }
    found
}

/// Every environment name `text` actually reads, with how many times it reads it. A read whose
/// argument is neither an inline literal nor a string constant defined in the same file is a
/// deliberate failure rather than a silent zero: it means the surface is no longer statically
/// declarable, which is itself the thing to argue about in review.
fn environment_reads(text: &str) -> BTreeMap<String, usize> {
    const CALLS: [&str; 2] = ["env::var(", "env::var_os("];
    let constants = string_constants(text);
    let mut found: BTreeMap<String, usize> = BTreeMap::new();
    for line in text.lines() {
        if is_comment(line) {
            continue;
        }
        for call in CALLS {
            let mut rest = line;
            while let Some(index) = rest.find(call) {
                let after = &rest[index + call.len()..];
                let end = after
                    .find(')')
                    .unwrap_or_else(|| panic!("an unterminated environment read in: {line}"));
                let argument = after[..end].trim();
                let name = match argument.strip_prefix('"') {
                    Some(literal) => literal
                        .find('"')
                        .map(|close| literal[..close].to_string())
                        .unwrap_or_else(|| panic!("an unterminated environment name in: {line}")),
                    None => constants.get(argument).cloned().unwrap_or_else(|| {
                        panic!(
                            "an environment read whose name this file cannot resolve statically: \
                             `{argument}` in: {line}"
                        )
                    }),
                };
                *found.entry(name).or_default() += 1;
                rest = &after[end..];
            }
        }
    }
    found
}

fn cg_cli_environment_reads() -> BTreeMap<String, usize> {
    let mut found: BTreeMap<String, usize> = BTreeMap::new();
    for source in rust_sources_below(&cg_cli_manifest_dir().join("src")) {
        let text = std::fs::read_to_string(&source)
            .unwrap_or_else(|error| panic!("read {}: {error}", source.display()));
        for (name, count) in environment_reads(&text) {
            *found.entry(name).or_default() += count;
        }
    }
    found
}

#[test]
fn the_binary_reads_exactly_the_environment_it_declares() {
    let read = cg_cli_environment_reads();
    let declared: BTreeSet<String> = DECLARED_INVENTORY
        .iter()
        .map(|(name, _)| (*name).to_string())
        .collect();
    let actually_read: BTreeSet<String> = read.keys().cloned().collect();

    let undeclared: Vec<&String> = actually_read.difference(&declared).collect();
    assert!(
        undeclared.is_empty(),
        "`cg` reads {undeclared:?} from the environment and does not declare it. An operator \
         packaging this binary cannot provision — or deliberately withhold — a name nobody told \
         them about. Add it to DECLARED_INVENTORY here AND to the inventory comment on \
         `data_root`, and say which of the two kinds it is: a LOCATION the install is configured \
         with, or a development switch. If it decides whether something RUNS or what a read may \
         SPEND, it is neither, and it does not belong in the environment at all."
    );

    let unread: Vec<&String> = declared.difference(&actually_read).collect();
    assert!(
        unread.is_empty(),
        "the inventory declares {unread:?}, which nothing reads any more. A stale entry is as \
         misleading as a missing one: it tells an operator to provision something that does \
         nothing."
    );

    // Named explicitly so the arithmetic above cannot pass on a coincidence.
    assert_eq!(
        read.keys().cloned().collect::<Vec<_>>(),
        vec![
            "CG_DATA_ROOT".to_string(),
            "CG_EVENT_SEARCH_TEST_HANDOFF_BARRIER".to_string(),
            "CG_EVENT_SEARCH_TEST_PRE_CATCH_UP_BARRIER".to_string(),
            FAULT_ENV.to_string(),
            "CG_FORGET_TEST_BEFORE_PURGE_SOCKET".to_string(),
            "CG_NO_UNDERSTAND_KICK".to_string(),
            "CG_USE_TEST_LEXICAL_EMBEDDER".to_string(),
            "CLAUDE_CONFIG_DIR".to_string(),
            "HOME".to_string(),
        ],
        "the whole environment surface of the cg binary, with read counts {read:?}"
    );
}

#[test]
fn the_owner_plane_location_is_read_once_in_the_layer_that_owns_it() {
    let read = cg_cli_environment_reads();
    assert!(
        !read.contains_key(OWNER_PLANE_LOCATION),
        "`{OWNER_PLANE_LOCATION}` must not be read here. One directory has to serve BOTH halves of \
         the owner plane, so it is read once in the layer that owns that plane; a second read in \
         the CLI is a second place for the two halves to disagree."
    );

    let owner_control = cg_cli_manifest_dir()
        .parent()
        .expect("the crates directory")
        .join("context-graph/src/owner_control.rs");
    let text = std::fs::read_to_string(&owner_control)
        .unwrap_or_else(|error| panic!("read {}: {error}", owner_control.display()));
    assert_eq!(
        environment_reads(&text).get(OWNER_PLANE_LOCATION),
        Some(&1),
        "`{OWNER_PLANE_LOCATION}` is read exactly once, in {}, so the owner's channel and a \
         reader's search for it come from one value: {:?}",
        owner_control.display(),
        environment_reads(&text)
    );
}

#[test]
fn the_inventory_comment_names_every_variable_the_binary_reads() {
    let main = cg_cli_manifest_dir().join("src/main.rs");
    let text = std::fs::read_to_string(&main).expect("read the cg entry point");
    let prose: String = text
        .lines()
        .filter(|line| is_comment(line))
        .collect::<Vec<_>>()
        .join("\n");

    for (name, purpose) in DECLARED_INVENTORY {
        assert!(
            prose.contains(name),
            "`{name}` ({purpose}) is read but never named in the binary's own inventory prose. A \
             reader of the source has to be able to find the whole list where the list is \
             declared, not by grepping."
        );
    }
    assert!(
        prose.contains(OWNER_PLANE_LOCATION),
        "the inventory names the SECOND location input too, even though it is read one layer \
         down — an operator provisioning a container needs both locations from one place."
    );
}

#[test]
fn the_fault_injection_read_is_compiled_out_of_every_build_that_is_not_a_test_build() {
    let main = cg_cli_manifest_dir().join("src/main.rs");
    let text = std::fs::read_to_string(&main).expect("read the cg entry point");

    let lines: Vec<&str> = text.lines().collect();
    let reads: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| !is_comment(line) && line.contains(&format!("\"{FAULT_ENV}\"")))
        .map(|(index, _)| index)
        .collect();
    assert_eq!(
        reads.len(),
        1,
        "the fault-injection variable is read in exactly one place, so one `#[cfg]` decides \
         whether it exists at all; found it at lines {reads:?}"
    );

    // Walk back to the function the read sits in, then to the attributes above it.
    let read_at = reads[0];
    let function_at = (0..read_at)
        .rev()
        .find(|index| lines[*index].starts_with("fn ") || lines[*index].starts_with("pub fn "))
        .expect("the read sits inside a top-level function");
    let guard = (0..function_at)
        .rev()
        .take_while(|index| lines[*index].starts_with("#["))
        .any(|index| lines[index].contains(&format!("feature = \"{FAULT_FEATURE}\"")));
    assert!(
        guard,
        "the function reading `{FAULT_ENV}` (line {}) must be gated on \
         `#[cfg(feature = \"{FAULT_FEATURE}\")]`. Without that gate the variable is present in a \
         developer's ordinary `cargo build` binary, where a stray export would silently change \
         what `cg status` reports.",
        function_at + 1
    );
}

#[test]
fn the_fault_injection_feature_is_turned_on_by_nothing_but_this_package_testing_itself() {
    let manifest_path = cg_cli_manifest_dir().join("Cargo.toml");
    let manifest = std::fs::read_to_string(&manifest_path).expect("read the cg-cli manifest");

    let mut section = String::new();
    let mut declared_in_features = false;
    let mut enabling_lines: Vec<(String, String)> = Vec::new();
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            section = trimmed.to_string();
            continue;
        }
        if trimmed.starts_with('#') || !trimmed.contains(FAULT_FEATURE) {
            continue;
        }
        if section == "[features]" && trimmed.starts_with(&format!("{FAULT_FEATURE} =")) {
            declared_in_features = true;
            assert_eq!(
                trimmed,
                format!("{FAULT_FEATURE} = []"),
                "the fault-injection feature turns nothing else on"
            );
            continue;
        }
        enabling_lines.push((section.clone(), trimmed.to_string()));
    }

    assert!(
        declared_in_features,
        "`{FAULT_FEATURE}` must be a declared feature of this package"
    );
    assert!(
        !manifest.contains("default = ["),
        "this package declares no default features, so the fault-injection door cannot be reached \
         by a plain `cargo build`"
    );
    assert_eq!(
        enabling_lines.len(),
        1,
        "exactly one place may turn the fault-injection door on; found {enabling_lines:?}"
    );
    let (section, line) = &enabling_lines[0];
    assert_eq!(
        section, "[dev-dependencies]",
        "the door is turned on ONLY by a dev-dependency, which is the boundary that keeps it out \
         of `cargo build` and `cargo install`; found it under {section} in `{line}`"
    );
    assert!(
        line.starts_with("context-graph-cli = { path = \".\""),
        "the dev-dependency that turns the door on is this package on ITSELF — a different \
         package enabling it would put the door in something that ships: `{line}`"
    );
}
