//! Test-estate discipline audit (see `AGENTS.md` "Fast test command and gate").
//!
//! ## The promise, in product terms
//!
//! A test suite erodes two ways nobody notices at the time: a new test synchronizes with
//! `sleep` instead of state (flaky on a slow machine, or silently not proving what it claims),
//! and a test quietly disappears (a refactor drops the file, a merge loses a module) while
//! everything still shows green. This audit turns both into a build failure instead of a slow
//! leak, with a mechanism that only ever tightens: a contributor removing a sleep site or adding
//! tests edits the applicable count in the same change, while a change that regresses cannot
//! silently raise those count baselines.
//!
//! ## The three arms
//!
//! 1. **Sleep and raw-clock sites.** `[sleep_and_clock_sites]` in
//!    `tests/fixtures/test_estate_baseline.toml` freezes, per file, the current count of
//!    `std::thread::sleep(`, `tokio::time::sleep(`, `SystemTime::now(`, `Instant::now(`, and
//!    `Utc::now(` call sites anywhere under `crates/*/src` or `crates/*/tests` — every one of
//!    them a persisted-timestamp or synchronization call that bypasses the seamed
//!    `contextdb_core::Wallclock::now()`. A file's live count rising past its baseline, or a new
//!    file carrying any of these calls at all, fails the audit. The fix is the seam
//!    (`Wallclock::test_clock_guard` for time, a counter/state/event wait for synchronization),
//!    never raising the number.
//! 2. **Default-tier test count per crate.** `[default_tier_test_count]` freezes, per crate, the
//!    exact number of `#[test]`/`#[tokio::test]` functions that run in the tier `cargo test` runs
//!    by default — excluding anything gated behind
//!    `#[cfg(context_graph_internal_test_support)]` and anything carrying `#[ignore]`. The entry
//!    moves with the estate in the same change: a change that adds tests raises the crate's entry
//!    to match in the same change, so the ratchet protects the tests it just added; a change that
//!    deliberately removes tests lowers the entry in the same change and says why. A live count
//!    that differs from its baseline in either direction without that same-change update fails
//!    the audit and reports the live count next to the frozen one — the mechanical version of
//!    "did this change quietly delete tests" and "did this change add tests nothing protects."
//!    The comparison runs over the UNION of the crates the tree ships and the rows the table
//!    carries, so a crate with no row is a failure rather than a crate quietly outside the
//!    ratchet, and deleting a row is not a way to drop a crate out of it. A crate with no
//!    default-tier tests declares `0` rather than omitting its row.
//! 3. **Gate wall-clock, compared and refused on exceed.** `[gate_wall_seconds]` records, per
//!    test binary, the wall-clock seconds captured from a gate's own junit; `[gate_wall_seconds_policy]`
//!    declares `tolerance_factor` and `slack_seconds`, and a binary's ceiling is
//!    `recorded seconds * tolerance_factor + slack_seconds`. Both terms are needed: the factor
//!    covers drift that scales with the binary, and the slack covers the part that does not — one
//!    process spawn per test, and a first file read a later run finds already cached — which on a
//!    binary whose whole run is a few spawns is many times the binary's own seconds. It is also
//!    what lets a measurement taken in one test pool be compared against a value frozen from a
//!    differently-sized one, which is not otherwise a like-for-like number. The comparison never reads a clock itself — it reads wall
//!    seconds someone else already measured, from the JUnit file named by `CG_GATE_JUNIT`, summing
//!    each binary's `<testcase>` seconds exactly the way the frozen values were derived. The
//!    ignored live arm (`gate_wall_seconds_never_exceed_the_frozen_baseline`, run in the gate step
//!    with `CG_GATE_JUNIT` set) fails when a binary exceeds that ceiling,
//!    and also when a binary this gate ran carries no row at all — an unratcheted binary has no
//!    ceiling, and deleting a row is how one loses its ceiling silently. A measurement that parses
//!    to no binaries at all is refused outright rather than read as a pass. It is `#[ignore]`d so a
//!    bare `cargo test --workspace` — which produces no fresh gate JUnit of its own — stays green
//!    rather than silently passing an ungiven measurement.
//!
//! Every baseline value here was captured on 2026-09-04 from this tree; the sleep/clock
//! counts and default-tier counts were derived by the exact algorithms this file implements
//! (source-level scanning, not a live `cargo test` run), so re-running this audit against an
//! unchanged tree reproduces them exactly.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Test/src files that never compile in the default tier at all: each is reached only through a
/// `mod` declaration in its crate's `tests/acceptance/main.rs` or `tests/integration/main.rs`
/// that itself carries `#[cfg(context_graph_internal_test_support)]`. Their contents — including
/// any `#[test]` functions and any sleep/clock call sites — are excluded from both counted arms
/// for the same reason: code that does not compile by default is not part of the default tier to
/// begin with, not a debt this audit is choosing to forgive.
const WHOLE_FILE_EXCLUDE: &[&str] = &[
    "crates/context-graph/tests/acceptance/local_entity_face_match_override.rs",
    "crates/context-graph/tests/acceptance/screenshot_memory_distinguishability.rs",
    "crates/context-graph/tests/acceptance/screenshot_memory_review.rs",
    "crates/context-graph/tests/integration/composite_query_edge_cases.rs",
    "crates/context-graph/tests/integration/cross_context_label_filter.rs",
    "crates/context-graph/tests/integration/cross_space_refusal.rs",
    "crates/context-graph/tests/integration/engine_probes/embedder_concurrency.rs",
    "crates/context-graph/tests/integration/explain_introspection.rs",
    "crates/context-graph/tests/integration/intra_space_modality_agnostic.rs",
    "crates/context-graph/tests/integration/reindex_concurrency.rs",
    "crates/context-graph/tests/integration/store_open_runtime_config.rs",
];

const SLEEP_CLOCK_NEEDLES: &[&str] = &[
    "std::thread::sleep(",
    "tokio::time::sleep(",
    "SystemTime::now(",
    "Instant::now(",
    "Utc::now(",
];

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate sits two levels below the repository root")
        .to_path_buf()
}

/// This audit's own source and its baseline fixture are excluded from every scan below: they
/// name the needle strings and attribute literals as data, not as call sites or test
/// declarations.
fn is_self(rel: &str) -> bool {
    rel == "crates/context-graph/tests/test_estate_audit.rs"
        || rel == "crates/context-graph/tests/fixtures/test_estate_baseline.toml"
}

fn crate_dirs(root: &Path) -> Vec<(String, PathBuf)> {
    let crates_dir = root.join("crates");
    let mut out = Vec::new();
    let entries = std::fs::read_dir(&crates_dir).expect("crates/ directory must exist");
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            out.push((name, path));
        }
    }
    out.sort();
    out
}

/// The Cargo package name for a crate directory, read from the `name = "..."` line inside its
/// `Cargo.toml`'s `[package]` table (falls back to the directory name if that line is somehow
/// missing, so a malformed manifest fails loudly elsewhere rather than silently here).
fn package_name(crate_dir: &Path) -> String {
    let manifest_path = crate_dir.join("Cargo.toml");
    let manifest = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", manifest_path.display()));
    let mut in_package = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed == "[package]" {
            in_package = true;
            continue;
        }
        if trimmed.starts_with('[') {
            in_package = false;
            continue;
        }
        if in_package {
            let name_value = trimmed
                .strip_prefix("name")
                .and_then(|rest| rest.trim_start().strip_prefix('='));
            if let Some(rest) = name_value {
                return rest.trim().trim_matches('"').to_string();
            }
        }
    }
    crate_dir.file_name().unwrap().to_string_lossy().to_string()
}

fn walk_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .expect("path must live under the repository root")
        .to_string_lossy()
        .replace('\\', "/")
}

// ---------------------------------------------------------------------------------------------
// Arm 1: sleep and raw-clock sites
// ---------------------------------------------------------------------------------------------

fn count_sleep_clock_sites(root: &Path) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, crate_dir) in crate_dirs(root) {
        for sub in ["src", "tests"] {
            let mut files = Vec::new();
            walk_rs_files(&crate_dir.join(sub), &mut files);
            for file in files {
                let rel = relative(root, &file);
                if is_self(&rel) || WHOLE_FILE_EXCLUDE.contains(&rel.as_str()) {
                    continue;
                }
                let Ok(content) = std::fs::read_to_string(&file) else {
                    continue;
                };
                let total: usize = SLEEP_CLOCK_NEEDLES
                    .iter()
                    .map(|needle| content.matches(needle).count())
                    .sum();
                if total > 0 {
                    counts.insert(rel, total);
                }
            }
        }
    }
    counts
}

#[test]
fn no_new_sleep_or_raw_clock_sites_beyond_the_frozen_baseline() {
    let root = repository_root();
    let baseline = load_baseline(&root);
    let live = count_sleep_clock_sites(&root);

    let mut violations = Vec::new();
    for (file, count) in &live {
        let allowed = baseline
            .sleep_and_clock_sites
            .get(file)
            .copied()
            .unwrap_or(0);
        if *count > allowed {
            violations.push(format!(
                "{file}: {count} sleep/raw-clock site(s), baseline allows {allowed} — route \
                 through Wallclock::test_clock_guard (timing) or a counter/state/event wait \
                 (synchronization), never raise the baseline"
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "test-estate audit: new sleep or raw-clock site(s) beyond \
         tests/fixtures/test_estate_baseline.toml:\n{}",
        violations.join("\n")
    );
}

// ---------------------------------------------------------------------------------------------
// Arm 2: default-tier test count per crate
// ---------------------------------------------------------------------------------------------

/// An attribute chain accumulated line-by-line above a candidate `fn`. Blank lines between
/// attributes and the item are tolerated (matches how `rustfmt` sometimes leaves them); any
/// other non-attribute, non-blank line resets the chain, since it means the attributes seen so
/// far decorated some earlier item, not this `fn`.
fn count_default_tier_tests_in_file(content: &str) -> usize {
    let mut count = 0usize;
    let mut chain: Vec<String> = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(inner) = strip_attribute(trimmed) {
            chain.push(inner.to_string());
            continue;
        }
        if trimmed.is_empty() {
            continue;
        }
        if is_fn_line(trimmed) {
            let is_test = chain.iter().any(|attribute| {
                attribute == "test"
                    || attribute == "tokio::test"
                    || attribute.starts_with("tokio::test(")
            });
            if is_test {
                let excluded = chain.iter().any(|a| {
                    a == "cfg(context_graph_internal_test_support)" || a.starts_with("ignore")
                });
                if !excluded {
                    count += 1;
                }
            }
            chain.clear();
            continue;
        }
        chain.clear();
    }
    count
}

fn strip_attribute(trimmed: &str) -> Option<&str> {
    let inner = trimmed
        .strip_prefix("#[")
        .or_else(|| trimmed.strip_prefix("#!["))?;
    let inner = inner.strip_suffix(']')?;
    Some(inner.trim())
}

fn is_fn_line(trimmed: &str) -> bool {
    let after_pub = trimmed
        .strip_prefix("pub(crate) ")
        .or_else(|| trimmed.strip_prefix("pub "))
        .unwrap_or(trimmed);
    let after_async = after_pub.strip_prefix("async ").unwrap_or(after_pub);
    after_async.starts_with("fn ")
}

fn count_default_tier_tests(root: &Path) -> BTreeMap<String, usize> {
    let mut per_crate = BTreeMap::new();
    for (_dir_name, crate_dir) in crate_dirs(root) {
        let pkg = package_name(&crate_dir);
        let mut total = 0usize;
        for sub in ["src", "tests"] {
            let mut files = Vec::new();
            walk_rs_files(&crate_dir.join(sub), &mut files);
            for file in files {
                let rel = relative(root, &file);
                if is_self(&rel) || WHOLE_FILE_EXCLUDE.contains(&rel.as_str()) {
                    continue;
                }
                let Ok(content) = std::fs::read_to_string(&file) else {
                    continue;
                };
                total += count_default_tier_tests_in_file(&content);
            }
        }
        per_crate.insert(pkg, total);
    }
    per_crate
}

#[test]
fn test_estate_audit_counts_parameterized_tokio_tests_and_detects_deletion() {
    let parameterized_tokio_test = r#"
        #[tokio::test(flavor = "current_thread")]
        async fn parameterized_tokio_test_stays_in_the_default_tier() {}
    "#;
    let counted = count_default_tier_tests_in_file(parameterized_tokio_test);
    assert_eq!(
        counted, 1,
        "a parameterized #[tokio::test(...)] is a default-tier test and must be counted"
    );

    let ignored_parameterized_tokio_test = r#"
        #[ignore = "manual only"]
        #[tokio::test(flavor = "current_thread")]
        async fn ignored_parameterized_tokio_test_stays_out_of_the_default_tier() {}
    "#;
    assert_eq!(
        count_default_tier_tests_in_file(ignored_parameterized_tokio_test),
        0,
        "an ignored parameterized Tokio test must remain outside the default tier"
    );

    let internal_parameterized_tokio_test = r#"
        #[cfg(context_graph_internal_test_support)]
        #[tokio::test(flavor = "current_thread")]
        async fn internal_parameterized_tokio_test_stays_out_of_the_default_tier() {}
    "#;
    assert_eq!(
        count_default_tier_tests_in_file(internal_parameterized_tokio_test),
        0,
        "an internal-only parameterized Tokio test must remain outside the default tier"
    );

    let after_deletion = count_default_tier_tests_in_file("");
    assert!(
        after_deletion < counted,
        "deleting a counted parameterized Tokio test must drop below the frozen baseline"
    );
}

/// Compares the live per-crate counts against the frozen ones over the UNION of both sides, so a
/// crate the tree ships with no `[default_tier_test_count]` row is a finding rather than a silent
/// exemption. Iterating the frozen table alone leaves a never-added row unprotected from birth and
/// makes deleting a row the way to drop a whole crate out of the ratchet.
fn default_tier_count_violations(
    live: &BTreeMap<String, usize>,
    baseline: &BTreeMap<String, usize>,
) -> Vec<String> {
    let mut violations = Vec::new();
    let mut crates: Vec<&String> = live.keys().chain(baseline.keys()).collect();
    crates.sort_unstable();
    crates.dedup();
    for crate_name in crates {
        let live_count = live.get(crate_name).copied();
        let baseline_count = baseline.get(crate_name).copied();
        match (live_count, baseline_count) {
            (Some(live_count), None) => violations.push(format!(
                "{crate_name}: {live_count} default-tier test(s) and no \
                 [default_tier_test_count] row — a crate freezes its count in the same change it \
                 ships, and a crate with no row can lose every test it has without failing \
                 anything; a crate with no tests declares 0 rather than omitting the row"
            )),
            (None, Some(baseline_count)) => violations.push(format!(
                "{crate_name}: frozen at {baseline_count} default-tier test(s) but the crate is \
                 no longer in the tree — remove its row in the same change that removes the crate"
            )),
            (Some(live_count), Some(baseline_count)) if live_count < baseline_count => violations
                .push(format!(
                    "{crate_name}: {live_count} default-tier test(s), baseline requires exactly \
                     {baseline_count} — a change that deliberately removes tests lowers the \
                     baseline in the same change and says why; one that does not means tests were \
                     lost"
                )),
            (Some(live_count), Some(baseline_count)) if live_count > baseline_count => violations
                .push(format!(
                    "{crate_name}: {live_count} default-tier test(s), baseline requires exactly \
                     {baseline_count} — a change that adds tests raises the baseline entry in the \
                     same change so the ratchet protects the tests it just added"
                )),
            _ => {}
        }
    }
    violations
}

#[test]
fn default_tier_test_count_moves_with_the_estate_in_the_same_change() {
    let root = repository_root();
    let baseline = load_baseline(&root);
    let live = count_default_tier_tests(&root);
    let violations = default_tier_count_violations(&live, &baseline.default_tier_test_count);
    assert!(
        violations.is_empty(),
        "test-estate audit: default-tier test count does not match \
         tests/fixtures/test_estate_baseline.toml:\n{}",
        violations.join("\n")
    );
}

/// The count ratchet is only a ratchet if a crate cannot step outside it. This drives the three
/// ways that happens — a crate the tree ships with no frozen row, a frozen row for a crate that is
/// gone, and a count that moved in either direction — against the comparison itself, so none of
/// them can regress into a silent pass.
#[test]
fn a_crate_outside_the_frozen_count_table_is_reported_rather_than_silently_unratcheted() {
    let baseline: BTreeMap<String, usize> = [
        ("kept-crate".to_string(), 12),
        ("shrunk-crate".to_string(), 9),
        ("grown-crate".to_string(), 3),
        ("removed-crate".to_string(), 4),
    ]
    .into_iter()
    .collect();
    let live: BTreeMap<String, usize> = [
        ("kept-crate".to_string(), 12),
        ("shrunk-crate".to_string(), 8),
        ("grown-crate".to_string(), 5),
        ("brand-new-crate".to_string(), 40),
    ]
    .into_iter()
    .collect();

    let violations = default_tier_count_violations(&live, &baseline);
    assert_eq!(
        violations.len(),
        4,
        "a shrunk crate, a grown crate, a crate with no frozen row and a frozen row with no crate \
         are all findings, got: {violations:?}"
    );
    let joined = violations.join("\n");
    assert!(
        joined.contains("brand-new-crate"),
        "a crate the tree ships with no [default_tier_test_count] row must be reported — \
         iterating only the frozen table leaves it unratcheted from birth, got: {violations:?}"
    );
    assert!(
        joined.contains("removed-crate"),
        "a frozen row whose crate is gone must be reported, so deleting a crate cannot leave a \
         stale ceiling behind, got: {violations:?}"
    );
    assert!(joined.contains("shrunk-crate") && joined.contains("grown-crate"));
    assert!(
        !joined.contains("kept-crate"),
        "a crate whose live count equals its frozen row is not a finding, got: {violations:?}"
    );

    assert!(
        default_tier_count_violations(&baseline, &baseline).is_empty(),
        "a tree that matches its frozen table exactly reports nothing"
    );
}

// ---------------------------------------------------------------------------------------------
// Arm 3: gate wall-clock per binary — compares against the frozen baseline and refuses on
// exceed. The comparison itself never reads a clock: it consumes wall seconds already captured
// in a JUnit file produced by an earlier gate step (`CG_GATE_JUNIT`), never `Instant::now()` or
// any other live timer.
// ---------------------------------------------------------------------------------------------

#[test]
fn gate_wall_seconds_baseline_is_well_formed() {
    let root = repository_root();
    let baseline = load_baseline(&root);
    assert!(
        !baseline.gate_wall_seconds.is_empty(),
        "the [gate_wall_seconds] baseline must record at least one test binary"
    );
    for (binary, seconds) in &baseline.gate_wall_seconds {
        assert!(
            seconds.is_finite() && *seconds >= 0.0,
            "{binary}: recorded wall seconds must be a finite, non-negative number, got \
             {seconds}"
        );
    }
    assert!(
        baseline.gate_wall_seconds_tolerance_factor.is_finite()
            && baseline.gate_wall_seconds_tolerance_factor >= 1.0,
        "[gate_wall_seconds_policy] tolerance_factor must be a finite number >= 1.0, got {}",
        baseline.gate_wall_seconds_tolerance_factor
    );
    assert!(
        baseline.gate_wall_seconds_slack.is_finite() && baseline.gate_wall_seconds_slack >= 0.0,
        "[gate_wall_seconds_policy] slack_seconds must be a finite, non-negative number, got {}",
        baseline.gate_wall_seconds_slack
    );
}

/// Wall seconds per test binary, read out of a JUnit XML document the way `cargo nextest` writes
/// one: it opens a `<testsuite name="crate::binary" ...>` element carrying NO `time` attribute of
/// its own, and records the measured seconds on each `<testcase ... time="..">` inside it. So a
/// binary's wall seconds are the sum of its testcases' seconds, which is exactly how the frozen
/// `[gate_wall_seconds]` values were derived. Reading a `time` attribute off the `<testsuite>` tag
/// instead finds nothing in a real nextest document and yields an empty map, which is a comparison
/// against nothing rather than a measurement.
///
/// Deliberately not a general XML parser: this audit only ever reads wall seconds someone else
/// already measured and wrote to a file, never a clock of its own.
fn parse_testsuite_wall_seconds(junit_xml: &str) -> BTreeMap<String, f64> {
    let mut out: BTreeMap<String, f64> = BTreeMap::new();
    let mut current_suite: Option<String> = None;
    for line in junit_xml.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("<testsuite ") {
            current_suite = extract_attr(trimmed, "name");
            if let Some(name) = &current_suite {
                out.entry(name.clone()).or_insert(0.0);
            }
            continue;
        }
        if trimmed.starts_with("</testsuite") {
            current_suite = None;
            continue;
        }
        if !trimmed.starts_with("<testcase ") {
            continue;
        }
        let (Some(name), Some(seconds)) = (
            current_suite.as_ref(),
            extract_attr(trimmed, "time").and_then(|value| value.parse::<f64>().ok()),
        ) else {
            continue;
        };
        *out.entry(name.clone()).or_insert(0.0) += seconds;
    }
    out
}

fn extract_attr(tag: &str, attr: &str) -> Option<String> {
    let needle = format!("{attr}=\"");
    let start = tag.find(&needle)? + needle.len();
    let rest = &tag[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// A binary's ceiling: `recorded seconds * tolerance_factor + slack_seconds`.
///
/// The factor covers drift that scales with the binary — the same tests against a busier machine.
/// The slack covers the part that does not scale: one process spawn per test under nextest, and a
/// first file read that a later run finds in the page cache. On a binary whose whole run is a
/// handful of spawns that fixed cost is many times the binary's own seconds, so a purely relative
/// ceiling reports the machine's scheduling as a runtime regression. It is also what makes a
/// measurement taken in one test pool comparable against a value frozen from a differently-sized
/// pool, which is otherwise a different amount of contention over the same cores and disk.
///
/// Same rule as the check beside it in `scripts/check-gate-wall-clock.py`; the two are updated
/// together.
fn wall_clock_ceiling(baseline_seconds: f64, tolerance_factor: f64, slack_seconds: f64) -> f64 {
    baseline_seconds * tolerance_factor + slack_seconds
}

/// REJECTS every binary present in both `live` and `baseline` whose live wall seconds exceed its
/// ceiling; ACCEPTS one within that ceiling; ignores a binary absent from either side (a binary
/// the gate didn't run, or one the baseline hasn't captured yet, is not this arm's finding).
fn wall_clock_violations(
    live: &BTreeMap<String, f64>,
    baseline: &BTreeMap<String, f64>,
    tolerance_factor: f64,
    slack_seconds: f64,
) -> Vec<String> {
    let mut violations = Vec::new();
    for (binary, baseline_seconds) in baseline {
        let Some(live_seconds) = live.get(binary) else {
            continue;
        };
        let ceiling = wall_clock_ceiling(*baseline_seconds, tolerance_factor, slack_seconds);
        if *live_seconds > ceiling {
            violations.push(format!(
                "{binary}: {live_seconds:.3}s exceeds the frozen baseline {baseline_seconds:.3}s \
                 x tolerance_factor {tolerance_factor} + slack {slack_seconds:.3}s = \
                 {ceiling:.3}s ceiling — lower the baseline only for a proven, deliberate change; \
                 otherwise this is a runtime regression"
            ));
        }
    }
    violations
}

/// REPORTS every binary this gate actually ran that carries no `[gate_wall_seconds]` row. A row
/// that was never added, or one that was deleted, gives that binary an unbounded runtime ceiling
/// while every direction check still prints a clean pass — so completeness is checked from the
/// LIVE side, never from the frozen table.
fn wall_clock_coverage_violations(
    live: &BTreeMap<String, f64>,
    baseline: &BTreeMap<String, f64>,
) -> Vec<String> {
    live.keys()
        .filter(|binary| !baseline.contains_key(*binary))
        .map(|binary| {
            format!(
                "{binary}: ran in this gate with no [gate_wall_seconds] row — record its measured \
                 wall seconds in the same change; an unratcheted binary has no ceiling at all, and \
                 deleting a row is how a binary loses one silently"
            )
        })
        .collect()
}

#[test]
fn wall_clock_comparison_rejects_a_binary_over_tolerance_and_accepts_one_within_it() {
    // Both fixtures are shaped the way cargo nextest actually writes JUnit: the `<testsuite>` tag
    // carries no `time` of its own and the seconds live on each `<testcase>`. A parser that reads
    // a `time` attribute off `<testsuite>` finds nothing here, which is the whole point — it would
    // then compare an empty live map against the baseline and report a clean pass forever.
    let baseline_junit = r#"<?xml version="1.0"?>
<testsuites name="nextest-run" tests="4">
    <testsuite name="crate::binary" tests="2" disabled="0" errors="0" failures="0">
        <testcase name="one" classname="crate::binary" time="6.0">
        </testcase>
        <testcase name="two" classname="crate::binary" time="4.0">
        </testcase>
    </testsuite>
    <testsuite name="crate::other" tests="2" disabled="0" errors="0" failures="0">
        <testcase name="one" classname="crate::other" time="2.0">
        </testcase>
        <testcase name="two" classname="crate::other" time="3.0">
        </testcase>
    </testsuite>
</testsuites>
"#;
    let live_junit = r#"<?xml version="1.0"?>
<testsuites name="nextest-run" tests="5">
    <testsuite name="crate::binary" tests="2" disabled="0" errors="0" failures="0">
        <testcase name="one" classname="crate::binary" time="20.5">
        </testcase>
        <testcase name="two" classname="crate::binary" time="0.5">
        </testcase>
    </testsuite>
    <testsuite name="crate::other" tests="2" disabled="0" errors="0" failures="0">
        <testcase name="one" classname="crate::other" time="2.5">
        </testcase>
        <testcase name="two" classname="crate::other" time="3.0">
        </testcase>
    </testsuite>
    <testsuite name="crate::absent_from_baseline" tests="1" disabled="0" errors="0" failures="0">
        <testcase name="one" classname="crate::absent_from_baseline" time="999.0">
        </testcase>
    </testsuite>
</testsuites>
"#;
    let baseline = parse_testsuite_wall_seconds(baseline_junit);
    let live = parse_testsuite_wall_seconds(live_junit);
    assert_eq!(
        baseline.get("crate::binary").copied(),
        Some(10.0),
        "a binary's wall seconds are the sum of its testcases' seconds, the way the frozen table \
         was derived; a parser reading the `<testsuite>` tag instead sees nothing here"
    );
    assert_eq!(baseline.get("crate::other").copied(), Some(5.0));
    assert_eq!(
        baseline.len(),
        2,
        "fixture parsing must find both baseline binaries"
    );
    assert_eq!(live.get("crate::binary").copied(), Some(21.0));
    assert_eq!(
        live.len(),
        3,
        "fixture parsing must find all three live binaries"
    );

    let violations = wall_clock_violations(&live, &baseline, 2.0, 0.0);
    assert_eq!(
        violations.len(),
        1,
        "only crate::binary (21.0s > 10.0s baseline x 2.0 tolerance = 20.0s) should be \
         reported; crate::other is within tolerance and crate::absent_from_baseline is the \
         coverage arm's finding, not this one, got: {violations:?}"
    );
    assert!(
        violations[0].contains("crate::binary"),
        "the one violation must name crate::binary, got: {violations:?}"
    );

    let coverage = wall_clock_coverage_violations(&live, &baseline);
    assert_eq!(
        coverage.len(),
        1,
        "a binary that ran with no frozen row must be reported, otherwise deleting its row is a \
         way to buy it an unbounded ceiling, got: {coverage:?}"
    );
    assert!(
        coverage[0].contains("crate::absent_from_baseline"),
        "the coverage violation must name the unratcheted binary, got: {coverage:?}"
    );
}

/// The slack exists because a relative tolerance is a ceiling measured in the binary's own
/// seconds: on a binary whose entire run is a few process spawns, `x2` is smaller than one spawn,
/// so the arm reports the machine's scheduling instead of the binary's runtime. This drives both
/// halves of that rule, because an allowance that only ever spares things is a check that can no
/// longer fail: a startup-sized overshoot inside the slack is spared, the SAME binary past it is
/// still reported, and a binary big enough to have a real ceiling is untouched by it.
#[test]
fn the_slack_term_spares_a_startup_sized_binary_without_sparing_a_real_regression() {
    let baseline: BTreeMap<String, f64> = [
        ("crate::three_trivial_tests".to_string(), 0.019),
        ("crate::real_work".to_string(), 10.0),
    ]
    .into_iter()
    .collect();
    let slack = 0.235;

    assert_eq!(
        wall_clock_ceiling(0.019, 2.0, slack),
        0.019 * 2.0 + slack,
        "a 0.019s row's ceiling is its x2 allowance plus the slack, which is what a few process \
         spawns actually cost"
    );
    assert_eq!(
        wall_clock_ceiling(10.0, 2.0, slack),
        20.0 + slack,
        "a 10.0s row keeps a ceiling the slack barely moves: an absolute allowance cannot loosen \
         a binary that already has a meaningful relative one"
    );

    let spared: BTreeMap<String, f64> = [
        ("crate::three_trivial_tests".to_string(), 0.260),
        ("crate::real_work".to_string(), 20.2),
    ]
    .into_iter()
    .collect();
    assert!(
        wall_clock_violations(&spared, &baseline, 2.0, slack).is_empty(),
        "0.260s against a 0.019s row is process startup, not a regression, and must be spared"
    );
    assert_eq!(
        wall_clock_violations(&spared, &baseline, 2.0, 0.0).len(),
        2,
        "without the slack those same measurements are both reported — which is the defect the \
         slack fixes"
    );

    let past_the_slack: BTreeMap<String, f64> = [
        ("crate::three_trivial_tests".to_string(), 0.274),
        ("crate::real_work".to_string(), 20.2),
    ]
    .into_iter()
    .collect();
    let still_caught = wall_clock_violations(&past_the_slack, &baseline, 2.0, slack);
    assert_eq!(
        still_caught.len(),
        1,
        "the slack is an allowance, not an exemption: the same binary past it is reported again, \
         got: {still_caught:?}"
    );
    assert!(still_caught[0].contains("crate::three_trivial_tests"));

    let real_regression: BTreeMap<String, f64> = [
        ("crate::three_trivial_tests".to_string(), 0.010),
        ("crate::real_work".to_string(), 20.3),
    ]
    .into_iter()
    .collect();
    let caught = wall_clock_violations(&real_regression, &baseline, 2.0, slack);
    assert_eq!(
        caught.len(),
        1,
        "a binary whose relative ceiling is far above the slack must still be refused when it \
         exceeds it, got: {caught:?}"
    );
    assert!(caught[0].contains("crate::real_work"));
}

/// The measured case this rule was sized for, kept as a fixture so it cannot come back: a
/// sub-second binary that did not change, frozen at 0.497s from one gate's own measurement and
/// measured at 0.999s in a differently-sized test pool. Under a bare x2 factor that was 5ms over
/// its ceiling and the repository's own gate went red on identical code. It must be spared now,
/// and a binary that genuinely takes many times its recorded seconds must still be refused.
#[test]
fn a_binary_remeasured_in_a_differently_sized_pool_is_spared_while_a_real_slowdown_is_refused() {
    let baseline: BTreeMap<String, f64> = [("crate::sub_second_binary".to_string(), 0.497)]
        .into_iter()
        .collect();
    let tolerance = 2.991;
    let slack = 0.235;

    let remeasured: BTreeMap<String, f64> = [("crate::sub_second_binary".to_string(), 0.999)]
        .into_iter()
        .collect();
    assert!(
        wall_clock_violations(&remeasured, &baseline, tolerance, slack).is_empty(),
        "0.999s against a 0.497s row is the difference between two test pools, not a slowdown"
    );
    assert_eq!(
        wall_clock_violations(&remeasured, &baseline, 2.0, 0.0).len(),
        1,
        "under the bare x2 factor that same pair was a violation, which is the defect being fixed"
    );

    let slowed: BTreeMap<String, f64> = [("crate::sub_second_binary".to_string(), 4.0)]
        .into_iter()
        .collect();
    let caught = wall_clock_violations(&slowed, &baseline, tolerance, slack);
    assert_eq!(
        caught.len(),
        1,
        "a 0.497s binary taking 4.0s is eight times its recorded seconds and must still be \
         refused, got: {caught:?}"
    );
    assert!(caught[0].contains("crate::sub_second_binary"));
}

/// The slack is declared in the fixture, never in code, and the shipped value is the one the arm
/// applies. A hardcoded constant here would let the fixture say one thing and the gate do another.
#[test]
fn the_declared_slack_is_the_one_the_arm_applies() {
    let baseline = load_baseline(&repository_root());
    let smallest = baseline
        .gate_wall_seconds
        .values()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert!(
        smallest.is_finite(),
        "the [gate_wall_seconds] table must record at least one binary"
    );
    assert_eq!(
        wall_clock_ceiling(
            smallest,
            baseline.gate_wall_seconds_tolerance_factor,
            baseline.gate_wall_seconds_slack
        ),
        smallest * baseline.gate_wall_seconds_tolerance_factor + baseline.gate_wall_seconds_slack,
        "the ceiling the arm applies is computed from the declared policy, not from a constant"
    );
    assert!(
        baseline.gate_wall_seconds_slack > 0.0,
        "the declared slack must be positive — a zero slack is the purely relative ceiling that \
         reports one process spawn as a runtime regression"
    );
    assert!(
        baseline.gate_wall_seconds_slack > smallest * baseline.gate_wall_seconds_tolerance_factor,
        "the declared slack must still be the larger part of the smallest binary's ceiling: below \
         that it has stopped covering the fixed cost it exists for"
    );
}

/// The wall-clock arm compares numbers it did not measure, so the one way it can go quietly hollow
/// is by parsing a real gate JUnit into nothing and reporting a clean pass over an empty map. This
/// drives that case directly: a nextest-shaped document must yield real per-binary seconds, and an
/// empty parse must be a hard failure rather than a pass.
#[test]
fn an_unreadable_gate_measurement_is_a_failure_rather_than_a_clean_pass() {
    let nextest_shaped = r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="nextest-run" tests="2" failures="0" errors="0" time="7.5">
    <testsuite name="crate::binary" tests="2" disabled="0" errors="0" failures="0">
        <testcase name="one" classname="crate::binary" timestamp="2026-01-01T00:00:00.000+00:00" time="2.5">
        </testcase>
        <testcase name="two" classname="crate::binary" timestamp="2026-01-01T00:00:00.000+00:00" time="5.0">
        </testcase>
    </testsuite>
</testsuites>
"#;
    let live = parse_testsuite_wall_seconds(nextest_shaped);
    assert_eq!(
        live.get("crate::binary").copied(),
        Some(7.5),
        "a document shaped exactly the way the gate's own writer shapes it must parse to real \
         per-binary seconds; parsing it to nothing is what turns every later comparison into a \
         comparison against nothing"
    );

    let baseline: BTreeMap<String, f64> =
        [("crate::binary".to_string(), 1.0)].into_iter().collect();
    assert_eq!(
        wall_clock_violations(&live, &baseline, 2.0, 0.0).len(),
        1,
        "the parsed measurement must actually drive the comparison: 7.5s against a 1.0s baseline \
         at tolerance 2.0 is a regression"
    );

    let empty = BTreeMap::new();
    assert!(
        wall_clock_violations(&empty, &baseline, 2.0, 0.0).is_empty(),
        "an empty live map produces no violations at all, which is exactly why the live arm must \
         refuse to accept one instead of reading it as a pass"
    );
    assert!(
        gate_measurement_is_unusable(&empty),
        "an empty live map must be classified as an unusable measurement"
    );
    assert!(
        !gate_measurement_is_unusable(&live),
        "a live map with real per-binary seconds must be classified as usable"
    );
}

/// A parsed gate measurement with no binaries in it is not a pass, it is a measurement that was
/// never taken — the live arm refuses it rather than comparing the frozen table against nothing.
fn gate_measurement_is_unusable(live: &BTreeMap<String, f64>) -> bool {
    live.is_empty()
}

/// The live arm: applies `wall_clock_violations` to the gate's own JUnit. Runs only when the
/// gate step supplies `CG_GATE_JUNIT`, never by default — a bare `cargo test --workspace` never
/// produces a fresh gate JUnit for itself to compare against, and treating an absent measurement
/// as a silent pass would hollow this arm out. When the gate step does not supply a readable
/// `CG_GATE_JUNIT`, the arm fails loud instead of skipping quietly.
#[test]
#[ignore = "runs in the gate step with CG_GATE_JUNIT"]
fn gate_wall_seconds_never_exceed_the_frozen_baseline() {
    let junit_path = std::env::var("CG_GATE_JUNIT").unwrap_or_else(|_| {
        panic!(
            "gate_wall_seconds_never_exceed_the_frozen_baseline was not given a gate \
             measurement: CG_GATE_JUNIT must name a readable JUnit file produced by this gate's \
             own test run. A silent pass here would hollow out the wall-clock arm."
        )
    });
    let junit_xml = std::fs::read_to_string(&junit_path).unwrap_or_else(|error| {
        panic!("CG_GATE_JUNIT names {junit_path}, which could not be read: {error}")
    });

    let root = repository_root();
    let baseline = load_baseline(&root);
    let live = parse_testsuite_wall_seconds(&junit_xml);
    assert!(
        !gate_measurement_is_unusable(&live),
        "CG_GATE_JUNIT names {junit_path}, but no test binary's wall seconds could be read out of \
         it. Comparing the frozen table against an empty measurement is a pass that proves \
         nothing, so this is a failure: the file is not the gate's own JUnit, or its shape changed \
         and the reader above no longer matches it."
    );
    let mut violations = wall_clock_violations(
        &live,
        &baseline.gate_wall_seconds,
        baseline.gate_wall_seconds_tolerance_factor,
        baseline.gate_wall_seconds_slack,
    );
    violations.extend(wall_clock_coverage_violations(
        &live,
        &baseline.gate_wall_seconds,
    ));
    assert!(
        violations.is_empty(),
        "test-estate audit: gate wall-clock does not match the frozen baseline in \
         tests/fixtures/test_estate_baseline.toml:\n{}",
        violations.join("\n")
    );
}

/// The ignored live arm above is only worth anything if it is actually invoked somewhere with
/// `CG_GATE_JUNIT` set. This proves it is named, by its exact invocation, in both the CI workflow
/// (after the workspace test step) and the repository's own gate script.
#[test]
fn the_ignored_wall_clock_arm_is_invoked_by_name_in_the_gate_script_and_ci() {
    const INVOCATION: &str = "cargo test -p context-graph --test test_estate_audit \
         gate_wall_seconds_never_exceed_the_frozen_baseline -- --ignored --exact";

    let root = repository_root();

    let ci_path = root.join(".github/workflows/ci.yml");
    let ci = std::fs::read_to_string(&ci_path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", ci_path.display()));
    let workspace_test_line = ci
        .lines()
        .position(|line| line.contains("cargo test --workspace"))
        .expect(
            "ci.yml must still run `cargo test --workspace` as the step the wall-clock arm runs \
             after",
        );
    let invocation_line = ci.lines().position(|line| line.contains(INVOCATION));
    assert!(
        invocation_line.is_some_and(|index| index > workspace_test_line),
        "ci.yml must run `{INVOCATION}` as a step after the `cargo test --workspace` step, with \
         CG_GATE_JUNIT set for it"
    );
    let ci_gate_junit_nearby = ci
        .lines()
        .skip(invocation_line.unwrap_or(0).saturating_sub(6))
        .take(8)
        .any(|line| line.contains("CG_GATE_JUNIT"));
    assert!(
        ci_gate_junit_nearby,
        "ci.yml's `{INVOCATION}` step must set CG_GATE_JUNIT for it — the arm was not given a \
         gate measurement otherwise"
    );

    const BASELINE_DIRECTION_INVOCATION: &str = "python3 scripts/check-gate-wall-clock.py --base";
    let baseline_direction_line = ci
        .lines()
        .position(|line| line.contains(BASELINE_DIRECTION_INVOCATION));
    assert!(
        baseline_direction_line.is_some_and(|index| index > workspace_test_line),
        "ci.yml must run `{BASELINE_DIRECTION_INVOCATION}` as a step after the \
         `cargo test --workspace` step, so a pull request that raises a `[gate_wall_seconds]` \
         baseline value is refused"
    );

    let gate_script_path = root.join("scripts/gate.sh");
    let gate_script = std::fs::read_to_string(&gate_script_path).unwrap_or_else(|error| {
        panic!(
            "reading {}: {error} — the repository's gate script must exist and run the \
             wall-clock arm after nextest",
            gate_script_path.display()
        )
    });
    let nextest_line = gate_script
        .lines()
        .position(|line| line.contains("cargo nextest run --workspace"))
        .expect("scripts/gate.sh must run `cargo nextest run --workspace`");
    let gate_invocation_line = gate_script
        .lines()
        .position(|line| line.contains(INVOCATION));
    assert!(
        gate_invocation_line.is_some_and(|index| index > nextest_line),
        "scripts/gate.sh must run `{INVOCATION}` after `cargo nextest run --workspace`, \
         with CG_GATE_JUNIT set for it"
    );
    let script_gate_junit_nearby = gate_script
        .lines()
        .skip(gate_invocation_line.unwrap_or(0).saturating_sub(6))
        .take(8)
        .any(|line| line.contains("CG_GATE_JUNIT"));
    assert!(
        script_gate_junit_nearby,
        "scripts/gate.sh's `{INVOCATION}` step must set CG_GATE_JUNIT for it — the arm was \
         not given a gate measurement otherwise"
    );
}

/// The README publishes the CI steps that enforce the wall-clock ceiling, as a yaml snippet a
/// reader is entitled to trust. The binding beside it checks the workflow's own invocation and
/// never the README's copy of it, which is how the published snippet came to describe a guard the
/// workflow does not have. This closes that: every line of the README's snippet must appear in the
/// shipped workflow, so a step that changes in one place and not the other is a build failure
/// rather than a page that quietly lies.
#[test]
fn the_readme_ci_snippet_names_the_wall_clock_steps_the_workflow_actually_runs() {
    const ARM: &str = "gate_wall_seconds_never_exceed_the_frozen_baseline -- --ignored --exact";

    let root = repository_root();
    let readme_path = root.join("crates/context-graph/README.md");
    let readme = std::fs::read_to_string(&readme_path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", readme_path.display()));
    let ci_path = root.join(".github/workflows/ci.yml");
    let ci = std::fs::read_to_string(&ci_path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", ci_path.display()));

    let snippet = readme
        .split("```")
        .find(|block| block.contains(ARM))
        .unwrap_or_else(|| {
            panic!(
                "{} must publish the CI snippet that runs the wall-clock arm",
                readme_path.display()
            )
        });

    let ci_lines: Vec<&str> = ci.lines().map(str::trim).collect();
    let mut missing = Vec::new();
    for line in snippet.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed == "yaml" {
            continue;
        }
        if !ci_lines.contains(&trimmed) {
            missing.push(trimmed.to_string());
        }
    }
    assert!(
        missing.is_empty(),
        "{} publishes CI steps that {} does not run — a reader is being shown a workflow that \
         does not exist. Lines with no match in the shipped workflow:\n{}",
        readme_path.display(),
        ci_path.display(),
        missing.join("\n")
    );
    assert!(
        snippet.contains(ARM),
        "the snippet this test compares must be the one that runs the wall-clock arm"
    );
}

// ---------------------------------------------------------------------------------------------
// Baseline fixture loading
// ---------------------------------------------------------------------------------------------

struct Baseline {
    sleep_and_clock_sites: BTreeMap<String, usize>,
    default_tier_test_count: BTreeMap<String, usize>,
    gate_wall_seconds: BTreeMap<String, f64>,
    gate_wall_seconds_tolerance_factor: f64,
    gate_wall_seconds_slack: f64,
}

fn load_baseline(root: &Path) -> Baseline {
    let path = root.join("crates/context-graph/tests/fixtures/test_estate_baseline.toml");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let value: toml::Value =
        toml::from_str(&text).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()));

    let sleep_and_clock_sites = table_of_usize(&value, "sleep_and_clock_sites");
    let default_tier_test_count = table_of_usize(&value, "default_tier_test_count");
    let gate_wall_seconds = table_of_f64(&value, "gate_wall_seconds");
    let policy = value
        .get("gate_wall_seconds_policy")
        .and_then(toml::Value::as_table)
        .unwrap_or_else(|| {
            panic!(
                "{} must have a [gate_wall_seconds_policy] table",
                path.display()
            )
        });
    let policy_number = |key: &str| -> f64 {
        policy
            .get(key)
            .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
            .unwrap_or_else(|| {
                panic!(
                    "[gate_wall_seconds_policy] {key} must be a number in {}",
                    path.display()
                )
            })
    };
    let gate_wall_seconds_tolerance_factor = policy_number("tolerance_factor");
    let gate_wall_seconds_slack = policy_number("slack_seconds");

    Baseline {
        sleep_and_clock_sites,
        default_tier_test_count,
        gate_wall_seconds,
        gate_wall_seconds_tolerance_factor,
        gate_wall_seconds_slack,
    }
}

fn table_of_usize(value: &toml::Value, section: &str) -> BTreeMap<String, usize> {
    value
        .get(section)
        .and_then(toml::Value::as_table)
        .unwrap_or_else(|| panic!("test_estate_baseline.toml must have a [{section}] table"))
        .iter()
        .map(|(k, v)| {
            let n = v
                .as_integer()
                .unwrap_or_else(|| panic!("[{section}] {k} must be an integer"));
            (k.clone(), n as usize)
        })
        .collect()
}

fn table_of_f64(value: &toml::Value, section: &str) -> BTreeMap<String, f64> {
    value
        .get(section)
        .and_then(toml::Value::as_table)
        .unwrap_or_else(|| panic!("test_estate_baseline.toml must have a [{section}] table"))
        .iter()
        .map(|(k, v)| {
            let n = v
                .as_float()
                .or_else(|| v.as_integer().map(|i| i as f64))
                .unwrap_or_else(|| panic!("[{section}] {k} must be a number"));
            (k.clone(), n)
        })
        .collect()
}
