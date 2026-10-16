#!/usr/bin/env python3
"""
Validate bidirectional docs <-> tests coverage for the context-graph repo and require an
enforcing test for every behavior claim in a file the current change touched.

A documented claim is bound to a test with an HTML comment placed at (or immediately after)
the end of the claim's paragraph:

    ... prepared value carrying the exact final token count.<!-- enforced by: `oversized_event_split_uses_selected_tokenizer_and_has_utf8_safe_exact_bidirectional_source_offsets` -->

This script reads those comments out of README.md, crates/*/README.md, docs/*.md and
docs/performance/**/*.md (if the directory exists) and skills/**/SKILL.md (if the directory exists); reads live default-tier test
function names out of the Rust test tree (`crates/*/tests/**/*.rs` and `crates/*/src/**/*.rs`
for `#[test]`/`#[tokio::test]` functions, including parameterized Tokio attributes); and reports:

  (a) per-doc-page tag coverage
  (b) doc pages with zero backing tests
(c) referenced names that are not live default-tier test functions (DEAD REFS -> nonzero exit)
  (d) test-tree tests referenced by no doc                          (inverse coverage, informational)
  (e) unbound claims in a file this change touched                  (UNBOUND -> nonzero exit)
  (f) crates whose src/ this change touched with no crates/<crate>/AGENTS.md (MISSING -> nonzero exit)

Exit status is nonzero when a dead reference is found (c), when an unbound claim is found
in a touched doc file (e), or when a crate this change edited carries no agent guide (f).
Unbacked doc pages and unreferenced tests are reported but do not fail the run: they track
work-to-do, not a broken binding.

## The crate-guide rule (f)

A contributor sent to a crate has to find the routing and invariant guide IN that crate. So a
change that touches any file under `crates/<crate>/src/` requires `crates/<crate>/AGENTS.md` to
exist; the run fails naming every such crate. Only `src/` counts — a change confined to a crate's
`tests/`, `Cargo.toml` or fixtures does not demand a guide — and a crate is named once however
many of its source files changed. Like (e), the rule needs a changed-file list, so it runs only
under `--base` or `--changed-files`. The Rust side of this same rule is
`a_changed_crate_with_no_agents_md_is_reported_missing_and_a_guided_one_is_not` in
`crates/context-graph/tests/doc_claim_bindings.rs`, and
`every_crate_this_change_touches_has_its_own_agents_md` in
`crates/context-graph/tests/agents_md_names_only_paths_that_exist.rs` requires the guides
unconditionally; all three must be changed together.

## What "default tier" means here, and where this validator's scope ends

A default-tier test is one an ordinary `cargo test --workspace` compiles and runs. Two separate
mechanisms keep a test out of that tier, and both are honoured:

  - the test function's own attributes: `#[ignore]`, or
    `#[cfg(context_graph_internal_test_support)]` on the function itself;
  - an ENCLOSING module the Rust resolver never compiles: a `mod x;` or an inline `mod x { ... }`
    declared behind `#[cfg(context_graph_internal_test_support)]`, whether in a multi-file test
    harness root (`crates/*/tests/*/main.rs`) or inside a module file reached from one. A test in
    such a module carries no exclusion marker of its own, so reading only the function's
    attributes would count a proof that never runs.

A name is default-tier when at least one of its definitions is reachable: defined under `src/`,
or under a `tests/` module chain with no gated `mod` declaration anywhere along it.

The scope stops there. This validator does not evaluate arbitrary `cfg` predicates, feature
gates, `#[cfg(test)]`, target gates, or conditional compilation inside a function body; the one
configuration flag it understands is `context_graph_internal_test_support`, which is how this
repository marks internal-only test support. `crates/context-graph/tests/doc_claim_bindings.rs`
enforces the always-on subset of this same rule inside the gate and must be changed with this
script.

## The "claim" heuristic (e), kept deliberately simple

A markdown file is split into blocks separated by one or more blank lines. A block is a
candidate "claim paragraph" when ALL of:
  - it is not a fenced code block (``` ... ```)
  - it does not start with '#' (a heading), '|' (a table row), or '<!--' (a bare comment)
  - stripped of any `<!-- enforced by: ... -->` tags it contains, it has at least 12
    whitespace-separated words

A candidate claim paragraph is "unbound" when no `<!-- enforced by: ... -->` tag appears
anywhere within it. This is a heuristic, not a parser for English: it will occasionally treat
a long descriptive aside as a claim, or miss a short one. It only runs against files named on
`--changed-files` (or derived from `git diff --name-only <base>..HEAD` with `--base`), so it
never penalizes a doc page nothing in this change touched. It is deliberately file-scoped, not
line-scoped: touching any part of a doc file surfaces every unbound claim in that file, not
only the lines the change added or altered. The script reports every unbound claim in a
touched file; determining which claims the change introduced requires human review.

Usage:
  python3 scripts/validate-doc-coverage.py --all
  python3 scripts/validate-doc-coverage.py --repo /path/to/context-graph
  python3 scripts/validate-doc-coverage.py --changed-files README.md,crates/context-graph/README.md
  python3 scripts/validate-doc-coverage.py --base HEAD~1
  python3 scripts/validate-doc-coverage.py --repo /tmp/copy --list-unreferenced
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Test source roots, relative to the repo root. `**` needs recursive glob.
TEST_GLOBS = [
    "crates/*/tests/**/*.rs",
    "crates/*/src/**/*.rs",
]

# Doc roots, relative to the repo root. docs/ and skills/ are scanned only when present.
DOC_FILES = ["README.md"]
DOC_GLOBS = ["crates/*/README.md", "docs/*.md", "docs/performance/**/*.md", "skills/**/SKILL.md"]

# A test line carries one of these attributes on the line immediately (or a few
# attribute-lines) above its `fn`. Tokio accepts parameters such as
# `#[tokio::test(flavor = "multi_thread")]`; those tests are in the default tier too.
TEST_ATTR_RE = re.compile(r"^\s*#\[(?:tokio::test(?:\s*\([^]]*\))?|test)\]\s*$")
FN_RE = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([a-z_][a-z0-9_]*)\s*[<(]")
# Any attribute line (so #[test] followed by #[ignore]/#[should_panic] still binds).
ATTR_LINE_RE = re.compile(r"^\s*#\[")
IGNORE_ATTR_RE = re.compile(r"^\s*#\[ignore(?:\s|\]|=)")
INTERNAL_TEST_SUPPORT_ATTR = "#[cfg(context_graph_internal_test_support)]"

ENFORCED_RE = re.compile(r"<!--\s*enforced by:\s*(.+?)\s*-->")
NAME_RE = re.compile(r"`?([a-z_][a-z0-9_]+)`?")


def find_repo_root(start: Path) -> Path:
    """Walk up from `start` until a dir has both README.md and crates/."""
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "README.md").exists() and (cand / "crates").is_dir():
            return cand
    return cur


MOD_DECL_RE = re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*(;|\{)")


def _scan_module_file(path: Path, module_dir: Path, inherited_gate: bool):
    """Scan one Rust module file for module structure.

    Returns `(children, gated_line_ranges)`:

      - `children` is `(child_path, child_module_dir, child_gate)` for every `mod x;` declaration,
        with `child_gate` true when the declaration is gated by
        `#[cfg(context_graph_internal_test_support)]`, sits inside a gated inline `mod x { ... }`
        block, or inherits a gate from the declaration that pulled this file in;
      - `gated_line_ranges` is the inclusive `(first, last)` line span of every gated inline module
        block, so the tests written inside one can be dropped.

    Nesting is tracked by counting braces, which is exact enough for module structure and is the
    documented limit of this scan.
    """
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return [], []

    children: list[tuple[Path, Path, bool]] = []
    gated_ranges: list[tuple[int, int]] = []
    # Open inline `mod` blocks: (depth outside the block, module dir for its children, gated,
    # first line).
    inline_stack: list[tuple[int, Path, bool, int]] = []
    depth = 0
    pending_gate = False

    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        current_dir = inline_stack[-1][1] if inline_stack else module_dir
        inside_gate = inherited_gate or (inline_stack[-1][2] if inline_stack else False)

        if stripped == INTERNAL_TEST_SUPPORT_ATTR:
            pending_gate = True
            continue

        m = MOD_DECL_RE.match(stripped)
        if m:
            name = m.group(1)
            gate = inside_gate or pending_gate
            pending_gate = False
            if m.group(2) == ";":
                for candidate, child_dir in (
                    (current_dir / f"{name}.rs", current_dir / name),
                    (current_dir / name / "mod.rs", current_dir / name),
                ):
                    if candidate.is_file():
                        children.append((candidate, child_dir, gate))
                        break
            else:
                inline_stack.append((depth, current_dir / name, gate, lineno))
        elif stripped and not stripped.startswith("#["):
            pending_gate = False

        depth += line.count("{") - line.count("}")
        while inline_stack and depth <= inline_stack[-1][0]:
            _, _, gated, first_line = inline_stack.pop()
            if gated:
                gated_ranges.append((first_line, lineno))

    for _, _, gated, first_line in inline_stack:
        if gated:
            gated_ranges.append((first_line, len(lines)))
    return children, gated_ranges


def module_exclusions(repo: Path):
    """Return `(excluded_files, gated_blocks)` for the multi-file test targets under `repo`.

    `excluded_files` are `.rs` files the Rust resolver never compiles into the default tier because
    a `mod` declaration reachable from their test target's root (`crates/*/tests/<target>/main.rs`)
    is gated by `#[cfg(context_graph_internal_test_support)]`; `gated_blocks` maps a compiled file
    to the inclusive line spans of gated inline `mod` blocks written inside it.
    """
    excluded_files: set[Path] = set()
    gated_blocks: dict[Path, list[tuple[int, int]]] = {}
    for main_rs in sorted(repo.glob("crates/*/tests/*/main.rs")):
        pending = [(main_rs, main_rs.parent, False)]
        seen: set[Path] = set()
        while pending:
            path, module_dir, gate = pending.pop()
            if path in seen:
                continue
            seen.add(path)
            if gate:
                excluded_files.add(path)
            children, gated_ranges = _scan_module_file(path, module_dir, gate)
            if gated_ranges and not gate:
                gated_blocks.setdefault(path, []).extend(gated_ranges)
            pending.extend(children)
    return excluded_files, gated_blocks


def extract_test_names(repo: Path):
    """Return live default-tier test function names and the subset defined under `tests/`.

    Proof aliases are function names only: a file name can survive after its test is deleted,
    while a default-tier test function is what `cargo test` can actually execute. A test is
    default-tier only when neither its own attributes nor any enclosing `mod` declaration reachable
    from its test target's root excludes it — see this file's module docstring for the exact scope.
    """
    name_to_files: dict[str, set[str]] = {}
    test_tree_names: set[str] = set()
    seen_files: set[Path] = set()
    excluded_module_files, gated_blocks = module_exclusions(repo)
    for glob in TEST_GLOBS:
        for path in sorted(repo.glob(glob)):
            if path in seen_files or not path.is_file():
                continue
            seen_files.add(path)
            if path in excluded_module_files:
                continue
            gated_spans = gated_blocks.get(path, [])
            rel = str(path.relative_to(repo))
            is_test_tree = "/tests/" in rel or rel.startswith("tests/")
            lines = path.read_text(errors="replace").splitlines()
            attributes: list[str] = []
            for lineno, line in enumerate(lines, 1):
                if any(first <= lineno <= last for first, last in gated_spans):
                    attributes.clear()
                    continue
                if ATTR_LINE_RE.match(line):
                    attributes.append(line.strip())
                    continue
                if attributes:
                    m = FN_RE.match(line)
                    if m:
                        name = m.group(1)
                        is_test = any(TEST_ATTR_RE.match(attribute) for attribute in attributes)
                        excluded = any(
                            IGNORE_ATTR_RE.match(attribute)
                            or attribute == INTERNAL_TEST_SUPPORT_ATTR
                            for attribute in attributes
                        )
                        if is_test and not excluded:
                            name_to_files.setdefault(name, set()).add(rel)
                            if is_test_tree:
                                test_tree_names.add(name)
                        attributes.clear()
                    elif line.strip() == "":
                        # Blank lines between attributes and `fn` are harmless.
                        continue
                    else:
                        # Something else broke the attribute->fn adjacency.
                        attributes.clear()
    return name_to_files, test_tree_names


def extract_doc_refs(repo: Path):
    """Return doc_relpath -> list of (name, lineno) references, for every doc file that
    exists (docs/ and skills/ globs contribute nothing when those directories are absent)."""
    per_file: dict[str, list[tuple[str, int]]] = {}
    doc_paths: list[Path] = []
    for f in DOC_FILES:
        p = repo / f
        if p.exists():
            doc_paths.append(p)
    for glob in DOC_GLOBS:
        doc_paths.extend(sorted(repo.glob(glob)))
    for p in doc_paths:
        rel = str(p.relative_to(repo))
        refs: list[tuple[str, int]] = []
        for lineno, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            for m in ENFORCED_RE.finditer(line):
                for nm in NAME_RE.finditer(m.group(1)):
                    refs.append((nm.group(1), lineno))
        per_file[rel] = refs  # keep even empty pages so we can report them
    return per_file


def all_doc_paths(repo: Path) -> list[Path]:
    paths: list[Path] = []
    for f in DOC_FILES:
        p = repo / f
        if p.exists():
            paths.append(p)
    for glob in DOC_GLOBS:
        paths.extend(sorted(repo.glob(glob)))
    return paths


def claim_paragraphs(text: str):
    """Yield (start_line, block_text) for each candidate claim paragraph, per the heuristic
    documented at the top of this file."""
    lines = text.splitlines()
    block: list[str] = []
    block_start = 0
    in_fence = False

    def flush():
        if not block:
            return None
        return (block_start, "\n".join(block))

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            result = flush()
            block.clear()
            if result:
                yield result
            continue
        if in_fence:
            continue
        if stripped == "":
            result = flush()
            block.clear()
            if result:
                yield result
            continue
        if not block:
            block_start = i
        block.append(line)
    result = flush()
    if result:
        yield result


def is_claim(block_text: str) -> bool:
    first = block_text.lstrip()
    if first.startswith("#") or first.startswith("|") or first.startswith("<!--"):
        return False
    stripped = ENFORCED_RE.sub("", block_text)
    words = stripped.split()
    return len(words) >= 12


def is_bound(block_text: str) -> bool:
    return ENFORCED_RE.search(block_text) is not None


def find_unbound_claims(repo: Path, changed_files: set[str]):
    """Return {relpath: [(start_line, first_60_chars), ...]} for unbound claim paragraphs in
    doc files present in `changed_files`."""
    unbound: dict[str, list[tuple[int, str]]] = {}
    for p in all_doc_paths(repo):
        rel = str(p.relative_to(repo))
        if rel not in changed_files:
            continue
        text = p.read_text(errors="replace")
        hits = []
        for start_line, block in claim_paragraphs(text):
            if is_claim(block) and not is_bound(block):
                preview = " ".join(block.split())[:60]
                hits.append((start_line, preview))
        if hits:
            unbound[rel] = hits
    return unbound


def crates_missing_an_agents_guide(repo: Path, changed_files: set[str]) -> list[str]:
    """Crate names whose `src/` this change touched and which carry no `crates/<crate>/AGENTS.md`,
    sorted and reported once per crate however many of its source files changed."""
    missing: set[str] = set()
    for path in changed_files:
        parts = path.split("/")
        if len(parts) < 4 or parts[0] != "crates" or parts[2] != "src":
            continue
        crate_name = parts[1]
        if not (repo / "crates" / crate_name / "AGENTS.md").exists():
            missing.add(crate_name)
    return sorted(missing)


AGENTS_GUIDE_STALENESS_EXCEPTIONS = (
    "crates/context-graph/tests/fixtures/agents_guide_staleness_exceptions.txt"
)


def declared_staleness_exceptions(repo: Path) -> set[str]:
    """Crate names an owner has declared exempt from the guide-staleness rule, one per line in
    `crates/context-graph/tests/fixtures/agents_guide_staleness_exceptions.txt` with the reason
    after a `#`. A tracked line, never a silent skip: an absent file means no exemptions."""
    path = repo / AGENTS_GUIDE_STALENESS_EXCEPTIONS
    if not path.exists():
        return set()
    declared: set[str] = set()
    for line in path.read_text(errors="replace").splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            declared.add(name)
    return declared


def crates_with_a_stale_agents_guide(repo: Path, changed_files: set[str]) -> list[str]:
    """Crate names whose `src/` this change touched while leaving `crates/<crate>/AGENTS.md`
    untouched, sorted, minus the declared exceptions.

    A crate guide states that crate's invariants and names the tests behind them. When the source
    moves and the guide does not, the guide keeps reading as authoritative while describing
    behavior that is gone — which is precisely the failure the guide machinery exists to prevent,
    and existence alone does not catch it."""
    exceptions = declared_staleness_exceptions(repo)
    stale: set[str] = set()
    for path in changed_files:
        parts = path.split("/")
        if len(parts) < 4 or parts[0] != "crates" or parts[2] != "src":
            continue
        crate_name = parts[1]
        if crate_name in exceptions:
            continue
        guide = f"crates/{crate_name}/AGENTS.md"
        if not (repo / guide).exists():
            continue  # reported by crates_missing_an_agents_guide instead
        if guide not in changed_files:
            stale.add(crate_name)
    return sorted(stale)


class ChangedFilesUnavailable(Exception):
    """The changed-file list was asked for and could not be produced.

    Every arm that needs it — the unbound-claim scan and the missing-guide scan — is silently
    vacuous over an empty list, so an unresolvable `--base` must stop the run rather than let both
    arms report a clean pass over nothing.
    """


def resolve_changed_files(args, repo: Path) -> set[str]:
    if args.all:
        try:
            out = subprocess.run(
                ["git", "ls-files", "-z"], cwd=repo, capture_output=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise ChangedFilesUnavailable(f"could not enumerate tracked files: {exc}") from exc
        tracked = {path for path in out.stdout.decode().split("\0") if path}
        return tracked | {path.relative_to(repo).as_posix() for path in all_doc_paths(repo)}
    if args.changed_files:
        return {f.strip() for f in args.changed_files.split(",") if f.strip()}
    if args.base:
        try:
            out = subprocess.run(
                ["git", "diff", "--name-only", f"{args.base}..HEAD"],
                cwd=repo,
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise ChangedFilesUnavailable(
                f"could not compute the changed-file list against {args.base!r}: {exc}"
            ) from exc
        return {line.strip() for line in out.stdout.splitlines() if line.strip()}
    return set()


def main() -> int:
    default_repo = find_repo_root(Path(__file__).resolve().parent)
    ap = argparse.ArgumentParser(description="context-graph docs <-> tests coverage")
    ap.add_argument("--repo", type=Path, default=default_repo,
                    help="context-graph repo root (auto-detected by default)")
    ap.add_argument("--changed-files", type=str, default=None,
                    help="comma-separated repo-relative paths of files this change touched")
    ap.add_argument("--base", type=str, default=None,
                    help="git ref to diff HEAD against for the changed-file list, when "
                         "--changed-files is not given directly")
    ap.add_argument("--all", action="store_true", help="audit every documentation page without a base ref")
    ap.add_argument("--list-unreferenced", action="store_true",
                    help="list every unreferenced test-tree test, not just counts")
    args = ap.parse_args()

    repo = args.repo.resolve()
    if not (repo / "README.md").exists():
        print(f"ERROR: {repo} does not look like the context-graph repo (no README.md)",
              file=sys.stderr)
        return 2

    name_to_files, test_tree_names = extract_test_names(repo)
    all_names = set(name_to_files)
    per_file = extract_doc_refs(repo)

    all_refs: set[str] = set()
    for refs in per_file.values():
        for name, _ in refs:
            all_refs.add(name)

    dead = sorted(r for r in all_refs if r not in all_names)
    unreferenced_test_tree = sorted(test_tree_names - all_refs)
    unbacked_pages = sorted(f for f, refs in per_file.items() if not refs)

    try:
        changed_files = resolve_changed_files(args, repo)
    except ChangedFilesUnavailable as exc:
        print(
            f"ERROR: {exc}. The unbound-claim and missing-guide arms both need that list and "
            f"pass over an empty one, so this run cannot report anything about them.",
            file=sys.stderr,
        )
        return 2
    unbound = find_unbound_claims(repo, changed_files)
    unguided = crates_missing_an_agents_guide(repo, changed_files)
    stale_guides = crates_with_a_stale_agents_guide(repo, changed_files)

    total_test_files = len({f for fs in name_to_files.values() for f in fs})

    print("context-graph docs <-> tests coverage")
    print("=" * 60)
    print(f"repo:                       {repo}")
    print(f"test functions discovered:  {len(all_names)}  (across {total_test_files} files)")
    print(f"  of which test-tree:       {len(test_tree_names)}")
    print(f"doc pages scanned:          {len(per_file)}")
    print(f"enforced-by references:     {sum(len(r) for r in per_file.values())}"
          f"  ({len(all_refs)} distinct test names)")
    print(f"dead references:            {len(dead)}")
    print(f"test-tree tests unref'd:    {len(unreferenced_test_tree)}")
    print(f"changed files considered:   {len(changed_files)}")
    print(f"unbound claims:             {sum(len(v) for v in unbound.values())}"
          f"  (in {len(unbound)} changed doc file(s))")
    print()

    print("--- (a) per-doc-page tag coverage ---")
    for fname in sorted(per_file):
        refs = per_file[fname]
        distinct = {n for n, _ in refs}
        valid = {n for n in distinct if n in all_names}
        invalid = {n for n in distinct if n not in all_names}
        line = f"  {fname}: {len(refs)} tags, {len(valid)} valid"
        if invalid:
            line += f", {len(invalid)} DEAD: {', '.join(sorted(invalid))}"
        print(line)
    print()

    print("--- (b) doc pages with zero backing tests ---")
    if unbacked_pages:
        for f in unbacked_pages:
            print(f"  {f}")
    else:
        print("  (none)")
    print()

    print("--- (c) dead references (test named by a doc but not found) ---")
    if dead:
        for name in dead:
            sources = []
            for f, refs in per_file.items():
                for n, ln in refs:
                    if n == name:
                        sources.append(f"{f}:{ln}")
            print(f"  {name}  (in: {', '.join(sources)})")
    else:
        print("  (none)")
    print()

    print("--- (d) test-tree tests referenced by no doc ---")
    print(f"  {len(unreferenced_test_tree)} of {len(test_tree_names)} test-tree tests are "
          f"unreferenced")
    if args.list_unreferenced:
        by_file: dict[str, list[str]] = {}
        for name in unreferenced_test_tree:
            for f in sorted(name_to_files[name]):
                by_file.setdefault(f, []).append(name)
        for f in sorted(by_file):
            print(f"  [{f}]")
            for name in sorted(by_file[f]):
                print(f"    {name}")
    else:
        print("  (re-run with --list-unreferenced to enumerate them)")
    print()

    print("--- (e) unbound claims in a changed doc file ---")
    if unbound:
        for f in sorted(unbound):
            for lineno, preview in unbound[f]:
                print(f"  {f}:{lineno}  {preview}...")
    else:
        print("  (none)")
    print()

    print("--- (f) crates this change edited with no agent guide ---")
    if unguided:
        for crate_name in unguided:
            print(f"  {crate_name}  (expected crates/{crate_name}/AGENTS.md)")
    else:
        print("  (none)")
    print()

    print("--- (g) crates whose source this change edited and whose agent guide it did not ---")
    if stale_guides:
        for crate_name in stale_guides:
            print(f"  {crate_name}  (crates/{crate_name}/AGENTS.md unchanged)")
    else:
        print("  (none)")
    print()

    if dead or unbound or unguided or stale_guides:
        if dead:
            print(f"FAIL: {len(dead)} dead reference(s) — a doc names a test that does not exist.")
        if unbound:
            total = sum(len(v) for v in unbound.values())
            print(f"FAIL: {total} unbound claim(s) in a file this change touched.")
        if unguided:
            print(
                f"FAIL: {len(unguided)} crate(s) whose src/ this change touched carry no "
                f"AGENTS.md: {', '.join(unguided)}. A contributor sent to a crate has to find "
                f"the routing and invariant guide in it."
            )
        if stale_guides:
            print(
                f"FAIL: {len(stale_guides)} crate(s) whose src/ this change touched with an "
                f"unchanged agent guide: {', '.join(stale_guides)}. Update the guide in the same "
                f"change, or declare the crate in "
                f"{AGENTS_GUIDE_STALENESS_EXCEPTIONS} with the reason."
            )
        return 1
    print("PASS: no dead references, no unbound claims in touched files, every edited crate "
          "carries an agent guide, and every edited crate's guide moved with its source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
