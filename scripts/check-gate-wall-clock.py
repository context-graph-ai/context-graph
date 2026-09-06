#!/usr/bin/env python3
"""
Refuse a gate whose test binaries got slower, and a baseline that rose.

`crates/context-graph/tests/fixtures/test_estate_baseline.toml` records, under
`[gate_wall_seconds]`, the wall-clock seconds one gate run measured for each test binary, and
under `[gate_wall_seconds_policy]` the `tolerance_factor` and the `slack_seconds` that together
set how far a fresh measurement may drift. A binary's ceiling is
`recorded seconds * tolerance_factor + slack_seconds`. Both terms are needed: the factor covers
drift that scales with the binary, while the slack covers the part that does not — one process
spawn per test, and a first file read a later run finds already cached. On a binary whose whole
run is a few spawns that fixed cost is many times the binary's own seconds, so a purely relative
ceiling reports the machine's scheduling as a runtime regression. The slack is also what lets a
measurement taken in one test pool be compared against a value frozen from a differently-sized
one, which is otherwise a different amount of contention over the same cores and disk. Neither
this script nor the Rust arm beside it ever reads a clock: both consume seconds someone else
already measured and wrote into a JUnit file.

Two modes, both of which exit nonzero on a violation and print every offending binary:

  --junit <path> --baseline <toml>   (--junit defaults to the CG_GATE_JUNIT environment variable)
      Compares the per-binary wall seconds in a gate's own JUnit against `[gate_wall_seconds]`.
      A binary present on both sides whose measurement exceeds
      `baseline seconds * tolerance_factor + slack_seconds` is a runtime regression. A binary
      whose whole ceiling is the slack is still recorded and still covered; it just cannot be
      failed on a number that carries no signal. A binary the gate RAN and the baseline has no
      row for is reported too: an unratcheted binary has no ceiling at all,
      and deleting a row is how a binary loses one silently. A row for a binary the gate did not
      run is ignored. A JUnit that parses to no binaries at all is a hard failure, never a pass:
      comparing the frozen table against nothing proves nothing. This is the same comparison
      `gate_wall_seconds_never_exceed_the_frozen_baseline` applies inside the gate.

  --base <ref>
      Compares the baseline file in the working tree against the same file at `<ref>`. The
      baseline only ever LOWERS: a value that rose is a ceiling raise, and the ordinary move on
      a regression is to make the binary fast again, not to buy room in the baseline. A binary
      the working tree adds is accepted (it has no earlier value to have risen from). One it
      REMOVES is a violation: a deleted row leaves that binary with no ceiling at all. The
      removal is accepted only when a JUnit for the same run shows the gate no longer runs that
      binary, which is the one case where the binary really is gone.

Both modes may be given at once; every violation from both is reported before exiting.

Usage:
  python3 scripts/check-gate-wall-clock.py --junit target/nextest/gate/junit.xml \
      --baseline crates/context-graph/tests/fixtures/test_estate_baseline.toml
  python3 scripts/check-gate-wall-clock.py --base origin/dev
"""

import argparse
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

DEFAULT_BASELINE = "crates/context-graph/tests/fixtures/test_estate_baseline.toml"

ATTR = re.compile(r'(\w+)="([^"]*)"')


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() and (candidate / "crates").is_dir():
            return candidate
    return start.parent


def attribute(tag: str, name: str) -> str | None:
    return dict(ATTR.findall(tag)).get(name)


def parse_testsuite_wall_seconds(junit_xml: str) -> dict[str, float]:
    """Wall seconds per test binary, read the way cargo nextest actually writes JUnit: it opens a
    `<testsuite name="crate::binary" ...>` element carrying NO `time` attribute of its own, and
    records the measured seconds on each `<testcase ... time="..">` inside it. A binary's wall
    seconds are therefore the SUM of its testcases' seconds, which is how the frozen
    `[gate_wall_seconds]` values were derived. Reading a `time` attribute off the `<testsuite>`
    tag instead finds nothing in a real document and yields an empty map — a comparison against
    nothing rather than a measurement.

    Deliberately not a general XML parser: this check only ever reads seconds someone else already
    measured and wrote to a file, never a clock of its own. Same rule as the Rust arm in
    `crates/context-graph/tests/test_estate_audit.rs`; the two are updated together."""
    measured: dict[str, float] = {}
    current_suite: str | None = None
    for line in junit_xml.splitlines():
        trimmed = line.lstrip()
        if trimmed.startswith("<testsuite "):
            current_suite = attribute(trimmed, "name")
            if current_suite is not None:
                measured.setdefault(current_suite, 0.0)
            continue
        if trimmed.startswith("</testsuite"):
            current_suite = None
            continue
        if not trimmed.startswith("<testcase "):
            continue
        seconds = attribute(trimmed, "time")
        if current_suite is None or seconds is None:
            continue
        try:
            measured[current_suite] = measured.get(current_suite, 0.0) + float(seconds)
        except ValueError:
            continue
    return measured


def measurement_is_unusable(live: dict[str, float]) -> bool:
    """A parsed gate measurement with no binaries in it is not a pass, it is a measurement that was
    never taken."""
    return not live


def wall_clock_ceiling(
    baseline_seconds: float, tolerance: float, slack_seconds: float
) -> float:
    """A binary's ceiling: its relative allowance plus the declared absolute slack. The factor
    covers drift that scales with the binary, the slack the fixed per-run cost that does not.
    Same rule as `wall_clock_ceiling` in `crates/context-graph/tests/test_estate_audit.rs`; the two
    are updated together."""
    return baseline_seconds * tolerance + slack_seconds


def read_recorded_seconds(text: str, source: str) -> dict[str, float]:
    """Just the `[gate_wall_seconds]` table. The direction check reads an EARLIER revision of this
    file, which legitimately predates any policy key added since, so it must not demand today's
    policy of yesterday's file — only the recorded seconds it is there to compare."""
    parsed = tomllib.loads(text)
    seconds = parsed.get("gate_wall_seconds")
    if not isinstance(seconds, dict):
        raise SystemExit(f"ERROR: {source} has no [gate_wall_seconds] table")
    return {name: float(value) for name, value in seconds.items()}


def read_baseline(text: str, source: str) -> tuple[dict[str, float], float, float]:
    parsed = tomllib.loads(text)
    seconds = parsed.get("gate_wall_seconds")
    if not isinstance(seconds, dict):
        raise SystemExit(f"ERROR: {source} has no [gate_wall_seconds] table")
    policy = parsed.get("gate_wall_seconds_policy")
    if not isinstance(policy, dict) or "tolerance_factor" not in policy:
        raise SystemExit(
            f"ERROR: {source} has no [gate_wall_seconds_policy] tolerance_factor"
        )
    if "slack_seconds" not in policy:
        raise SystemExit(
            f"ERROR: {source} has no [gate_wall_seconds_policy] slack_seconds"
        )
    tolerance = float(policy["tolerance_factor"])
    if tolerance < 1.0:
        raise SystemExit(
            f"ERROR: {source} declares tolerance_factor {tolerance}, which is below 1.0"
        )
    slack_seconds = float(policy["slack_seconds"])
    if slack_seconds < 0.0:
        raise SystemExit(
            f"ERROR: {source} declares slack_seconds {slack_seconds}, which is below 0.0"
        )
    return (
        {name: float(value) for name, value in seconds.items()},
        tolerance,
        slack_seconds,
    )


def compare_against_junit(
    junit_path: Path, baseline_path: Path, live: dict[str, float]
) -> list[str]:
    baseline, tolerance, slack_seconds = read_baseline(
        baseline_path.read_text(), str(baseline_path)
    )
    violations = []
    for binary, baseline_seconds in sorted(baseline.items()):
        if binary not in live:
            continue
        ceiling = wall_clock_ceiling(baseline_seconds, tolerance, slack_seconds)
        if live[binary] > ceiling:
            violations.append(
                f"{binary}: {live[binary]:.3f}s exceeds the frozen baseline "
                f"{baseline_seconds:.3f}s x tolerance_factor {tolerance} "
                f"+ slack {slack_seconds:.3f}s = {ceiling:.3f}s "
                f"ceiling — lower the baseline only for a proven, deliberate change; "
                f"otherwise this is a runtime regression"
            )
    # Completeness is checked from the LIVE side, never from the frozen table: a row that was
    # never added, or one that was deleted, gives that binary an unbounded ceiling while every
    # direction check still prints a clean pass.
    for binary in sorted(live):
        if binary in baseline:
            continue
        violations.append(
            f"{binary}: ran in this gate with no [gate_wall_seconds] row — record its measured "
            f"wall seconds in the same change; an unratcheted binary has no ceiling at all, and "
            f"deleting a row is how a binary loses one silently"
        )
    return violations


def compare_against_base(
    repo: Path, base: str, baseline_path: Path, live: dict[str, float] | None
) -> list[str]:
    relative = baseline_path.relative_to(repo).as_posix()

    def git(*arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *arguments], cwd=repo, capture_output=True, text=True
        )

    if git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}").returncode != 0:
        raise SystemExit(f"ERROR: {base} does not resolve to a commit in this repository")
    if git("cat-file", "-e", f"{base}:{relative}").returncode != 0:
        print(
            f"  {relative} did not exist at {base}; every value in it is new, and a new value "
            f"has no earlier value to have risen from"
        )
        return []
    earlier = git("show", f"{base}:{relative}").stdout
    base_seconds = read_recorded_seconds(earlier, f"{relative} at {base}")
    now_seconds, _, _ = read_baseline(baseline_path.read_text(), relative)
    violations = []
    for binary, earlier_value in sorted(base_seconds.items()):
        if binary not in now_seconds:
            # A removed row leaves that binary with no ceiling at all. The one honest reason to
            # remove one is that the binary itself is gone, and the only evidence for that is a
            # measurement of the same run that does not list it.
            if live is not None and binary not in live:
                continue
            violations.append(
                f"{binary}: had a [gate_wall_seconds] row of {earlier_value:.3f}s at {base} and "
                f"has none now — a removed row is a removed ceiling. Remove the row only when the "
                f"binary itself is gone, and prove that with a gate measurement that does not "
                f"list it"
            )
            continue
        if now_seconds[binary] > earlier_value:
            violations.append(
                f"{binary}: baseline raised from {earlier_value:.3f}s at {base} to "
                f"{now_seconds[binary]:.3f}s — [gate_wall_seconds] only ever lowers; fix the "
                f"binary's runtime instead of buying room in the ceiling"
            )
    return violations


def main() -> int:
    repo = find_repo_root(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(
        description="compare gate wall-clock seconds against the frozen baseline"
    )
    parser.add_argument("--repo", type=Path, default=repo, help="repository root")
    parser.add_argument(
        "--junit",
        type=Path,
        default=None,
        help="JUnit XML produced by this gate's own test run; defaults to CG_GATE_JUNIT",
    )
    parser.add_argument(
        "--baseline", type=Path, default=None, help=f"baseline TOML (default {DEFAULT_BASELINE})"
    )
    parser.add_argument(
        "--base", type=str, default=None, help="git ref the baseline may not have risen since"
    )
    args = parser.parse_args()

    junit = args.junit
    if junit is None and os.environ.get("CG_GATE_JUNIT"):
        junit = Path(os.environ["CG_GATE_JUNIT"])

    repo = args.repo.resolve()
    baseline_path = (
        args.baseline.resolve() if args.baseline else (repo / DEFAULT_BASELINE)
    )
    if not baseline_path.exists():
        print(f"ERROR: baseline {baseline_path} does not exist", file=sys.stderr)
        return 2
    if junit is None and args.base is None:
        print(
            "ERROR: give --junit (with --baseline) or --base, or both",
            file=sys.stderr,
        )
        return 2

    violations: list[str] = []
    live: dict[str, float] | None = None
    if junit is not None:
        if not junit.exists():
            print(f"ERROR: JUnit {junit} does not exist", file=sys.stderr)
            return 2
        live = parse_testsuite_wall_seconds(junit.read_text())
        if measurement_is_unusable(live):
            print(
                f"ERROR: {junit} yielded no test binary's wall seconds. Comparing the frozen "
                f"table against an empty measurement is a pass that proves nothing, so this is a "
                f"failure: the file is not the gate's own JUnit, or its shape changed and the "
                f"reader above no longer matches it.",
                file=sys.stderr,
            )
            return 1
        found = compare_against_junit(junit, baseline_path, live)
        print(
            f"--- gate wall-clock vs {baseline_path.name}: {len(live)} binary/binaries measured, "
            f"{len(found)} over the ceiling or unratcheted ---"
        )
        violations += found
    if args.base is not None:
        found = compare_against_base(repo, args.base, baseline_path, live)
        print(
            f"--- baseline direction vs {args.base}: {len(found)} raised or removed value(s) ---"
        )
        violations += found

    for violation in violations:
        print(f"  {violation}")
    if violations:
        print(f"FAIL: {len(violations)} gate wall-clock violation(s).")
        return 1
    print(
        "PASS: every measured binary is ratcheted and under its ceiling, no baseline value "
        "raised or removed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
