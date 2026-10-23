#!/usr/bin/env python3
"""Check and run the Python test shards of fork-tests.yml.

The shard map (python-shards.json) has one list of shards for each pass
("gpu" and "cpu"). A shard is a list of entries. Each entry runs in its own
child process, and the child writes its own log. An entry is one of these:

  test_x                  all tests in python/tests/test_x.py
  test_x.Class.test_y     one test (or a class with test_x.Class)
  test_x:rest             the tests in test_x.py that no other entry names

Commands:

  check                    Check the map without importing mlx. Every
                           python/tests/test*.py file must be in the map of
                           each pass exactly one time: as one whole-file
                           entry, or as named entries with at most one rest
                           entry. The command fails on any other case.
  run PASS SHARD COUNT     Check the map, expand the split files to test IDs
                           (this imports them), and fail if the named IDs of
                           a split file are not a partition of its tests.
                           Then run the entries of shard SHARD, JOBS at a
                           time. COUNT is the number of shard jobs in the
                           workflow, and it must equal the shards in the map.
  verify DIR               Read the test ID files that the shard jobs wrote
                           and fail if a test ran in two shards of a pass or
                           if a shard result is missing.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TESTS = Path("python/tests")
MAP = Path(__file__).with_name("python-shards.json")
PASSES = ("gpu", "cpu")


def parse(entry):
    """Return (file, kind) for a map entry."""
    if entry.endswith(":rest"):
        return entry[: -len(":rest")], "rest"
    if "." in entry:
        return entry.split(".", 1)[0], "named"
    return entry, "whole"


def load_map():
    with open(MAP) as f:
        data = json.load(f)
    if sorted(data) != sorted(PASSES):
        sys.exit(f"error: {MAP} must have exactly the keys {PASSES}")
    return data


def check_static(shards):
    """Return a list of errors. Does not import mlx or the test files."""
    errors = []
    files = sorted(p.stem for p in TESTS.glob("test*.py"))
    for name in PASSES:
        uses = {}
        entries = []
        for index, shard in enumerate(shards[name]):
            if not shard:
                errors.append(f"{name} shard {index} has no entries")
            for entry in shard:
                file, kind = parse(entry)
                uses.setdefault(file, []).append((index, kind, entry))
                entries.append(entry)
        for entry in sorted({e for e in entries if entries.count(e) > 1}):
            errors.append(f"{name}: the entry {entry} is in the map more than one time")
        for file in files:
            found = uses.pop(file, [])
            kinds = [kind for _, kind, _ in found]
            where = ", ".join(f"shard {i}: {e}" for i, _, e in found)
            if not found:
                errors.append(f"{name}: {TESTS}/{file}.py is in no shard")
            elif "whole" in kinds and len(found) > 1:
                errors.append(f"{name}: {TESTS}/{file}.py is in more than one shard or entry ({where})")
            elif kinds.count("rest") > 1:
                errors.append(f"{name}: {TESTS}/{file}.py has more than one rest entry ({where})")
        for file, found in sorted(uses.items()):
            errors.append(
                f"{name}: {', '.join(e for _, _, e in found)} names no file {TESTS}/{file}.py"
            )
    return errors


def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


def test_ids(module):
    suite = unittest.defaultTestLoader.loadTestsFromName(module)
    ids = [test.id() for test in flatten(suite)]
    bad = [i for i in ids if not i.startswith(module + ".")]
    if bad:
        sys.exit(f"error: {module} did not load: {bad}")
    return ids


def expand(shards, name):
    """Expand the entries of one pass to child arguments.

    Returns a list (one item for each shard) of lists of (entry, names),
    where names is the list of unittest names for the child. Also returns
    a list of errors.
    """
    errors = []
    split = {}
    for shard in shards[name]:
        for entry in shard:
            file, kind = parse(entry)
            if kind != "whole":
                split.setdefault(file, []).append(entry)
    sys.path.insert(0, str(TESTS))
    owner = {}
    rest_ids = {}
    for file, entries in sorted(split.items()):
        ids = test_ids(file)
        count = {i: 0 for i in ids}
        for entry in entries:
            if parse(entry)[1] != "named":
                continue
            matched = [i for i in ids if i == entry or i.startswith(entry + ".")]
            if not matched:
                errors.append(f"{name}: the entry {entry} matches no test in {file}.py")
            for i in matched:
                count[i] += 1
                owner[i] = entry
        twice = [i for i, c in count.items() if c > 1]
        if twice:
            errors.append(f"{name}: these tests are in more than one entry: {twice}")
        missing = [i for i, c in count.items() if c == 0]
        if f"{file}:rest" in entries:
            rest_ids[file] = missing
        elif missing:
            errors.append(f"{name}: these tests of {file}.py are in no shard: {missing}")
    plan = []
    for shard in shards[name]:
        units = []
        for entry in shard:
            file, kind = parse(entry)
            if kind == "whole":
                units.append((entry, [entry]))
            elif kind == "rest":
                units.append((entry, rest_ids.get(file, [])))
            else:
                units.append((entry, sorted(i for i, e in owner.items() if e == entry)))
        plan.append(units)
    return plan, errors


def child(log_dir, unit, names):
    """Run the tests in names in this process. Used by run."""
    # The same settings as python/tests/run.py.
    os.environ["MLX_ENABLE_TF32"] = "0"
    os.environ["MLX_ENABLE_CACHE_THRASHING_CHECK"] = "0"
    sys.path.insert(0, str(TESTS))
    import mlx.core as mx

    exclude = set(os.environ.get("CPU_EXCLUDE", "").split())
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    tests = [t for t in flatten(suite) if t.id() not in exclude]
    ran = []

    class Result(unittest.TextTestResult):
        def startTest(self, test):
            ran.append(test.id())
            super().startTest(test)

    runner = unittest.TextTestRunner(
        sys.stdout, verbosity=2, durations=0, resultclass=Result
    )
    result = runner.run(unittest.TestSuite(tests))
    with open(Path(log_dir) / f"{unit}.ids", "w") as f:
        json.dump(ran, f)
    mx.clear_streams()
    sys.exit(0 if result.wasSuccessful() else 1)


def file_name(entry):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", entry)


def run(args):
    shards = load_map()
    errors = check_static(shards)
    if errors:
        report(errors)
    if args.count != len(shards[args.name]):
        report([
            f"the workflow runs {args.count} {args.name} shard jobs, "
            f"but {MAP.name} has {len(shards[args.name])} {args.name} shards"
        ])
    plan, errors = expand(shards, args.name)
    if errors:
        report(errors)
    units = plan[args.shard]
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    results = {}

    def one(unit):
        entry, names = unit
        name = file_name(entry)
        start = time.monotonic()
        with open(logs / f"{name}.log", "w") as log:
            code = subprocess.run(
                [sys.executable, "-u", __file__, "child", str(logs), name, *names],
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        seconds = time.monotonic() - start
        with lock:
            results[entry] = (code, seconds)
            state = "passed" if code == 0 else f"failed (exit code {code})"
            print(f"{entry}: {state} in {seconds:.1f} s", flush=True)
            # Print the end of a failed log at once. The full logs print
            # after all entries finish, and a killed job never gets there.
            if code != 0:
                tail = (logs / f"{name}.log").read_text(errors="replace").splitlines()[-80:]
                print("\n".join(f"  | {line}" for line in tail), flush=True)

    print(f"{args.name} shard {args.shard}: {len(units)} entries, {args.jobs} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(one, units))

    ran = {}
    failed = []
    for entry, _ in units:
        name = file_name(entry)
        print(f"::group::Python tests ({args.name}): {entry}")
        print((logs / f"{name}.log").read_text(errors="replace"), end="")
        print("::endgroup::")
        if results[entry][0] != 0:
            failed.append(entry)
        ids_file = logs / f"{name}.ids"
        ran[entry] = json.loads(ids_file.read_text()) if ids_file.exists() else []
    out = {
        "pass": args.name,
        "shard": args.shard,
        "count": args.count,
        "entries": {
            e: {"exit": results[e][0], "seconds": round(results[e][1], 1), "tests": ran[e]}
            for e, _ in units
        },
    }
    with open(logs / f"testids-{args.name}-{args.shard}.json", "w") as f:
        json.dump(out, f)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"### Python tests, {args.name} pass, shard {args.shard}\n\n")
            f.write("| Entry | Result | Tests | Seconds |\n|---|---|---|---|\n")
            for e, _ in sorted(units, key=lambda u: -results[u[0]][1]):
                code, seconds = results[e]
                state = "passed" if code == 0 else f"failed ({code})"
                f.write(f"| `{e}` | {state} | {len(ran[e])} | {seconds:.1f} |\n")
            total = sum(len(t) for t in ran.values())
            f.write(f"\nTests started: {total}\n\n")
    if failed:
        print(f"::error::Python test entries that failed ({args.name} pass): {' '.join(failed)}")
        sys.exit(1)


def verify(args):
    shards = load_map()
    errors = []
    lines = ["### Python tests in all shards", "", "| Pass | Shards | Tests started |", "|---|---|---|"]
    for name in PASSES:
        seen = {}
        found = 0
        for index in range(len(shards[name])):
            path = Path(args.dir) / f"testids-{name}-{index}.json"
            if not path.exists():
                errors.append(f"{name} shard {index} wrote no result ({path.name})")
                continue
            found += 1
            data = json.loads(path.read_text())
            for entry, info in data["entries"].items():
                for test in info["tests"]:
                    seen.setdefault(test, []).append(f"shard {index} {entry}")
        twice = {t: w for t, w in seen.items() if len(w) > 1}
        for test, where in sorted(twice.items()):
            errors.append(f"{name}: {test} ran more than one time: {where}")
        lines.append(f"| {name} | {found} of {len(shards[name])} | {len(seen)} |")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("\n".join(lines) + "\n\n")
    print("\n".join(lines))
    if errors:
        report(errors)


def report(errors):
    for error in errors:
        print(f"::error::{error}")
    sys.exit(1)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3], sys.argv[4:])
        return
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    p = sub.add_parser("run")
    p.add_argument("name", choices=PASSES)
    p.add_argument("shard", type=int)
    p.add_argument("count", type=int)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--logs", required=True)
    p = sub.add_parser("verify")
    p.add_argument("dir")
    args = parser.parse_args()
    if args.command == "check":
        errors = check_static(load_map())
        if errors:
            report(errors)
        print(f"{MAP.name}: every test file is in exactly one place in each pass.")
    elif args.command == "run":
        if not 0 <= args.shard < args.count:
            report([f"shard {args.shard} is not in 0..{args.count - 1}"])
        run(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
