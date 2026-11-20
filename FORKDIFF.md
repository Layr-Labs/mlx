# Fork diff: what this fork changes, and keeping that page honest

This repository is a fork of [`ml-explore/mlx`](https://github.com/ml-explore/mlx).
Everything it changes relative to upstream is published as a browsable page:

**https://layr-labs.github.io/mlx/**

The page is rendered by [`protolambda/forkdiff`](https://github.com/protolambda/forkdiff)
from [`fork.yaml`](fork.yaml) at the repo root, in the style of
[op-geth's go-ethereum fork diff](https://op-geth.optimism.io/). `fork.yaml`
groups the changed files into sections with a paragraph each, names the exact
upstream commit the fork is based on (`base.hash`), and lists files that are
not code (`ignore`). Every push to `main` re-renders and redeploys it
(`.github/workflows/forkdiff-pages.yml`).

**One-time setup.** GitHub Pages has to be switched on by a repo admin:
Settings → Pages → Source: **GitHub Actions**. The workflow token cannot do
this itself. Until it's done the deploy workflow still builds the page (kept
as a run artifact, `forkdiff-page`) and exits with a notice instead of failing.

## Native CI

The Linux lint job, ASAN/UBSAN tests, fork-diff check and page renderer use only
Darkbloom Linux ARM64 runners. All eight Mac jobs in `fork-tests.yml` use only
Darkbloom macOS ARM64 runners. Each names its VM size ([Job sizes](#job-sizes)).
They keep their original steps, tests and job timeouts.
The sanitizer jobs use the qualified image's compilers and dependencies, with a
tool preflight in place of installing packages through sudo, and are capped
at 60 minutes. When compatible capacity is busy they wait; these jobs have no GitHub-hosted or Blacksmith
fallback. The current profile qualification lifetime is one hour, and the
existing watchdog can cancel or retry a job after its 30-minute wait threshold.
The watchdog can rerun a stranded workflow once, then cancel another stranded
attempt. These jobs keep native labels on a rerun; the watchdog cannot send them
to hosted runners. These bounds are not a promise of a 24-hour native queue.
The `DARKBLOOM_ROUTING` variable controls the separate dynamic route workflow;
it does not turn off these native-only jobs.

The Fedora aarch64 job runs the unchanged Fedora script on a Darkbloom Linux
ARM64 runner. The sandbox has no Docker, so `.github/scripts/fedora-proot/`
unpacks the archived Fedora 42 container rootfs (pinned by digest) and runs the
script under PRoot, built from pinned source with its missing ARM64 `faccessat2`
entry added. PRoot fakes root through ptrace and is not an isolation boundary;
the job VM is. The Fedora x86_64 job keeps its hosted runner: emulating x86 on
ARM was not qualified. The GitHub-hosted Pages job still packages and publishes
the rendered artifact through its protected environment. If its first deploy
fails before it creates a deployment (the packaged artifact is not listed yet),
it waits 20 seconds and tries once more. The disabled native CodeQL workflow keeps its separate routing
and activation rules.

### Job sizes

Each Darkbloom macOS job names its VM size in its runner label, for example
`darkbloom-macos-arm64-4x16` (4 vCPUs, 16 GiB). A job without a size gets the
size that the approval pins (12 vCPUs, 32 GiB). The image pipeline signs
4x16, 4x24, 8x16 and 12x32. A job gets the smallest signed size that has 30%
headroom over its measured memory peak (used memory: app + wired +
compressed). A job that took less than 10 minutes must stay under 10 minutes.

The "Peak used" column is the sampler's peak of app + wired + compressed
memory in pull request run 44750715218. The idle VM uses about 6 GiB.

| Job | Size | Wall time on 12 vCPUs (runs 43368507444 to 43569480732) | Hosted 3 vCPUs, 7 GiB | Run 44750715218: size, wall time, peak used |
|---|---|---|---|---|
| `build-wheel` | 4x16 | 153 to 354 s | 575 s (run 43120201853, `macos-26`) | 4x16, 230 s, 10.65 GiB |
| `build-cpp` | 4x16 | 277 to 620 s; p90 about 480 s. The slow runs spend up to 190 s in the build cache action. | 940 s, of which the compile is 632 s (run 43120201853) | 8x16, 302 s, 10.42 GiB (`air-lld` 6.52). The sampler scales the job's 282 sampled seconds to about 335 s on 4 vCPUs (+19%), so the estimated p90 on 4x16 is about 540 s. |
| `python-gpu` (2 shards) | 4x16 | 171 to 541 s | 297 to 635 s (run 43531506709, `xcode-27`) | 4x16, 214 and 288 s, 11.65 and 10.29 GiB |
| `python-cpu` (3 shards) | 4x24 | 152 to 444 s | 279 to 546 s | 4x16, 194 to 258 s, 12.37 to 13.20 GiB, with 5.2 to 7.1 GiB compressed and memory pressure at warning once. 16 GiB is less than 1.3 times that peak, so the shards get 24 GiB. |
| `metal-build-and-test` | 4x16 | 34 to 190 s | 20 to 23 s | 4x16, 36 s, 6.01 GiB |

`test_init_reduce_large_output` needs a 4 GiB Metal buffer on the GPU and a
`memory_size` of 16 GiB on the CPU. The 4x16 GPU shards (a Metal buffer can
have half of the 16 GiB) and the 4x24 CPU shards give both, so the test still
runs. Run 44750715218 ran it on both, at 4x16.

Every macOS job runs `.github/scripts/ci-resource-sampler.py`, a copy of the
sampler of Layr-Labs/d-inference (wrapped to this repository's ruff rules), from
its checkout on. Its last step,
"Report the resource samples", prints the memory peak of each class, the
three largest processes, the busy vCPUs and the sampled time scaled to 4 and
8 vCPUs. Use these numbers to change a size. Check Lint runs
`.github/scripts/test-runner-sizes.py`. It fails when a macOS job names no
size, a size that is not signed, or a size other than the one its table
gives.

## The gate

A fork-diff page is only useful while it is true, and two things make it go
stale silently. `scripts/check_forkdiff.py` runs on every PR and on every push
to `main` (`.github/workflows/forkdiff-check.yml`) and fails on both:

| Drift | Check | Why it matters |
|-------|-------|----------------|
| **Merge of newer upstream** | `base.hash` must equal `git merge-base HEAD upstream/main` and be an ancestor of upstream `main` | After a merge the old base makes upstream's own commits look like fork changes: thousands of files, none of them ours. |
| **New fork change nobody described** | every path in `git diff --name-only base.hash HEAD` must match a section glob or a global `ignore` | An undescribed file is a change the page can't explain. |
| **Section describing code we no longer carry** | every glob must match at least one changed path | Stale sections are as misleading as missing ones. |

The check also renders the page, so a `fork.yaml` that forkdiff itself rejects
cannot merge. The deploy workflow runs the same check before publishing, so a
stale analysis is never served.

## Day to day

**Adding or changing fork files in a PR.** If the check lists uncovered files,
add each to the section that explains it in `fork.yaml` (or start a new
section with a short description). Files that are not code go under the
top-level `ignore`. Keep globs specific: a `mlx/**` catch-all would swallow
upstream's changes after a bad rebase and defeat the gate.

**Upstream sync policy (2026-10-22).** The fork merges upstream `main` when
David directs it. Use a merge commit. Do not rebase. Each merge does these steps:

1. Set `base.hash` to the new merge-base, `git merge-base HEAD upstream/main`.
2. Read each `fork.yaml` section again. Remove a section when upstream now has
   the same change. Change a section when the merge changed the fork code.
3. Delete again the upstream workflows and actions that the fork removes. The
   "Fork tooling" section lists them.
4. Run the check below until it passes.

Between merges, port an upstream fix that the fork needs as its own commit.
Describe the port in its own `fork.yaml` section. The check compares the fork
with `base.hash`:

```
python3 -m pip install pyyaml
python3 scripts/check_forkdiff.py --upstream-ref refs/remotes/upstream/main
```

**Previewing the page locally** (Go 1.21+):

```
go run github.com/protolambda/forkdiff@v0.1.1 -repo . -fork fork.yaml -out tmp/index.html
open tmp/index.html
```

## Design notes

- `base.hash` is a full 40-hex commit id, never a branch name: a symbolic base
  would move underneath the page and the gate alike.
- Section `ignore` lists count as coverage (forkdiff still lists those files,
  grayed out); the top-level `ignore` is for things that aren't code at all.
- The glob semantics are forkdiff's: `*` and `?` stop at `/`, `**` spans
  directories (and may match none), `[!x]` negates a class.
- The check is pure git + PyYAML so the same command runs locally and in CI.
