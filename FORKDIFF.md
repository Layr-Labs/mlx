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
Darkbloom macOS ARM64 runners with the approved 12-core, 32-GiB profile.
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
