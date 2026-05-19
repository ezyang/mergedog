mergedog is a tool for making it easier to land OSS PR contributions to
pytorch/pytorch.  The primary design goals:

- **Adoption.** Once we have acknowledged a PR as good, we take full control
  over the rest of the process of shepherding the PR in.  The original author
  is not involved.

- **Security.** In fact, it is actively harmful if the external contributor
  intervenes after we start the shepherding process, as they can introduce
  unsafe code after approval.  Interventions from an untrusted user are ignored,
  or, in the worst case, this halts the merge process and we have to take over.

- **Autonomous.** I want to ask for a PR to be merged, and then I don't want
  to touch it until it's done.

## How it works

1. We mark a PR as approved.  The exact commit which we approved indicates trust.
   We cannot land an untrusted commit.

2. We get CI to be green.  To have been approved, this indicates that we trust an
   LLM to solve any pending CI problems.  This is the bulk of the autonomous process;
   mergedog:

   a. Approves the CI run
   b. Polls for CI results
   c. Decides if the CI results indicate real failures, and if they do, adds a commit
      with the changes, clearly annotated with `[MERGEDOG]` in the commit title.

   We assume any CI failures can be addressed without having to test locally.
   If mergedog fails to successfully one-shot a CI failure, this means it is too
   complicated and we should quit out for human intervention.

   When plain CI is passing, mergedog applies `ciflow/trunk` and same applies.

3. We wait for a human to review the mergedog commits and manually trigger a
   pytorchbot merge.

mergedog supports both regular fork PRs and ghstack PRs. For ghstack, fixes
are amended into the contributor's orig commit and re-uploaded with
`ghstack submit -m`; staleness is resolved by rebasing onto `origin/main`
rather than by merging.

mergedog is implemented as a traditional software harness that shells into
a local LLM CLI to actually issue the fixes.  Codex is the default; Claude
Code and MetaCode are also supported.

## Running mergedog

### mux (recommended)

The mux is a Textual-based TUI that supervises multiple shepherd processes.
Each PR gets its own subprocess; the mux shows a live table of PR status,
accepts commands, and auto-prunes PRs that merge or close.

```
python -m mergedog.mux [<pr>...] [--repo OWNER/NAME] [--resume-known|--no-resume-known] [--ignore-sev] [--manage-mergedog-label] [--root DIR]
```

- With no PR arguments, the mux restarts every job in the tracked list
  (`~/.mergedog/mux-jobs.json`, falling back to `~/.mergedog/mux-prs.json`).
- `--resume-known` also includes tracked jobs when PRs are provided.
- `--no-resume-known` starts only the PRs provided on the command line, or no
  jobs if none are provided.
- `--ignore-sev` tells all spawned shepherds to skip the `ci: sev` check.
- `--manage-mergedog-label` tells all spawned shepherds to add the `mergedog`
  label at startup and remove it on exit. By default, shepherds do not mutate
  this label.
- `--root DIR` redirects all on-disk state to a different directory (also
  settable via `MERGEDOG_ROOT` env var).
- `--repo OWNER/NAME` targets a GitHub repo other than the default
  `pytorch/pytorch`.

By default mergedog targets `pytorch/pytorch`. `--repo` is the preferred way
to select another GitHub repo; `MERGEDOG_REPO` / `MERGEDOG_REPO_SLUG` remain
available as environment defaults for scripts. `MERGEDOG_REPO_SSH_URL`
overrides the clone URL; otherwise mergedog derives
`git@github.com:owner/name.git`. Non-PyTorch repos skip the PyTorch-only hooks
(`ci: sev`, `ciflow/trunk`, Dr. CI, autolabeling, and `pytorchmergebot`
handoff recovery).

TUI commands (type in the input bar at the bottom):

| Command | Effect |
|---|---|
| `add <pr>` or just `<pr>` | Start shepherding a PR |
| `stack <pr>` or `stack add <pr>` | Start shepherding a ghstack stack |
| `stack rebase <pr>` | Start shepherding a stack with `--rebase` |
| `stack restart <pr>` | Kill and re-spawn a stack |
| `stack cancel <pr>` | SIGTERM the stack shepherd (keeps state) |
| `stack remove <pr>` | SIGTERM + wipe the `stack-<pr>` worktree, state, context |
| `stack log <pr>` | Print the stack log file path |
| `cancel <pr>` | SIGTERM the shepherd (keeps state) |
| `restart <pr>` | Kill and re-spawn |
| `restart all` | Kill all shepherds and respawn |
| `restart dead` | Re-spawn only crashed shepherds |
| `remove <pr>` | SIGTERM + wipe worktree, state, context |
| `rebase <pr>` | Shorthand for `add <pr> --rebase` |
| `rebase all` | Kill all shepherds and respawn with `--rebase` |
| `reassess <pr>` | Re-invoke Claude for previously-spurious failures |
| `mark-spurious <pr>` | Snapshot current failed/cancelled checks as spurious and restart |
| `log <pr>` | Print the log file path |
| `ignore-sev [on\|off]` | Toggle SEV parking for future spawns |
| `mergedog-label [on\|off]` | Toggle `mergedog` label management for future spawns |
| `migrate` | Print resume instructions for moving to another host |
| `quit` | Terminate everything |

### Natural-language interface (Claude Code)

Talk to a running mux in plain English via Claude Code:

```
python -m mergedog.chat
```

This launches Claude Code with the mergedog MCP server pre-configured.
You can say things like "watch PR 182367", "what's happening with my
PRs?", "rebase everything", or "why did 182500 fail?" and Claude
translates that into mux commands and log reads.  It shows you the
deterministic command it ran, so you gradually learn the CLI.

Requires `claude` on your PATH and a running mux instance.  Honors
`--root` / `MERGEDOG_ROOT` if you're using a non-default root.

### Single PR (one-off)

Run a shepherd for a single PR in the foreground:

```
mergedog <pr_number_or_url> [--repo OWNER/NAME] [flags]
# or equivalently:
python -m mergedog <pr_number_or_url> [--repo OWNER/NAME] [flags]
```

This is useful for one-off runs, debugging, or when you want to pass flags
that are specific to a particular PR (especially `--extra-context`).

### ghstack stacks

```
mergedog stack <any_pr_in_stack> [flags]
```

Discovers the full stack from the PR body and drives every PR bottom-up in
a single process.  Additional flag: `--force-ghstack` bypasses ghstack's
anti-clobber check.

### Rage bundle

```
mergedog rage <pr_number_or_url> [--root DIR]
mergedog rage stack <any_pr_in_stack> [--root DIR]
```

Creates a private markdown paste with redacted diagnostics for the PR.  The
bundle includes the mux/shepherd log, persisted trust/state JSON, context
sidecar, mux tracking list, pushed-commit records for that PR, and local
regular/stack worktree branch/HEAD/status summaries.  For non-bottom stack
members, the bundle also lists existing `worktrees/stack-*` candidates
because stack mode names the shared worktree after the bottom PR.  Before
upload, mergedog applies a best-effort credential scrub over the entire
bundle.

For ghstack runs, use `mergedog rage stack <any_pr_in_stack>`.  It resolves
the full stack, uses one stack log keyed by the bottom PR
(`logs/stack-<bottom_pr>.log`), and includes every member's state/context plus
stack-wide pushed commits and the shared stack worktree summary.

### Common flags

These work on both the single-PR and stack entry points:

| Flag | Purpose |
|---|---|
| `--rebase` | Upfront merge/rebase onto `origin/main` before polling CI |
| `--accept-divergence` | Proceed even if PR head differs from the approval commit |
| `--ignore-sev` | Don't park on open `ci: sev` issues |
| `--reassess` | Re-invoke Claude for failures previously judged spurious |
| `--manage-mergedog-label` | Add the `mergedog` label at startup and remove it on exit |
| `--extra-context TEXT` | Operator hint injected into Claude's fix prompt (trusted) |
| `--extra-context-file PATH` | Same, but reads from a file (mutually exclusive with above) |
| `--root DIR` | Override on-disk root (`~/.mergedog`) |
| `--repo OWNER/NAME` | Target a GitHub repo other than `pytorch/pytorch` |

### LLM provider

By default, mergedog shells out to `codex` with the provider CLI's default
model. To switch future agent invocations, persist the provider in
`~/.mergedog/config.json`:

```bash
mergedog config llm codex
mergedog config llm metacode --model provider/model
mergedog config llm claude --clear-model
```

`mergedog config llm` prints the current setting.  Supported providers are
`claude`, `codex`, and `metacode`.

## Customizing Claude's prompt

The prompts Claude sees are in `mergedog/prompts.py`. They are not
user-configurable files — they're code.

For per-run steering, use `--extra-context` (or `--extra-context-file`).
The text you provide is injected into a trusted "operator context" section
of the fix prompt, so Claude treats it as authoritative instructions.
Examples:

```bash
# Tell Claude to ignore a known pre-existing lint failure
mergedog 12345 --extra-context "The mypy error in torch/_dynamo/foo.py is pre-existing; ignore it."

# Point Claude at a detailed playbook
mergedog 12345 --extra-context-file ~/playbooks/inductor-ci-fixes.md
```

Inside the mux, extra-context isn't directly exposed per-PR. If you need it,
run the PR as a one-off outside the mux, or contribute a mux command for it.

## On-disk layout

Everything lives under `~/.mergedog/` (override with `--root` or `MERGEDOG_ROOT`):

```
~/.mergedog/
├── repo/                     # shared bare clone of the configured repo
├── worktrees/<pr>/           # per-PR git worktrees
├── worktrees/stack-<pr>/     # shared worktree for ghstack stacks
├── state/<pr>.json           # trust DB, spurious verdicts, last failure
├── contexts/<pr>.md          # sidecar: PR title/body/comments (untrusted, fed to Claude)
├── logs/<pr>.log             # per-PR shepherd stdout/stderr (written by mux)
├── logs/stack-<pr>.log       # stack shepherd log, named by bottom PR
├── mux-prs.json              # backwards-compatible regular PR tracking list
├── mux-jobs.json             # mux's tracked regular PR + stack job list
├── mux.lock                  # flock'd by the running mux (IPC discovery)
├── mux.sock                  # Unix socket for IPC (same commands as TUI)
├── config.json               # persistent operator settings (LLM provider/model)
├── label-cache.json          # cached repo labels for autolabeling (24h TTL)
├── lintrunner-venv/          # shared lintrunner virtualenv
└── pushed-commits.log        # append-only log of pushed commits (stack mode)
```

## Security: taint tracking

All data from GitHub (PR body, comments, commit messages, CI logs) is wrapped
in `TaintedStr` — a `str` subclass that propagates taint through string
operations.  Prompt construction sites call `assert_untainted()`, so a
`TaintError` is raised if untrusted data reaches a prompt without going through
an explicit declassification point (`untaint()`).  This prevents prompt
injection from external contributors.  See `mergedog/taint.py` for the full
design and the "Handling TaintError" guide.
