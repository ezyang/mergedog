# Mergedog UX workstreams

Prompted by dogfooding feedback (2026-05-06). Each section is a
self-contained prompt you can hand to a future Claude session.

---

## 1. Natural-language CLI wrapper (claude-over-mux)

**Problem:** Users don't want to learn `add`, `rebase`, `cancel`, etc.
They want to say "watch this PR" or "what's going on with 182367?" and
get an answer. A Claude layer on top of the mux also serves as a
teaching tool: it can explain what it's doing in deterministic-CLI
terms, gradually building the user's mental model until they prefer the
fast path.

**Prompt:**

> Design and implement a way for users to interact with a running
> mergedog mux instance via natural language, mediated by Claude.
>
> Context: mergedog has a Textual TUI (`mergedog/mux.py`) with a
> command input bar. The mux accepts commands like `add <pr>`, `cancel
> <pr>`, `rebase <pr>`, `remove <pr>`, `log <pr>`, etc. (see the
> module docstring for the full list). Each PR has a log file at
> `~/.mergedog/logs/<pr>.log` and state at `~/.mergedog/state/<pr>.json`.
>
> The goal is a second entry point — probably `python -m mergedog.chat`
> or a subcommand — where a user can type things like:
>
> - "watch PR 182367"
> - "what's happening with my PRs?"
> - "rebase everything"
> - "why did 182500 fail?"
> - "is anything ready to merge?"
>
> and Claude translates that into mux commands + reads log/state files
> to answer questions. Critically, Claude should **show the user what
> deterministic command it ran** (e.g. "I ran `rebase 182367` for you —
> next time you can type that directly"). This is the teaching loop.
>
> Design constraints:
> - The mux is a long-running Textual app. The chat interface needs to
>   either (a) share the same process and add a second input mode, or
>   (b) be a separate process that sends commands to the mux via some
>   IPC (unix socket, file-based, stdin pipe, etc.).
> - Option (b) is probably simpler and more robust. The mux already
>   writes state to disk; a read-only query path just needs to read
>   those files. For mutations, the simplest approach might be a command
>   file or socket the mux polls.
> - Don't over-build. Start with the query side (reading logs + state
>   to answer "what's happening?") and a thin mutation layer (writing
>   commands to a file the mux picks up). The teaching UX (showing what
>   command was used) is the most important part.
>
> Start by proposing the IPC mechanism and the chat-side architecture.
> Don't implement until we agree on the approach.

**Open questions:**
- Should this be a Claude Code MCP server instead of a standalone CLI?
  That way users who already have Claude Code get mergedog as a tool.
- Should the chat wrapper also be able to trigger `@pytorchbot merge`
  on the user's behalf? (See workstream 5.)

---

## 2. Self-healing mux / auto-debug loop

**Problem:** When shepherds HALT, the operator today has to manually
read the log, figure out what went wrong, and either fix mergedog code
or restart. During early development, the workflow was "copy the error,
paste it to Claude, apply the fix." This pattern could be partially
automated: when a shepherd HALTs, offer to have Claude diagnose and
optionally patch.

**Prompt:**

> Design a "self-heal" mode for the mergedog mux.
>
> Context: each shepherd subprocess can HALT (exit non-zero) when it
> hits a condition it can't handle. The mux shows this as a 🔴 row.
> Today the operator reads `~/.mergedog/logs/<pr>.log`, diagnoses the
> issue, and either fixes mergedog code or restarts the shepherd.
>
> The idea: when a shepherd HALTs, the mux could optionally invoke
> Claude to read the log, diagnose the failure, and suggest (or apply)
> a fix. This is explicitly a development-time / early-adopter feature,
> not a production autopilot.
>
> Design constraints:
> - **Safety first.** Claude should never auto-apply code patches to
>   mergedog without human approval. The flow should be: diagnose →
>   show proposed fix → wait for confirmation → apply + restart.
> - This is different from the per-PR shepherd's Claude invocations
>   (which fix CI failures in pytorch). This is Claude fixing mergedog
>   itself. Keep the two completely separate.
> - The simplest version: a new mux command `debug <pr>` that tails
>   the log, feeds it to Claude with mergedog source context, and
>   prints a diagnosis. No auto-patching in v1.
> - Consider: should this just be documentation telling users to paste
>   the log into Claude Code, rather than a built-in feature? The
>   built-in version is slicker but the docs version is zero code.
>
> Propose the approach. Think about what's actually valuable here vs.
> what's over-engineering.

**Risk:** This is the most dangerous workstream. An auto-debug loop
that patches its own code is a footgun. v1 should be diagnosis-only.

---

## 3. Structured shepherd→mux IPC channel

**Problem:** The shepherd communicates with the mux via its last stderr
log line. This is clever but bandwidth-limited: the mux can only show
one line of text and has to parse `[APPROVED]`/`[MERGING]` prefixes
out of it. For richer display (phase icons, progress bars, structured
state), we need a proper data channel.

**Prompt:**

> Add a structured status channel from shepherd subprocesses to the
> mux, alongside the existing log-line channel.
>
> Context: today the mux reads the last line of each shepherd's stderr
> log (`~/.mergedog/logs/<pr>.log`) every 2 seconds. The shepherd
> embeds phase info as text prefixes (`[APPROVED]`, `[MERGING]`). See
> `mergedog/log.py` for the prefix logic and `mergedog/mux.py`
> `_refresh()` for the reader side.
>
> The goal: each shepherd writes a small JSON status file (e.g.
> `~/.mergedog/status/<pr>.json`) that the mux reads on each refresh.
> This file contains structured fields the mux can use for display:
>
> ```json
> {
>   "phase": "polling_ci",
>   "ci_done": 45,
>   "ci_total": 52,
>   "ci_failed": 2,
>   "approved": true,
>   "merging": false,
>   "last_claude_session": "2026-05-06T14:30:00",
>   "fix_attempts": 1,
>   "max_fix_attempts": 3
> }
> ```
>
> Design constraints:
> - The log-line channel stays as-is (it's the human-readable audit
>   trail). The JSON file is a parallel channel for machine-readable
>   state.
> - Write atomically (write to tmp + rename) to avoid partial reads.
> - The shepherd already has natural points where phase changes: see
>   `_refresh_status_prefix()` in `shepherd.py`, the CI polling loop,
>   Claude invocations, and the handoff flow in `handoff.py`.
> - The mux should use this to replace the text-prefix parsing and
>   enable richer display (workstream 4 will consume this).
> - Keep it minimal. Only add fields that the mux will actually use
>   in its display. Don't build a general-purpose event system.
>
> Implement the status file writer (shepherd side) and reader (mux
> side). Don't change the mux display yet — just wire up the data.

---

## 4. Mux display redesign (user journey)

**Problem:** The mux display was built bottom-up ("what's easy to
show") not top-down ("what does the user need to know"). Users can't
tell when they need to act, what phase a PR is in, or what mergedog
is doing. The traffic light (🟢/🔴) only means "process alive/dead"
which is an implementation detail, not a user-facing concept.

**Depends on:** Workstream 3 (structured IPC), at least partially.

**Prompt:**

> Redesign the mux display to be organized around the user's journey
> rather than implementation details.
>
> Context: the mux is a Textual TUI (`mergedog/mux.py`) showing a
> DataTable with columns: PR, Title, status-emoji, Last-log-line.
> The status emoji is 🟢 (alive), ✅ (exited 0), 🔴 (exited non-zero).
> Users don't know what these mean — a dogfooder thought 🟢 meant
> "ready to merge."
>
> Redesign goals:
>
> 1. **Phase, not process state.** Replace the traffic light with a
>    phase indicator that maps to user-meaningful states. Proposed
>    phases (use short text labels, not emoji overload):
>    - `CI` — polling/waiting for CI
>    - `FIX` — Claude is working on a failure
>    - `READY` — all green, waiting for human to trigger merge
>    - `MERGE` — pytorchmergebot is merging
>    - `HALT` — needs operator attention
>    Keep it to ~5 phases max. The log line provides detail.
>
> 2. **"Needs your action" stands out.** `READY` and `HALT` rows
>    should be visually distinct (Textual Rich styling — bold, color).
>    These are the only states where the user needs to do something.
>
> 3. **Don't overdo it.** The design pillar is "status pane, not app."
>    One row per PR, no modals, no mouse. More log space is better
>    than more columns. The phase label should be narrow (5-6 chars).
>
> 4. **Merge from the UI.** Add a `merge <pr>` command that posts
>    `@pytorchbot merge` as a comment on the PR. This is the obvious
>    action when a PR shows `READY`. The user still has to explicitly
>    type it — no auto-merge.
>
> 5. **Help command.** `help` in the input bar prints a quick reference
>    of phases and commands. Users shouldn't need external docs.
>
> If workstream 3 (structured IPC) isn't done yet, you can parse the
> phase from the existing log-line prefixes + process state as a
> stopgap. But design the display assuming structured data is coming.
>
> UX note from ezyang: "I don't want too many emojis, that makes it
> hard to parse." Prefer short text labels over emoji for phases.

---

## 5. Notifications and the merge trigger

**Problem:** gchat notifications exist (`mergedog/notify.py`) but
haven't been tested. Also, the handoff moment ("your PR is ready,
trigger merge") has no push notification — you have to be watching
the terminal.

**Prompt:**

> Wire up gchat notifications for the key user-facing moments, not
> just HALT.
>
> Context: `mergedog/notify.py` sends a gchat DM via `meta
> google.chat.message send` when a shepherd HALTs. The `--gchat-to`
> flag is passed through from the mux to each shepherd. This hasn't
> been tested in production yet.
>
> Extend notifications to cover:
>
> 1. **READY (handoff):** "PR #12345 is all green and ready for
>    `@pytorchbot merge`." Include the PR URL. This is the most
>    important notification — it's the moment the user needs to act.
>
> 2. **MERGING → merged:** "PR #12345 has been merged." Nice-to-have
>    confirmation.
>
> 3. **HALT** (already exists): keep as-is, but test it.
>
> Implementation:
> - `notify.py` currently only exports `notify_halt()`. Add
>   `notify_ready()` and optionally `notify_merged()`.
> - Call `notify_ready()` from the handoff flow in `handoff.py` (right
>   after posting the handoff comment).
> - Call `notify_merged()` from `watch_post_handoff` when the PR
>   closes (before the `die(EXIT_PR_NOT_ACTIONABLE)` call).
> - Keep the same fire-and-forget pattern (never raise on failure).
>
> Also: test the existing HALT notification manually end-to-end. Run
> a shepherd with `--gchat-to <your-username>`, force a HALT, and
> verify the DM arrives.

---

## 6. User education and WIP-PR positioning

**Problem:** Two related issues. First, users think mergedog is only
for merge-ready PRs, but it's also useful for WIP CI tending. Second,
users might blindly accept MERGEDOG commits without reading them,
which is the disempowerment failure mode the trust model is designed
to prevent.

**Prompt:**

> Improve mergedog's first-run experience and in-flow education to
> address two problems:
>
> 1. **Users don't know mergedog works on WIP PRs.** They think they
>    need reviewer approval before adding a PR. In reality, mergedog
>    can tend CI on any open PR — rebasing, classifying failures,
>    requesting trunk CI — even while the PR is still in review.
>    Communicate this in the README, the `help` command (workstream 4),
>    and the initial "added PR" log message.
>
> 2. **Users need to actually read MERGEDOG commits.** The handoff
>    comment already lists what Claude did, but users might skip it.
>    Consider:
>    - Making the handoff comment more prominent about this. E.g.:
>      "**You must review the following commits before merging:**"
>      followed by a diff summary or commit list with links.
>    - Adding a `review <pr>` mux command that shows the MERGEDOG
>      commits and their diffs in the terminal, so reviewing is easy.
>    - README/docs language about the operator's responsibility.
>
> This is primarily a docs + messaging task, not a code task. The main
> code change would be the `review` command if we decide to add it.
> Start by drafting the user-facing copy (README section, handoff
> comment wording, help text) and propose whether `review <pr>` is
> worth building.

---

## 7. Contribution model for prompts

**Problem:** The repo is vibe-coded so code quality doesn't matter
much, but the prompts in `mergedog/prompts.py` are load-bearing and
need real review. Also need a way to verify prompt changes work (hard
to unit test — you need to run against real CI failures).

**Prompt:**

> Design a contribution and review process for mergedog's prompts.
>
> Context: `mergedog/prompts.py` contains the system/user prompts that
> drive Claude's CI-fixing behavior. These are the most important
> "code" in the repo — a bad prompt change can cause mergedog to
> produce wrong fixes, miss failures, or violate trust invariants.
> The rest of the codebase is vibe-coded and churn is fine.
>
> The challenge: prompt changes can't be verified by unit tests. You
> need to run the shepherd against real failing CI and see if the fix
> is correct. The test matrix is effectively "every type of CI failure
> pytorch has."
>
> Propose:
>
> 1. **A lightweight prompt review checklist.** What should a reviewer
>    look for? (Injection risk, trust model violations, regression on
>    known failure classes, etc.)
>
> 2. **A testing protocol.** How should a contributor demonstrate their
>    prompt change works? Minimum: run against N real CI failures and
>    show before/after Claude output. Could we maintain a corpus of
>    "golden" CI logs + expected Claude behavior?
>
> 3. **What goes in CONTRIBUTING.md.** The message to contributors:
>    the code is whatever, the prompts matter, here's how to test.
>
> This is a docs/process task. Draft the artifacts, don't write code.
