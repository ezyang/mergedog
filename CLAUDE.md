# Python environment

If no virtual environment is active, activate `.venv` before running Python or installing packages:

```
source .venv/bin/activate
```

# Project memories

Before doing repo-specific operations, check for matching project memories under `~/.codex/memories/` (for example, push command shapes or local workflow exceptions).

# Temporary scripts

Put temporary scripts and scratch files under `agent_space/`.

# Network access

Codex shell commands may not have working external internet access, even when the local machine does. Do not assume GitHub/API failures from Codex-run commands reproduce in the production instance.

A separately running `python -m mergedog.mux` process may have internet access and can successfully run shepherd operations that Codex cannot run directly.

# Commit cadence

Commit proactively at natural breakpoints — don't wait to be asked.

**Why:** ezyang Ctrl-Cs often (uncommitted work is at risk) and reads commit logs as status reports.

**How to apply:** after each coherent unit (fix, refactor pass, feature step), commit. Match repo style: short imperative title, optional one-line body — no PR-style paragraphs. Prefer too-small splits over too-large.

**Authorization:** committing locally is pre-authorized — just do it, don't ask. (Pushing still requires confirmation.)

# Production logs

The "production instance" is just the local `python -m mergedog.mux` process running on this machine. Each shepherd's stdout/stderr is captured to `~/.mergedog/logs/<pr>.log` (see `mergedog/mux.py:_spawn`). On-disk per-PR state lives at `~/.mergedog/state/<pr>.json` (head branch + clone URL + trusted SHAs). Read these directly when investigating a HALT — no remote system to log into.
