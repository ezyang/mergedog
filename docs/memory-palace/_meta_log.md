---
name: Memory palace meta log
description: Tracks which transcript sessions have been ingested by UUID, so refresh works across multiple machines.
type: reference
originSessionId: 99555959-243d-47e3-a3cf-e0d88b509da7
---

This memory palace lives at `docs/memory-palace/` inside the mergedog repo
(synced via git). Transcripts live *outside* the repo at
`~/.claude/projects/-Users-ezyang-Nursery-mergedog/*.jsonl` and are
**machine-local** — each computer has its own set. We track ingested
sessions by UUID, not mtime, so a refresh on any machine just diffs that
machine's session UUIDs against the list below.

## Refresh procedure

1. `ls ~/.claude/projects/-Users-ezyang-Nursery-mergedog/*.jsonl`
2. Each filename's stem is the session UUID. Skip any UUID listed in
   "Ingested sessions" below.
3. For each new jsonl: extract `type=user` entries whose `message.content`
   is a string and is not `<command-name>`/`<local-command-caveat>`
   wrapped.
4. Fold new themes into existing files (edit, don't duplicate). If
   nothing new, just append the UUID to the list.
5. Append the UUID(s) to "Ingested sessions" with the date and a
   one-phrase theme.

## Ingested sessions

| Session UUID | Date | Theme |
|---|---|---|
| b135fc0f-abb3-41ee-94f3-28215568842a | 2026-05-03 | gh CLI capability check |
| 8136a079-4082-4cd3-9221-c51148d79874 | 2026-05-03 | Initial impl, mux/TUI, claude log filtering |
| d81a99f0-69cb-45d6-bf4b-035885cf7b8f | 2026-05-03 | Prompt-injection model, sanitizer, no fast mode |
| bdedbb54-dd2c-471f-b4e6-ef7a1fc2fc9d | 2026-05-03 | CI status, action_required gates |
| 81423a19-7a41-4225-afe3-f99ae48cad31 | 2026-05-03 | Cleanup/refactor pass |
| 6a47a9a8-dc0c-4abe-a1b9-e5957e504380 | 2026-05-03 | Edits-by-maintainers exception, handoff metadata |
| 9e88e8a6-eaa5-42b9-a39e-953a7d37ce65 | 2026-05-03 | Idempotency: don't re-post handoff |
| 99555959-243d-47e3-a3cf-e0d88b509da7 | 2026-05-03 | Memory-palace build |
