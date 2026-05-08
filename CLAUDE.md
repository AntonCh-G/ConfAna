@AGENTS.md

## Claude Code

- Use plan mode for non-trivial changes or when modifying multiple modules.
- Treat unresolved TODO items in AGENTS.md as blocking scientific assumptions; do not guess them.
- Prefer project skills under `.claude/skills/` when relevant instead of repeating long procedures here.
- Keep edits modular and reviewable.
- Run targeted tests after each major module change.

## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Issues once a GitHub remote is configured. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default five-label triage vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repo using root `CONTEXT.md` and root `docs/adr/`. See `docs/agents/domain.md`.
