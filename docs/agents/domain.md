# Domain Docs

This is a single-context repo.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root.
- **`docs/adr/`** for architectural decision records relevant to the area being changed.

If any of these files don't exist, proceed silently. Don't flag their absence or suggest creating them upfront; the producer skill (`/grill-with-docs`) creates them lazily when terms or decisions get resolved.

## Use the glossary's vocabulary

When output names a domain concept in an issue title, refactor proposal, hypothesis, or test name, use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept needed isn't in the glossary yet, either reconsider the language or note the gap for `/grill-with-docs`.

## Flag ADR conflicts

If output contradicts an existing ADR, surface the conflict explicitly rather than silently overriding it.
