# research/

Literature grounding for the project. Every number the models do **not** learn from data lives
here with a citation. **Chemistry and materials only — biology is out of scope.**

- `connectors.md` — the research MCPs to connect, and how to use them.
- `references/` — one file per domain, each a table of sourced numbers.
  - `concrete.md` — the live concrete app's factors (ported from `public/model/factors.json`).
  - `polymers.md`, `corrosives.md`, `cleaning.md`, `insulators.md` — the ambitious new domains.
  - `materials.md`, `chemistry.md` — general stubs.
- `templates/reference-entry.md` — the citation format, so sources stay consistent.

**Rule:** a factor without an entry here does not ship. Cross-check each with a second,
independent search before recording it.
