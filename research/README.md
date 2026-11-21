# research/

Literature grounding for the project. The concrete CO2 factors and densities are here with a
citation each. Not everything on the live page is cited: 176 of the 232 numbers in the materials
table are handbook-level values recalled by the AI, and the preset strengths, ACI class C1, the
wear minimum and the service-life rule are uncited. The page labels all of them (see the root
README). **Chemistry and materials only — biology is out of scope.**

- `*.json` — raw output of the research agents behind the live page: exposure limits, the
  materials table and the frontier brief, with citations, quotes and an evidence label per value.
- `connectors.md` — the research MCPs to connect, and how to use them.
- `references/` — one file per domain, each a table of sourced numbers.
  - `concrete.md` — the live concrete app's factors (ported from `public/model/factors.json`).
  - `polymers.md`, `corrosives.md`, `cleaning.md`, `insulators.md` — the ambitious new domains.
  - `materials.md`, `chemistry.md` — general stubs.
- `templates/reference-entry.md` — the citation format, so sources stay consistent.

The six new-domain files hold no entries yet.

**Rule for the concrete factors and for new domains:** a factor without an entry here does not ship. Cross-check each with a second,
independent search before recording it.
