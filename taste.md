# taste.md: visual rules for this site

Paste this into any coding agent before it touches the UI. It is the short form of Hudson's
measured UI taste profile (24 forced-choice rounds, validated 3 of 3), adapted to this project.

**How to read it:** MUST rules are hard constraints. DEFAULT rules hold unless there is a stated
reason. TRY rules are working guesses; change them if the result looks wrong.

## The look in one line

Light, calm and flat, like Stripe or Notion: white cards, hairline borders, one vivid accent.
Not dark, not dense.

## MUST

- **Light theme only.** No dark mode.
- **No drop shadows on surfaces.** Separate things with 1px borders (`--line`).
- **Exactly two glows are allowed:** the soft coloured glow under the primary button, and the
  ambient glow behind the 3D stage. Nothing else glows.
- **One vivid accent**, used for the primary button, links, the selected mix and the highlighted
  ingredient in 3D. Everything else is ink, grey or white.
- **Numbers stay modest:** key figures are about 1.4x body size, weight 700. Emphasis comes from
  position and a small chip next to the number, never from huge numerals.
- **Honest labels.** Anything illustrative says so next to it (the 3D model does).

## DEFAULT

- **Accent colour:** `#6D4AFF` with `#B14AFF` for the button gradient. The appetite for a vivid
  accent is measured; this exact purple is a working value and can be swapped.
- **Type:** sans-serif everywhere in the product.
- **Borders:** `#E3E5EE`. Page background `#FBFBFD`. Text `#14161F`. Muted text `#5D6475`.
- **Spacing:** generous. Sections sit 32px apart; cards have 18 to 22px padding.

## TRY (context rules)

- **Corners:** soft (12px, pills for chips) on things a person types into or presses; sharp (6px)
  on data: result panel, tables, caveat box.
- **Serif:** only the page title, as the one "marketing moment". Never in data or controls.
- **Motion:** expressive on things you touch or that react to you; still on things you read.

## Where the theatrics go

All of the drama lives in the 3D stage, so the data stays readable:

1. 216 particles fly in from a wide sphere and pack into a one-cubic-metre cube (about 2 s).
2. The cube's outline fades in after they settle, and the cube turns slowly; dragging rotates it.
3. Cement is drawn in the accent colour and pulses, because it is most of the CO2.
4. Picking another mix (a headline mix or a table row) rebuilds the cube, so you see the cement
   particles thin out as slag and fly ash replace them.

The tables, the numbers and the "How this can be wrong" box do not animate.
Everything respects `prefers-reduced-motion`.

## Do not

- Add a dark hero, neon, glassmorphism or particle backgrounds.
- Animate table rows or count numbers up.
- Add a second accent colour.
- Hide or shrink the "How this can be wrong" box. It is part of the design.
