# taste.md: visual rules for this site

Paste this into any coding agent before it touches the UI. It is the short form of Hudson's
measured UI taste profile (24 forced-choice rounds, validated 3 of 3), adapted to this project.

**How to read it:** MUST rules are hard constraints. DEFAULT rules hold unless there is a stated
reason. TRY rules are working guesses; change them if the result looks wrong.

## The look in one line

Light, calm and flat, like a printed lab report: white cards on warm paper, hairline borders, one
flat blue accent. Not dark, not dense.

**Revised after feedback that the first version looked AI-generated.** The purple gradient button,
both glows, the uppercase accent-coloured eyebrow, the serif hero and the row of pill chips with
status badges are gone. Simulations are now picked from the three-bar menu at the top left.

## MUST

- **Light theme only.** No dark mode.
- **No drop shadows, glows or gradients.** Separate things with 1px borders (`--line`).
- **One accent, a flat colour**, used for the primary button, links, the selected mix and the
  highlighted ingredient in 3D. Everything else is ink, grey or white.
- **Simulations (families of materials) are chosen from the three-bar menu in the top bar**, not
  from a row of pills in the page.
- **Numbers stay modest:** key figures are about 1.4x body size, weight 700. Emphasis comes from
  position and a small chip next to the number, never from huge numerals.
- **Honest labels.** Anything illustrative says so next to it (the 3D model does).

## DEFAULT

- **Accent colour:** `#1F3FA8` (hover `#183289`). A working value that can be swapped, but keep it
  one flat colour. It is also the colour of cement in the 3D stage.
- **Type:** the system sans-serif stack everywhere. No webfont, no serif.
- **Borders:** `#DEDCD3`. Page background `#F6F5F1`. Text `#1C1C19`. Muted text `#5E5D56`.
- **Spacing:** generous. Sections sit 32px apart; cards have 18 to 22px padding.

## TRY (context rules)

- **Corners:** 4px everywhere, on controls and on data alike. No pills.
- **Headings:** plain bold sans. No marketing hero, no uppercase tracked eyebrow labels.
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

- Add a dark hero, neon, glassmorphism, gradients, glows or particle backgrounds.
- Add pill chips or status badges to navigation. Put them in the menu as plain text.
- Animate table rows or count numbers up.
- Add a second accent colour.
- Hide or shrink the "How this can be wrong" box. It is part of the design.
