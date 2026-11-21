# Corrosive solutions & material compatibility

Given a material, propose solution chemistries (or inhibitors) that keep corrosion below a limit —
or, given a solution, rank which materials survive it.

## 1. The expensive outcome
Corrosion tests are slow: immersion / electrochemical cells run for days to months per
material–solution pair. Predictable targets:
- **Corrosion rate** (mm/year or mass loss) for a material in a solution.
- **Material compatibility class** (recommended / caution / not recommended).

## 2. Dataset (confirm current access via the research MCPs)
Corrosion data is scattered; assemble carefully and cite each source:
- Published immersion / electrochemical corrosion-rate tables (find via OpenAlex / Wiley).
- Standard material-compatibility charts (treat as reference data, cite the publisher).
- The electrochemical series and Pourbaix context for sanity checks.
Features: material identity + solution composition (acid/base/salt, concentration), temperature, pH.

## 3. Candidate space
Vary acid/base/salt identity and concentration, inhibitor dose, and temperature within tested
ranges; keep combinations physically sensible.

## 4. Models
Expected corrosion rate + a cautious **upper** bound (here caution means *over*-estimating attack).
Reuse either model family. Split so a material–solution system is never in both train and test.

## 5. Cost objective
Annualised material loss, hazard of the solution (GHS class), and cost. Factors cited in
`../../research/references/corrosives.md`.

## 6. Lab-supported check
Flag only when nearest measured material–solution systems agree.

---
**Safety:** this domain describes *corrosion resistance*, not how to cause harm. Keep it to
material-selection and protection. **Ground rules:** predictions as predictions; numbers cited;
never biology. See `../../docs/AGENTS.md`.
