# Corrosive solutions & material compatibility

> **Status: raw archive parsed into a normalised table; one EXPLORATORY reference-grouped baseline. No product
> model.** **These are historical observations, not safety advice.** NIST states that each CORR-DATA record is
> one observation of one material under one set of conditions, "not ... advice on the usability of any material
> in any environment". Nothing here, measured or predicted, says a material is safe in a service environment.

Given a material, propose solution chemistries (or inhibitors) that keep corrosion below a limit, or, given a
solution, rank which materials survive it (long-term goal; original plan below).

## Data and parsing

`work/data/CORR-DATA_Database.zip` (NIST CORR-DATA, NIST Open Data licence; SHA-256 in `work/data/README.md`)
is read **in memory** by `work/parse_corrdata.py`; it is never unpacked into `work/data/` (`--cache` copies the
CSV to the gitignored `work/cache/` if you want to look at it). The parser uses the archive's
`columnDefinitions.txt` and writes:

- `work/cleaned/corrdata_normalized.csv` (24,721 rows, 6.5 MB): material, material class (Material Group with
  spelling and spacing variants merged: 16 raw values to 10), family, UNS; environment; concentration (value or range, unit, qualifier, raw
  text); temperature in C (value or range, qualifier, source column, raw C and F text); exposure duration in
  hours (raw text kept); rate in mm/yr with the raw value, raw unit and conversion factor; the letter rating
  kept separately; localized attack and comments as text; `reference_number` as the group.
- `work/cleaned/corrdata_references.csv`: reference number to citation.
- `work/reproductions/corrdata_parse_report.json`: all counts below, conversion factors, rules and examples of
  unparsed values.

Rules, in short: only plain numbers become point values; "x max" is an upper bound, "x min" a lower bound,
"x-y" a range; words (Boiling, Room, Saturated, Concentrated ...) stay qualifiers with no invented number; values
that look like spreadsheet date artefacts ("5-Oct") are not reinterpreted. mils/yr are converted with
1 mil/yr = 0.0254 mm/yr, deg F with (F - 32) x 5/9, durations with 1 d = 24 h and 1 yr = 8,766 h. Ratings A-D are
**not** converted to numbers; their source definitions (A < 0.002 in/yr, B < 0.020, C 0.020-0.050, D > 0.050) are
recorded in the report.

### Coverage

| Field | Parsed (numeric) | Qualitative only | Blank | Unparsed |
| --- | --- | --- | --- | --- |
| Rate | 7,718 (5,914 point values, 1,563 upper bounds, 223 lower bounds, 18 ranges) | 16,648 letter ratings (A 9,694, B 2,486, C 1,675, D 2,793) | 350 | 5 |
| Temperature | 14,422 (13,331 point, 838 range, 253 bounds or "room to x"; 63 taken from the F column) | 6,931 (Boiling 6,026, Room 711, Hot, Molten, Cold ...) | 3,356 | 12 |
| Concentration | 12,089 (11,752 in the column's default vol %, 234 "%", 73 g/L, 14 wt %, 12 "m") | 3,065 (Saturated, All, Anhydrous, Concentrated ...) | 9,519 | 48 |
| Duration | 4,541 (h, d, yr) | | 20,175 | 5 |

- Numeric rates by source unit: 7,684 from the mm/yr column, 34 converted from mils/yr.
- Where both rate columns are plain numbers (5,805 rows), 5,640 agree within rounding, 38 differ by less than 2x
  and **127 differ by more than 2x** (flagged in `rate_columns_consistent`; the mm/yr value is kept).
- Distinct: 240 reference numbers (25 rows have none), 702 material names, 288 environments, 10 material
  classes, 88 families. Material names are not harmonised ("316", "Type 316" stay distinct).

## Exploratory baseline (`work/reproductions/corrdata_baseline_group_cv.json`)

Target: log10 rate in mm/yr. Rows: 5,761 point values > 0 (excluded: 1,804 bounds or ranges, 23 zeros, 127 with
large column disagreement, 3 without a reference). Features: one-hot material class, family, material and
environment (categories seen fewer than 10 times in a training fold pooled), concentration in vol % when that is
the unit, temperature (value or range midpoint), log hours, and the concentration and temperature qualifiers.
Encoders and imputers are fitted inside each training fold. 5-fold `GroupKFold`.

| Model | By Reference # (207 groups, largest 851 rows) | By material name (557 groups) |
| --- | --- | --- |
| mean | 1.03 | 1.03 |
| Ridge alpha 10 | **0.78 (selected)**, R² 0.39 | 0.75 |
| Ridge alpha 1 | 0.79 | 0.76 |
| ExtraTrees 300, leaf 3 | 0.80 | **0.69 (selected)**, R² 0.49 |

MAE in log10 units. Holding out whole source documents, a linear model on simple encodings improves on the mean
by 0.25 log units (reference-bootstrap 95% interval 0.17 to 0.33); 0.78 log units is still a typical factor of
about 6 in rate. Trees do not help when whole sources are held out, which suggests much of the signal is
source-specific. Holding out material names is easier (0.69), partly because near-identical alloys under
different names stay on both sides.

## Limits, in plain language

- **Not safety advice.** Single historical observations (the programme ended in 1997) of stochastic processes;
  small changes in environment, impurities, aeration, flow, welds or heat treatment change the outcome.
- The regression uses only 23% of rows. Most records are letter ratings or bounds; an ordinal or censored
  model is needed to use them, and none has been built yet.
- Point rates span 2.5e-5 to 58,000 mm/yr; extreme values are kept as recorded and were not audited.
- "Boiling" and "Room" temperatures carry no number, so the model cannot use them except as flags.
- Environment names are free text (288 values); concentrations mix units, and many are blank.
- This is exploratory: the model settings were fixed by hand and compared on the same folds they are reported on.

## Reproduce

```
python3 -I domains/corrosives/work/parse_corrdata.py      # ~3 s; reads the zip in memory
python3 -I domains/corrosives/work/baseline_corrdata.py   # ~1 min on 4 shared CPUs
```

Outputs are never silently replaced: an existing file with different content gets a new time-stamped name, and
every baseline run writes its own report (data SHA-256, command, versions, folds, per-model metrics, selected
model, timing).

## Original plan

## 3. Candidate space
Vary acid/base/salt identity and concentration, inhibitor dose, and temperature within tested
ranges; keep combinations physically sensible.

## 4. Models
Expected corrosion rate + a cautious **upper** bound (here caution means *over*-estimating attack).
Reuse either model family. Split so a material–solution system is never in both train and test.

## 5. Cost objective
Annualised material loss, hazard of the solution (GHS class), and cost. Factors cited in
`../../research/references/corrosives.md`.

## 6. Neighbour check
Flag only when nearest measured material–solution systems agree.

---
**Safety:** this domain describes *corrosion resistance*, not how to cause harm. Keep it to
material-selection and protection. **Ground rules:** predictions as predictions; numbers cited;
never biology. See `../../docs/AGENTS.md`.
