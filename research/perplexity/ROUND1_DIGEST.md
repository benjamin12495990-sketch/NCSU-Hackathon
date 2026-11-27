# Perplexity round 1: what it found and what it changes

Run on 2026-11-27 by the project owner: prompts F1 to F9, X1 to X3 and P1 from
`research/perplexity_prompts.md`. The raw answers are in `round1/`.

**Status of everything below: leads, not findings.** In almost every answer Perplexity says it ran
one round of searches and opened landing pages, not data files. Nothing here may be trained on or
cited as fact until someone has opened the data record and read its licence.
`computer_prompt_2_verify.txt` asks Perplexity's Computer mode to do exactly that.

Two of its claims could be checked against files in this repo, and both held:

- **Polymer file source.** Perplexity says `experimental_dataset.csv` comes from Zenodo
  10.5281/zenodo.19242815 (Bhati et al., Schrödinger, 2026; CC BY 4.0), and that the file is 64.1 kB
  with 666 points. Our copy is 64,139 bytes and has 666 data rows. [Likely] it is the same file. A
  hash comparison would settle it; ours is SHA-256 `e4ce0b9b…c939`.
- **Alloy file.** Perplexity mentions a "1,545-composition HEA dataset". That matches the 1,545
  rows of our `MPEA_dataset.csv` (Borg et al. 2020).

## Six things that change the plan

1. **Polymers: the big dataset is blocked by rights.** Perplexity reports that PoLyInfo's terms
   forbid reproduction or distribution without written permission from NIMS, and forbid bulk
   download. PolyMetriX has
   7,367 Tg rows, and 7,170 of them trace to PoLyInfo through GREA. Those rows stay out of the repo
   unless NIMS agrees in writing. The usable part is the 197 non-GREA rows, of which 182 are new to
   us.
2. **Solid electrolytes: there is no easy doubling.** OBELiX was built from the Liverpool LiIon
   database plus the Laskowski table, so merging LiIon adds few room-temperature rows. LiIon's terms
   allow academic use and state no open licence. Perplexity found no open data with conductivity at
   several temperatures, activation energies, or bulk and grain-boundary values kept apart. The one
   large lead is DigBat: 3,816 measured electrolytes and 27,645 conductivity entries according to a
   press release, terms unknown.
3. **Electrolytes: part of the error cannot be learned from composition.** One study reports
   conductivity rising from 3×10⁻⁴ to 1.1×10⁻³ S/cm as pellet density went from 85% to 98%. That is
   0.56 log units for the same material. Bulk conductivity is typically more than ten times the
   grain-boundary value. [Likely] a model that sees only composition has an error floor of about
   half a log unit or more. Our 90% range is about ±4 log units, so there is still much to gain, but
   not all of it.
4. **Crystals: the DFT label is now the larger error.** JARVIS's own comparison with experiment, as
   Perplexity reports it:

   | Property | DFT (OptB88vdW) against experiment | Materials compared |
   | --- | --- | --- |
   | Formation energy | MAE 0.128 eV/atom | 1,317 |
   | Band gap | MAE 1.33 eV (TBmBJ: 0.51 eV) | 54 |
   | Bulk modulus | MAE 5.75 GPa | 21 |

   Our network's formation-energy error against DFT is 0.048 eV/atom on held-out crystal systems.
   So against reality the DFT label is the bigger part of the error, and more DFT rows of the same
   kind buy little. [Likely] the better move for band gap is to train on the 21,559 TBmBJ gaps
   already on disk, because that label is much closer to experiment. These numbers still need to be
   confirmed at the source.
5. **Concrete: three new candidates, none confirmed for late ages.**
   - Blended Cement Concrete Database, Zenodo 10.5281/zenodo.18080351: 8,979 records from 260
     papers, with curing temperature and humidity and a source DOI per row. It was extracted by a
     language model, with over 80% checked by hand. Licence CC BY-NC 4.0, which forbids commercial
     use.
   - Pavement Concrete Datasets, Mendeley 10.17632/pz38ys9pdx.1: 1,430 strength records from
     transport-agency reports, CC BY 4.0.
   - UHPC mix-design database, Mendeley 10.17632/7wk5nd9jz8.1: 1,809 formulations, CC BY 4.0. It is
     a different strength regime, so it would be a separate task.

   Nobody has seen their age ranges. Under the seal rule, nobody on the project opens their strength
   values until the test rows are set aside.
6. **Noise floors: none of our errors is near one yet.**

   | Property | Spread between laboratories | Our error | Reading |
   | --- | --- | --- | --- |
   | Concrete cylinder strength (ASTM C39) | about 5% coefficient of variation | 9.07 to 11.50 MPa RMSE on the 56-to-365-day sealed test | Far above the floor |
   | Aqueous solubility | 0.24 log units for careful shake-flask work; 0.5 to 0.6 between typical labs | 0.761 log units MAE | Above the floor. [Likely] cleaner labels help more than more rows |
   | Polymer Tg by DSC | 2 to 5 °C (95% limit, same sample) | not compared here | The spread between sources is far larger than the instrument's |

   The C39 figures came from unofficial copies of the standard that disagree in one digit. The
   "0.5 to 0.6 between typical labs" figure is from our own research (Llompart et al. 2024), not
   from Perplexity.

## Leads by family

Licences are as Perplexity reported them. "Opened" says what Perplexity read.

| Family | Lead | Licence as reported | What it would add | Opened |
| --- | --- | --- | --- | --- |
| Concrete | Zenodo 10.5281/zenodo.18080351 | CC BY-NC 4.0 | 8,979 records; curing temperature and humidity | landing page |
| Concrete | Mendeley 10.17632/pz38ys9pdx.1 | CC BY 4.0 | 1,430 pavement-concrete strengths | landing page |
| Concrete | Mendeley 10.17632/7wk5nd9jz8.1 | CC BY 4.0 | 1,809 UHPC mixes, specimen geometry per row | landing page |
| Alloys | Kaggle, Raiipa tempering data | CC BY 4.0 | 1,466 steel tempering tests, hardness in HRC | file preview |
| Alloys | Zenodo record 23215479 | not seen | steel tempering, time and temperature | snippet only |
| Polymers | Zenodo 10.5281/zenodo.19242815 | CC BY 4.0 | the source and licence of the file we already use | landing page |
| Solubility | SolProp, zenodo.org/records/5970538 | CC BY 4.0 | aqueous solubility for 11,804 solutes; overlap with AqSolDB unknown | landing page |
| Solubility | Bradley melting points, figshare 1031637 | CC0 | 28,645 melting-point measurements, a feature for the general solubility equation | landing page |
| Surfactants | P2SURF, github.com/snlai4chembiogroup/P2SURF | LICENSE file not read | measured CMC, single and binary | landing page |
| Surfactants | PendantProp, Zenodo 10.5281/zenodo.17232763 | CC BY 4.0 | 11 surfactants, one robot, full isotherms | landing page |
| Corrosion | Mendeley 10.17632/ysr7wy3y7z.1 | CC BY 4.0 | 617 + 106 first-year atmospheric rates for carbon steel | landing page |
| Electrolytes | DigBat (digbat.org) | not found | 27,645 conductivity entries, if the terms allow | press release |
| Insulators | Starrydata2 dielectric subset | not stated | over 20,000 measured samples with temperature and frequency | repo page |
| Crystals | MinX in UniFFBench, github.com/M3RG-IITD/UniFFBench | MIT on the code; data not stated | about 1,500 minerals with measured elastic properties | repo page |

Perplexity found nothing open for these:

- hot hardness;
- copolymer Tg beyond the 351 rows we already have;
- aqueous solubility at several temperatures;
- cleaning performance of formulations;
- immersion corrosion data that states its test standard;
- measured breakdown strength;
- measured band gaps with a structure ID, method and temperature.

## Definitions to record before merging anything (X3)

- Concrete: cylinder strength is about 0.65 to 0.90 of cube strength, and the ratio changes with
  strength. Store shape and size; never convert with one factor.
- Hardness: nanoindentation read 10 to 30% above microhardness in one study. Keep the scales apart.
- Surfactants: for one surfactant, CMC was 1.8 mM by surface tension and 2.7 mM by fluorescence.
  Record the method and the added salt.
- Tg: DSC midpoint, DSC onset and the DMA peaks are different numbers. One unrated source puts the
  DMA tan-delta peak 10 to 20 °C above the DSC midpoint.

## Physics constraints worth testing (P1)

Perplexity's list of laws is in `round1/P1.md`. The parts it verified with numbers are few:

- The general solubility equation, with no fitted parameters, reached RMSE 0.53 log units on 1,026
  non-electrolytes. Our own research found it far worse on AqSolDB (Llompart et al. 2024), so it is
  a baseline to beat, not a shortcut.
- Monotonic constraints hold only near the training data and fade when extrapolating (a Sandia
  study on monotonic Gaussian processes).
- It found no verified numbers for constraints on concrete, Tg, CMC, corrosion or conductivity.
