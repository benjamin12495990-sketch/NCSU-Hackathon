# General chemistry (molecules & mixtures)

> **Status: a water-solubility (LogS) model was trained and verified but is not on the site yet: its
> training file is about 2 MB (about 660 kB compressed), too slow for a phone.** `prepare.py`,
> `nn_generic.js`, `result.json` and `skeptic_check.json` here are the agent's files, copied unchanged;
> the data file and weights are not in the repo. The verified test numbers are in
> `public/model/chemistry/family.json`. Sources were not checked again through MCPs. The rest of this
> page is the original plan.

Molecular and mixture properties as the surrogate target — the non-biological, physico-chemical
end of chemistry.

- **Outcome:** solubility, boiling point, viscosity, reaction yield.
- **Datasets (confirm via MCPs):** QM9 (quantum properties), ESOL / Delaney (aqueous solubility),
  the **physico-chemical** subsets of MoleculeNet. Avoid its toxicity/bio tasks — out of scope.
- **Candidate space:** perturb molecular structure within the dataset's chemical space.
- **Models:** either model family, expected + cautious.
- **Cost:** synthesis difficulty, cost — cited in `../../research/references/chemistry.md`.

**Out of scope:** anything biological (binding, toxicity, metabolism). Keep to physics-and-chemistry
properties of molecules and mixtures.
