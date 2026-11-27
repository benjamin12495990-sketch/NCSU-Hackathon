// The families of materials the project is scoped for (docs/SCOPE.md).
// Families with live: true are built (concrete has its own page; the trained families use
// FamilyPage.jsx with files in public/model/<id>/). pipeline: true marks a family trained and scored
// offline in the GPU pipeline, with its numbers in public/model/<id>/pipeline.json; noPlan: true means docs/SCOPE.md has
// no design plan for it yet. The rest are plans.
export const DOMAINS = [
  { id: 'concrete', label: 'Concrete', live: true, predict: 'Compressive strength at a chosen curing age', vary: 'Mix proportions', rankBy: 'Embodied CO₂' },
  { id: 'polymers', label: 'Polymers (glass transition)', live: true, predict: 'Glass transition temperature, tensile strength, thermal stability', vary: 'Monomer choice, chain length and composition', rankBy: 'Synthesis cost, CO₂ and recyclability' },
  { id: 'corrosives', label: 'Corrosive solutions', pipeline: true, predict: 'Corrosion rate and material compatibility', vary: 'Acid, base, salt and inhibitor, with concentration and temperature', rankBy: 'Material loss, hazard and cost' },
  { id: 'cleaning', label: 'Cleaning products', pipeline: true, predict: 'Cleaning efficacy, critical micelle concentration, surface tension', vary: 'Surfactant blend, builders and solvent', rankBy: 'Cost, biodegradability and irritancy' },
  { id: 'crystals', label: 'Crystals (DFT-calculated)', pipeline: true, noPlan: true, predict: 'Bulk and shear modulus, formation energy and band gap, all DFT-calculated', vary: 'No candidate space is defined yet', rankBy: 'No ranking objective is defined yet' },
  { id: 'insulators', label: 'Insulators (dielectric constant)', live: true, predict: 'Thermal conductivity and dielectric breakdown strength', vary: 'Composition, porosity and processing', rankBy: 'Cost, weight and CO₂' },
  { id: 'materials', label: 'Alloys (hardness)', live: true, predict: 'Any measured property, such as hardness, conductivity or density', vary: 'Composition and processing', rankBy: 'Cost, scarcity and CO₂' },
  { id: 'chemistry', label: 'General chemistry', trainedOffline: true, predict: 'Solubility, boiling point, reaction yield, viscosity', vary: 'Molecular structure and mixture', rankBy: 'Synthesis difficulty and cost' },
]
