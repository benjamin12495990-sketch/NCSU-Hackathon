// The families of materials the project is scoped for (docs/SCOPE.md).
// Families with live: true are built (concrete has its own page; the trained families use
// FamilyPage.jsx with files in public/model/<id>/; the GPU-pipeline families use sim/SimPage.jsx with files in public/pipeline/).
// pipeline: true marks a family trained in the GPU pipeline (its "Trained offline" card in public/model/<id>/pipeline.json is
// shown only if its page is not live); noPlan: true means docs/SCOPE.md has no design plan for it yet. The rest are plans.
export const DOMAINS = [
  { id: 'nightly', label: 'All simulations: nightly results', live: true, overview: true, predict: 'Every task the nightly pipeline trains', vary: '-', rankBy: '-' },
  { id: 'concrete', label: 'Concrete', live: true, predict: 'Compressive strength at a chosen curing age', vary: 'Mix proportions', rankBy: 'Embodied CO₂' },
  { id: 'polymers', label: 'Polymers (glass transition)', live: true, predict: 'Glass transition temperature, tensile strength, thermal stability', vary: 'Monomer choice, chain length and composition', rankBy: 'Synthesis cost, CO₂ and recyclability' },
  { id: 'corrosives', label: 'Corrosive solutions', live: true, pipeline: true, predict: 'Corrosion rate and material compatibility', vary: 'Acid, base, salt and inhibitor, with concentration and temperature', rankBy: 'Material loss, hazard and cost' },
  { id: 'cleaning', label: 'Cleaning products (surfactants)', live: true, pipeline: true, predict: 'Cleaning efficacy, critical micelle concentration, surface tension', vary: 'Surfactant blend, builders and solvent', rankBy: 'Cost, biodegradability and irritancy' },
  { id: 'crystals', label: 'Crystals (DFT-calculated)', live: true, pipeline: true, noPlan: true, predict: 'Bulk and shear modulus, formation energy and band gap, all DFT-calculated', vary: 'No candidate space is defined yet', rankBy: 'No ranking objective is defined yet' },
  { id: 'insulators', label: 'Insulators (dielectric constant)', live: true, predict: 'Thermal conductivity and dielectric breakdown strength', vary: 'Composition, porosity and processing', rankBy: 'Cost, weight and CO₂' },
  { id: 'materials', label: 'Alloys (hardness)', live: true, predict: 'Any measured property, such as hardness, conductivity or density', vary: 'Composition and processing', rankBy: 'Cost, scarcity and CO₂' },
  { id: 'electrolytes', label: 'Solid electrolytes (Li-ion conductivity)', live: true, pipeline: true, predict: 'Room-temperature ionic conductivity of lithium solid electrolytes', vary: 'Composition (element fractions)', rankBy: 'Predicted conductivity, cautious estimate' },
  { id: 'chemistry', label: 'General chemistry (water solubility)', live: true, trainedOffline: true, predict: 'Solubility, boiling point, reaction yield, viscosity', vary: 'Molecular structure and mixture', rankBy: 'Synthesis difficulty and cost' },
]
