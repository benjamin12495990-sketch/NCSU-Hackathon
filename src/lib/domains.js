// The families of materials the project is scoped for (docs/SCOPE.md).
// Only concrete is built. The rest are plans: no dataset, model or references.
export const DOMAINS = [
  { id: 'concrete', label: 'Concrete', live: true, predict: 'Compressive strength at a chosen curing age', vary: 'Mix proportions', rankBy: 'Embodied CO₂' },
  { id: 'polymers', label: 'Carbon-based polymers', predict: 'Glass transition temperature, tensile strength, thermal stability', vary: 'Monomer choice, chain length and composition', rankBy: 'Synthesis cost, CO₂ and recyclability' },
  { id: 'corrosives', label: 'Corrosive solutions', predict: 'Corrosion rate and material compatibility', vary: 'Acid, base, salt and inhibitor, with concentration and temperature', rankBy: 'Material loss, hazard and cost' },
  { id: 'cleaning', label: 'Cleaning products', predict: 'Cleaning efficacy, critical micelle concentration, surface tension', vary: 'Surfactant blend, builders and solvent', rankBy: 'Cost, biodegradability and irritancy' },
  { id: 'insulators', label: 'Insulators', predict: 'Thermal conductivity and dielectric breakdown strength', vary: 'Composition, porosity and processing', rankBy: 'Cost, weight and CO₂' },
  { id: 'materials', label: 'General materials', predict: 'Any measured property, such as hardness, conductivity or density', vary: 'Composition and processing', rankBy: 'Cost, scarcity and CO₂' },
  { id: 'chemistry', label: 'General chemistry', predict: 'Solubility, boiling point, reaction yield, viscosity', vary: 'Molecular structure and mixture', rankBy: 'Synthesis difficulty and cost' },
]
