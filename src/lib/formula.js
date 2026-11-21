// Concrete is a mixture, so it has no true chemical formula. This gives the
// next best thing: the bulk composition of the paste (cement + slag + fly ash
// + water) written per silicon atom, like an empirical formula.
// Each binder's oxide analysis (mass percent) comes from oxides.json.
// Aggregates and superplasticizer are left out.
const OXIDES = ['CaO', 'SiO2', 'Al2O3', 'Fe2O3', 'MgO', 'SO3']

export function pasteFormula(mix, oxides) {
  const M = oxides.molar_masses_g_per_mol
  const mass = { cement: mix[0], slag: mix[1], flyash: mix[2] }
  // moles of each oxide in the binder (any consistent unit; only ratios matter)
  const n = Object.fromEntries(OXIDES.map((ox) => [ox, 0]))
  for (const [binder, kg] of Object.entries(mass)) {
    for (const ox of OXIDES) n[ox] += (kg * (oxides.binders[binder][ox] ?? 0)) / 100 / M[ox]
  }
  const atoms = {
    Ca: n.CaO, Si: n.SiO2, Al: 2 * n.Al2O3, Fe: 2 * n.Fe2O3, Mg: n.MgO, S: n.SO3,
    O: n.CaO + 2 * n.SiO2 + 3 * n.Al2O3 + 3 * n.Fe2O3 + n.MgO + 3 * n.SO3,
  }
  const perSi = (v) => v / atoms.Si
  return {
    // [element, atoms per silicon atom], leaving out anything under 0.005
    parts: ['Ca', 'Si', 'Al', 'Fe', 'Mg', 'S', 'O'].map((el) => [el, perSi(atoms[el])]).filter(([, v]) => v >= 0.005),
    water: perSi(mix[3] / M.H2O), // H2O molecules per silicon atom
    caSi: perSi(atoms.Ca),
  }
}

// Plain-text form, for example "Ca1.32 Si Al0.35 Fe0.05 Mg0.11 O4.02 · 2.61 H2O".
export function formulaText(f) {
  return f.parts.map(([el, v]) => (el === 'Si' ? 'Si' : el + v.toFixed(2))).join(' ') + ` · ${f.water.toFixed(2)} H2O`
}
