import { pasteFormula, formulaText } from '../lib/formula.js'

// Shows the paste formula of a mix with real subscripts, for example
// Ca₁.₃₂SiAl₀.₃₅O₄.₀₂ · 2.61 H₂O. Silicon is the reference atom, so it has no number.
export default function Formula({ mix, oxides }) {
  const f = pasteFormula(mix, oxides)
  return (
    <span className="chem" aria-label={`Paste formula ${formulaText(f)}`}>
      {f.parts.map(([el, v]) => (
        <span key={el}>{el}{el !== 'Si' && <sub>{v.toFixed(2)}</sub>}</span>
      ))}
      {' · '}{f.water.toFixed(2)} H<sub>2</sub>O
    </span>
  )
}
