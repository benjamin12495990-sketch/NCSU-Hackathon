import { useEffect, useState } from 'react'
import { loadPipeline } from '../../lib/pipeline.js'
import PipelineResults from './PipelineResults.jsx'
import CorrosionDesigner from './CorrosionDesigner.jsx'
import ElectrolyteDesigner from './ElectrolyteDesigner.jsx'
import SurfactantFinder from './SurfactantFinder.jsx'
import CrystalFinder from './CrystalFinder.jsx'
import PropertyFinder from './PropertyFinder.jsx'
import '../family.css'
import './sim.css'

// The page for a family trained in the GPU pipeline (corrosion, cleaning products, solid electrolytes, crystals, chemistry):
// first the family's designer, which is the interactive part, like the concrete designer: say what you need, get ranked candidates
// with a cautious estimate and their measured neighbours. Then the pipeline's own results for every task of the family: the network,
// the strongest trees and their blend on rows they never trained on. Numbers come from public/pipeline/ (gpu/export_site.py).
const DESIGNERS = { corrosives: CorrosionDesigner, electrolytes: ElectrolyteDesigner, cleaning: SurfactantFinder, crystals: CrystalFinder, chemistry: PropertyFinder }

const INTRO = {
  corrosives: 'Say what the part will be exposed to. The page ranks the materials in NIST CORR-DATA by their predicted corrosion rate in that environment, with a cautious upper estimate, and shows the measured records behind each one.',
  electrolytes: 'Say how conductive a lithium solid electrolyte must be. The page proposes compositions made by nudging measured ones, predicts their room-temperature ionic conductivity, and shows the nearest measured compositions.',
  cleaning: 'Say what the surfactant must do: how low its critical micelle concentration and how low the surface tension it reaches. The page lists SURFPRO surfactants that meet it, measured where measured and predicted (marked) where not.',
  crystals: 'Say which DFT-calculated properties a crystal must have. The page screens the crystals the pipeline held out of training, with the network, the trees and their blend, and shows each unit cell in 3D.',
  chemistry: 'Say how soluble in water a compound must be. The page lists compounds the model never trained on that are predicted to meet it, with a cautious estimate, beside their measured values.',
}

export default function SimPage({ family, domain, onBack }) {
  const [index, setIndex] = useState(undefined)
  useEffect(() => {
    const ctl = new AbortController()
    loadPipeline('index.json', ctl.signal).then(setIndex, () => {})
    return () => ctl.abort()
  }, [])
  const Designer = DESIGNERS[family]
  const tasks = (index?.tasks ?? []).filter((t) => t.site === family)
  return (
    <>
      <section className="card fam-head">
        <h1>{domain?.label ?? family}</h1>
        <p className="lede">{INTRO[family]}</p>
        {index?.rehearsal && <p className="flag low">Rehearsal export: these numbers come from a tiny test run and mean nothing.</p>}
      </section>
      {Designer && <Designer family={family} index={index} tasks={tasks} />}
      {index === null && (
        <section className="card fam-section">
          <p className="note">The pipeline&apos;s results have not been exported to the site yet. On the training laptop, after a nightly run:
            {' '}<code>python -m gpu.export_site</code>, then commit <code>public/pipeline/</code>.</p>
        </section>
      )}
      {tasks.map((t) => <PipelineResults key={t.id} task={t} />)}
      <section className="card fam-section warn-box">
        <h2>How this can be wrong</h2>
        <ul>
          <li>Every number is a prediction from data that already exists. It narrows the list of experiments; it does not replace one.</li>
          <li>The 90% ranges are calibrated on rows like the ones tested. A material unlike anything in the data can fall outside them more often.</li>
          {family === 'crystals' && <li>All crystal labels are DFT calculations, not measurements, and this test set was scored in earlier cycles (exploratory).</li>}
          {family === 'corrosives' && <li>CORR-DATA rates are historical point values of mixed quality. This is not safety advice.</li>}
          {family === 'electrolytes' && <li>The typical miss is large (see the interval): good for a direction, not for ranking close candidates.</li>}
          {family === 'cleaning' && <li>These are properties of single pure surfactants, not the cleaning performance of a product.</li>}
        </ul>
      </section>
      <button type="button" className="chip" onClick={onBack}>Open the live concrete designer</button>
    </>
  )
}
