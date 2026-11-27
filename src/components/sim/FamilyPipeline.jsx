import { useEffect, useState } from 'react'
import { loadPipeline } from '../../lib/pipeline.js'
import PipelineResults from './PipelineResults.jsx'
import PropertyFinder from './PropertyFinder.jsx'
import './sim.css'

// Added to the existing family pages (alloys, insulators, polymers): a requirement-driven finder over the rows the pipeline held out,
// then the pipeline's own results (network, strongest trees, their blend). Renders nothing until an export exists.
export default function FamilyPipeline({ family }) {
  const [index, setIndex] = useState(undefined)
  useEffect(() => {
    const ctl = new AbortController()
    loadPipeline('index.json', ctl.signal).then(setIndex, () => {})
    return () => ctl.abort()
  }, [])
  const tasks = (index?.tasks ?? []).filter((t) => t.site === family)
  if (!tasks.length) return null
  return (
    <>
      {index?.rehearsal && <section className="card fam-section"><p className="flag low">Rehearsal export: the pipeline numbers below mean nothing.</p></section>}
      <PropertyFinder family={family} index={index} tasks={tasks} />
      {tasks.map((t) => <PipelineResults key={t.id} task={t} />)}
    </>
  )
}
