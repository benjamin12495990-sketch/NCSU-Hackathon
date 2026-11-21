// A worker process for evolve2.mjs: receives { id, type: 'pretrain' | 'train', ... } and replies with the result.
// Each job is a pure function of its message, so results do not depend on which worker ran it or when.
import './guard.mjs'
import { buildPretrain, trainIndividual } from './evo2_core.mjs'
import { openArchive } from './archive.mjs'

process.on('message', (job) => {
  try {
    if (job.type === 'pretrain') { buildPretrain(job.genome, job.r); process.send({ id: job.id, ok: true }); return }
    let parentReplicas = null
    if (job.parent) parentReplicas = openArchive(job.parent.archive).loadWeights(job.parent.id, job.parent.sha).replicas
    const res = trainIndividual(job.genome, parentReplicas)
    process.send({ id: job.id, ok: true, res })
  } catch (e) {
    process.send({ id: job.id, ok: false, error: String(e && e.stack || e) })
  }
})
