// One network trainer for the lab. seeded() and Mlp are copied unchanged from src/lib/nnFamily.js (itself a
// copy of src/lib/nn.js); inputs() is copied from nn.js createTrainer, so with options {} this trainer gives
// exactly the page's predictions (npm run lab:parity proves it). Added: getWeights()/setWeights() for warm
// starts, a self-contained weights object (fromWeights) for scoring later, and a pretrain(rows) seam.
import './guard.mjs'

// Seeded random numbers, so the same press gives the same training run.
function seeded(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Fully connected layers with ReLU between them and one linear output.
class Mlp {
  constructor(sizes, rand) {
    this.sizes = sizes
    this.L = sizes.length - 1
    const normal = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand())
    this.W = []; this.b = []; this.gW = []; this.gb = []
    this.mW = []; this.vW = []; this.mb = []; this.vb = []
    this.acts = [null]; this.deltas = [null]
    for (let l = 0; l < this.L; l++) {
      const nIn = sizes[l], nOut = sizes[l + 1]
      const W = new Float64Array(nIn * nOut)
      const scale = Math.sqrt(2 / nIn) // He initialisation, suited to ReLU
      for (let k = 0; k < W.length; k++) W[k] = normal() * scale
      this.W.push(W); this.b.push(new Float64Array(nOut))
      this.gW.push(new Float64Array(W.length)); this.gb.push(new Float64Array(nOut))
      this.mW.push(new Float64Array(W.length)); this.vW.push(new Float64Array(W.length))
      this.mb.push(new Float64Array(nOut)); this.vb.push(new Float64Array(nOut))
      this.acts.push(new Float64Array(nOut)); this.deltas.push(new Float64Array(nOut))
    }
    this.t = 0
  }

  // Runs one input through the layers and returns the single output.
  forward(x) {
    this.acts[0] = x
    let a = x
    for (let l = 0; l < this.L; l++) {
      const nIn = this.sizes[l], nOut = this.sizes[l + 1], W = this.W[l], b = this.b[l], out = this.acts[l + 1]
      const hidden = l < this.L - 1
      for (let j = 0; j < nOut; j++) {
        let s = b[j]
        const off = j * nIn
        for (let i = 0; i < nIn; i++) s += W[off + i] * a[i]
        out[j] = hidden && s < 0 ? 0 : s
      }
      a = out
    }
    return a[0]
  }

  // Backpropagation for the last forward(): adds this sample's gradients.
  backward(error) {
    this.deltas[this.L][0] = error
    for (let l = this.L - 1; l >= 0; l--) {
      const nIn = this.sizes[l], nOut = this.sizes[l + 1]
      const a = this.acts[l], W = this.W[l], gW = this.gW[l], gb = this.gb[l]
      const d = this.deltas[l + 1], dPrev = this.deltas[l]
      if (l > 0) dPrev.fill(0)
      for (let j = 0; j < nOut; j++) {
        const dj = d[j]
        if (dj === 0) continue
        gb[j] += dj
        const off = j * nIn
        for (let i = 0; i < nIn; i++) {
          gW[off + i] += dj * a[i]
          if (l > 0) dPrev[i] += dj * W[off + i]
        }
      }
      if (l > 0) for (let i = 0; i < nIn; i++) if (a[i] <= 0) dPrev[i] = 0 // ReLU passes no gradient when off
    }
  }

  // One Adam update from the gradients gathered over a batch.
  step(lr, batchSize, decay) {
    this.t++
    const b1 = 0.9, b2 = 0.999, eps = 1e-8
    const c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t
    const update = (w, g, m, v, shrink) => {
      for (let k = 0; k < w.length; k++) {
        const grad = g[k] / batchSize + shrink * w[k]
        m[k] = b1 * m[k] + (1 - b1) * grad
        v[k] = b2 * v[k] + (1 - b2) * grad * grad
        w[k] -= (lr * (m[k] / c1)) / (Math.sqrt(v[k] / c2) + eps)
        g[k] = 0
      }
    }
    for (let l = 0; l < this.L; l++) {
      update(this.W[l], this.gW[l], this.mW[l], this.vW[l], decay)
      update(this.b[l], this.gb[l], this.mb[l], this.vb[l], 0)
    }
  }
}

export { seeded, Mlp }

// Copied from src/lib/nn.js (net_parity.mjs checks the copy is identical).
export const PANEL_DESIGN = { features: 'physics', physicsOn: 'base', physics: ['abrams', 'bolomey', 'feret', 'bolomeyAge'], decay: 0.003, loss: 'huber' }
export const DEFAULTS = { hidden: [64, 32], epochs: 300, batch: 32, lr: 0.004, decay: 1e-3, seed: 11, features: 'base', loss: 'squared', kSlag: 0.9, kFly: 0.4 }
const FEATURES = ['cement', 'slag', 'flyash', 'water', 'superplasticizer', 'coarse', 'fine', 'age', 'water_binder_ratio']

// The network's inputs for one row of the nine features, exactly as src/lib/nn.js computes them.
// 'base' = nine inputs + ln(age); 'rich' adds engineer's ratios; 'physics' adds textbook strength laws.
export function makeInputs(opts) {
  const at = (name) => FEATURES.indexOf(name)
  const [iC, iS, iF, iW, iSp, iCa, iFa, iAge] = ['cement', 'slag', 'flyash', 'water', 'superplasticizer', 'coarse', 'fine', 'age'].map(at)
  return (row) => {
    const base = [...row, Math.log(row[iAge])]
    if (opts.features !== 'rich' && opts.features !== 'physics') return base
    const binder = row[iC] + row[iS] + row[iF], agg = row[iCa] + row[iFa]
    const rich = [...base, binder, row[iW] / row[iC], row[iS] / binder, row[iF] / binder, row[iSp] / binder, agg / binder, row[iFa] / agg, Math.sqrt(row[iAge])]
    if (opts.features !== 'physics') return rich
    const w = row[iW], t = row[iAge], ceq = row[iC] + opts.kSlag * row[iS] + opts.kFly * row[iF]
    const vBinder = row[iC] / 3.15 + row[iS] / 2.9 + row[iF] / 2.3 // litres of binder solids per cubic metre
    const fast = Math.exp(0.25 * (1 - Math.sqrt(28 / t))), slow = Math.exp(0.5 * (1 - Math.sqrt(28 / t))) // fib/CEB-FIP hardening curves
    const laws = {
      abrams: w / ceq, bolomey: ceq / w, feret: (vBinder / (vBinder + w)) ** 2, paste: (vBinder + w) / 1000, ageCurve: fast,
      bolomeyAge: (ceq / w) * fast, cementAge: (row[iC] / w) * fast, slagAge: (row[iS] / w) * slow, flyAge: (row[iF] / w) * slow,
    }
    return [...(opts.physicsOn === 'base' ? base : rich), ...(opts.physics || Object.keys(laws)).map((name) => laws[name])]
  }
}

// Trains one network on train = { X: rows of the nine features, y: MPa }. Same loop as nn.js createTrainer:
// standardise with the training rows, He init from seeded(seed), shuffle each epoch, Adam, lr decays to 10%.
// options.features: 'base' | 'rich' | 'physics'; options.loss: 'squared' | 'huber' (clip at 1 sd).
export function createNet(train, options = {}) {
  const opts = { ...DEFAULTS, ...options }
  const inputs = makeInputs(opts)
  const rawX = train.X.map(inputs)
  const n = rawX.length, d = rawX[0].length
  const mean = new Float64Array(d), sd = new Float64Array(d)
  for (const x of rawX) for (let i = 0; i < d; i++) mean[i] += x[i] / n
  for (const x of rawX) for (let i = 0; i < d; i++) sd[i] += (x[i] - mean[i]) ** 2 / n
  for (let i = 0; i < d; i++) sd[i] = Math.sqrt(sd[i]) || 1
  const yMean = train.y.reduce((s, v) => s + v, 0) / n
  const ySd = Math.sqrt(train.y.reduce((s, v) => s + (v - yMean) ** 2, 0) / n)
  return build(opts, { mean: [...mean], sd: [...sd], yMean, ySd }, train)
}

// Rebuilds a trained network from getWeights() output (no training rows needed): used to score saved models.
export function fromWeights(w) {
  const net = build(w.options, w.scale, null)
  net.setWeights(w)
  return net
}

function build(opts, sc, train) {
  const inputs = makeInputs(opts)
  const mean = Float64Array.from(sc.mean), sd = Float64Array.from(sc.sd), { yMean, ySd } = sc
  const d = mean.length
  const scale = (x) => Float64Array.from(x, (v, i) => (v - mean[i]) / sd[i])
  const X = train ? train.X.map((r) => scale(inputs(r))) : []
  const y = train ? train.y.map((v) => (v - yMean) / ySd) : []
  const n = X.length
  const rand = seeded(opts.seed)
  const sizes = [d, ...opts.hidden, 1]
  const net = new Mlp(sizes, rand)
  const order = Array.from({ length: n }, (_, i) => i)
  const predictScaled = (x) => net.forward(x) * ySd + yMean
  const trainer = {
    options: opts, layers: sizes, epoch: 0, totalEpochs: opts.epochs, warmStarted: false,
    weightCount: net.W.reduce((s, w) => s + w.length, 0) + net.b.reduce((s, b) => s + b.length, 0),
    // Seam for physics pre-training on synthetic rows (planned). For now it does nothing: it does not touch
    // the weights or draw from the random stream, so runs with and without it are identical.
    pretrain(rows) { return { rows: rows ? rows.length : 0, applied: false } },
    runEpochs(count) {
      let loss = 0
      for (let c = 0; c < count && trainer.epoch < opts.epochs; c++) {
        for (let i = n - 1; i > 0; i--) { // shuffle the rows
          const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]
        }
        const lr = opts.lr * (1 - 0.9 * (trainer.epoch / opts.epochs)) // slow down as it settles
        loss = 0
        for (let start = 0; start < n; start += opts.batch) {
          const end = Math.min(n, start + opts.batch)
          for (let k = start; k < end; k++) {
            const idx = order[k]
            const error = net.forward(X[idx]) - y[idx]
            loss += error * error
            net.backward(opts.loss === 'huber' ? Math.max(-1, Math.min(1, error)) : error)
          }
          net.step(lr, end - start, opts.decay)
        }
        trainer.epoch++
      }
      return { epoch: trainer.epoch, loss: (loss / Math.max(1, n)) * ySd * ySd }
    },
    train() { return trainer.runEpochs(opts.epochs - trainer.epoch) },
    predict(row) { return predictScaled(scale(inputs(row))) },
    predictMany(rows) { return rows.map((r) => predictScaled(scale(inputs(r)))) },
    // Everything needed to rebuild the trained network: options, input scaling, layer sizes, weights.
    getWeights() {
      return { format: 'lab-net-1', options: opts, scale: { mean: [...mean], sd: [...sd], yMean, ySd }, layers: sizes, W: net.W.map((w) => [...w]), b: net.b.map((b) => [...b]) }
    },
    // Warm start: copy weights of the same layer sizes. Adam's moments restart from zero (new optimiser run).
    setWeights(w) {
      if (!sameShape(w.layers, sizes)) throw new Error(`setWeights: layers ${w.layers} do not match ${sizes}`)
      w.W.forEach((src, l) => net.W[l].set(src)); w.b.forEach((src, l) => net.b[l].set(src))
      trainer.warmStarted = true
    },
  }
  return trainer
}

export const sameShape = (a, b) => a.length === b.length && a.every((v, i) => v === b[i])
// The layer sizes a genome or option set would give, without building a network.
export function layersFor(options) {
  const opts = { ...DEFAULTS, ...options }
  return [makeInputs(opts)(Array(9).fill(1)).length, ...opts.hidden, 1]
}
