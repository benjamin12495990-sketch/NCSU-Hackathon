// Generic copy of the small from-scratch network used by the concrete model.
// seeded() and the Mlp class below are copied verbatim from
// /workspace/ncsu-hackathon/src/lib/nn.js (lines 5 to 98).

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

// ---------------------------------------------------------------------------
// Generic trainer around Mlp (everything above this line is copied verbatim
// from src/lib/nn.js). No dataset-specific inputs: it takes rows of numbers.
//   train = { X: [[...]], y: [...] }
//   options: hidden, epochs, batch, lr, decay, seed, loss ('squared'|'huber'),
//            logTarget (fit ln(y), report exp of the output),
//            inputClip (0 = off), clampOutput (false = off)
// Inputs and target are standardised with the training rows only.
function createTrainer(train, options = {}) {
  const opts = { hidden: [64, 32], epochs: 300, batch: 32, lr: 0.004, decay: 1e-3, seed: 11, loss: 'squared', logTarget: false, ...options }
  const n = train.X.length, d = train.X[0].length
  const mean = new Float64Array(d), sd = new Float64Array(d)
  for (const x of train.X) for (let i = 0; i < d; i++) mean[i] += x[i] / n
  for (const x of train.X) for (let i = 0; i < d; i++) sd[i] += (x[i] - mean[i]) ** 2 / n
  for (let i = 0; i < d; i++) sd[i] = Math.sqrt(sd[i]) || 1
  const toTarget = opts.logTarget ? Math.log : (v) => v
  const fromTarget = opts.logTarget ? Math.exp : (v) => v
  const t = train.y.map(toTarget)
  const yMean = t.reduce((s, v) => s + v, 0) / n
  const ySd = Math.sqrt(t.reduce((s, v) => s + (v - yMean) ** 2, 0) / n) || 1
  // Two optional guards for rows unlike anything in training (an alloy with a
  // lot of an element the training rows barely contain): cap each standardised
  // input at +-inputClip, and keep the prediction inside the range of targets
  // seen in training. Without them a ReLU network extrapolates without limit.
  const clip = opts.inputClip > 0 ? opts.inputClip : Infinity
  const yLo = Math.min(...train.y), yHi = Math.max(...train.y)
  const scale = (x) => Float64Array.from(x, (v, i) => Math.max(-clip, Math.min(clip, (v - mean[i]) / sd[i])))
  const X = train.X.map(scale)
  const y = t.map((v) => (v - yMean) / ySd)

  const rand = seeded(opts.seed)
  const net = new Mlp([d, ...opts.hidden, 1], rand)
  const order = Array.from({ length: n }, (_, i) => i)

  const trainer = {
    totalEpochs: opts.epochs,
    epoch: 0,
    trainRows: n,
    weights: net.W.reduce((s, w) => s + w.length, 0) + net.b.reduce((s, b) => s + b.length, 0),
    layers: [d, ...opts.hidden, 1],
    history: [],
    done: false,
    runEpochs(count) {
      for (let c = 0; c < count && trainer.epoch < opts.epochs; c++) {
        for (let i = n - 1; i > 0; i--) { // shuffle the rows
          const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]
        }
        const lr = opts.lr * (1 - 0.9 * (trainer.epoch / opts.epochs)) // slow down as it settles
        let loss = 0
        for (let start = 0; start < n; start += opts.batch) {
          const end = Math.min(n, start + opts.batch)
          for (let k = start; k < end; k++) {
            const idx = order[k]
            const error = net.forward(X[idx]) - y[idx]
            loss += error * error
            // Huber loss: beyond one standard deviation a miss counts linearly.
            net.backward(opts.loss === 'huber' ? Math.max(-1, Math.min(1, error)) : error)
          }
          net.step(lr, end - start, opts.decay)
        }
        trainer.epoch++
        trainer.history.push({ epoch: trainer.epoch, loss: loss / n }) // in standardised target units
      }
      trainer.done = trainer.epoch >= opts.epochs
      return trainer.history[trainer.history.length - 1]
    },
    // Prediction in the target's own units for one row of raw features.
    predict(row) {
      const p = fromTarget(net.forward(scale(row)) * ySd + yMean)
      return opts.clampOutput ? Math.max(yLo, Math.min(yHi, p)) : p
    },
  }
  return trainer
}

// Several networks that differ only in their random starting weights; the
// prediction is their average. Same interface as createTrainer.
function createEnsemble(train, options = {}) {
  const { members = 5, seed = 11, ...rest } = options
  const nets = Array.from({ length: members }, (_, i) => createTrainer(train, { ...rest, seed: seed + 101 * i }))
  const ensemble = {
    members,
    totalEpochs: nets[0].totalEpochs,
    epoch: 0,
    trainRows: nets[0].trainRows,
    weights: nets[0].weights * members,
    layers: nets[0].layers,
    history: [],
    done: false,
    runEpochs(count) {
      let last
      for (const t of nets) last = t.runEpochs(count)
      ensemble.epoch = nets[0].epoch
      ensemble.history.push({ epoch: ensemble.epoch, loss: last.loss })
      ensemble.done = nets[0].done
      return ensemble.history[ensemble.history.length - 1]
    },
    predict: (row) => nets.reduce((s, t) => s + t.predict(row), 0) / nets.length,
  }
  return ensemble
}

// Mean absolute error, root mean squared error and R2 of predictions p against y.
function metrics(p, y) {
  const n = y.length, m = y.reduce((s, v) => s + v, 0) / n
  let abs = 0, sq = 0, tot = 0
  for (let i = 0; i < n; i++) { const e = p[i] - y[i]; abs += Math.abs(e); sq += e * e; tot += (y[i] - m) ** 2 }
  return { mae: abs / n, rmse: Math.sqrt(sq / n), r2: 1 - sq / tot }
}

export { seeded, Mlp, createTrainer, createEnsemble, metrics }
