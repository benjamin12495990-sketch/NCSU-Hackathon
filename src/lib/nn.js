// A small neural network written from scratch, trained in the browser.
// It starts from random weights and learns to predict concrete strength from
// the same training rows the trees used. Nothing is downloaded pre-trained.

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

// Sets up a training run on nn_data.json. Call runEpochs(n) repeatedly; each
// call trains n more passes over the training rows and reports the error on
// the held-out test mixes, which the network never trains on.
export function createTrainer(nnData, options = {}) {
  const opts = { hidden: [64, 32], epochs: 300, batch: 32, lr: 0.004, decay: 1e-3, seed: 11, ...options }
  const ageIndex = nnData.features.indexOf('age')
  // The network gets the tree model's nine inputs plus ln(age): strength
  // rises roughly with the logarithm of curing time.
  const inputs = (row) => [...row, Math.log(row[ageIndex])]
  const trainX = nnData.train.X.map(inputs), testX = nnData.test.X.map(inputs)
  const n = trainX.length, d = trainX[0].length

  // Scale every input and the target using the training rows only.
  const mean = new Float64Array(d), sd = new Float64Array(d)
  for (const x of trainX) for (let i = 0; i < d; i++) mean[i] += x[i] / n
  for (const x of trainX) for (let i = 0; i < d; i++) sd[i] += (x[i] - mean[i]) ** 2 / n
  for (let i = 0; i < d; i++) sd[i] = Math.sqrt(sd[i]) || 1
  const yMean = nnData.train.y.reduce((s, v) => s + v, 0) / n
  const ySd = Math.sqrt(nnData.train.y.reduce((s, v) => s + (v - yMean) ** 2, 0) / n)
  const scale = (x) => Float64Array.from(x, (v, i) => (v - mean[i]) / sd[i])
  const X = trainX.map(scale), Xt = testX.map(scale)
  const y = nnData.train.y.map((v) => (v - yMean) / ySd)

  const rand = seeded(opts.seed)
  const net = new Mlp([d, ...opts.hidden, 1], rand)
  const order = Array.from({ length: n }, (_, i) => i)
  const predictScaled = (x) => net.forward(x) * ySd + yMean

  const testError = () => {
    let abs = 0, sq = 0, tot = 0
    const yt = nnData.test.y, m = yt.reduce((s, v) => s + v, 0) / yt.length
    for (let i = 0; i < Xt.length; i++) {
      const e = predictScaled(Xt[i]) - yt[i]
      abs += Math.abs(e); sq += e * e; tot += (yt[i] - m) ** 2
    }
    return { mae: abs / Xt.length, r2: 1 - sq / tot }
  }

  const trainer = {
    totalEpochs: opts.epochs,
    epoch: 0,
    trainRows: n,
    testRows: Xt.length,
    weights: net.W.reduce((s, w) => s + w.length, 0) + net.b.reduce((s, b) => s + b.length, 0),
    layers: [d, ...opts.hidden, 1],
    history: [{ epoch: 0, loss: null, ...testError() }], // before any training: random weights
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
            net.backward(error)
          }
          net.step(lr, end - start, opts.decay)
        }
        trainer.epoch++
        trainer.history.push({ epoch: trainer.epoch, loss: (loss / n) * ySd * ySd, ...testError() })
      }
      trainer.done = trainer.epoch >= opts.epochs
      return trainer.history[trainer.history.length - 1]
    },
    // Strength in MPa for one row of the tree model's nine features.
    predict(row) {
      return predictScaled(scale(inputs(row)))
    },
  }
  return trainer
}
