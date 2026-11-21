// Synthetic (mix, age) -> strength rows from textbook physics, for pre-training. Two generators:
//  'powers': Powers' gel-space law f = s * A * x^n, x = 0.68 a / (0.32 a + W / c_eq(t)), with the degree of
//            hydration a(t) = a_u * exp(-(tau / 24t)^beta) and an equivalent cement c_eq(t) = C + k_s(t) S + k_f(t) F.
//  'fib':    f(t) = f28 * exp(s (1 - sqrt(28 / t))), s = 0.20 + 0.35 * (slag + fly ash share of binder), where f28
//            is the Powers strength at 28 days times its own scale.
// The only free number in each generator (the scale s) is fitted by least squares on TRAIN rows (ages 1-7 d), never
// on SELECT. Synthetic mixes are convex combinations of two TRAIN mixes with the same set of ingredients present
// (slag yes/no, fly ash yes/no, superplasticizer yes/no), so they stay inside TRAIN's ranges and combinations.
// Ages are log-uniform from 1 to 365 days. Each parameter carries its confidence label and citation (PARAMS),
// copied from the lab's physics research notes (physics.json); meta() returns them with the calibration.
import '../guard.mjs'
import { seeded } from '../net.mjs'

const WL = 'Wang X.-Y., Luan Y. (2018) Int J Concr Struct Mater, doi:10.1186/s40069-018-0241-z'
const SF = 'Schindler A.K., Folliard K.J. (2005) ACI Mater J 102(1), doi:10.14359/14246; Mills R.H. (1966) HRB Special Report 90'
export const PARAMS = {
  A_cement_MPa: { value: 104.9, confidence: 'confirmed_by_tool', citation: WL },
  A_slag_MPa: { value: 151.9, confidence: 'confirmed_by_tool', citation: WL },
  A_flyash_MPa: { value: 104.9, confidence: 'own assumption (WL2018 gives no fly-ash value; cement value used, fly ash enters through k_f)', citation: WL },
  n_cement: { value: 2.21, confidence: 'confirmed_by_tool', citation: WL },
  n_slag: { value: 2.6, confidence: 'confirmed_by_tool', citation: WL },
  n_flyash: { value: 2.21, confidence: 'own assumption (cement value)', citation: WL },
  blend_rule: { value: 'A and n = binder-mass-fraction weighted', confidence: 'confirmed_by_tool', citation: WL },
  gel_space_form: { value: 'x = 0.68 a / (0.32 a + w/c)', confidence: 'partly_confirmed_form_confirmed_constants_from_memory', citation: 'Powers (1958) via ' + WL },
  alpha_u: { value: 'min(1, 1.031 w/b / (0.194 + w/b) + 0.50 p_FA + 0.30 p_SLAG)', confidence: 'from_memory_unconfirmed', citation: SF },
  tau_h: { value: 13, range: [10, 20], confidence: 'from_memory_unconfirmed (generic OPC value)', citation: SF },
  beta: { value: 0.8, range: [0.6, 1.1], confidence: 'from_memory_unconfirmed (generic OPC value)', citation: SF },
  k_slag_t: { value: '0.95 (1 - exp(-t/6)) (1.15 - 0.6 p_SLAG) + 0.05', confidence: 'own heuristic, shaped on Bougara et al. 2010 (confirmed_by_tool: k 0.7 at 5 d, 1.25 at 28 d for 50% slag)', citation: 'Bougara et al. (2010) Cem Concr Compos' },
  k_flyash_t: { value: '0.15 + 0.75 (1 - exp(-t/45)) (1.1 - p_FA)', confidence: 'own heuristic, shaped on Murumi & Gupta 2015 (confirmed_by_tool: k 0.54 at 25%, 0.35 at 45%, 28 d)', citation: 'Murumi & Gupta (2015); Babu & Rao (1996) doi:10.1016/0008-8846(96)00011-7' },
  fib_curve: { value: 'exp(s (1 - sqrt(28/t)))', confidence: 'partly_confirmed_form_confirmed_constants_from_memory', citation: 'fib Model Code 2010; EN 1992-1-1 cl. 3.1.2' },
  fib_s: { value: '0.20 + 0.35 p_SCM', confidence: 'own heuristic, untested (fitted by eye to BOxCrete 22 C age ratios, confirmed_by_tool)', citation: 'Baten et al. (2026) arXiv:2603.21525' },
  per_mix_variation: { value: 'tau ~ U(10, 20) h, beta ~ U(0.65, 0.95), n +- U(-0.1, 0.1), multiplicative noise CV 5%', confidence: 'design choice, following the research note: randomise so the network learns a family, not one curve', citation: 'lab physics notes, section 8' },
}

const B = (m) => m[0] + m[1] + m[2]
// Unscaled Powers strength (MPa) of mix m = [C, S, F, W, ...] at age t days, with kinetic parameters k.
export function powersRaw(m, t, k = { tau: 13, beta: 0.8, dn: 0 }) {
  const [C, S, F, W] = m, b = B(m), ps = S / b, pf = F / b, wb = W / b
  const alphaU = Math.min(1, (1.031 * wb) / (0.194 + wb) + 0.5 * pf + 0.3 * ps)
  const alpha = alphaU * Math.exp(-Math.pow(k.tau / (24 * t), k.beta))
  const ks = 0.95 * (1 - Math.exp(-t / 6)) * (1.15 - 0.6 * ps) + 0.05
  const kf = 0.15 + 0.75 * (1 - Math.exp(-t / 45)) * (1.1 - pf)
  const ceq = C + ks * S + kf * F
  const x = (0.68 * alpha) / (0.32 * alpha + W / ceq)
  const A = (C * PARAMS.A_cement_MPa.value + S * PARAMS.A_slag_MPa.value + F * PARAMS.A_flyash_MPa.value) / b
  const n = (C * PARAMS.n_cement.value + S * PARAMS.n_slag.value + F * PARAMS.n_flyash.value) / b + k.dn
  return A * Math.pow(x, n)
}
export const fibS = (m) => 0.2 + 0.35 * ((m[1] + m[2]) / B(m))
export const fibRaw = (m, t) => powersRaw(m, 28) * Math.exp(fibS(m) * (1 - Math.sqrt(28 / t)))

const lsScale = (g, y) => g.reduce((s, v, i) => s + v * y[i], 0) / g.reduce((s, v) => s + v * v, 0)
const err = (p, y) => ({ rmse: Math.sqrt(p.reduce((s, v, i) => s + (v - y[i]) ** 2, 0) / y.length), mae: p.reduce((s, v, i) => s + Math.abs(v - y[i]), 0) / y.length })

// Fits each generator's scale on TRAIN. Returns { scale, predict(kind)(rows), meta }.
export function calibrate(train) {
  const out = {}
  for (const [kind, raw] of [['powers', powersRaw], ['fib', fibRaw]]) {
    const g = train.X.map((x) => raw(x, x[7]))
    const scale = lsScale(g, train.y)
    out[kind] = { scale, train: err(g.map((v) => v * scale), train.y) }
  }
  const predict = (kind) => (rows) => rows.map((x) => out[kind].scale * (kind === 'powers' ? powersRaw(x, x[7]) : fibRaw(x, x[7])))
  return { powers: out.powers, fib: out.fib, predict }
}

// n synthetic rows (mixes x agesPerMix) from generator kind, deterministic for seed.
export function synthesize(train, { kind = 'powers', mixes = 300, agesPerMix = 6, seed = 2026 } = {}) {
  const cal = calibrate(train)
  const rng = seeded(seed)
  const uniq = [...new Map(train.X.map((x) => [x.slice(0, 7).join(','), x.slice(0, 7)])).values()]
  const pattern = (m) => `${m[1] > 0}${m[2] > 0}${m[4] > 0}`
  const byPattern = new Map()
  for (const m of uniq) (byPattern.get(pattern(m)) || byPattern.set(pattern(m), []).get(pattern(m))).push(m)
  const normal = () => Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng())
  const X = [], y = []
  for (let i = 0; i < mixes; i++) {
    const a = uniq[Math.floor(rng() * uniq.length)], peers = byPattern.get(pattern(a)), b = peers[Math.floor(rng() * peers.length)]
    const lam = rng()
    const m = a.map((v, j) => lam * v + (1 - lam) * b[j])
    const k = { tau: 10 + 10 * rng(), beta: 0.65 + 0.3 * rng(), dn: -0.1 + 0.2 * rng() }
    for (let j = 0; j < agesPerMix; j++) {
      const t = Math.exp(Math.log(365) * rng())
      const clean = kind === 'powers' ? cal.powers.scale * powersRaw(m, t, k) : cal.fib.scale * powersRaw(m, 28, k) * Math.exp(fibS(m) * (1 - Math.sqrt(28 / t)))
      X.push([...m, t, m[3] / B(m)])
      y.push(Math.max(0, clean * (1 + 0.05 * normal())))
    }
  }
  return { X, y }
}

// Everything a reader needs to judge the synthetic data: parameters, labels, citations, calibration, cross-checks.
export function meta(train) {
  const cal = calibrate(train)
  const opc = [380, 0, 0, 190, 0, 1000, 800], scm = [200, 120, 80, 180, 6, 950, 780]
  const ratio = (m) => Object.fromEntries([1, 3, 7, 90, 365].map((t) => [t, Number((powersRaw(m, t) / powersRaw(m, 28)).toFixed(3))]))
  return {
    generators: { powers: 'f = s * A x^n, Powers gel-space with Mills a_u, generic tau/beta, efficiency factors k_s(t), k_f(t)', fib: 'f = s * Powers28 * exp((0.20 + 0.35 p_SCM)(1 - sqrt(28/t)))' },
    params: PARAMS,
    calibration: { on: 'TRAIN rows only (ages 1, 3, 7 d)', powersScale: cal.powers.scale, powersTrain: cal.powers.train, fibScale: cal.fib.scale, fibTrain: cal.fib.train },
    sampling: 'convex combination of two TRAIN mixes with the same ingredients present; age log-uniform 1-365 d',
    crossCheck: { 'powers f(t)/f(28), OPC w/c 0.50': ratio(opc), 'powers f(t)/f(28), 50% SCM w/b 0.45': ratio(scm), 'fib s for that SCM mix': fibS(scm) },
  }
}
