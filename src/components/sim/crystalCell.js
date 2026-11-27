// Geometry of one unit cell from structures.json, for CrystalViewer.jsx. No React and no three.js here, so it can be checked alone.
// A cell is { formula, L, X, E }: L holds the three lattice vectors as rows (angstrom), X the fractional coordinates of the atoms
// (already wrapped into the cell by the exporter) and E their element symbols.

// Element colours: the page has eight (the CSS variables --cv-el-1 to --cv-el-8 in CrystalViewer.css, so a theme can change them).
// They are handed out per crystal, in the order its elements first appear, so the legend under the picture is the only key: the same
// element can get another colour in another crystal. This file only says which of the eight (`slot`, 0 to 7) an element gets.
export const PALETTE_SIZE = 8

const EDGE = 1e-3 // an atom this close (in fractional units) to a face of the cell is drawn again on the opposite face
const isNum = (v) => typeof v === 'number' && Number.isFinite(v)
const triple = (v) => Array.isArray(v) && v.length === 3 && v.every(isNum)

export function validCell(cell) {
  return !!cell && Array.isArray(cell.L) && cell.L.length === 3 && cell.L.every(triple) && Array.isArray(cell.X) && Array.isArray(cell.E) &&
    cell.X.length === cell.E.length && cell.E.length > 0 && cell.X.every(triple)
}

// Lattice lengths (angstrom) and angles (degrees), and how many atoms of each element the cell holds.
export function cellSummary(cell) {
  if (!validCell(cell)) return null
  const L = cell.L
  const len = (v) => Math.hypot(v[0], v[1], v[2])
  const angle = (u, v) => (Math.acos(Math.max(-1, Math.min(1, (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (len(u) * len(v))))) * 180) / Math.PI
  const counts = new Map()
  for (const e of cell.E) counts.set(e, (counts.get(e) ?? 0) + 1)
  return {
    a: len(L[0]), b: len(L[1]), c: len(L[2]), alpha: angle(L[1], L[2]), beta: angle(L[0], L[2]), gamma: angle(L[0], L[1]),
    atoms: cell.E.length,
    elements: [...counts].map(([el, n], i) => ({ el, n, slot: i % PALETTE_SIZE })),
  }
}

// What the viewer draws, centred on the middle of the cell, in angstrom:
//   elements  [{ el, slot (which of the eight colours), n (atoms in one cell), positions: [[x, y, z], ...] (with the copies on opposite faces) }]
//   edges     the 12 edges of the cell as 24 points, flattened
//   atomRadius one sphere size for every atom (a share of the shortest distance between two atoms, so spheres never touch)
//   radius    of the sphere around the origin that holds everything drawn
export function cellModel(cell) {
  if (!validCell(cell)) return null
  const [a, b, c] = cell.L
  const mid = [0, 1, 2].map((k) => (a[k] + b[k] + c[k]) / 2)
  const cart = (f) => [0, 1, 2].map((k) => f[0] * a[k] + f[1] * b[k] + f[2] * c[k])
  const centred = (f) => cart(f).map((v, k) => v - mid[k])

  const index = new Map()
  const elements = []
  const frac = cell.X.map((x) => x.map((v) => v - Math.floor(v)))
  frac.forEach((f, i) => {
    const el = String(cell.E[i])
    if (!index.has(el)) {
      index.set(el, elements.length)
      elements.push({ el, slot: elements.length % PALETTE_SIZE, n: 0, positions: [] })
    }
    const e = elements[index.get(el)]
    e.n++
    const shifts = f.map((v) => (v < EDGE ? [0, 1] : v > 1 - EDGE ? [0, -1] : [0]))
    for (const sx of shifts[0]) for (const sy of shifts[1]) for (const sz of shifts[2]) e.positions.push(centred([f[0] + sx, f[1] + sy, f[2] + sz]))
  })

  // Shortest distance between two atoms of the crystal: every pair in the cell, against the 27 neighbouring copies of the second one.
  const pos = frac.map(cart)
  const moves = []
  for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (let k = -1; k <= 1; k++) moves.push([i, j, k, ...cart([i, j, k])])
  let nearest = Infinity
  for (let i = 0; i < pos.length; i++) {
    for (let j = i; j < pos.length; j++) {
      for (const m of moves) {
        if (i === j && m[0] === 0 && m[1] === 0 && m[2] === 0) continue
        const d = Math.hypot(pos[j][0] + m[3] - pos[i][0], pos[j][1] + m[4] - pos[i][1], pos[j][2] + m[5] - pos[i][2])
        if (d < nearest) nearest = d
      }
    }
  }

  const corners = []
  for (let i = 0; i <= 1; i++) for (let j = 0; j <= 1; j++) for (let k = 0; k <= 1; k++) corners.push([i, j, k])
  const edges = []
  for (let p = 0; p < 8; p++) {
    for (let q = p + 1; q < 8; q++) {
      const differ = [0, 1, 2].filter((k) => corners[p][k] !== corners[q][k]).length
      if (differ === 1) edges.push(...centred(corners[p]), ...centred(corners[q]))
    }
  }
  const half = Math.max(...corners.map((f) => Math.hypot(...centred(f))))
  const atomRadius = Number.isFinite(nearest) && nearest > 0 ? Math.min(0.45 * nearest, Math.max(0.3 * nearest, 0.06 * half)) : 0.12 * half
  return {
    elements, edges, atomRadius, radius: half + atomRadius, nearest: Number.isFinite(nearest) ? nearest : null,
    atoms: cell.E.length, drawn: elements.reduce((s, e) => s + e.positions.length, 0),
  }
}
