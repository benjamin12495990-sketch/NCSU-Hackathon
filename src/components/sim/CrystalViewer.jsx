import { useEffect, useMemo, useRef, useState } from 'react'
import { fmt } from '../../lib/pipeline.js'
import { useTheme } from '../../lib/theme.js'
import { PALETTE_SIZE, cellModel, cellSummary } from './crystalCell.js'
import './CrystalViewer.css'

// The unit cell of one crystal in 3D: every atom of the cell as a sphere coloured by element (legend below), the twelve edges of the
// cell, drag to rotate. The positions and the cell come from structures.json and are drawn to scale; sphere sizes and colours are
// not physical, and the caption says so. three.js is fetched only when this component first mounts, so it stays out of the page's
// main bundle. The stage is built once and reused as the visitor picks other crystals; everything is disposed on unmount.
// No colour is written in this file: the element colours and the light from below are CSS variables of CrystalViewer.css, the cell's
// edges take --muted from index.css, and all are read again when the theme changes.

const FOV = 35
const VIEW = [1, 0.62, 1.3] // where the camera starts, as a direction from the middle of the cell
const INTRO_MS = 420
const ZOOM_MIN = 0.45
const ZOOM_MAX = 1.7
const reducedMotion = () => typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

// The colours the stage needs, read from the CSS variables in force on the viewer's own element (so a theme can change any of them).
function readColors(el) {
  const css = getComputedStyle(el)
  const get = (name) => css.getPropertyValue(name).trim()
  const line = get('--muted')
  return { line, ground: get('--cv-ground'), elements: Array.from({ length: PALETTE_SIZE }, (_, i) => get(`--cv-el-${i + 1}`) || line) }
}

// Builds the WebGL stage inside `mount`. Throws when the browser has no WebGL.
function createStage(THREE, OrbitControls, mount, { still, onGrab }) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
  mount.appendChild(renderer.domElement)

  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 1000)
  camera.position.set(...VIEW)
  scene.add(camera)
  // Both lights are three.js's default white; the tint of the light from below comes from the CSS (setColors).
  const sky = new THREE.HemisphereLight()
  sky.intensity = 1.4
  scene.add(sky)
  const lamp = new THREE.DirectionalLight() // rides with the camera, so the lit side always faces the visitor
  lamp.intensity = 2.1
  lamp.position.set(-2, 4, 5)
  camera.add(lamp)

  const controls = new OrbitControls(camera, renderer.domElement)
  controls.enableZoom = false // the wheel keeps scrolling the page; the + and − buttons zoom
  controls.enablePan = false
  controls.enableDamping = true
  controls.autoRotate = !still
  controls.autoRotateSpeed = 1.1
  // OrbitControls turns touch scrolling off on its canvas; give vertical swipes back to the page (a sideways drag still rotates).
  renderer.domElement.style.touchAction = 'pan-y'
  controls.addEventListener('start', onGrab)

  const sphere = new THREE.SphereGeometry(1, 28, 18)
  const lineMaterial = new THREE.LineBasicMaterial()
  const matrix = new THREE.Matrix4()
  const turn = new THREE.Quaternion()
  const at = new THREE.Vector3()
  const size = new THREE.Vector3()
  const orbit = new THREE.Spherical()

  let group = null
  let palette = [] // the element colours by slot, from the CSS (setColors)
  let meshes = []
  let lineGeometry = null
  let radius = 1
  let atomRadius = 0.3
  let zoom = 1
  let intro = null
  let dirty = true
  let frame = 0
  let onScreen = true

  const paint = (mesh) => {
    const value = palette[mesh.userData.slot]
    if (value) mesh.material.color.set(value)
  }

  // Every sphere at its place, at `scale` of its full size (the short grow-in when a crystal is picked).
  const place = (scale) => {
    size.setScalar(atomRadius * scale)
    for (const mesh of meshes) {
      mesh.userData.positions.forEach((p, i) => {
        matrix.compose(at.set(p[0], p[1], p[2]), turn, size)
        mesh.setMatrixAt(i, matrix)
      })
      mesh.instanceMatrix.needsUpdate = true
    }
  }

  // Camera distance at which the whole cell fits the canvas, whatever its shape.
  const fitDistance = () => {
    const v = (FOV * Math.PI) / 180
    const h = 2 * Math.atan(Math.tan(v / 2) * camera.aspect)
    return (radius / Math.sin(Math.min(v, h) / 2)) * 1.04
  }
  const frameCamera = () => {
    const d = fitDistance()
    camera.near = d / 50
    camera.far = d * 20
    camera.position.sub(controls.target).setLength(d * zoom).add(controls.target)
    camera.updateProjectionMatrix()
    dirty = true
  }

  const clear = () => {
    if (!group) return
    scene.remove(group)
    for (const mesh of meshes) {
      mesh.material.dispose()
      mesh.dispose()
    }
    lineGeometry?.dispose()
    group = null
    meshes = []
    lineGeometry = null
  }

  const tick = (now) => {
    frame = requestAnimationFrame(tick)
    let growing = false
    if (intro !== null) {
      const p = Math.min(1, Math.max(0, (now - intro) / INTRO_MS))
      place(0.25 + 0.75 * (1 - (1 - p) ** 3))
      if (p >= 1) intro = null
      growing = true
    }
    const moved = controls.update()
    if (moved || growing || dirty) {
      renderer.render(scene, camera)
      dirty = false
    }
  }
  const run = () => {
    if (!frame && onScreen) frame = requestAnimationFrame(tick)
  }
  const pause = () => {
    cancelAnimationFrame(frame)
    frame = 0
  }

  const resize = () => {
    const { clientWidth: w, clientHeight: h } = mount
    if (!w || !h) return
    renderer.setSize(w, h)
    camera.aspect = w / h
    frameCamera()
    renderer.render(scene, camera) // resizing clears the canvas, so draw at once instead of flashing empty for a frame
    dirty = false
  }
  const resizeWatch = new ResizeObserver(resize)
  resizeWatch.observe(mount)
  resize()
  // Nothing is drawn while the stage is scrolled off the screen.
  const screenWatch = typeof IntersectionObserver === 'undefined' ? null : new IntersectionObserver((entries) => {
    onScreen = entries.some((e) => e.isIntersecting)
    if (onScreen) { dirty = true; run() } else pause()
  })
  screenWatch?.observe(mount)
  run()

  return {
    // Shows another crystal (or nothing, for null): the old spheres and lines are freed first.
    setModel(model) {
      clear()
      if (model) {
        radius = model.radius
        atomRadius = model.atomRadius
        group = new THREE.Group()
        group.rotation.x = -Math.PI / 2 // crystal z (the c axis of most cells) points up on the screen
        for (const e of model.elements) {
          const mesh = new THREE.InstancedMesh(sphere, new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.05 }), e.positions.length)
          mesh.userData.positions = e.positions
          mesh.userData.slot = e.slot
          paint(mesh)
          mesh.frustumCulled = false
          meshes.push(mesh)
          group.add(mesh)
        }
        lineGeometry = new THREE.BufferGeometry()
        lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(model.edges, 3))
        group.add(new THREE.LineSegments(lineGeometry, lineMaterial))
        scene.add(group)
        intro = still ? null : performance.now()
        place(still ? 1 : 0.25)
        zoom = 1
        frameCamera()
      }
      dirty = true
    },
    // The colours of the page as they are now: the cell's edges, the light from below and the element colours by slot.
    setColors({ line, ground, elements }) {
      if (line) lineMaterial.color.set(line)
      if (ground) sky.groundColor.set(ground)
      palette = elements
      meshes.forEach(paint)
      dirty = true
    },
    setTurning(on) {
      controls.autoRotate = on
    },
    // Arrow keys: move the camera around the cell by these angles (radians).
    nudge(dTheta, dPhi) {
      orbit.setFromVector3(at.copy(camera.position).sub(controls.target))
      orbit.theta += dTheta
      orbit.phi = Math.min(Math.PI - 0.05, Math.max(0.05, orbit.phi + dPhi))
      camera.position.setFromSpherical(orbit).add(controls.target)
      camera.lookAt(controls.target)
      dirty = true
    },
    zoomBy(factor) {
      zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zoom * factor))
      frameCamera()
    },
    reset() {
      zoom = 1
      controls.target.set(0, 0, 0)
      camera.position.set(...VIEW)
      frameCamera()
    },
    dispose() {
      pause()
      onScreen = false
      resizeWatch.disconnect()
      screenWatch?.disconnect()
      controls.removeEventListener('start', onGrab)
      controls.dispose()
      clear()
      sphere.dispose()
      lineMaterial.dispose()
      renderer.dispose()
      renderer.forceContextLoss()
      renderer.domElement.remove()
    },
  }
}

// A lattice angle to one decimal place (108.53 -> 108.5, 90.0000001 -> 90).
const degrees = (v) => String(Number(v.toFixed(1)))
const list = (items) => (items.length > 1 ? `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}` : items[0] ?? '')

export default function CrystalViewer({ cell, name }) {
  const rootRef = useRef(null)
  const mountRef = useRef(null)
  const stageRef = useRef(null)
  const [status, setStatus] = useState('loading') // loading | ready | nowebgl | failed
  const [turning, setTurning] = useState(() => !reducedMotion())
  const theme = useTheme()
  const model = useMemo(() => cellModel(cell), [cell])
  const summary = useMemo(() => cellSummary(cell), [cell])

  // Fetch three.js and build the stage once; free everything when the viewer leaves the page.
  useEffect(() => {
    const mount = mountRef.current
    let gone = false
    let stage = null
    const start = async () => {
      let THREE, OrbitControls
      try {
        [THREE, { OrbitControls }] = await Promise.all([import('three'), import('three/examples/jsm/controls/OrbitControls.js')])
      } catch {
        if (!gone) setStatus('failed')
        return
      }
      if (gone) return
      try {
        stage = createStage(THREE, OrbitControls, mount, { still: reducedMotion(), onGrab: () => setTurning(false) })
      } catch {
        setStatus('nowebgl') // no WebGL on this device; the legend and the cell's numbers below still show
        return
      }
      stageRef.current = stage
      setStatus('ready')
    }
    start()
    return () => {
      gone = true
      stageRef.current = null
      stage?.dispose()
    }
  }, [])

  // The colours come from the CSS and are read again when the theme changes. This effect stands before the one that builds the
  // spheres, so the first crystal is drawn in colour at once.
  useEffect(() => {
    if (status === 'ready' && rootRef.current) stageRef.current?.setColors(readColors(rootRef.current))
  }, [status, theme])
  useEffect(() => {
    if (status === 'ready') stageRef.current?.setModel(model)
  }, [status, model])
  useEffect(() => {
    if (status === 'ready') stageRef.current?.setTurning(turning)
  }, [status, turning])

  const live = status === 'ready' && !!model
  const onKey = (e) => {
    const stage = stageRef.current
    if (!live || !stage) return
    const step = 0.16
    const move = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[e.key]
    if (move) {
      e.preventDefault()
      setTurning(false)
      stage.nudge(move[0], move[1])
    } else if (e.key === '+' || e.key === '=') {
      stage.zoomBy(0.85)
    } else if (e.key === '-') {
      stage.zoomBy(1 / 0.85)
    }
  }
  const what = name ? `the unit cell of ${name}` : 'the unit cell'
  const label = model
    ? `3D model of ${what}: ${model.atoms} ${model.atoms === 1 ? 'atom' : 'atoms'} (${list(model.elements.map((e) => `${e.n} ${e.el}`))}). Drag or use the arrow keys to rotate.`
    : `3D model of ${what}: not available.`

  return (
    <div className="stage cv" ref={rootRef}>
      <div className="canvas cv-canvas" role="img" aria-label={label} tabIndex={live ? 0 : -1} onKeyDown={onKey}>
        <div className="cv-mount" ref={mountRef} />
        {!model && <p className="note cv-msg">This crystal&apos;s unit cell could not be read from the export.</p>}
        {model && status === 'loading' && <p className="note cv-msg">Loading the 3D view…</p>}
        {model && status === 'nowebgl' && <p className="note cv-msg">The 3D view needs WebGL, which this browser has turned off. The cell is described below.</p>}
        {model && status === 'failed' && <p className="note cv-msg">The 3D code did not load. Reload the page to try again; the cell is described below.</p>}
      </div>
      {model && (
        <>
          <div className="cv-tools">
            <button type="button" className="chip" aria-pressed={turning} disabled={!live} onClick={() => setTurning((t) => !t)}>Turn slowly</button>
            <button type="button" className="chip" disabled={!live} onClick={() => stageRef.current?.reset()}>Reset view</button>
            <button type="button" className="chip cv-zoom" disabled={!live} aria-label="Zoom in" onClick={() => stageRef.current?.zoomBy(0.8)}>+</button>
            <button type="button" className="chip cv-zoom" disabled={!live} aria-label="Zoom out" onClick={() => stageRef.current?.zoomBy(1.25)}>−</button>
          </div>
          <ul className="legend cv-legend" aria-label="Elements and their number of atoms in one cell">
            {model.elements.map((e) => (
              <li key={e.el}><span className="dot" style={{ background: `var(--cv-el-${e.slot + 1})` }} />{e.el} <b>{e.n}</b></li>
            ))}
            <li className="cv-count">{model.atoms} {model.atoms === 1 ? 'atom' : 'atoms'} in one cell</li>
          </ul>
          {summary && (
            <p className="fine cv-lattice">
              Cell edges a {fmt(summary.a)} Å, b {fmt(summary.b)} Å, c {fmt(summary.c)} Å; angles α {degrees(summary.alpha)}°, β {degrees(summary.beta)}°, γ {degrees(summary.gamma)}°.
            </p>
          )}
          <p className="fine">
            Atom positions and the cell are to scale, as stored for this crystal in JARVIS-DFT. Sphere sizes are not atomic radii, no bonds are
            drawn, and the colours are handed out per crystal from a colour-blind-safe palette, so read them with the legend.
            {model.drawn > model.atoms && ` Atoms on a face, edge or corner of the cell are drawn again on the opposite side (${model.drawn} spheres for ${model.atoms} atoms).`}
            {model.elements.length > 8 && ' This cell has more than eight elements, so some colours repeat.'}
            {' '}Drag to rotate; with the picture focused, the arrow keys turn it.
          </p>
        </>
      )}
    </div>
  )
}
