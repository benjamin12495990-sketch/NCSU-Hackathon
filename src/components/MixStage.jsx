import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { buildParticles } from '../lib/particles.js'

// The 3D stage: one cubic metre of the chosen mix, drawn with Three.js.
// Particles fly in and pack into the cube, the cube turns slowly, and the
// ingredient named in `highlight` pulses. Drag to rotate.
export default function MixStage({ mix, factors, ingredients, highlight }) {
  const mountRef = useRef(null)
  const [failed, setFailed] = useState(false)
  const model = useMemo(() => buildParticles(mix, factors, ingredients), [mix, factors, ingredients])

  useEffect(() => {
    const mount = mountRef.current
    let renderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    } catch {
      setFailed(true) // no WebGL on this device; the rest of the page still works
      return
    }
    const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mount.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100)
    camera.position.set(11, 7, 14.5)
    scene.add(new THREE.HemisphereLight(0xffffff, 0xb9c0d4, 1.5))
    const sun = new THREE.DirectionalLight(0xffffff, 2.2)
    sun.position.set(4, 9, 7)
    scene.add(sun)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableZoom = false
    controls.enablePan = false
    controls.enableDamping = true
    controls.autoRotate = !still
    controls.autoRotateSpeed = 1.2

    const sphere = new THREE.SphereGeometry(1, 24, 16)
    const materials = new Map()
    for (const { name, color } of model.legend) {
      const hot = name === highlight
      materials.set(
        name,
        new THREE.MeshStandardMaterial({
          color,
          roughness: name === 'water' ? 0.15 : 0.55,
          metalness: 0.05,
          transparent: name === 'water',
          opacity: name === 'water' ? 0.75 : 1,
          emissive: hot ? 0x6d4aff : 0x000000,
          emissiveIntensity: hot ? 0.22 : 0,
        }),
      )
    }

    // Each particle starts scattered on a big sphere and flies to its site.
    const count = model.particles.length
    const meshes = model.particles.map((p, i) => {
      const mesh = new THREE.Mesh(sphere, materials.get(p.name))
      const a = i * 2.399963 // golden angle spreads the start points evenly
      const y = 1 - (2 * (i + 0.5)) / count
      const ring = Math.sqrt(1 - y * y)
      mesh.userData = {
        from: new THREE.Vector3(Math.cos(a) * ring, y, Math.sin(a) * ring).multiplyScalar(16),
        to: new THREE.Vector3(...p.pos),
        radius: p.radius,
        hot: p.name === highlight,
      }
      scene.add(mesh)
      return mesh
    })

    // Outline of the one-cubic-metre box, faded in once the particles settle.
    const boxGeometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(model.half * 2 + 0.5, model.half * 2 + 0.5, model.half * 2 + 0.5))
    const boxMaterial = new THREE.LineBasicMaterial({ color: 0x6d4aff, transparent: true, opacity: 0 })
    scene.add(new THREE.LineSegments(boxGeometry, boxMaterial))

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = mount
      if (!w || !h) return
      renderer.setSize(w, h)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    const observer = new ResizeObserver(resize)
    observer.observe(mount)
    resize()

    const start = performance.now()
    const gap = 0.006
    let frame
    const tick = (now) => {
      const t = (now - start) / 1000
      for (let i = 0; i < meshes.length; i++) {
        const { from, to, radius, hot } = meshes[i].userData
        const p = still ? 1 : Math.min(1, Math.max(0, (t - i * gap) / 0.9))
        const ease = 1 - (1 - p) ** 3
        meshes[i].position.lerpVectors(from, to, ease)
        const pulse = hot && !still ? 1 + 0.16 * Math.sin(t * 4.5) : 1
        meshes[i].scale.setScalar(radius * ease * pulse)
      }
      const settle = count * gap + 0.7
      boxMaterial.opacity = still ? 0.45 : 0.45 * Math.min(1, Math.max(0, (t - settle) / 0.5))
      controls.update()
      renderer.render(scene, camera)
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
      controls.dispose()
      sphere.dispose()
      boxGeometry.dispose()
      boxMaterial.dispose()
      for (const material of materials.values()) material.dispose()
      renderer.dispose()
      renderer.forceContextLoss()
      renderer.domElement.remove()
    }
  }, [model, highlight])

  return (
    <div className="stage">
      <div className="canvas" ref={mountRef} aria-label="Rotating 3D model of one cubic metre of the mix">
        {failed && <p className="note">The 3D view needs WebGL, which this browser has turned off.</p>}
      </div>
      <ul className="legend">
        {model.legend.map(({ name, label, color, kg, share }) => (
          <li key={name} className={(name === highlight ? 'hot' : '') + (kg === 0 ? ' none' : '')}>
            <span className="dot" style={{ background: color }} />
            {label} <b>{kg} kg</b> · {(share * 100).toFixed(share < 0.01 && kg > 0 ? 1 : 0)}%
          </li>
        ))}
      </ul>
      <p className="fine">
        Illustration of one cubic metre: the number of particles of each colour follows that
        ingredient's share of the volume. Particle sizes and positions are not to scale. Drag to rotate.
      </p>
    </div>
  )
}
