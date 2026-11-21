import { useEffect, useRef, useState } from 'react'
import './domainmenu.css'

// The top bar. The three-bar button at the top left opens the list of
// simulations (the families of materials); choosing one switches the page.
export default function DomainMenu({ domains, activeId, onSelect }) {
  const [open, setOpen] = useState(false)
  const burger = useRef(null)
  const drawer = useRef(null)
  const active = domains.find((d) => d.id === activeId) ?? domains[0]

  // While the menu is open: start on the current simulation, and let Escape close it.
  useEffect(() => {
    if (!open) return undefined
    drawer.current?.querySelector('[aria-current="true"]')?.focus()
    const onKey = (e) => {
      if (e.key !== 'Escape') return
      setOpen(false)
      burger.current?.focus()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open])

  const choose = (id) => {
    onSelect(id)
    setOpen(false)
    burger.current?.focus()
  }

  // Tabbing out of the open menu closes it. A click can report no related
  // target (Safari), so only a real target outside the menu counts.
  const onBlur = (e) => {
    const next = e.relatedTarget
    if (next && !e.currentTarget.contains(next) && next !== burger.current) setOpen(false)
  }

  return (
    <>
      <header className="topbar">
        <button
          ref={burger}
          type="button"
          className="burger"
          aria-label="Choose a simulation"
          aria-expanded={open}
          aria-controls="simulation-menu"
          onClick={() => setOpen((o) => !o)}
        >
          <span className="bars" aria-hidden="true"><i /><i /><i /></span>
        </button>
        <p className="brand">
          Formulation Designer
          <span className="crumb">{active.label}</span>
        </p>
      </header>
      {open && (
        <>
          <div className="scrim" aria-hidden="true" onClick={() => setOpen(false)} />
          <nav id="simulation-menu" ref={drawer} className="drawer" aria-label="Simulations" onBlur={onBlur}>
            <p className="drawer-title">Simulations</p>
            <ul>
              {domains.map((d) => (
                <li key={d.id}>
                  <button type="button" aria-current={d.id === activeId ? 'true' : undefined} onClick={() => choose(d.id)}>
                    <span>{d.label}</span>
                    <span className={d.live ? 'status live' : 'status'}>{d.live ? 'Live' : 'Planned'}</span>
                  </button>
                </li>
              ))}
            </ul>
            <p className="drawer-note">
              Engineer the Tools of Scientific Discovery. Only concrete is built; the others are plans.
            </p>
          </nav>
        </>
      )}
    </>
  )
}
