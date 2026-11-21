import { setTheme, useTheme } from '../lib/theme.js'
import './themetoggle.css'

// A switch at the right of the top bar. Flat, like the rest of the page.
export default function ThemeToggle() {
  const dark = useTheme() === 'dark'
  return (
    <button
      type="button"
      role="switch"
      aria-checked={dark}
      aria-label="Dark mode"
      className="themetoggle"
      onClick={() => setTheme(dark ? 'light' : 'dark')}
    >
      <span className="themelabel">Dark mode</span>
      <span className="track" aria-hidden="true"><i /></span>
    </button>
  )
}
