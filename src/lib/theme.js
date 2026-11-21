import { useSyncExternalStore } from 'react'

// The theme lives on <html data-theme="light|dark">. index.html sets it before
// the page paints (saved choice, else the system setting); this file changes it
// afterwards and remembers the choice.
const KEY = 'theme'

export const getTheme = () => (document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light')

export function setTheme(theme) {
  document.documentElement.dataset.theme = theme
  try {
    localStorage.setItem(KEY, theme)
  } catch {
    // Storage can be blocked (private window); the choice then lasts until reload.
  }
}

const subscribe = (notify) => {
  const watcher = new MutationObserver(notify)
  watcher.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
  return () => watcher.disconnect()
}

// Components that must react to the theme (the toggle, the 3D stage) read it here.
export const useTheme = () => useSyncExternalStore(subscribe, getTheme, () => 'light')
