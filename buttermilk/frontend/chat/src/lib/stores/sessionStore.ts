import { browser } from "$app/environment"
import { goto } from "$app/navigation"
import { writable } from "svelte/store"

// Create a writable store with localStorage persistence and URL sync
function createSessionStore() {
  // Default empty session ID
  const defaultValue = ""

  // Read initial value from localStorage if available
  const initialValue = browser
    ? window.localStorage.getItem("sessionId") || defaultValue
    : defaultValue

  // Create writable store with the initial value
  const sessionStore = writable<string>(initialValue)

  // Subscribe to changes and update localStorage + URL
  if (browser) {
    sessionStore.subscribe(value => {
      if (value) {
        // Save to localStorage as fallback
        window.localStorage.setItem("sessionId", value)

        // Update URL if we're on a terminal session page and ID changed
        const currentPath = window.location.pathname
        if (currentPath.startsWith("/terminal/")) {
          const currentSessionId = currentPath.split("/")[2]
          if (currentSessionId !== value) {
            // Update URL to reflect new session ID
            goto(`/terminal/${value}`, { replaceState: true })
          }
        }
      } else {
        window.localStorage.removeItem("sessionId")
      }
    })
  }

  return {
    ...sessionStore,
    // Method to clear the session
    clear: () => {
      sessionStore.set("")
      if (browser) {
        window.localStorage.removeItem("sessionId")
      }
    },
    // Method to set session from URL (doesn't trigger URL update)
    setFromUrl: (value: string) => {
      sessionStore.set(value)
      if (browser && value) {
        window.localStorage.setItem("sessionId", value)
      }
    },
  }
}

export const sessionId = createSessionStore()
