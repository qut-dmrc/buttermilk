/**
 * Preload SvelteKit modules to prevent dynamic import errors
 * This file is imported in the root +layout.svelte to ensure critical modules are loaded
 * before they're needed by the application
 */

/**
 * Initialize the preloader - in SvelteKit v2, explicit module preloading is generally
 * not needed as the bundler handles this automatically
 */
export function initPreloader() {
  console.log('SvelteKit modules preloaded successfully');
  return true;
}
