<script lang="ts">
  import { browser } from '$app/environment';
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';

  // Generate a new session ID and redirect to session route
  onMount(async () => {
    if (browser) {
      // Check if we're actually at /terminal (not a sub-route)
      // This prevents creating a new session if we're actually loading /terminal/{sessionId}
      const path = window.location.pathname;
      if (path === '/terminal' || path === '/terminal/') {
        // Only create new session if we're at the base /terminal route
        const newSessionId = crypto.randomUUID();
        
        // Redirect to the session-specific route
        await goto(`/terminal/${newSessionId}`, { replaceState: true });
      }
    }
  });
</script>

<!-- Redirecting to session page -->
<div class="h-100 flex items-center justify-center">
  <div class="text-center">
    <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 mx-auto mb-4"></div>
    <p class="text-gray-600">Creating new session...</p>
  </div>
</div>
