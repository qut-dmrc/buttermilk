<script lang="ts">
  import { browser } from '$app/environment';
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';

  // Get a new session ID from backend and redirect to session route
  onMount(async () => {
    if (browser) {
      // Check if we're actually at /terminal (not a sub-route)
      // This prevents creating a new session if we're actually loading /terminal/{sessionId}
      const path = window.location.pathname;
      if (path === '/terminal' || path === '/terminal/') {
        try {
          // Get session ID from backend
          const response = await fetch('/api/session');
          if (!response.ok) {
            throw new Error(`Failed to create session: ${response.statusText}`);
          }
          const data = await response.json();
          const newSessionId = data.session_id;
          
          // Redirect to the session-specific route
          await goto(`/terminal/${newSessionId}`, { replaceState: true });
        } catch (error) {
          console.error('Failed to create session:', error);
          // Fallback: show error or retry
        }
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
