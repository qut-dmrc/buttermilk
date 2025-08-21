<script lang="ts">
  import { browser } from '$app/environment';
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';

  // Generate a new session ID and redirect to session route
  onMount(async () => {
    if (browser) {
      // Generate a new session ID using built-in crypto API
      const newSessionId = crypto.randomUUID();
      console.log('Redirecting to new session:', newSessionId);
      
      // Redirect to the session-specific route
      await goto(`/session/${newSessionId}`, { replaceState: true });
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
