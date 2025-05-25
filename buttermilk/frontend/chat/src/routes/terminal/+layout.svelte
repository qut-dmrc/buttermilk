<script lang="ts">
	import { onMount } from 'svelte';
	import { browser } from '$app/environment'; 
	import '$lib/styles/app.scss'; // Import main SCSS file
	import MessageSidebar from '$lib/components/MessageSidebar.svelte';
	import { messageStore } from '$lib/stores/messageStore';
	
	onMount(() => {
		if (browser) {
			import('bootstrap/dist/js/bootstrap.bundle.min.js');
		}
	});
</script>

	<div class="terminal-column">
	
		<slot />
	</div>
<style>
	.terminal-column {
		height: 100%;
	}
	
</style>
