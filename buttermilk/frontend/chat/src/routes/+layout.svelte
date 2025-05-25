<script lang="ts">
	import { initPreloader } from '$lib/preload-modules.js';
	import { onMount } from 'svelte';
	import { browser } from '$app/environment'; // Import browser check
	import Nav from '$lib/components/layout/Nav.svelte'; // Placeholder
	import Header from '$lib/components/layout/Header.svelte'; // Placeholder
	import Sidebar from '$lib/components/layout/Sidebar.svelte'; // Placeholder
	import '$lib/styles/app.scss'; // Import the main SCSS file
    import { flowRunning } from '$lib/stores/apiStore';
	onMount(() => {
		// Initialize module preloading
		initPreloader();
		// Bootstrap JS needed for components like navbar toggler
		if (browser) { // Ensure this only runs on the client
			import('bootstrap/dist/js/bootstrap.bundle.min.js');
		}
	});
	
</script>
<div class="container-lg">
	<Nav />
	<Header />

	<div class="row align-items-top">

		{#if !$flowRunning}
		<div class="col-sm-3 col-md-3 col-xxl-2 col-xl-2 col-lg-3">
			<Sidebar />
		</div>

		{/if}
		<div class={!$flowRunning ? 'col-lg-9' : 'col-12'}>
			<slot />
		</div>
	</div>
</div>