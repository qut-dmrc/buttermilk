<script lang="ts">
	import { onMount } from 'svelte';
	import { browser } from '$app/environment';
	import '$lib/styles/app.scss';

	onMount(() => {
		if (browser) {
			import('bootstrap/dist/js/bootstrap.bundle.min.js');
		}
	});
</script>

<div class="demo-layout">
	<slot />
</div>

<style>
	.demo-layout {
		height: 100%;
		background: #1a1a1a;
	}
</style>
