<script lang="ts">
	import { onMount } from 'svelte';
	import { browser } from '$app/environment';

	interface DemoSession {
		session_id: string;
		title: string;
		description: string;
		duration: string;
		agents: string[];
	}

	let demos: DemoSession[] = [];
	let loading = true;
	let error = '';

	onMount(async () => {
		if (browser) {
			try {
				const response = await fetch('/api/demo/list');
				if (response.ok) {
					const data = await response.json();
					demos = data.demos || [];
				} else {
					error = 'Failed to load demo list';
				}
			} catch (err) {
				error = 'Error loading demos';
				console.error('Failed to fetch demo list:', err);
			} finally {
				loading = false;
			}
		}
	});
</script>

<svelte:head>
	<title>automod.cc - Demo Transcripts</title>
</svelte:head>

<div class="demo-page">
	<header class="demo-header">
		<h1>automod.cc</h1>
		<p class="tagline">Multi-Agent Content Moderation Research Platform</p>
	</header>

	<section class="demo-intro">
		<h2>Demo Transcripts</h2>
		<p>
			Watch pre-recorded sessions of our multi-agent groupchat system in action.
			These demos showcase how multiple AI agents collaborate to analyze content
			and reach consensus on moderation decisions.
		</p>
	</section>

	{#if loading}
		<div class="loading">
			<div class="spinner"></div>
			<p>Loading available demos...</p>
		</div>
	{:else if error}
		<div class="error">
			<p>{error}</p>
		</div>
	{:else if demos.length === 0}
		<div class="no-demos">
			<p>No demo transcripts available yet.</p>
		</div>
	{:else}
		<div class="demo-grid">
			{#each demos as demo}
				<a href="/demo/{demo.session_id}" class="demo-card">
					<h3>{demo.title}</h3>
					<p class="description">{demo.description}</p>
					<div class="meta">
						<span class="duration">{demo.duration}</span>
						<span class="agents">{demo.agents.length} agents</span>
					</div>
					<div class="agent-list">
						{#each demo.agents as agent}
							<span class="agent-tag">{agent}</span>
						{/each}
					</div>
				</a>
			{/each}
		</div>
	{/if}
</div>

<style>
	.demo-page {
		max-width: 1200px;
		margin: 0 auto;
		padding: 2rem;
		font-family: 'Roboto Mono', monospace;
		color: #b8c5b8;
		background: #1a1a1a;
		min-height: 100vh;
	}

	.demo-header {
		text-align: center;
		margin-bottom: 3rem;
		padding-bottom: 2rem;
		border-bottom: 1px solid #333;
	}

	.demo-header h1 {
		font-size: 2.5rem;
		color: #c9b458;
		margin-bottom: 0.5rem;
	}

	.tagline {
		font-size: 1rem;
		color: #888;
	}

	.demo-intro {
		margin-bottom: 2rem;
	}

	.demo-intro h2 {
		color: #c9b458;
		margin-bottom: 1rem;
	}

	.demo-intro p {
		line-height: 1.6;
		color: #aaa;
	}

	.loading, .error, .no-demos {
		text-align: center;
		padding: 3rem;
	}

	.spinner {
		width: 40px;
		height: 40px;
		border: 3px solid #333;
		border-top-color: #c9b458;
		border-radius: 50%;
		animation: spin 1s linear infinite;
		margin: 0 auto 1rem;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	.error {
		color: #dc3545;
	}

	.demo-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
		gap: 1.5rem;
	}

	.demo-card {
		background: #252525;
		border: 1px solid #333;
		border-radius: 8px;
		padding: 1.5rem;
		text-decoration: none;
		color: inherit;
		transition: border-color 0.2s, transform 0.2s;
	}

	.demo-card:hover {
		border-color: #c9b458;
		transform: translateY(-2px);
	}

	.demo-card h3 {
		color: #c9b458;
		margin-bottom: 0.75rem;
		font-size: 1.1rem;
	}

	.description {
		font-size: 0.9rem;
		color: #aaa;
		margin-bottom: 1rem;
		line-height: 1.4;
	}

	.meta {
		display: flex;
		gap: 1rem;
		margin-bottom: 1rem;
		font-size: 0.8rem;
		color: #666;
	}

	.agent-list {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
	}

	.agent-tag {
		background: #1a1a1a;
		border: 1px solid #444;
		padding: 0.25rem 0.5rem;
		border-radius: 4px;
		font-size: 0.75rem;
		color: #888;
	}
</style>
