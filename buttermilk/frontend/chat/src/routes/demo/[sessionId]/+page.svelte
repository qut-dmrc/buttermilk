<script lang="ts">
	import { browser } from '$app/environment';
	import { page } from '$app/stores';
	import { onMount } from 'svelte';
	import ChatTerminal from '$lib/ChatTerminal.svelte';
	import { sessionId as sessionIdStore } from '$lib/stores/sessionStore';

	// State
	let isLoading = true;
	let error = '';
	let sessionMetadata: any = null;
	let websocketTerminal: ChatTerminal | null = null;
	let pendingMessages: any[] = [];
	let messagesProcessed = false;
	let playbackSpeed = 1; // 1x, 2x, 0.5x speed
	let isPlaying = false;
	let currentMessageIndex = 0;

	// Get session ID from URL
	$: urlSessionId = $page.params.sessionId;

	// Playback controls
	function startPlayback() {
		if (pendingMessages.length === 0 || isPlaying) return;
		isPlaying = true;
		playNextMessage();
	}

	function pausePlayback() {
		isPlaying = false;
	}

	function resetPlayback() {
		isPlaying = false;
		currentMessageIndex = 0;
		// Clear terminal messages would require a reset method
	}

	function playNextMessage() {
		if (!isPlaying || currentMessageIndex >= pendingMessages.length) {
			isPlaying = false;
			return;
		}

		const message = pendingMessages[currentMessageIndex];
		if (websocketTerminal) {
			websocketTerminal.handleMessage(message);
		}
		currentMessageIndex++;

		// Calculate delay based on message type and playback speed
		let delay = 500 / playbackSpeed; // Base delay
		if (message.type === 'chat_message') {
			delay = 1500 / playbackSpeed; // Longer for chat messages
		} else if (message.type === 'system_prompt') {
			delay = 2000 / playbackSpeed; // Even longer for prompts
		}

		setTimeout(playNextMessage, delay);
	}

	function playAllImmediately() {
		if (!websocketTerminal || messagesProcessed) return;

		for (const message of pendingMessages) {
			websocketTerminal.handleMessage(message);
		}
		messagesProcessed = true;
		currentMessageIndex = pendingMessages.length;
	}

	// Load demo session
	async function loadDemoSession(sessionId: string) {
		isLoading = true;
		error = '';

		try {
			sessionIdStore.set(sessionId);

			const response = await fetch(`/api/demo/${sessionId}`);
			if (!response.ok) {
				if (response.status === 404) {
					error = 'Demo transcript not found';
				} else {
					error = 'Failed to load demo transcript';
				}
				return;
			}

			const data = await response.json();
			sessionMetadata = data.session_metadata || {};
			pendingMessages = data.messages || [];

			console.log(`Loaded demo with ${pendingMessages.length} messages`);
		} catch (err) {
			console.error('Failed to load demo:', err);
			error = 'Error loading demo transcript';
		} finally {
			isLoading = false;
		}
	}

	onMount(async () => {
		if (browser && urlSessionId) {
			await loadDemoSession(urlSessionId);
		}
	});

	// Handle terminal ready event
	function handleTerminalReady(event: CustomEvent) {
		websocketTerminal = event.detail;
		console.log('Terminal ready for demo playback');
	}
</script>

<svelte:head>
	<title>automod.cc - Demo: {urlSessionId}</title>
</svelte:head>

{#if isLoading}
	<div class="loading-container">
		<div class="spinner"></div>
		<p>Loading demo transcript...</p>
	</div>
{:else if error}
	<div class="error-container">
		<p class="error">{error}</p>
		<a href="/demo" class="back-link">Back to demos</a>
	</div>
{:else}
	<div class="demo-viewer">
		<!-- Playback Controls -->
		<div class="playback-controls">
			<a href="/demo" class="back-btn">[ back ]</a>

			<div class="controls-center">
				<button
					class="control-btn"
					on:click={resetPlayback}
					title="Reset"
				>
					[ reset ]
				</button>

				{#if isPlaying}
					<button
						class="control-btn"
						on:click={pausePlayback}
					>
						[ pause ]
					</button>
				{:else}
					<button
						class="control-btn"
						on:click={startPlayback}
						disabled={messagesProcessed}
					>
						[ play ]
					</button>
				{/if}

				<button
					class="control-btn"
					on:click={playAllImmediately}
					disabled={messagesProcessed}
				>
					[ show all ]
				</button>

				<select
					class="speed-select"
					bind:value={playbackSpeed}
					disabled={isPlaying}
				>
					<option value={0.5}>0.5x</option>
					<option value={1}>1x</option>
					<option value={2}>2x</option>
					<option value={4}>4x</option>
				</select>
			</div>

			<div class="progress">
				{currentMessageIndex} / {pendingMessages.length} messages
			</div>
		</div>

		<!-- Terminal Display -->
		<ChatTerminal
			wsUrl=""
			selectedFlow=""
			selectedRecord=""
			readonly={true}
			currentSessionId={urlSessionId}
			sessionStatus="demo"
			isResumable={false}
			bind:this={websocketTerminal}
			on:ready={handleTerminalReady}
		/>
	</div>
{/if}

<style>
	.loading-container, .error-container {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		height: 100vh;
		background: #1a1a1a;
		color: #b8c5b8;
		font-family: 'Roboto Mono', monospace;
	}

	.spinner {
		width: 40px;
		height: 40px;
		border: 3px solid #333;
		border-top-color: #c9b458;
		border-radius: 50%;
		animation: spin 1s linear infinite;
		margin-bottom: 1rem;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	.error {
		color: #dc3545;
		margin-bottom: 1rem;
	}

	.back-link {
		color: #c9b458;
		text-decoration: none;
	}

	.back-link:hover {
		text-decoration: underline;
	}

	.demo-viewer {
		display: flex;
		flex-direction: column;
		height: 100vh;
		background: #1a1a1a;
	}

	.playback-controls {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.75rem 1rem;
		background: #252525;
		border-bottom: 1px solid #333;
		font-family: 'Roboto Mono', monospace;
	}

	.back-btn {
		color: #c9b458;
		text-decoration: none;
		font-size: 0.9rem;
	}

	.back-btn:hover {
		text-decoration: underline;
	}

	.controls-center {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.control-btn {
		background: transparent;
		border: 1px solid #444;
		color: #b8c5b8;
		padding: 0.4rem 0.8rem;
		font-family: 'Roboto Mono', monospace;
		font-size: 0.8rem;
		cursor: pointer;
		transition: border-color 0.2s, color 0.2s;
	}

	.control-btn:hover:not(:disabled) {
		border-color: #c9b458;
		color: #c9b458;
	}

	.control-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.speed-select {
		background: #1a1a1a;
		border: 1px solid #444;
		color: #b8c5b8;
		padding: 0.4rem;
		font-family: 'Roboto Mono', monospace;
		font-size: 0.8rem;
	}

	.progress {
		color: #666;
		font-size: 0.8rem;
	}
</style>
