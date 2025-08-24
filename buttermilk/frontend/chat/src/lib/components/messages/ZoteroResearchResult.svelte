<script lang="ts">
	import { getModelColor, type Message } from '$lib/utils/messageUtils';
	import { marked } from 'marked';
	import BasicMessage from './BasicMessage.svelte';
	import Modal from '../Modal.svelte';
	import DOMPurify from 'dompurify';

	// Props
	export let message: Message;
	export let expanded = false;

	let showModal = false;

	function sanitize(html: string): string {
		return DOMPurify.sanitize(html);
	}

	// Parse markdown content
	function parseMarkdown(content: string): string {
		if (!content) return 'No content available';
		try {
			return marked.parse(content, { async: false }) as string;
		} catch (e) {
			console.error('Error parsing markdown:', e);
			return content;
		}
	}

	// Get Zotero data from message outputs
	$: zoteroData = message.outputs || {};
	$: summary = zoteroData.summary || '';
	$: response = zoteroData.response || '';
	$: citation = zoteroData.citation || '';
	$: modelBasedColor = getModelColor(message.agent_info?.parameters?.model);
</script>

<div class="message-terminal" style="color: {modelBasedColor}">
	<BasicMessage {message}>
		<svelte:fragment slot="agentNick">[ZOTERO]</svelte:fragment>

		<svelte:fragment slot="messageContent">
				{@html sanitize(parseMarkdown(summary))}
				<button class="btn btn-sm btn-link" on:click={() => (showModal = true)}
					>View full response</button
				>
			{#if citation}
				<div class="citation">
					{@html sanitize(parseMarkdown(citation))}
				</div>
			{/if}
		</svelte:fragment>
	</BasicMessage>
</div>

<Modal bind:show={showModal} title="Full Response">
	<svelte:fragment slot="body">
		{@html sanitize(parseMarkdown(response))}
	</svelte:fragment>
</Modal>

<style>
	.citation {
		font-size: 0.8rem;
		color: #bbbbbb;
		margin-top: 2px;
		border-left: 2px solid #555;
		padding-left: 5px;
		margin-left: 5px;
	}
</style>
