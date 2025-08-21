<script lang="ts">
	import { type Message } from '$lib/utils/messageUtils';
	import { marked } from 'marked';
	import BasicMessage from './BasicMessage.svelte';
	import DOMPurify from 'dompurify';

	// Props
	export let message: Message;
	export let expanded = false;

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
	$: citation = zoteroData.citation || '';
</script>

<div class="message-terminal">
	<BasicMessage {message}>
		<svelte:fragment slot="agentNick">[ZOTERO_REF]</svelte:fragment>

		<svelte:fragment slot="messageContent">
			{#if citation}
				<div class="citation">
					{@html sanitize(parseMarkdown(citation))}
				</div>
			{:else}
				No content available
			{/if}
		</svelte:fragment>
	</BasicMessage>
</div>

<style>
	.citation {
		font-size: 0.8rem;
		color: #bbbbbb;
	}
</style>
