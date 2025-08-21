<script lang="ts">
	import { type ManagerResponse, type Message } from '$lib/utils/messageUtils';
	import { createEventDispatcher } from 'svelte';
	// Import modular components
	import JudgeMessage from './messages/JudgeMessage.svelte'; // Import JudgeMessage directly
	import AssessmentMessage from './messages/AssessmentMessage.svelte';
	import AgentMessage from './messages/AgentMessage.svelte';
	import * as DifferencesMessageComponent from './messages/DifferencesMessage.svelte';
	import ManagerRequestMessage from './messages/ManagerRequestMessage.svelte';
	import * as RecordMessageComponent from './messages/RecordMessage.svelte';
	import * as ResearcherMessageComponent from './messages/ResearcherMessage.svelte';
	import * as SummaryMessageComponent from './messages/SummaryMessage.svelte';
	import * as ZoteroResearchResultComponent from './messages/ZoteroResearchResult.svelte';
	import * as ZoteroRefResultComponent from './messages/ZoteroRefResult.svelte';

	const RecordMessage = RecordMessageComponent.default;
	const SummaryMessage = SummaryMessageComponent.default;
	const ResearcherMessage = ResearcherMessageComponent.default;
	const DifferencesMessage = DifferencesMessageComponent.default;
	const ZoteroResearchResult = ZoteroResearchResultComponent.default;
	const ZoteroRefResult = ZoteroRefResultComponent.default;

	// Props
	export let message: Message;
	export let expanded = false;

	$: messageType = message.type;

	// Forward the managerResponse event
	const dispatch = createEventDispatcher();

	// Explicitly type the event parameter
	function forwardManagerResponse(event: CustomEvent<ManagerResponse>) {
		dispatch('managerResponse', event.detail);
	}
</script>

{#if messageType === 'judge_reasons'}
	<JudgeMessage {message} {expanded} />
{:else if messageType === 'assessments'}
	<AssessmentMessage {message} {expanded} />
{:else if messageType === 'record'}
	<RecordMessage {message} {expanded} />
{:else if messageType === 'research_result'}
	<ResearcherMessage {message} {expanded} />
{:else if messageType === 'zotero_research_result'}
	<ZoteroResearchResult {message} {expanded} />
{:else if messageType === 'zotero_ref_result'}
	<ZoteroRefResult {message} {expanded} />
{:else if messageType === 'ui_message'}
	<ManagerRequestMessage {message} on:managerResponse={forwardManagerResponse} />
{:else if messageType === 'summary_result'}
	<SummaryMessage {message} {expanded} />
{:else if messageType === 'differences'}
	<DifferencesMessage {message} {expanded} />
{:else if messageType === 'system_update'}{:else if messageType === 'system_error'}
	<AgentMessage {message} {expanded} />
{:else if messageType === 'user'}
	<AgentMessage {message} {expanded} />
{:else}{/if}
