<script lang="ts">
  import { onMount } from 'svelte';
  import {
    getAgentStyle,
    getAgentEmoji,
    type Message,
    type RecordData,
    type MessageType,
    formatUncertainty} from '$lib/utils/messageUtils';
  import { marked } from 'marked';
  
  // Import common expandable details styles
  import '$lib/styles/expandable-details.scss';
	import BasicMessage from './BasicMessage.svelte';

  // Props
  export let message: Message;
  export let expanded = false;

  // Local state
  let isContentExpanded = false;
  // let isReasonsExpanded = false; // Removed: only used for judge reasons

  // Parse markdown content with newline fixing
  function parseMarkdown(content: string): string {
    if (!content) return 'No content available';
    
    // Replace escaped newlines with actual newlines
    const cleanedContent = content.replace(/\\n/g, '\n');
    
    try {
      // Use marked.parse synchronously to ensure we return a string
      return marked.parse(cleanedContent, { async: false }) as string;
    } catch (e) {
      console.error('Error parsing markdown:', e);
      return cleanedContent;
    }
  }

  // Get agent styling
  $: agentStyle = getAgentStyle(message.agent_info?.agent_name || 'System');
  $: agentEmoji = getAgentEmoji(message.agent_info?.agent_name || 'System', 
    message.agent_info?.agent_id || ''
  );
  
  // Get record data from message outputs
  // $: recordData = message.outputs as RecordData; // Removed explicit cast

  // Format metadata for display - filtering out unnecessary fields
  function formatMetadata(metadata: any): string[] {
    if (!metadata) return [];
    
    // Only display specific metadata fields and skip duplicates/fetch fields
    const allowedFields = ['outlet'];
    
    return Object.entries(metadata)
      .filter(([key]) => {
        // Skip title, content, text, fetch_ fields, and date (handled separately)
        return allowedFields.includes(key.toLowerCase()) && 
               !key.toLowerCase().startsWith('fetch_') && 
               key.toLowerCase() !== 'date' &&
               key.toLowerCase() !== 'title' && 
               key.toLowerCase() !== 'content' && 
               key.toLowerCase() !== 'text';
      })
      .map(([key, value]) => `${key.charAt(0).toUpperCase() + key.slice(1)}: ${value}`);
  }

  // Toggle content expansion
  function toggleContent() {
    isContentExpanded = !isContentExpanded;
  }

  // Format date if available (date only, no time)
  function formatDate(dateStr: string): string {
    if (!dateStr) return '';
    
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString();
    } catch (e) {
      // Return the first 10 characters if it's in ISO format YYYY-MM-DD
      if (typeof dateStr === 'string' && dateStr.match(/^\d{4}-\d{2}-\d{2}/)) {
        return dateStr.substring(0, 10);
      }
      return dateStr;
    }
  }
</script>

<div class="message-terminal">
  <BasicMessage message={message}>

    <svelte:fragment slot="agentNick">[RECORD]</svelte:fragment>
    <svelte:fragment slot="messagePrefix">
      <span class="record-title-inline">{message.outputs?.metadata?.title}</span>

      <!-- Compact metadata - only show relevant fields -->
      {#if message.outputs?.metadata}
        <span class="metadata-inline">
          {#if message.outputs.metadata.date}| date: {formatDate(message.outputs.metadata.date)}{/if}
          {#each formatMetadata(message.outputs.metadata) as item}| {item}{/each}
        </span>
      {/if}
    </svelte:fragment>

    <svelte:fragment slot="messageContent">
      <div class="record-message">
        <span class="record-inline">


          <!-- Content Toggle (Inline button) -->
          <button
            class="content-toggle-inline"
            on:click={toggleContent}
            title={isContentExpanded ? "Collapse content" : "Expand content"}
          >
            <i class="bi {isContentExpanded ? 'bi-dash-circle-fill' : 'bi-plus-circle-fill'}"></i>
            {isContentExpanded ? '[-]' : '[+]'} details
          </button>

          <!-- URL as separate line if exists, to avoid excessively long lines -->
          {#if message.outputs?.metadata?.url}
            <div class="url-line">
              <span class="url-label">URL:</span>
              <a href={message.outputs.metadata.url} target="_blank" rel="noopener noreferrer" class="url-link">{message.outputs.metadata.url}</a>
            </div>
          {/if}
          
        </span>
        <div class="{isContentExpanded ? 'expanded-content' : 'truncated-content'} markdown-content">
          {@html parseMarkdown(message.outputs?.content)}
        </div>

      </div>
    </svelte:fragment>
  </BasicMessage>
</div>

<!-- Sidebar Card (only if this message should appear in sidebar) -->
{#if expanded}
  <div class="sidebar-card mt-2 mb-3">
    <div class="card">
      <div class="card-header d-flex justify-content-between align-items-center"
           style="background-color: {agentStyle.background}; border-color: {agentStyle.border};">
        <span>
          {agentEmoji} <small>{message.agent_info?.agent_name || 'System'}</small>
        </span>
        <span class="badge bg-light text-dark">
          {#if message.type === 'record'}
            Record
          {:else if message.type === 'judge_reasons'}
            Judge
          {:else}
            Message
          {/if}
        </span>
      </div>

      <div class="card-body p-2">
        <div class="tiny-text">
          <div class="text-truncate">
            <b>
              {#if message.type === 'record'}
                {message.outputs?.metadata?.title || 'Untitled Record'}
              {:else if message.type === 'judge_reasons'}
                Judge Output
              {:else}
                {message.agent_info?.agent_name || 'System Message'}
              {/if}
            </b>
          </div>
          {#if message.type === 'record' && message.outputs?.metadata?.source}
            <div>Source: <span class="text-muted">{message.outputs.metadata.source}</span></div>
          {/if}
          {#if message.type === 'judge_reasons' && message.outputs?.conclusion}
             <div>Conclusion: <span class="text-muted text-truncate">{message.outputs.conclusion}</span></div>
          {/if}
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .record-message .record-inline {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: center;
    color: #f0f0f0;
    max-width: 100%;
  }

  .record-message .record-tag {
    color: #5fad4e;
    font-weight: bold;
    margin-right: 8px;
  }

  .record-message .record-title-inline {
    font-weight: bold;
    margin-right: 8px;
    color: #f0f0f0;
  }

  .record-message .metadata-inline {
    color: #bbbbbb;
    font-size: 0.9em;
  }

  .record-message .url-line {
    padding-left: 24px;
    font-size: 0.9em;
    color: #bbbbbb;
  }

  .record-message .url-label {
    color: #888888;
  }

  .record-message .url-link {
    color: #2a9fd6;
    text-decoration: none;
  }

  .record-message .url-link:hover {
    text-decoration: underline;
  }

  .record-message .truncated-content {
    /*max-height: 120px; */
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 4; /* Show x lines */
  }
  /* Bootstrap Icons Adjustments */
  .bi {
    vertical-align: -0.125em;
  }


  .record-message .record-header {
    margin-bottom: 0.2em;
    border-bottom: 1px solid #555;
  }

  .record-message .record-agent-info {
    font-weight: bold;
    color: #f0f0f0;
  }


</style>
