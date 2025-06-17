<!-- 
OSB Query Input Component

Enhanced input component specifically for OSB queries with:
- Case metadata fields (case number, priority, content type)
- Query suggestions based on OSB vector store content
- Validation for query length and required fields
- Quick action buttons for common OSB operations
-->

<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();

  // Input state
  export let query = '';
  export let isConnected = false;
  export let isProcessing = false;

  // OSB-specific metadata
  let caseNumber = '';
  let casePriority = 'medium';
  let contentType = '';
  let platform = '';
  let userContext = '';

  // Advanced options (collapsed by default)
  let showAdvancedOptions = false;
  let enableMultiAgentSynthesis = true;
  let enableCrossValidation = true;
  let enablePrecedentAnalysis = true;
  let includePolicyReferences = true;
  let enableStreamingResponse = true;

  // Validation state
  $: isQueryValid = query.trim().length > 0 && query.length <= 2000;
  $: canSubmit = isConnected && !isProcessing && isQueryValid;

  // Priority options
  const priorityOptions = [
    { value: 'low', label: 'Low', class: 'text-info' },
    { value: 'medium', label: 'Medium', class: 'text-warning' },
    { value: 'high', label: 'High', class: 'text-danger' },
    { value: 'critical', label: 'Critical', class: 'text-danger fw-bold' }
  ];

  // Content type suggestions
  const contentTypes = [
    'social_media_post',
    'forum_comment', 
    'article_content',
    'video_content',
    'image_content',
    'user_profile',
    'advertisement',
    'other'
  ];

  // Platform suggestions
  const platforms = [
    'twitter',
    'facebook', 
    'instagram',
    'youtube',
    'tiktok',
    'reddit',
    'linkedin',
    'other'
  ];

  // Query suggestions for OSB analysis
  const querySuggestions = [
    "What are the policy implications of this content regarding hate speech?",
    "Does this content violate community standards and what actions should be taken?",
    "Analyze this content for potential harmful misinformation and provide recommendations.",
    "Review this content for compliance with platform policies on harassment.",
    "Evaluate the policy precedents that apply to this content moderation case.",
    "What are the cultural and contextual factors to consider for this content?",
    "Assess the balance between free expression and safety concerns for this content."
  ];

  function handleSubmit() {
    if (!canSubmit) return;

    // Create OSB query message
    const osbQuery = {
      type: 'run_flow',
      flow: 'osb',
      query: query.trim(),
      
      // OSB metadata
      case_number: caseNumber.trim() || undefined,
      case_priority: casePriority,
      content_type: contentType || undefined,
      platform: platform || undefined,
      user_context: userContext.trim() || undefined,

      // Processing options
      enable_multi_agent_synthesis: enableMultiAgentSynthesis,
      enable_cross_validation: enableCrossValidation,
      enable_precedent_analysis: enablePrecedentAnalysis,
      include_policy_references: includePolicyReferences,
      enable_streaming_response: enableStreamingResponse,
      max_processing_time: 60
    };

    dispatch('submit', osbQuery);
    
    // Clear query but keep metadata for next query
    query = '';
  }

  function handleKeyPress(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
    }
  }

  function insertQuerySuggestion(suggestion: string) {
    query = suggestion;
  }

  function generateCaseNumber() {
    const date = new Date();
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const time = String(date.getHours()).padStart(2, '0') + String(date.getMinutes()).padStart(2, '0');
    caseNumber = `OSB-${year}${month}${day}-${time}`;
  }
</script>

<div class="osb-query-input">
  <!-- Query Input Area -->
  <div class="query-section mb-3">
    <div class="input-group">
      <textarea
        bind:value={query}
        on:keypress={handleKeyPress}
        class="form-control query-textarea"
        placeholder="Enter your OSB policy analysis query... (Shift+Enter for new line)"
        rows="3"
        maxlength="2000"
        disabled={!isConnected || isProcessing}
      ></textarea>
      <button
        type="button"
        class="btn btn-primary"
        disabled={!canSubmit}
        on:click={handleSubmit}
      >
        {#if isProcessing}
          <span class="spinner-border spinner-border-sm me-1" role="status"></span>
          Processing...
        {:else}
          Analyze
        {/if}
      </button>
    </div>
    
    <!-- Character count and validation -->
    <div class="d-flex justify-content-between align-items-center mt-1">
      <small class="text-muted">
        {query.length}/2000 characters
      </small>
      {#if query.length > 1800}
        <small class="text-warning">Approaching character limit</small>
      {/if}
      {#if !isQueryValid && query.length > 0}
        <small class="text-danger">Query too long or empty</small>
      {/if}
    </div>
  </div>

  <!-- Case Metadata -->
  <div class="metadata-section mb-3">
    <div class="row g-2">
      <!-- Case Number -->
      <div class="col-md-4">
        <div class="input-group input-group-sm">
          <span class="input-group-text">Case#</span>
          <input
            type="text"
            class="form-control"
            placeholder="OSB-2025-001"
            bind:value={caseNumber}
          />
          <button 
            type="button" 
            class="btn btn-outline-secondary"
            title="Generate case number"
            on:click={generateCaseNumber}
          >
            🎲
          </button>
        </div>
      </div>

      <!-- Priority -->
      <div class="col-md-3">
        <select class="form-select form-select-sm" bind:value={casePriority}>
          {#each priorityOptions as option}
            <option value={option.value} class={option.class}>
              {option.label}
            </option>
          {/each}
        </select>
      </div>

      <!-- Content Type -->
      <div class="col-md-3">
        <input
          type="text"
          class="form-control form-control-sm"
          placeholder="Content type"
          list="content-types"
          bind:value={contentType}
        />
        <datalist id="content-types">
          {#each contentTypes as type}
            <option value={type}>{type.replace(/_/g, ' ')}</option>
          {/each}
        </datalist>
      </div>

      <!-- Platform -->
      <div class="col-md-2">
        <input
          type="text"
          class="form-control form-control-sm"
          placeholder="Platform"
          list="platforms"
          bind:value={platform}
        />
        <datalist id="platforms">
          {#each platforms as plat}
            <option value={plat}>{plat}</option>
          {/each}
        </datalist>
      </div>
    </div>

    <!-- User Context -->
    {#if showAdvancedOptions}
      <div class="row mt-2">
        <div class="col-12">
          <input
            type="text"
            class="form-control form-control-sm"
            placeholder="User context (e.g., public figure, minor, verified account)"
            bind:value={userContext}
          />
        </div>
      </div>
    {/if}
  </div>

  <!-- Advanced Options Toggle -->
  <div class="advanced-section mb-3">
    <button
      type="button"
      class="btn btn-link btn-sm p-0 text-decoration-none"
      on:click={() => showAdvancedOptions = !showAdvancedOptions}
    >
      {showAdvancedOptions ? '▼' : '▶'} Advanced Options
    </button>

    {#if showAdvancedOptions}
      <div class="advanced-options mt-2">
        <div class="row g-2">
          <div class="col-md-6">
            <div class="form-check form-check-sm">
              <input
                type="checkbox"
                class="form-check-input"
                id="multiAgentSynthesis"
                bind:checked={enableMultiAgentSynthesis}
              />
              <label class="form-check-label text-muted" for="multiAgentSynthesis">
                Multi-agent synthesis
              </label>
            </div>
            <div class="form-check form-check-sm">
              <input
                type="checkbox"
                class="form-check-input"
                id="crossValidation"
                bind:checked={enableCrossValidation}
              />
              <label class="form-check-label text-muted" for="crossValidation">
                Cross-validation
              </label>
            </div>
          </div>
          <div class="col-md-6">
            <div class="form-check form-check-sm">
              <input
                type="checkbox"
                class="form-check-input"
                id="precedentAnalysis"
                bind:checked={enablePrecedentAnalysis}
              />
              <label class="form-check-label text-muted" for="precedentAnalysis">
                Precedent analysis
              </label>
            </div>
            <div class="form-check form-check-sm">
              <input
                type="checkbox"
                class="form-check-input"
                id="streamingResponse"
                bind:checked={enableStreamingResponse}
              />
              <label class="form-check-label text-muted" for="streamingResponse">
                Streaming responses
              </label>
            </div>
          </div>
        </div>
      </div>
    {/if}
  </div>

  <!-- Query Suggestions -->
  <div class="suggestions-section">
    <details class="text-muted">
      <summary class="small">Query Suggestions</summary>
      <div class="suggestions-list mt-2">
        {#each querySuggestions as suggestion}
          <button
            type="button"
            class="btn btn-outline-light btn-sm me-1 mb-1"
            on:click={() => insertQuerySuggestion(suggestion)}
            disabled={!isConnected}
          >
            {suggestion.substring(0, 60)}...
          </button>
        {/each}
      </div>
    </details>
  </div>
</div>

<style>
  .osb-query-input {
    background-color: rgba(0, 20, 40, 0.3);
    border: 1px solid rgba(100, 150, 200, 0.3);
    border-radius: 6px;
    padding: 16px;
    font-family: 'Courier New', monospace;
  }

  .query-textarea {
    font-family: 'Courier New', monospace;
    background-color: rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(100, 150, 200, 0.4);
    color: #e0e0e0;
    resize: vertical;
    min-height: 80px;
  }

  .query-textarea:focus {
    background-color: rgba(0, 0, 0, 0.6);
    border-color: rgba(100, 150, 200, 0.8);
    box-shadow: 0 0 0 0.2rem rgba(100, 150, 200, 0.25);
    color: #ffffff;
  }

  .form-control:disabled {
    opacity: 0.6;
  }

  .input-group-text {
    background-color: rgba(100, 150, 200, 0.2);
    border-color: rgba(100, 150, 200, 0.4);
    color: #e0e0e0;
    font-size: 0.85em;
  }

  .form-select,
  .form-control {
    background-color: rgba(0, 0, 0, 0.3);
    border-color: rgba(100, 150, 200, 0.3);
    color: #e0e0e0;
    font-size: 0.85em;
  }

  .form-select:focus,
  .form-control:focus {
    background-color: rgba(0, 0, 0, 0.5);
    border-color: rgba(100, 150, 200, 0.7);
    color: #ffffff;
  }

  .btn-link {
    color: rgba(100, 150, 200, 0.8);
    font-size: 0.85em;
  }

  .btn-link:hover {
    color: rgba(100, 150, 200, 1);
  }

  .form-check-input:checked {
    background-color: rgba(100, 150, 200, 0.8);
    border-color: rgba(100, 150, 200, 0.8);
  }

  .form-check-label {
    font-size: 0.8em;
  }

  .suggestions-list .btn {
    font-size: 0.75em;
    padding: 0.25rem 0.5rem;
    border-color: rgba(100, 150, 200, 0.3);
    color: rgba(200, 200, 200, 0.8);
  }

  .suggestions-list .btn:hover:not(:disabled) {
    background-color: rgba(100, 150, 200, 0.2);
    border-color: rgba(100, 150, 200, 0.5);
    color: #ffffff;
  }

  details summary {
    cursor: pointer;
    user-select: none;
    font-size: 0.85em;
  }

  details summary:hover {
    color: #ffffff !important;
  }

  .spinner-border-sm {
    width: 0.875rem;
    height: 0.875rem;
  }
</style>