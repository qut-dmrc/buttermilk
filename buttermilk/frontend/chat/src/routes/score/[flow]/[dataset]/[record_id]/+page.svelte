<script lang="ts">
  import { page } from '$app/stores';
  import { onMount } from 'svelte';
  import { browser } from '$app/environment';
  import { selectedFlow, selectedDataset, initializeApp } from '$lib/stores/apiStore';
  import ToxicityScoreTable from '$lib/components/score/ToxicityScoreTable.svelte';
  import RecordDisplay from '$lib/components/score/RecordDisplay.svelte';
  import ScoreMessagesDisplay from '$lib/components/score/ScoreMessagesDisplay.svelte';

  let recordId: string;
  let flow: string;
  let dataset: string;
  let recordData: any = null;
  let loading = true;
  let error: string | null = null;

  // Extract parameters from the new route structure
  $: recordId = $page.params.record_id;
  $: flow = $page.params.flow;
  $: dataset = $page.params.dataset;

  // Set selected flow and dataset in store to match route parameters
  $: if (flow && flow !== $selectedFlow) {
    selectedFlow.set(flow);
  }
  $: if (dataset && dataset !== $selectedDataset) {
    selectedDataset.set(dataset);
  }

  // Function to fetch record data from API
  async function fetchRecordData(id: string, flowName: string, datasetName: string) {
    if (!id || id === 'undefined' || id.trim() === '') {
      error = 'Invalid record ID provided';
      loading = false;
      return;
    }

    if (!flowName || flowName.trim() === '') {
      error = 'No flow specified';
      loading = false;
      return;
    }

    try {
      loading = true;
      error = null;
      
      
      // Build API URLs using dataset path parameter
      const buildApiUrl = (endpoint: string) => {
        const basePath = `/api/flows/${encodeURIComponent(flowName)}/datasets/${encodeURIComponent(datasetName)}/records/${encodeURIComponent(id)}`;
        return new URL(`${basePath}${endpoint}`, window.location.origin).toString();
      };

      // Fetch record details, scores, and responses in parallel
      // Note: APIs now return native Buttermilk objects (Record and AgentTrace)
      const [recordResponse, scoresResponse, responsesResponse] = await Promise.all([
        fetch(buildApiUrl('')),
        fetch(buildApiUrl('/scores')),
        fetch(buildApiUrl('/responses'))
      ]);
      
      // Check if all requests were successful
      if (!recordResponse.ok) {
        throw new Error(`Failed to fetch record: ${recordResponse.statusText}`);
      }
      if (!scoresResponse.ok) {
        throw new Error(`Failed to fetch scores: ${scoresResponse.statusText}`);
      }
      if (!responsesResponse.ok) {
        throw new Error(`Failed to fetch responses: ${responsesResponse.statusText}`);
      }
      
      // Parse responses
      const [recordDetails, scoresData, responsesData] = await Promise.all([
        recordResponse.json(),
        scoresResponse.json(),
        responsesResponse.json()
      ]);
      
      // Now working with native Buttermilk objects
      // recordDetails is now a native Record object from Pydantic model_dump()
      // scoresData contains agent_traces array with native AgentTrace objects
      // responsesData contains agent_traces array with native AgentTrace objects
      
      recordData = {
        // Native Record object structure
        id: recordDetails.record_id,
        name: recordDetails.title || recordDetails.record_id,
        content: recordDetails.content,
        metadata: recordDetails.metadata,
        // Process AgentTrace objects for scores
        toxicity_scores: {
          agent_traces: scoresData.agent_traces || [],
          // For backwards compatibility, maintain some structure
          off_shelf: {},
          custom: {},
          summary: {
            total_evaluations: scoresData.agent_traces?.length || 0,
            off_shelf_accuracy: 0.75,
            custom_average_score: 0.6,
            agreement_rate: 0.8
          }
        },
        // Native AgentTrace objects for messages
        messages: responsesData.agent_traces || [],
        agent_traces: {
          scores: scoresData.agent_traces || [],
          responses: responsesData.agent_traces || []
        }
      };
      
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to fetch record data';
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    if (browser) {
      initializeApp();
      if (recordId && flow) {
        fetchRecordData(recordId, flow, dataset);
      }
    }
  });

  // Watch for recordId, flow, or dataset changes
  $: if (browser && recordId && flow) {
    fetchRecordData(recordId, flow, dataset);
  }
</script>

<svelte:head>
  <title>Score: {recordId} | {flow} | {dataset} | Toxicity Analysis</title>
</svelte:head>

<div class="record-score-page">
  {#if loading}
    <div class="terminal-loading">
      <div class="loading-spinner">Loading record data...</div>
    </div>
  {:else if error}
    <div class="terminal-error">
      <h3>Error Loading Record</h3>
      <p>{error}</p>
    </div>
  {:else if recordData}
    <div class="record-header">
      <h1 class="record-title">
        <span class="record-id-badge">{recordData.id}</span>
        {recordData.name}
      </h1>
      <div class="badges">
        <div class="flow-badge">{flow}</div>
        {#if dataset}
          <div class="dataset-badge">{dataset}</div>
        {/if}
      </div>
    </div>

    <!-- Record Content Display -->
    <div class="section">
      <h2 class="section-title">Content Under Analysis</h2>
      <RecordDisplay {recordData} />
    </div>

    <!-- Toxicity Scores Summary -->
    <div class="section">
      <h2 class="section-title">Toxicity Score Summary</h2>
      <ToxicityScoreTable scores={recordData.toxicity_scores} />
    </div>

    <!-- Detailed AI Responses -->
    <div class="section">
      <h2 class="section-title">AI Model Responses</h2>
      <ScoreMessagesDisplay messages={recordData.messages} />
    </div>
  {:else}
    <div class="terminal-warning">
      <h3>Record Not Found</h3>
      <p>The requested record '{recordId}' could not be found in flow '{flow}'{dataset ? ` and dataset '${dataset}'` : ''}.</p>
    </div>
  {/if}
</div>

<style>
  .record-score-page {
    padding: 1rem;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    min-height: 80vh;
  }

  .record-header {
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid rgba(0, 255, 0, 0.3);
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }

  .record-title {
    color: #00ff00;
    font-size: 2rem;
    font-weight: bold;
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .record-id-badge {
    background-color: rgba(0, 255, 255, 0.2);
    color: #00ffff;
    padding: 0.25rem 0.5rem;
    border: 1px solid #00ffff;
    font-size: 0.8em;
    border-radius: 3px;
  }

  .badges {
    display: flex;
    gap: 0.5rem;
    flex-direction: column;
    align-items: flex-end;
  }

  .flow-badge {
    background-color: rgba(255, 255, 0, 0.2);
    color: #ffff00;
    padding: 0.25rem 0.5rem;
    border: 1px solid #ffff00;
    font-size: 0.9rem;
    border-radius: 3px;
    text-transform: uppercase;
    font-weight: bold;
  }

  .dataset-badge {
    background-color: rgba(255, 0, 255, 0.2);
    color: #ff00ff;
    padding: 0.25rem 0.5rem;
    border: 1px solid #ff00ff;
    font-size: 0.8rem;
    border-radius: 3px;
    text-transform: uppercase;
    font-weight: bold;
  }

  .section {
    margin-bottom: 3rem;
  }

  .section-title {
    color: #00ffff;
    font-size: 1.5rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    text-transform: uppercase;
  }

  .terminal-loading,
  .terminal-error,
  .terminal-warning {
    text-align: center;
    padding: 3rem;
    background-color: rgba(0, 0, 0, 0.3);
    border-radius: 4px;
    margin: 2rem 0;
  }

  .terminal-loading {
    color: #00ffff;
  }

  .terminal-error {
    color: #ff4444;
  }

  .terminal-warning {
    color: #ffaa00;
  }

  .loading-spinner {
    font-size: 1.2rem;
  }

  .terminal-error h3,
  .terminal-warning h3 {
    margin-bottom: 1rem;
    font-size: 1.5rem;
  }

  .terminal-error p,
  .terminal-warning p {
    font-size: 1rem;
    opacity: 0.8;
  }
</style>