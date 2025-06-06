<script lang="ts">
  import { page } from '$app/stores';
  import { onMount } from 'svelte';
  import { browser } from '$app/environment';
  import ToxicityScoreTable from '$lib/components/score/ToxicityScoreTable.svelte';
  import RecordDisplay from '$lib/components/score/RecordDisplay.svelte';
  import ScoreMessagesDisplay from '$lib/components/score/ScoreMessagesDisplay.svelte';

  let recordId: string;
  let recordData: any = null;
  let loading = true;
  let error: string | null = null;

  $: recordId = $page.params.record_id;

  // Mock function to fetch record data - replace with actual API call
  async function fetchRecordData(id: string) {
    try {
      loading = true;
      error = null;
      
      // TODO: Replace with actual API endpoint
      // For now, use mock data based on the reference example
      const mockData = {
        id: id,
        name: id.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase()),
        content: "Sample content that will be analyzed for toxicity. This represents the text that was evaluated by various AI models for potential policy violations.",
        toxicity_scores: {
          off_shelf: {
            'GPT-4': { correct: true, score: 0.85, label: 'TOXIC' },
            'Claude-3': { correct: false, score: 0.42, label: 'SAFE' },
            'Gemini': { correct: true, score: 0.78, label: 'TOXIC' },
            'LLaMA-2': { correct: true, score: 0.91, label: 'TOXIC' }
          },
          custom: {
            'Judge-GPT4': { step: 'judge', score: 0.88 },
            'Judge-Claude': { step: 'judge', score: 0.45 },
            'Synth-GPT4': { step: 'synth', score: 0.82 },
            'Synth-Claude': { step: 'synth', score: 0.39 }
          }
        },
        messages: [
          {
            agent: 'Judge-GPT4',
            type: 'judge',
            content: 'This content violates our community guidelines regarding hate speech targeting specific groups.',
            score: 0.88,
            reasoning: 'The language used contains derogatory terms and promotes harmful stereotypes.'
          },
          {
            agent: 'Judge-Claude',
            type: 'judge', 
            content: 'While the content discusses sensitive topics, it appears to be educational in nature.',
            score: 0.45,
            reasoning: 'The context suggests academic analysis rather than promoting harmful behavior.'
          }
        ]
      };
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 500));
      
      recordData = mockData;
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to fetch record data';
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    if (browser && recordId) {
      fetchRecordData(recordId);
    }
  });

  // Watch for recordId changes
  $: if (browser && recordId) {
    fetchRecordData(recordId);
  }
</script>

<svelte:head>
  <title>Score: {recordId} | Toxicity Analysis</title>
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
      <p>The requested record '{recordId}' could not be found.</p>
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