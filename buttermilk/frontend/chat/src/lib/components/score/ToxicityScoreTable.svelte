<script lang="ts">
  export let scores: {
    off_shelf: Record<string, { correct: boolean; score: number; label: string }>;
    custom: Record<string, { step: string; score: number }>;
  };

  // Generate ASCII progress bar
  function generateScoreBar(score: number): string {
    if (score === undefined || score === null) return "░░░░░";
    
    const normalizedScore = Math.min(Math.max(score, 0), 1);
    const fullBlocks = Math.floor(normalizedScore * 5);
    const remainder = (normalizedScore * 5) - fullBlocks;
    
    let bar = "";
    
    // Add full blocks
    for (let i = 0; i < fullBlocks; i++) {
      bar += "█";
    }
    
    // Add partial block if needed
    if (remainder > 0.25 && remainder < 0.75) {
      bar += "▌";
    } else if (remainder >= 0.75) {
      bar += "█";
    }
    
    // Add empty blocks
    const emptyBlocks = 5 - bar.length;
    for (let i = 0; i < emptyBlocks; i++) {
      bar += "░";
    }
    
    return bar;
  }

  // Get color based on score
  function getScoreColor(score: number): string {
    if (score === undefined || score === null) return "#666";
    if (score >= 0.8) return "#ff4444";
    if (score >= 0.6) return "#ff8844";
    if (score >= 0.4) return "#ffaa00";
    if (score >= 0.2) return "#88ff44";
    return "#00ff00";
  }

  // Calculate percentage correctly from what's shown
  function formatPercentage(score: number): string {
    if (score === undefined || score === null) return "N/A";
    return `${(score * 100).toFixed(0)}%`;
  }

  // Calculate summary stats
  $: offShelfResults = scores?.off_shelf || {};
  $: customResults = scores?.custom || {};
  
  $: offShelfCorrect = Object.values(offShelfResults).filter((r: any) => r.correct).length;
  $: offShelfTotal = Object.keys(offShelfResults).length;
  $: offShelfAccuracy = offShelfTotal > 0 ? (offShelfCorrect / offShelfTotal) : 0;
  
  $: customAverage = Object.values(customResults).length > 0 
    ? Object.values(customResults).reduce((sum: number, r: any) => sum + (r.score || 0), 0) / Object.values(customResults).length
    : 0;
</script>

<div class="toxicity-score-table">
  <!-- Off-the-shelf Results Table -->
  <div class="score-section">
    <h3 class="section-header">
      <span class="section-icon">🤖</span>
      Off-the-Shelf Model Results
      <span class="section-summary">({formatPercentage(offShelfAccuracy)} correct)</span>
    </h3>
    
    <div class="table-container">
      <table class="score-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Prediction</th>
            <th>Score</th>
            <th>Label</th>
            <th>Accuracy</th>
          </tr>
        </thead>
        <tbody>
          {#each Object.entries(offShelfResults) as [model, result]}
            {@const score = result.score}
            {@const scoreColor = getScoreColor(score)}
            <tr>
              <td class="model-name">{model}</td>
              <td class="prediction-cell">
                <span class="prediction-icon" class:correct={result.correct}>
                  {result.correct ? '✓' : '✗'}
                </span>
              </td>
              <td class="score-cell" style="color: {scoreColor}">
                <span class="score-value">{formatPercentage(score)}</span>
                <span class="score-bar">{generateScoreBar(score)}</span>
              </td>
              <td class="label-cell">
                <span class="label-badge" class:toxic={result.label === 'TOXIC'}>
                  {result.label}
                </span>
              </td>
              <td class="accuracy-cell">
                <span class="accuracy-icon" class:correct={result.correct}>
                  {result.correct ? '✓' : '✗'}
                </span>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Custom Results Table -->
  <div class="score-section">
    <h3 class="section-header">
      <span class="section-icon">⚡</span>
      Custom Model Results
      <span class="section-summary">(avg: {formatPercentage(customAverage)})</span>
    </h3>
    
    <div class="table-container">
      <table class="score-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Step</th>
            <th>Score</th>
            <th>Performance</th>
          </tr>
        </thead>
        <tbody>
          {#each Object.entries(customResults) as [model, result]}
            {@const score = result.score}
            {@const scoreColor = getScoreColor(score)}
            <tr>
              <td class="model-name">{model}</td>
              <td class="step-cell">
                <span class="step-badge" class:judge={result.step === 'judge'} class:synth={result.step === 'synth'}>
                  {result.step}
                </span>
              </td>
              <td class="score-cell" style="color: {scoreColor}">
                <span class="score-value">{formatPercentage(score)}</span>
                <span class="score-bar">{generateScoreBar(score)}</span>
              </td>
              <td class="performance-cell">
                <span class="performance-indicator" style="color: {scoreColor}">
                  {score >= 0.8 ? 'Excellent' : score >= 0.6 ? 'Good' : score >= 0.4 ? 'Fair' : 'Poor'}
                </span>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>
</div>

<style>
  .toxicity-score-table {
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  }

  .score-section {
    margin-bottom: 2rem;
    background-color: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
  }

  .section-header {
    background-color: rgba(0, 255, 255, 0.1);
    color: #00ffff;
    padding: 1rem;
    margin: 0;
    font-size: 1.2rem;
    border-bottom: 1px solid rgba(0, 255, 255, 0.3);
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .section-icon {
    font-size: 1.5rem;
  }

  .section-summary {
    margin-left: auto;
    font-size: 0.9rem;
    color: #aaa;
  }

  .table-container {
    overflow-x: auto;
  }

  .score-table {
    width: 100%;
    border-collapse: collapse;
    background-color: transparent;
  }

  .score-table th {
    background-color: rgba(255, 255, 255, 0.05);
    color: #00ff00;
    padding: 0.75rem;
    text-align: left;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    font-weight: bold;
    text-transform: uppercase;
    font-size: 0.85rem;
  }

  .score-table td {
    padding: 0.75rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    color: #ccc;
  }

  .score-table tr:hover {
    background-color: rgba(255, 255, 255, 0.02);
  }

  .model-name {
    font-weight: bold;
    color: #fff;
  }

  .prediction-cell,
  .accuracy-cell {
    text-align: center;
  }

  .prediction-icon,
  .accuracy-icon {
    font-size: 1.2rem;
  }

  .prediction-icon.correct,
  .accuracy-icon.correct {
    color: #00ff00;
  }

  .prediction-icon:not(.correct),
  .accuracy-icon:not(.correct) {
    color: #ff4444;
  }

  .score-cell {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .score-value {
    min-width: 50px;
    font-weight: bold;
  }

  .score-bar {
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    letter-spacing: -1px;
    font-size: 1.1rem;
  }

  .label-badge {
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    font-size: 0.8rem;
    font-weight: bold;
    text-transform: uppercase;
  }

  .label-badge.toxic {
    background-color: rgba(255, 68, 68, 0.2);
    color: #ff4444;
    border: 1px solid #ff4444;
  }

  .label-badge:not(.toxic) {
    background-color: rgba(0, 255, 0, 0.2);
    color: #00ff00;
    border: 1px solid #00ff00;
  }

  .step-badge {
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    font-size: 0.8rem;
    font-weight: bold;
    text-transform: uppercase;
  }

  .step-badge.judge {
    background-color: rgba(255, 170, 0, 0.2);
    color: #ffaa00;
    border: 1px solid #ffaa00;
  }

  .step-badge.synth {
    background-color: rgba(0, 255, 255, 0.2);
    color: #00ffff;
    border: 1px solid #00ffff;
  }

  .performance-indicator {
    font-weight: bold;
    text-transform: uppercase;
    font-size: 0.85rem;
  }

  @media (max-width: 768px) {
    .score-table {
      font-size: 0.8rem;
    }
    
    .score-table th,
    .score-table td {
      padding: 0.5rem;
    }
  }
</style>