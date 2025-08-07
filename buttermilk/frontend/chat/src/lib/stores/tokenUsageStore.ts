import { writable, derived } from 'svelte/store';
import type { Message } from '$lib/utils/messageUtils';

// Store for tracking cumulative token usage
interface TokenUsage {
  totalPromptTokens: number;
  totalCompletionTokens: number;
  totalCostUsd: number;
  messageCount: number;
}

// Create a writable store for token usage
const tokenUsageData = writable<TokenUsage>({
  totalPromptTokens: 0,
  totalCompletionTokens: 0,
  totalCostUsd: 0,
  messageCount: 0
});

// Helper function to update token usage when a new message arrives
export function updateTokenUsage(message: Message) {
  if (message.prompt_tokens || message.completion_tokens || message.cost_usd) {
    tokenUsageData.update(usage => ({
      totalPromptTokens: usage.totalPromptTokens + (message.prompt_tokens || 0),
      totalCompletionTokens: usage.totalCompletionTokens + (message.completion_tokens || 0),
      totalCostUsd: usage.totalCostUsd + (message.cost_usd || 0),
      messageCount: usage.messageCount + 1
    }));
  }
}

// Helper function to reset token usage
export function resetTokenUsage() {
  tokenUsageData.set({
    totalPromptTokens: 0,
    totalCompletionTokens: 0,
    totalCostUsd: 0,
    messageCount: 0
  });
}

// Derived store for formatted display values
export const tokenUsageDisplay = derived(tokenUsageData, $usage => {
  const totalTokens = $usage.totalPromptTokens + $usage.totalCompletionTokens;
  
  return {
    totalTokens: totalTokens.toLocaleString(),
    promptTokens: $usage.totalPromptTokens.toLocaleString(),
    completionTokens: $usage.totalCompletionTokens.toLocaleString(),
    cost: `$${$usage.totalCostUsd.toFixed(4)}`,
    messageCount: $usage.messageCount
  };
});

// Export the raw store for direct access if needed
export const tokenUsage = {
  subscribe: tokenUsageData.subscribe,
  update: updateTokenUsage,
  reset: resetTokenUsage
};