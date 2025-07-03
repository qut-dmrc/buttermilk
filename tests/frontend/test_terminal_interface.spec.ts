/**
 * Flow-Agnostic Terminal Interface E2E Tests
 * 
 * Comprehensive Playwright test suite for the Buttermilk terminal interface
 * supporting any flow configuration. Tests include:
 * 
 * - Basic terminal functionality and WebSocket connections
 * - Flow selection and agent interaction testing
 * - Message validation and error handling
 * - Performance and responsiveness testing
 * - Cross-browser compatibility testing
 * 
 * These tests validate the complete frontend experience for any YAML-configured
 * flow, ensuring the terminal interface works seamlessly with the backend
 * WebSocket infrastructure.
 */

import { test, expect, Page } from '@playwright/test';

// Flow configurations for testing (should match backend configurations)
const TEST_FLOWS = [
  {
    name: 'osb',
    displayName: 'OSB Interactive Flow',
    agents: ['researcher', 'policy_analyst', 'fact_checker', 'explorer'],
    sampleQuery: 'Analyze this content for policy violations: "This is a test social media post"'
  },
  {
    name: 'content_moderation',
    displayName: 'Content Moderation Flow',
    agents: ['classifier', 'reviewer'],
    sampleQuery: 'Moderate this content: "User submitted content for review"'
  },
  {
    name: 'research',
    displayName: 'Research Flow',
    agents: ['researcher', 'analyst', 'synthesizer'],
    sampleQuery: 'Research topic: "Climate change impact on agriculture"'
  }
];

// Test configuration
const TERMINAL_URL = 'http://localhost:5173'; // SvelteKit dev server
const API_BASE_URL = 'http://localhost:8000'; // FastAPI backend

class TerminalTestHelper {
  constructor(private page: Page) {}

  async navigateToTerminal(): Promise<void> {
    await this.page.goto(TERMINAL_URL);
    await this.page.waitForLoadState('networkidle');
  }

  async waitForWebSocketConnection(): Promise<void> {
    // Wait for WebSocket connection indicator (adjust selector based on UI)
    await this.page.waitForSelector('[data-testid="connection-status"][data-status="connected"]', { 
      timeout: 10000 
    });
  }

  async selectFlow(flowName: string): Promise<void> {
    // Select flow from dropdown/selector (adjust based on UI implementation)
    const flowSelector = this.page.locator('[data-testid="flow-selector"]');
    await flowSelector.click();
    await this.page.locator(`[data-testid="flow-option-${flowName}"]`).click();
  }

  async sendMessage(message: string): Promise<void> {
    const messageInput = this.page.locator('[data-testid="message-input"]');
    await messageInput.fill(message);
    await messageInput.press('Enter');
  }

  async waitForResponse(timeout: number = 30000): Promise<void> {
    // Wait for any agent response message to appear
    await this.page.waitForSelector('[data-testid="agent-response"]', { timeout });
  }

  async getLastMessage(): Promise<string> {
    const messages = this.page.locator('[data-testid="message"]');
    const lastMessage = messages.last();
    return await lastMessage.textContent() || '';
  }

  async getAgentResponses(): Promise<string[]> {
    const agentResponses = this.page.locator('[data-testid="agent-response"]');
    return await agentResponses.allTextContents();
  }

  async clearTerminal(): Promise<void> {
    // Clear terminal if UI provides this functionality
    const clearButton = this.page.locator('[data-testid="clear-terminal"]');
    if (await clearButton.isVisible()) {
      await clearButton.click();
    }
  }
}

test.describe('Terminal Interface - Basic Functionality', () => {
  let helper: TerminalTestHelper;

  test.beforeEach(async ({ page }) => {
    helper = new TerminalTestHelper(page);
    await helper.navigateToTerminal();
  });

  test('should load terminal interface successfully', async ({ page }) => {
    // Verify terminal UI elements are present
    await expect(page.locator('[data-testid="terminal-container"]')).toBeVisible();
    await expect(page.locator('[data-testid="message-input"]')).toBeVisible();
    await expect(page.locator('[data-testid="flow-selector"]')).toBeVisible();
  });

  test('should establish WebSocket connection', async ({ page }) => {
    await helper.waitForWebSocketConnection();
    
    // Verify connection status indicator
    const connectionStatus = page.locator('[data-testid="connection-status"]');
    await expect(connectionStatus).toHaveAttribute('data-status', 'connected');
  });

  test('should display available flows', async ({ page }) => {
    const flowSelector = page.locator('[data-testid="flow-selector"]');
    await flowSelector.click();
    
    // Verify all configured flows are available
    for (const flow of TEST_FLOWS) {
      const flowOption = page.locator(`[data-testid="flow-option-${flow.name}"]`);
      await expect(flowOption).toBeVisible();
      await expect(flowOption).toContainText(flow.displayName);
    }
  });
});

test.describe('Terminal Interface - Flow Interaction', () => {
  let helper: TerminalTestHelper;

  test.beforeEach(async ({ page }) => {
    helper = new TerminalTestHelper(page);
    await helper.navigateToTerminal();
    await helper.waitForWebSocketConnection();
  });

  for (const flow of TEST_FLOWS) {
    test(`should handle ${flow.name} flow interaction`, async ({ page }) => {
      // Select the flow
      await helper.selectFlow(flow.name);
      
      // Verify flow is selected
      const selectedFlow = page.locator('[data-testid="selected-flow"]');
      await expect(selectedFlow).toContainText(flow.displayName);
      
      // Send a query
      await helper.sendMessage(flow.sampleQuery);
      
      // Wait for response
      await helper.waitForResponse();
      
      // Verify response was received
      const responses = await helper.getAgentResponses();
      expect(responses.length).toBeGreaterThan(0);
      
      // Verify message appears in terminal
      const lastMessage = await helper.getLastMessage();
      expect(lastMessage).toContain(flow.sampleQuery);
    });
  }

  test('should handle flow switching', async ({ page }) => {
    // Start with first flow
    await helper.selectFlow(TEST_FLOWS[0].name);
    await helper.sendMessage(TEST_FLOWS[0].sampleQuery);
    
    // Switch to second flow
    await helper.selectFlow(TEST_FLOWS[1].name);
    
    // Verify flow change
    const selectedFlow = page.locator('[data-testid="selected-flow"]');
    await expect(selectedFlow).toContainText(TEST_FLOWS[1].displayName);
    
    // Send query in new flow
    await helper.sendMessage(TEST_FLOWS[1].sampleQuery);
    await helper.waitForResponse();
    
    // Verify response
    const responses = await helper.getAgentResponses();
    expect(responses.length).toBeGreaterThan(0);
  });
});

test.describe('Terminal Interface - Agent Responses', () => {
  let helper: TerminalTestHelper;

  test.beforeEach(async ({ page }) => {
    helper = new TerminalTestHelper(page);
    await helper.navigateToTerminal();
    await helper.waitForWebSocketConnection();
  });

  test('should display agent status updates', async ({ page }) => {
    await helper.selectFlow('osb');
    await helper.sendMessage('Test query for agent status tracking');
    
    // Wait for status updates to appear
    await page.waitForSelector('[data-testid="agent-status"]', { timeout: 30000 });
    
    // Verify status updates for different agents
    const statusUpdates = page.locator('[data-testid="agent-status"]');
    const statusTexts = await statusUpdates.allTextContents();
    
    // Should have status updates for multiple agents
    expect(statusTexts.length).toBeGreaterThan(0);
    
    // Verify status contains agent names
    const agentNames = TEST_FLOWS.find(f => f.name === 'osb')?.agents || [];
    const hasAgentStatus = statusTexts.some(status => 
      agentNames.some(agent => status.includes(agent))
    );
    expect(hasAgentStatus).toBeTruthy();
  });

  test('should display partial responses during processing', async ({ page }) => {
    await helper.selectFlow('osb');
    await helper.sendMessage('Test query for partial response tracking');
    
    // Wait for partial responses
    await page.waitForSelector('[data-testid="partial-response"]', { timeout: 30000 });
    
    // Verify partial responses appear
    const partialResponses = page.locator('[data-testid="partial-response"]');
    const responseCount = await partialResponses.count();
    
    expect(responseCount).toBeGreaterThan(0);
  });

  test('should display final synthesis result', async ({ page }) => {
    await helper.selectFlow('osb');
    await helper.sendMessage('Test query for synthesis result');
    
    // Wait for final synthesis
    await page.waitForSelector('[data-testid="synthesis-result"]', { timeout: 60000 });
    
    // Verify synthesis result is displayed
    const synthesisResult = page.locator('[data-testid="synthesis-result"]');
    await expect(synthesisResult).toBeVisible();
    
    // Verify synthesis contains summary
    const synthesisText = await synthesisResult.textContent();
    expect(synthesisText).toBeTruthy();
    expect(synthesisText!.length).toBeGreaterThan(10);
  });
});

test.describe('Terminal Interface - Error Handling', () => {
  let helper: TerminalTestHelper;

  test.beforeEach(async ({ page }) => {
    helper = new TerminalTestHelper(page);
    await helper.navigateToTerminal();
    await helper.waitForWebSocketConnection();
  });

  test('should handle empty messages gracefully', async ({ page }) => {
    await helper.selectFlow('osb');
    
    // Try to send empty message
    const messageInput = page.locator('[data-testid="message-input"]');
    await messageInput.press('Enter');
    
    // Should not send empty message or should show error
    const errorMessage = page.locator('[data-testid="error-message"]');
    if (await errorMessage.isVisible()) {
      await expect(errorMessage).toContainText('empty');
    }
  });

  test('should handle very long messages', async ({ page }) => {
    await helper.selectFlow('osb');
    
    // Send very long message
    const longMessage = 'x'.repeat(2500); // Exceeds typical limits
    await helper.sendMessage(longMessage);
    
    // Should either truncate, show error, or handle gracefully
    const errorMessage = page.locator('[data-testid="error-message"]');
    if (await errorMessage.isVisible()) {
      await expect(errorMessage).toContainText('too long');
    }
  });

  test('should handle WebSocket disconnection', async ({ page }) => {
    await helper.selectFlow('osb');
    
    // Simulate network interruption by blocking WebSocket requests
    await page.route('ws://localhost:8000/ws/*', route => route.abort());
    
    // Send message after connection is blocked
    await helper.sendMessage('Test message after disconnection');
    
    // Should show connection error
    const connectionStatus = page.locator('[data-testid="connection-status"]');
    await expect(connectionStatus).toHaveAttribute('data-status', 'disconnected');
  });
});

test.describe('Terminal Interface - Performance', () => {
  let helper: TerminalTestHelper;

  test.beforeEach(async ({ page }) => {
    helper = new TerminalTestHelper(page);
    await helper.navigateToTerminal();
  });

  test('should load within performance thresholds', async ({ page }) => {
    const startTime = Date.now();
    
    await helper.waitForWebSocketConnection();
    
    const loadTime = Date.now() - startTime;
    
    // Should load within 5 seconds
    expect(loadTime).toBeLessThan(5000);
  });

  test('should handle rapid message sending', async ({ page }) => {
    await helper.waitForWebSocketConnection();
    await helper.selectFlow('osb');
    
    // Send multiple messages quickly
    const messages = [
      'Quick test 1',
      'Quick test 2', 
      'Quick test 3',
      'Quick test 4',
      'Quick test 5'
    ];
    
    for (const message of messages) {
      await helper.sendMessage(message);
      // Small delay to avoid overwhelming
      await page.waitForTimeout(100);
    }
    
    // All messages should be sent without errors
    // Terminal should remain responsive
    const messageInput = page.locator('[data-testid="message-input"]');
    await expect(messageInput).toBeEnabled();
  });

  test('should maintain responsiveness during long workflows', async ({ page }) => {
    await helper.waitForWebSocketConnection();
    await helper.selectFlow('osb');
    
    // Send query that triggers long workflow
    await helper.sendMessage('Complex analysis requiring all agents');
    
    // Terminal should remain responsive during processing
    const messageInput = page.locator('[data-testid="message-input"]');
    
    // Wait a bit then test responsiveness
    await page.waitForTimeout(2000);
    await expect(messageInput).toBeEnabled();
    
    // Should be able to send another message
    await messageInput.fill('Second message during processing');
    await expect(messageInput).toHaveValue('Second message during processing');
  });
});

test.describe('Terminal Interface - Accessibility', () => {
  let helper: TerminalTestHelper;

  test.beforeEach(async ({ page }) => {
    helper = new TerminalTestHelper(page);
    await helper.navigateToTerminal();
  });

  test('should support keyboard navigation', async ({ page }) => {
    // Focus should start on message input
    const messageInput = page.locator('[data-testid="message-input"]');
    await expect(messageInput).toBeFocused();
    
    // Tab should navigate to flow selector
    await page.keyboard.press('Tab');
    const flowSelector = page.locator('[data-testid="flow-selector"]');
    await expect(flowSelector).toBeFocused();
    
    // Enter should open flow selector
    await page.keyboard.press('Enter');
    // First flow option should be focusable
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('Enter');
  });

  test('should have proper ARIA labels', async ({ page }) => {
    // Verify important elements have ARIA labels
    const messageInput = page.locator('[data-testid="message-input"]');
    await expect(messageInput).toHaveAttribute('aria-label');
    
    const flowSelector = page.locator('[data-testid="flow-selector"]');
    await expect(flowSelector).toHaveAttribute('aria-label');
  });

  test('should support screen reader announcements', async ({ page }) => {
    await helper.waitForWebSocketConnection();
    await helper.selectFlow('osb');
    
    // Send message and verify live region updates
    await helper.sendMessage('Test for screen reader');
    
    // Should have live region for status updates
    const liveRegion = page.locator('[aria-live="polite"]');
    await expect(liveRegion).toBeVisible();
  });
});

test.describe('Terminal Interface - Cross-Browser Compatibility', () => {
  // These tests would run across different browsers configured in playwright.config.ts
  
  test('should work consistently across browsers', async ({ page, browserName }) => {
    const helper = new TerminalTestHelper(page);
    await helper.navigateToTerminal();
    await helper.waitForWebSocketConnection();
    
    // Basic functionality should work in all browsers
    await helper.selectFlow('osb');
    await helper.sendMessage('Cross-browser compatibility test');
    await helper.waitForResponse();
    
    const responses = await helper.getAgentResponses();
    expect(responses.length).toBeGreaterThan(0);
    
    console.log(`✅ Terminal interface working in ${browserName}`);
  });
});

// Performance measurement utilities
test.describe('Terminal Interface - Performance Metrics', () => {
  test('should measure WebSocket connection time', async ({ page }) => {
    const helper = new TerminalTestHelper(page);
    
    const startTime = Date.now();
    await helper.navigateToTerminal();
    await helper.waitForWebSocketConnection();
    const connectionTime = Date.now() - startTime;
    
    console.log(`WebSocket connection established in ${connectionTime}ms`);
    expect(connectionTime).toBeLessThan(3000); // Under 3 seconds
  });

  test('should measure message round-trip time', async ({ page }) => {
    const helper = new TerminalTestHelper(page);
    await helper.navigateToTerminal();
    await helper.waitForWebSocketConnection();
    await helper.selectFlow('osb');
    
    const startTime = Date.now();
    await helper.sendMessage('Performance test message');
    await helper.waitForResponse();
    const roundTripTime = Date.now() - startTime;
    
    console.log(`Message round-trip completed in ${roundTripTime}ms`);
    expect(roundTripTime).toBeLessThan(30000); // Under 30 seconds
  });
});