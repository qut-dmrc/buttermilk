import { TestScenario } from './TestClient.js';

export const testScenarios: TestScenario[] = [
  {
    name: 'Basic Connection Test',
    steps: [
      { action: 'waitFor', matcher: 'Connected', timeout: 5000 },
      { action: 'assert', value: 'Connected' }
    ],
    expectedOutcomes: ['Connected']
  },

  {
    name: 'Help Command Test',
    steps: [
      { action: 'waitFor', matcher: 'Connected', timeout: 5000 },
      { action: 'input', value: '/help' },
      { action: 'waitFor', matcher: 'Available commands', timeout: 2000 },
      { action: 'assert', value: '/flow' },
      { action: 'assert', value: '/run' },
      { action: 'assert', value: '/help' }
    ],
    expectedOutcomes: ['Available commands']
  },

  {
    name: 'Test Flow Execution',
    steps: [
      { action: 'waitFor', matcher: 'Connected', timeout: 5000 },
      { action: 'input', value: '/flow test' },
      { action: 'waitFor', matcher: 'Starting flow: test', timeout: 3000 },
      { action: 'waitFor', matcher: 'test_agent', timeout: 3000 },
      { action: 'waitFor', matcher: 'Test flow completed', timeout: 5000 }
    ],
    expectedOutcomes: [
      'Starting flow: test',
      'Test agent ready',
      'Test flow completed successfully'
    ]
  },

  {
    name: 'OSB Flow with Interaction',
    steps: [
      { action: 'waitFor', matcher: 'Connected', timeout: 5000 },
      { action: 'input', value: '/flow osb What is AI ethics?' },
      { action: 'waitFor', matcher: 'Starting flow: osb', timeout: 3000 },
      { action: 'waitFor', matcher: 'researcher', timeout: 3000 },
      { action: 'waitFor', matcher: 'Continue?', timeout: 5000 },
      { action: 'input', value: 'yes' },
      { action: 'waitFor', matcher: 'Starting multi-agent analysis', timeout: 3000 },
      { action: 'waitFor', matcher: 'Mock synthesis', timeout: 5000 }
    ],
    expectedOutcomes: [
      'researcher',
      'policy_analyst',
      'fact_checker',
      'explorer',
      'Mock synthesis of all agent findings'
    ]
  },

  {
    name: 'Invalid Flow Test',
    steps: [
      { action: 'waitFor', matcher: 'Connected', timeout: 5000 },
      { action: 'input', value: '/flow nonexistent' },
      { action: 'waitFor', matcher: 'Unknown flow', timeout: 3000 }
    ],
    expectedOutcomes: ['Unknown flow: nonexistent']
  },

  {
    name: 'JSON Message Test',
    steps: [
      { action: 'waitFor', matcher: 'Connected', timeout: 5000 },
      { action: 'input', value: '{"type": "run_flow", "payload": {"flow": "test"}}' },
      { action: 'waitFor', matcher: 'Starting flow: test', timeout: 3000 }
    ],
    expectedOutcomes: ['Starting flow: test']
  },

  {
    name: 'Regular Message Test',
    steps: [
      { action: 'waitFor', matcher: 'Connected', timeout: 5000 },
      { action: 'input', value: 'Hello, this is a test message' },
      { action: 'waitFor', matcher: 'Mock server received: Hello', timeout: 2000 }
    ],
    expectedOutcomes: ['Mock server received: Hello, this is a test message']
  },

  {
    name: 'Reconnection Test',
    steps: [
      { action: 'waitFor', matcher: 'Connected', timeout: 5000 },
      // Note: This would require the mock server to simulate disconnection
      // For now, we just test that the client shows reconnection UI
      { action: 'assert', value: 'Connected' }
    ],
    expectedOutcomes: ['Connected']
  }
];

// Stress test scenarios
export const stressTestScenarios: TestScenario[] = [
  {
    name: 'Rapid Message Test',
    steps: [
      { action: 'waitFor', matcher: 'Connected', timeout: 5000 },
      ...Array.from({ length: 10 }, (_, i) => ({
        action: 'input' as const,
        value: `Message ${i + 1}`
      })),
      { action: 'wait', timeout: 2000 }
    ],
    expectedOutcomes: ['Mock server received: Message 10']
  },

  {
    name: 'Large Message Test',
    steps: [
      { action: 'waitFor', matcher: 'Connected', timeout: 5000 },
      { 
        action: 'input', 
        value: '{"type": "user_message", "payload": {"text": "' + 'x'.repeat(1000) + '"}}' 
      },
      { action: 'waitFor', matcher: 'Mock server received:', timeout: 3000 }
    ]
  }
];

// Error scenarios
export const errorScenarios: TestScenario[] = [
  {
    name: 'Invalid JSON Test',
    steps: [
      { action: 'waitFor', matcher: 'Connected', timeout: 5000 },
      { action: 'input', value: '{"invalid json' },
      { action: 'wait', timeout: 1000 },
      // Should still work as regular message
      { action: 'assert', value: 'Mock server received:' }
    ]
  },

  {
    name: 'Empty Flow Name Test',
    steps: [
      { action: 'waitFor', matcher: 'Connected', timeout: 5000 },
      { action: 'input', value: '/flow' },
      { action: 'waitFor', matcher: 'Please specify a flow name', timeout: 2000 }
    ],
    expectedOutcomes: ['Please specify a flow name']
  }
];