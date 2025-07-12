#!/usr/bin/env node
import { TestClient, ScenarioRunner, TestScenario } from './TestClient.js';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Real server test scenarios
const realServerScenarios: TestScenario[] = [
  {
    name: 'Real Server Connection Test',
    steps: [
      { action: 'waitFor', matcher: 'Connected', timeout: 10000 },
      { action: 'wait', timeout: 1000 },
      { action: 'input', value: '/help' },
      { action: 'waitFor', matcher: 'Available commands', timeout: 2000 }
    ],
    expectedOutcomes: ['Connected', 'Available commands']
  },

  {
    name: 'Test Flow Execution on Real Server',
    steps: [
      { action: 'waitFor', matcher: 'Connected', timeout: 10000 },
      { action: 'wait', timeout: 500 },
      { action: 'input', value: '/flow test' },
      { action: 'waitFor', matcher: /flow.*test|Starting flow|Initializing|test/, timeout: 10000 },
      { action: 'wait', timeout: 5000 }  // Wait for flow to complete
    ]
  },

  {
    name: 'OSB Flow with Real Backend',
    steps: [
      { action: 'waitFor', matcher: 'Connected', timeout: 10000 },
      { action: 'wait', timeout: 500 },
      { action: 'input', value: '/flow osb What are the ethical implications of AI?' },
      { action: 'waitFor', matcher: /osb|OSB|Starting flow/, timeout: 10000 },
      // Wait for agents to join
      { action: 'waitFor', matcher: /researcher|policy_analyst|fact_checker|explorer/, timeout: 15000 },
      // Wait for UI prompt
      { action: 'waitFor', matcher: /Continue\?|Proceed\?|confirm|ready/i, timeout: 20000 },
      { action: 'wait', timeout: 1000 },
      { action: 'input', value: 'yes' },
      // Wait for analysis to start
      { action: 'waitFor', matcher: /analysis|processing|working/i, timeout: 10000 },
      // Wait for results (this might take a while)
      { action: 'wait', timeout: 30000 }
    ]
  },

  {
    name: 'User Message Interaction',
    steps: [
      { action: 'waitFor', matcher: 'Connected', timeout: 10000 },
      { action: 'wait', timeout: 500 },
      { action: 'input', value: 'Hello, this is a test message from the automated test suite!' },
      { action: 'wait', timeout: 2000 },
      { action: 'input', value: 'Can you see my messages?' },
      { action: 'wait', timeout: 2000 }
    ]
  }
];

async function runRealServerTests() {
  console.log('🌐 Buttermilk CLI Real Server Integration Tests');
  console.log('==============================================\n');

  const host = process.env.BUTTERMILK_HOST || 'localhost';
  const port = parseInt(process.env.BUTTERMILK_PORT || '8080');
  
  console.log(`📡 Testing against server at ${host}:${port}`);
  console.log('   (Set BUTTERMILK_HOST and BUTTERMILK_PORT to test different servers)\n');

  // First, check if server is reachable
  try {
    const http = await import('http');
    await new Promise((resolve, reject) => {
      const req = http.request({
        hostname: host,
        port: port,
        path: '/api/session',
        method: 'GET',
        timeout: 5000
      }, (res) => {
        if (res.statusCode === 200) {
          resolve(true);
        } else {
          reject(new Error(`Server returned status ${res.statusCode}`));
        }
      });
      
      req.on('error', reject);
      req.on('timeout', () => reject(new Error('Connection timeout')));
      req.end();
    });
    
    console.log('✅ Server is reachable\n');
  } catch (error) {
    console.error('❌ Cannot reach server:', error);
    console.log('\nMake sure the Buttermilk backend is running on the specified host and port.');
    process.exit(1);
  }

  let totalTests = 0;
  let passedTests = 0;
  let failedTests = 0;

  for (const scenario of realServerScenarios) {
    totalTests++;
    
    console.log(`\n🏃 Running: ${scenario.name}`);
    console.log('─'.repeat(50));
    
    const client = new TestClient({
      cliPath: path.join(__dirname, '../../dist/cli.js'),
      host,
      port,
      debug: process.argv.includes('--debug')
    });

    const runner = new ScenarioRunner(client);

    try {
      await client.start();
      const startTime = Date.now();
      
      const result = await runner.runScenario(scenario);
      const duration = ((Date.now() - startTime) / 1000).toFixed(1);
      
      if (result.success) {
        console.log(`✅ PASSED in ${duration}s`);
        passedTests++;
        
        if (process.argv.includes('--verbose')) {
          console.log('\n📝 Key messages:');
          const output = result.output;
          // Extract interesting messages
          const lines = output.split('\n').filter(line => 
            line.includes('Connected') ||
            line.includes('flow') ||
            line.includes('agent') ||
            line.includes('OSB') ||
            line.includes('complete')
          ).slice(0, 10);
          lines.forEach(line => console.log(`   ${line.trim()}`));
        }
      } else {
        console.log(`❌ FAILED in ${duration}s`);
        result.errors.forEach(error => console.log(`   - ${error}`));
        failedTests++;
        
        if (process.argv.includes('--debug')) {
          console.log('\n📝 Full output:');
          console.log(result.output);
        }
      }
    } catch (error) {
      console.log(`❌ FAILED with error: ${error}`);
      failedTests++;
    } finally {
      await client.stop();
    }
    
    // Small delay between tests
    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  // Summary
  console.log('\n\n📊 Test Summary');
  console.log('═'.repeat(50));
  console.log(`Total Tests: ${totalTests}`);
  console.log(`✅ Passed: ${passedTests}`);
  console.log(`❌ Failed: ${failedTests}`);
  console.log(`Success Rate: ${((passedTests / totalTests) * 100).toFixed(1)}%`);
  
  console.log('\n💡 Tips:');
  console.log('   - Use --debug to see all CLI output');
  console.log('   - Use --verbose to see key messages from passed tests');
  console.log('   - Set BUTTERMILK_HOST and BUTTERMILK_PORT for remote servers');
  
  process.exit(failedTests === 0 ? 0 : 1);
}

// Command line interface
if (import.meta.url === `file://${process.argv[1]}`) {
  if (process.argv.includes('--help')) {
    console.log(`
Buttermilk CLI Real Server Test

Usage: node real-server-test.js [options]

Options:
  --debug     Show all CLI output
  --verbose   Show key messages from passed tests
  --help      Show this help message

Environment Variables:
  BUTTERMILK_HOST    Server host (default: localhost)
  BUTTERMILK_PORT    Server port (default: 8080)

This test suite runs against a real Buttermilk backend server.
Make sure the server is running before executing tests.
    `);
    process.exit(0);
  }

  runRealServerTests().catch(console.error);
}

export { runRealServerTests };