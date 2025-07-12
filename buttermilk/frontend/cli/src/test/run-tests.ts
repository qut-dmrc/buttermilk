#!/usr/bin/env node
import { MockWebSocketServer, MockHTTPServer } from './MockWebSocketServer.js';
import { TestClient, ScenarioRunner } from './TestClient.js';
import { testScenarios, stressTestScenarios, errorScenarios } from './scenarios.js';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface TestConfig {
  httpPort: number;
  wsPort: number;
  sessionId: string;
  debug: boolean;
}

const defaultConfig: TestConfig = {
  httpPort: 8080,
  wsPort: 8080,
  sessionId: 'test-session-123',
  debug: process.env.DEBUG === 'true'
};

async function runTests(config: TestConfig = defaultConfig) {
  console.log('🧪 Starting Buttermilk CLI Tests');
  console.log('================================\n');

  // Start mock servers
  const httpServer = new MockHTTPServer(config.httpPort, config.sessionId);
  const wsServer = new MockWebSocketServer({
    port: config.wsPort,
    sessionId: config.sessionId,
    delayMs: 50
  });

  try {
    console.log('📡 Starting mock servers...');
    await httpServer.start();
    await wsServer.start();
    console.log('✅ Mock servers started\n');

    // Run different test suites
    const suites = [
      { name: 'Basic Tests', scenarios: testScenarios },
      { name: 'Error Handling', scenarios: errorScenarios },
      { name: 'Stress Tests', scenarios: stressTestScenarios }
    ];

    let totalTests = 0;
    let passedTests = 0;
    let failedTests = 0;

    for (const suite of suites) {
      console.log(`\n📋 Running ${suite.name}`);
      console.log('─'.repeat(40));

      for (const scenario of suite.scenarios) {
        totalTests++;
        
        // Create new client for each test
        const client = new TestClient({
          cliPath: path.join(__dirname, '../../dist/cli.js'),
          host: 'localhost',
          port: config.httpPort,
          debug: config.debug
        });

        const runner = new ScenarioRunner(client);

        try {
          console.log(`\n🏃 Running: ${scenario.name}`);
          await client.start();
          
          const result = await runner.runScenario(scenario);
          
          if (result.success) {
            console.log(`✅ PASSED: ${scenario.name}`);
            passedTests++;
          } else {
            console.log(`❌ FAILED: ${scenario.name}`);
            result.errors.forEach(error => console.log(`   - ${error}`));
            failedTests++;
            
            if (config.debug) {
              console.log('\n📝 Output:');
              console.log(result.output);
            }
          }
        } catch (error) {
          console.log(`❌ FAILED: ${scenario.name} - ${error}`);
          failedTests++;
        } finally {
          await client.stop();
        }
      }
    }

    // Summary
    console.log('\n\n📊 Test Summary');
    console.log('═'.repeat(40));
    console.log(`Total Tests: ${totalTests}`);
    console.log(`✅ Passed: ${passedTests}`);
    console.log(`❌ Failed: ${failedTests}`);
    console.log(`Success Rate: ${((passedTests / totalTests) * 100).toFixed(1)}%`);

    // Message log summary
    if (config.debug) {
      console.log('\n\n📨 Message Log Summary');
      console.log('─'.repeat(40));
      const messageLog = wsServer.getMessageLog();
      console.log(`Total messages: ${messageLog.length}`);
      
      const messageTypes = new Map<string, number>();
      messageLog.forEach(entry => {
        const type = entry.message.type;
        messageTypes.set(type, (messageTypes.get(type) || 0) + 1);
      });
      
      console.log('\nMessage type breakdown:');
      messageTypes.forEach((count, type) => {
        console.log(`  ${type}: ${count}`);
      });
    }

    // Exit code based on test results
    const exitCode = failedTests === 0 ? 0 : 1;
    process.exit(exitCode);

  } catch (error) {
    console.error('💥 Test runner error:', error);
    process.exit(1);
  } finally {
    console.log('\n🛑 Stopping mock servers...');
    await wsServer.stop();
    await httpServer.stop();
  }
}

// Command line interface
if (import.meta.url === `file://${process.argv[1]}`) {
  const args = process.argv.slice(2);
  const config = { ...defaultConfig };

  // Parse command line arguments
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--debug':
      case '-d':
        config.debug = true;
        break;
      case '--http-port':
        config.httpPort = parseInt(args[++i]);
        break;
      case '--ws-port':
        config.wsPort = parseInt(args[++i]);
        break;
      case '--help':
      case '-h':
        console.log(`
Buttermilk CLI Test Runner

Usage: node run-tests.js [options]

Options:
  --debug, -d         Enable debug output
  --http-port <port>  HTTP server port (default: 8080)
  --ws-port <port>    WebSocket server port (default: 8080)
  --help, -h          Show this help message

Environment Variables:
  DEBUG=true          Enable debug output
        `);
        process.exit(0);
    }
  }

  runTests(config).catch(console.error);
}

export { runTests };