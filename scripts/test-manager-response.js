#!/usr/bin/env node

const WebSocket = require('ws');

class TestClient {
  constructor(url) {
    this.url = url;
    this.ws = null;
  }

  connect() {
    return new Promise((resolve, reject) => {
      console.log(`🔌 Connecting to ${this.url}...`);
      
      this.ws = new WebSocket(this.url);
      
      this.ws.on('open', () => {
        console.log('✅ Connected to WebSocket');
        resolve();
      });

      this.ws.on('message', (data) => {
        try {
          const message = JSON.parse(data);
          console.log(`📥 [RECV] Type: ${message.type}`);
          if (message.preview) {
            console.log(`📄 Preview: ${message.preview}`);
          }
          if (message.outputs?.content) {
            console.log(`💬 Content: ${message.outputs.content}`);
          }
        } catch (e) {
          console.log(`📥 [RECV] Raw: ${data}`);
        }
      });

      this.ws.on('error', (error) => {
        console.error('❌ WebSocket error:', error.message);
        reject(error);
      });

      this.ws.on('close', () => {
        console.log('🔌 WebSocket connection closed');
      });
    });
  }

  sendMessage(message) {
    const jsonMessage = JSON.stringify(message);
    console.log(`📤 [SEND] ${jsonMessage}`);
    this.ws.send(jsonMessage);
  }

  async runTests() {
    try {
      await this.connect();

      console.log('\n🧪 Test 1: Correct manager_response format');
      this.sendMessage({ 
        type: 'manager_response', 
        content: 'Hello from test script using correct format!'
      });

      // Wait 3 seconds
      await new Promise(resolve => setTimeout(resolve, 3000));

      console.log('\n🧪 Test 2: Start a flow with manager_response');
      this.sendMessage({ 
        type: 'run_flow', 
        flow: 'zot',
        prompt: 'What is AI?'
      });

      // Wait 5 seconds for flow to start
      await new Promise(resolve => setTimeout(resolve, 5000));

      console.log('\n🧪 Test 3: Send user input to flow');
      this.sendMessage({ 
        type: 'manager_response', 
        content: 'Tell me more about machine learning'
      });

      // Wait for response
      await new Promise(resolve => setTimeout(resolve, 10000));

      console.log('\n✅ All tests completed');
      this.ws.close();
    } catch (error) {
      console.error('❌ Test failed:', error);
      process.exit(1);
    }
  }
}

async function main() {
  // Get session first
  const fetch = (await import('node-fetch')).default;
  
  try {
    const response = await fetch('http://localhost:8000/api/session');
    const data = await response.json();
    const sessionId = data.session_id;
    
    console.log(`🆔 Got session ID: ${sessionId}`);
    
    const wsUrl = `ws://localhost:8000/ws/${sessionId}`;
    const client = new TestClient(wsUrl);
    await client.runTests();
    
  } catch (error) {
    console.error('❌ Failed to get session:', error.message);
    process.exit(1);
  }
}

main().catch(console.error);