#!/usr/bin/env node
import WebSocket from 'ws';
import http from 'http';

console.log('🧪 Testing Zot Flow');
console.log('==================\n');

// Get session ID
const getSession = () => new Promise((resolve, reject) => {
  http.get('http://localhost:8000/api/session', (res) => {
    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => {
      try {
        resolve(JSON.parse(data).session_id);
      } catch (e) {
        reject(e);
      }
    });
  }).on('error', reject);
});

async function testZotFlow() {
  try {
    const sessionId = await getSession();
    console.log(`✅ Session: ${sessionId}\n`);

    const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
    
    ws.on('open', () => {
      console.log('✅ Connected to WebSocket\n');
      
      // Send zot flow request
      const message = {
        type: 'run_flow',
        flow: 'zot',
        prompt: 'What are the latest developments in artificial intelligence research?'
      };
      
      console.log('📤 Sending zot flow request:');
      console.log(JSON.stringify(message, null, 2));
      ws.send(JSON.stringify(message));
    });

    ws.on('message', (data) => {
      const msg = JSON.parse(data.toString());
      const time = new Date().toLocaleTimeString();
      
      console.log(`\n[${time}] 📨 ${msg.type}`);
      
      // Log message details based on type
      if (msg.payload || msg.message) {
        const content = msg.payload || msg;
        console.log('   ', JSON.stringify(content).substring(0, 200));
      }
      
      // Look for specific zot-related messages
      if (msg.type === 'agent_announcement' && msg.payload?.agent_id?.includes('zot')) {
        console.log('   🎯 Zot agent detected!');
      }
    });

    ws.on('error', (error) => {
      console.error('❌ WebSocket error:', error);
    });

    ws.on('close', () => {
      console.log('\n🔌 Connection closed');
    });

    // Keep running for 30 seconds
    setTimeout(() => {
      console.log('\n⏱️  Test complete');
      ws.close();
      process.exit(0);
    }, 30000);

  } catch (error) {
    console.error('❌ Test failed:', error);
    process.exit(1);
  }
}

testZotFlow();