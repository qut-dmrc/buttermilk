#!/usr/bin/env node
import WebSocket from 'ws';
import http from 'http';

const getSession = () => new Promise((resolve, reject) => {
  http.get('http://localhost:8000/api/session', (res) => {
    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => resolve(JSON.parse(data).session_id));
  }).on('error', reject);
});

async function testUIMessage() {
  console.log('🧪 Testing ui_message type fix');
  console.log('==============================\n');
  
  const sessionId = await getSession();
  console.log(`✅ Session: ${sessionId}\n`);
  
  const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
  
  ws.on('open', () => {
    console.log('✅ Connected to WebSocket\n');
    
    // Test 1: Send ui_message
    console.log('📤 Test 1: Sending ui_message...');
    ws.send(JSON.stringify({
      type: 'ui_message',
      payload: { text: "What's digital constitutionalism?" }
    }));
    
    // Test 2: Send old user_message (should fail)
    setTimeout(() => {
      console.log('\n📤 Test 2: Sending user_message (should fail)...');
      ws.send(JSON.stringify({
        type: 'user_message',
        payload: { text: "This should trigger unknown message warning" }
      }));
    }, 2000);
    
    // Test 3: Send manager_response
    setTimeout(() => {
      console.log('\n📤 Test 3: Sending manager_response...');
      ws.send(JSON.stringify({
        type: 'manager_response',
        payload: { text: "yes" }
      }));
    }, 4000);
  });

  ws.on('message', (data) => {
    const msg = JSON.parse(data.toString());
    console.log(`\n📨 Received: ${msg.type}`);
    if (msg.payload) {
      console.log(`   Payload: ${JSON.stringify(msg.payload).substring(0, 100)}`);
    }
  });

  ws.on('error', (error) => {
    console.error('❌ WebSocket error:', error);
  });

  // Close after 6 seconds
  setTimeout(() => {
    ws.close();
    console.log('\n✅ Test complete');
    console.log('\nCheck server logs for:');
    console.log('- ui_message should be processed without warnings');
    console.log('- user_message should show "Unknown message type" warning');
  }, 6000);
}

testUIMessage().catch(console.error);