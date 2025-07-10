#!/usr/bin/env node
import WebSocket from 'ws';
import http from 'http';

const HOST = process.env.BUTTERMILK_HOST || 'localhost';
const PORT = parseInt(process.env.BUTTERMILK_PORT || '8000');

console.log(`🧪 Simple Flow Test against ${HOST}:${PORT}`);
console.log('=====================================\n');

async function getSessionId(): Promise<string> {
  return new Promise((resolve, reject) => {
    const req = http.request({
      hostname: HOST,
      port: PORT,
      path: '/api/session',
      method: 'GET'
    }, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const json = JSON.parse(data);
          resolve(json.session_id);
        } catch (e) {
          reject(e);
        }
      });
    });
    req.on('error', reject);
    req.end();
  });
}

async function runFlowTest() {
  try {
    // 1. Get session ID
    console.log('1️⃣  Getting session ID...');
    const sessionId = await getSessionId();
    console.log(`   ✅ Got session: ${sessionId}\n`);

    // 2. Connect WebSocket
    console.log('2️⃣  Connecting WebSocket...');
    const ws = new WebSocket(`ws://${HOST}:${PORT}/ws/${sessionId}`);
    
    const messages: any[] = [];
    
    ws.on('open', () => {
      console.log('   ✅ WebSocket connected\n');
      
      // 3. Send test flow command
      console.log('3️⃣  Starting test flow...');
      ws.send(JSON.stringify({
        type: 'run_flow',
        flow: 'test'
      }));
      console.log('   📤 Sent: run_flow test\n');
    });

    ws.on('message', (data) => {
      const msg = JSON.parse(data.toString());
      messages.push(msg);
      
      console.log(`   📨 Received: ${msg.type}`);
      
      // Log specific message details
      switch (msg.type) {
        case 'flow_progress_update':
          console.log(`      Status: ${msg.payload?.status} - ${msg.payload?.message}`);
          break;
        case 'agent_announcement':
          console.log(`      Agent: ${msg.payload?.agent_id} ${msg.payload?.action}`);
          break;
        case 'agent_output':
          console.log(`      Output: ${msg.payload?.content || JSON.stringify(msg.payload).substring(0, 100)}`);
          break;
        case 'system_message':
        case 'system_update':
          console.log(`      Message: ${msg.payload?.message || JSON.stringify(msg.payload).substring(0, 100)}`);
          break;
      }
      
      // Check for completion
      if (msg.type === 'flow_progress_update' && msg.payload?.status === 'COMPLETED') {
        console.log('\n4️⃣  Flow completed successfully! 🎉');
        
        // 5. Try OSB flow with interaction
        console.log('\n5️⃣  Starting OSB flow...');
        ws.send(JSON.stringify({
          type: 'run_flow',
          flow: 'osb',
          prompt: 'What are the benefits of automated testing?'
        }));
        console.log('   📤 Sent: run_flow osb with prompt\n');
      }
      
      // Handle UI messages that need response
      if (msg.type === 'ui_message' && msg.payload?.requires_response) {
        console.log('\n6️⃣  UI is asking for confirmation...');
        console.log(`   Question: ${msg.payload?.message}`);
        
        setTimeout(() => {
          console.log('   📤 Sending: yes');
          ws.send(JSON.stringify({
            type: 'manager_response',
            payload: { text: 'yes' }
          }));
        }, 1000);
      }
    });

    ws.on('error', (error) => {
      console.error('❌ WebSocket error:', error);
    });

    ws.on('close', () => {
      console.log('\n🔌 WebSocket closed');
      console.log(`\n📊 Total messages received: ${messages.length}`);
      
      // Summary of message types
      const typeCounts = messages.reduce((acc, msg) => {
        acc[msg.type] = (acc[msg.type] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);
      
      console.log('\n📈 Message type breakdown:');
      Object.entries(typeCounts).forEach(([type, count]) => {
        console.log(`   ${type}: ${count}`);
      });
      
      process.exit(0);
    });

    // Timeout after 60 seconds
    setTimeout(() => {
      console.log('\n⏱️  Test timeout - closing connection');
      ws.close();
    }, 60000);

  } catch (error) {
    console.error('❌ Test failed:', error);
    process.exit(1);
  }
}

runFlowTest();