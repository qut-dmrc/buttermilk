#!/bin/bash

echo "🤖 Automated Buttermilk Flow Testing Process"
echo "==========================================="
echo ""

# Step 1: Check if server is running
echo "1️⃣ Checking server status..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "   ❌ Server not running. Starting server in debug mode..."
    cd /workspaces/buttermilk
    make debug &
    SERVER_PID=$!
    echo "   ⏳ Waiting for server to start (PID: $SERVER_PID)..."
    echo "   📝 Server logs: /tmp/buttermilk-debug.log"
    
    # Wait for server to be ready
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "   ✅ Server started successfully!"
            break
        fi
        sleep 1
    done
else
    echo "   ✅ Server is already running"
    echo "   💡 To see server logs, restart with: make debug"
fi

# Step 2: Build CLI if needed
echo ""
echo "2️⃣ Building CLI client..."
cd /workspaces/buttermilk/buttermilk/frontend/cli
npm run build > /dev/null 2>&1
echo "   ✅ CLI built"

# Step 3: Test connection
echo ""
echo "3️⃣ Testing API connection..."
SESSION=$(curl -s http://localhost:8000/api/session | jq -r .session_id)
if [ -n "$SESSION" ]; then
    echo "   ✅ API responding (session: $SESSION)"
else
    echo "   ❌ API not responding"
    exit 1
fi

# Step 4: Run zot flow test
echo ""
echo "4️⃣ Testing Zot flow..."
cat > test-zot-automated.js << 'EOF'
import WebSocket from 'ws';
import http from 'http';

const getSession = () => new Promise((resolve, reject) => {
  http.get('http://localhost:8000/api/session', (res) => {
    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => resolve(JSON.parse(data).session_id));
  }).on('error', reject);
});

async function runTest() {
  const sessionId = await getSession();
  console.log(`   Session: ${sessionId}`);
  
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
    const messages = [];
    
    ws.on('open', () => {
      console.log('   Connected to WebSocket');
      
      const msg = {
        type: 'run_flow',
        flow: 'zot',
        prompt: 'What are the latest AI research trends?'
      };
      
      console.log('   Sending zot flow request...');
      ws.send(JSON.stringify(msg));
    });

    ws.on('message', (data) => {
      const msg = JSON.parse(data.toString());
      messages.push(msg);
      console.log(`   Received: ${msg.type}`);
      
      if (msg.type === 'system_error' || msg.payload?.error) {
        console.log(`   ❌ Error: ${JSON.stringify(msg.payload)}`);
      }
    });

    ws.on('error', (error) => {
      console.error('   ❌ WebSocket error:', error.message);
      reject(error);
    });

    // Wait 10 seconds then close
    setTimeout(() => {
      ws.close();
      console.log(`   Total messages received: ${messages.length}`);
      resolve(messages);
    }, 10000);
  });
}

runTest().catch(console.error);
EOF

node test-zot-automated.js

# Step 5: Check flow metrics
echo ""
echo "5️⃣ Checking flow execution metrics..."
curl -s http://localhost:8000/monitoring/metrics/flows | jq '.zot' | grep -E "(total_executions|failed_executions|error_rate)"

# Step 6: Summary
echo ""
echo "📊 Test Summary:"
echo "================"

# Check if zot flow has any successful executions
SUCCESS=$(curl -s http://localhost:8000/monitoring/metrics/flows | jq -r '.zot.successful_executions // 0')
FAILED=$(curl -s http://localhost:8000/monitoring/metrics/flows | jq -r '.zot.failed_executions // 0')
TOTAL=$(curl -s http://localhost:8000/monitoring/metrics/flows | jq -r '.zot.total_executions // 0')

echo "   Total executions: $TOTAL"
echo "   Successful: $SUCCESS"
echo "   Failed: $FAILED"

if [ "$SUCCESS" -gt 0 ]; then
    echo "   ✅ Zot flow is working!"
else
    echo "   ❌ Zot flow is not working properly"
    echo ""
    echo "   Debugging tips:"
    echo "   - Check server logs: tail -f /tmp/buttermilk-debug.log"
    echo "   - Check for missing dependencies or configuration"
    echo "   - Verify ChromaDB is configured for zot flow"
fi

# Cleanup
rm -f test-zot-automated.js

echo ""
echo "🤖 Automated test complete!"
