#!/bin/bash

echo "🔍 Buttermilk Flow Debugging"
echo "==========================="
echo ""

# 1. Check server health
echo "1️⃣ Server Health Check:"
curl -s http://localhost:8000/health | jq
echo ""

# 2. Check available flows
echo "2️⃣ Available Flows:"
curl -s http://localhost:8000/api/flows | jq
echo ""

# 3. Check monitoring status
echo "3️⃣ Monitoring Status:"
curl -s http://localhost:8000/monitoring/system/status | jq
echo ""

# 4. Check for any alerts
echo "4️⃣ System Alerts:"
curl -s http://localhost:8000/monitoring/alerts | jq '.alerts | length' | xargs -I {} echo "   {} active alerts"
echo ""

# 5. Check flow metrics
echo "5️⃣ Flow Metrics:"
curl -s http://localhost:8000/monitoring/metrics/flows | jq
echo ""

# 6. Look for recent logs
echo "6️⃣ Recent Log Files:"
find /workspaces/buttermilk -name "*.log" -type f -mmin -10 2>/dev/null | while read log; do
    echo "   📄 $log"
    echo "      Last 5 lines:"
    tail -5 "$log" | sed 's/^/      /'
    echo ""
done

# 7. Check if WebSocket is working
echo "7️⃣ WebSocket Test:"
SESSION=$(curl -s http://localhost:8000/api/session | jq -r .session_id)
echo "   Session: $SESSION"

# 8. Try a simple HTTP flow request
echo ""
echo "8️⃣ Direct Flow Request (HTTP):"
curl -s -X POST "http://localhost:8000/flow/zot" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test query"}' | jq . || echo "   Failed"

echo ""
echo "🔍 Debug Summary:"
echo "=================="
ps aux | grep buttermilk | grep -v grep | wc -l | xargs -I {} echo "✅ {} Buttermilk processes running"
netstat -tlnp 2>/dev/null | grep 8000 > /dev/null && echo "✅ Port 8000 is listening" || echo "❌ Port 8000 not listening"
