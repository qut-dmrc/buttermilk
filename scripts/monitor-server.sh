#!/bin/bash

# Monitor Buttermilk server output
echo "🔍 Monitoring Buttermilk server..."
echo "================================"

# Find the server process
SERVER_PID=$(ps aux | grep "buttermilk.runner.cli" | grep -v grep | awk '{print $2}' | tail -1)

if [ -z "$SERVER_PID" ]; then
    echo "❌ Server not running"
    exit 1
fi

echo "✅ Found server process: PID $SERVER_PID"
echo ""

# Use strace to monitor the process output
# This captures system calls and signals
echo "📋 Recent activity:"
timeout 10s strace -p $SERVER_PID -s 200 2>&1 | grep -E "(write|read|send|recv)" | tail -20

# Alternative: Check for log files
echo ""
echo "📁 Checking for log files..."
find /workspaces/buttermilk -name "*.log" -type f -mmin -5 2>/dev/null | head -5

# Check API endpoints
echo ""
echo "🌐 API Status:"
curl -s http://localhost:8000/health || echo "No health endpoint"
echo ""
curl -s http://localhost:8000/api/session | jq || echo "Session endpoint failed"
