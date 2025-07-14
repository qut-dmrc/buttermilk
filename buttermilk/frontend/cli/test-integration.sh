#!/bin/bash

# Integration test script for Buttermilk CLI
# This tests against a real backend

set -e

echo "🧪 Buttermilk CLI Integration Test"
echo "=================================="
echo ""

# Check if backend is running
if ! curl -s http://localhost:8000/api/session > /dev/null; then
    echo "❌ Backend not running on localhost:8000"
    echo "   Please start the Buttermilk backend first"
    exit 1
fi

echo "✅ Backend is running"
echo ""

# Build the CLI
echo "🔨 Building CLI..."
npm run build

# Create test script
cat > test-commands.txt << EOF
/help
/flow test
Hello, this is a test message
/flow osb What is artificial intelligence?
yes
EOF

echo ""
echo "🏃 Running CLI with test commands..."
echo ""

# Run CLI with test commands
timeout 30s node dist/cli.js --debug < test-commands.txt || true

echo ""
echo "✅ Integration test completed"

# Cleanup
rm -f test-commands.txt