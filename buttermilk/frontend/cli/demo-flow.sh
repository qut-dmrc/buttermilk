#!/bin/bash

echo "🎬 Buttermilk CLI Flow Demo"
echo "=========================="
echo ""
echo "This demo will show:"
echo "1. Connecting to the backend"
echo "2. Running a test flow" 
echo "3. Running an OSB flow with interaction"
echo ""
echo "Press Enter to start..."
read

# Start the CLI
echo "Starting CLI..."
node dist/cli.js --host localhost --port 8000