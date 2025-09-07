#!/bin/bash

# Default number of lines to show
DEFAULT_LINES=50

# Allow override via command line argument
if [ -n "$1" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    DEFAULT_LINES=$1
    echo "Using custom line count: $DEFAULT_LINES"
fi

echo "📋 Buttermilk Server Log Viewer (Structured Logs)"
echo "=================================================="
echo ""

# Find the most recent structured log file (JSONL format)
LOG_FILE=$(ls -t /tmp/buttermilk_*.jsonl 2>/dev/null | head -n 1)

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ No structured log files found in /tmp/"
    echo ""
    echo "To create debug logs:"
    echo "1. Stop the current server: pkill -f buttermilk.runner.cli"
    echo "2. Start in debug mode: make debug"
    echo ""
    echo "Note: For better log access, use: uv run python -m buttermilk.debug.ws_debug_cli logs -n $DEFAULT_LINES"
    echo ""
    exit 1
fi

# Show log info
LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
LOG_LINES=$(wc -l < "$LOG_FILE")
LAST_MODIFIED=$(date -r "$LOG_FILE" "+%Y-%m-%d %H:%M:%S")

echo "📄 Log file: $LOG_FILE"
echo "📏 Size: $LOG_SIZE"
echo "📝 Lines: $LOG_LINES"
echo "🕒 Last modified: $LAST_MODIFIED"
echo ""

# Recommend using the proper structured log tools
echo "⚠️  DEPRECATION NOTICE: This script works with structured logs but has limitations."
echo "For full structured log support, use: uv run python -m buttermilk.debug.ws_debug_cli logs -n $DEFAULT_LINES"
echo ""

# Menu
echo "Options:"
echo "1. Show last $DEFAULT_LINES lines (basic)"
echo "2. Show errors only (basic text search)"
echo "3. Show warnings and errors (basic text search)"
echo "4. Use proper structured log tool (RECOMMENDED)"
echo "5. Search for pattern (basic text search)"
echo ""

read -p "Choose option (1-5): " choice

case $choice in
    1)
        echo -e "\n📋 Last $DEFAULT_LINES lines:\n"
        tail -$DEFAULT_LINES "$LOG_FILE"
        ;;
    2)
        echo -e "\n❌ Errors (basic text search - use ws_debug_cli for proper structured filtering):\n"
        grep -i "error\|exception\|traceback" "$LOG_FILE" | tail -$DEFAULT_LINES
        ;;
    3)
        echo -e "\n⚠️  Warnings and Errors (basic text search):\n"
        grep -i "warn\|error\|exception" "$LOG_FILE" | tail -$DEFAULT_LINES
        ;;
    4)
        echo -e "\n🔧 Using proper structured log tool:\n"
        cd /src/buttermilk
        uv run python -m buttermilk.debug.ws_debug_cli logs -n $DEFAULT_LINES
        ;;
    5)
        read -p "Enter search pattern: " pattern
        echo -e "\n🔍 Searching for '$pattern' (basic text search):\n"
        grep -i "$pattern" "$LOG_FILE" | tail -$DEFAULT_LINES   
        ;;
    *)
        echo "Invalid option"
        ;;
esac