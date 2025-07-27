#!/bin/bash

echo "📋 Buttermilk Server Log Viewer"
echo "==============================="
echo ""

LOG_FILE=$(./scripts/mcp_debug/getlog.sh)

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ No debug log found at $LOG_FILE"
    echo ""
    echo "To create debug logs:"
    echo "1. Stop the current server: pkill -f buttermilk.runner.cli"
    echo "2. Start in debug mode: make debug"
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

# Menu
echo "Options:"
echo "1. Show last 50 lines"
echo "2. Show errors only"
echo "3. Show warnings and errors"
echo "4. Follow log (tail -f)"
echo "5. Search for pattern"
echo "6. Show WebSocket messages"
echo ""

read -p "Choose option (1-6): " choice

case $choice in
    1)
        echo -e "\n📋 Last 20 lines:\n"
        tail -20 "$LOG_FILE"
        ;;
    2)
        echo -e "\n❌ Errors:\n"
        grep -i "error\|exception\|traceback" "$LOG_FILE" | tail -20
        ;;
    3)
        echo -e "\n⚠️  Warnings and Errors:\n"
        grep -i "warn\|error\|exception" "$LOG_FILE" | tail -20
        ;;
    4)
        echo -e "\n👀 Following log (Ctrl+C to stop):\n"
        tail -f "$LOG_FILE"
        ;;
    5)
        read -p "Enter search pattern: " pattern
        echo -e "\n🔍 Searching for '$pattern':\n"
        grep -i "$pattern" "$LOG_FILE" | tail -20   
        ;;
    6)
        echo -e "\n🌐 WebSocket messages:\n"
        grep -i "websocket\|ws\|message_service" "$LOG_FILE" | tail -20
        ;;
    *)
        echo "Invalid option"
        ;;
esac