#!/bin/bash

# Buttermilk Log Viewer Tool
# View and analyze Buttermilk server logs

MODE="$1"
LINES="${2:-50}"
PATTERN="$3"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Find the most recent log file
find_log_file() {
    # First check for debug log
    if [ -f "/tmp/buttermilk-debug.log" ]; then
        echo "/tmp/buttermilk-debug.log"
        return
    fi
    
    # Then check for timestamped logs
    local latest_log=$(ls -t /tmp/buttermilk_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo "$latest_log"
        return
    fi
    
    # No log found
    echo ""
}

LOG_FILE=$(find_log_file)

if [ -z "$LOG_FILE" ]; then
    echo -e "${RED}❌ No Buttermilk log files found${NC}"
    echo ""
    echo "To create logs:"
    echo "1. Start server in debug mode: make debug"
    echo "2. Or run with verbose: uv run python -m buttermilk.runner.cli +run=api verbose=true"
    exit 1
fi

# Show log info
echo -e "${GREEN}📋 Buttermilk Log Viewer${NC}"
echo "========================"
echo -e "Log file: ${BLUE}$LOG_FILE${NC}"

if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
    LOG_LINES=$(wc -l < "$LOG_FILE")
    LAST_MODIFIED=$(date -r "$LOG_FILE" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "Unknown")
    
    echo "Size: $LOG_SIZE | Lines: $LOG_LINES | Modified: $LAST_MODIFIED"
fi

echo ""

# Execute based on mode
case "$MODE" in
    "tail")
        echo -e "${YELLOW}📜 Last $LINES lines:${NC}"
        echo ""
        tail -n "$LINES" "$LOG_FILE"
        ;;
    
    "errors")
        echo -e "${RED}❌ Errors (last $LINES):${NC}"
        echo ""
        grep -i "error\|exception\|traceback\|failed" "$LOG_FILE" | tail -n "$LINES"
        ;;
    
    "warnings")
        echo -e "${YELLOW}⚠️  Warnings and Errors (last $LINES):${NC}"
        echo ""
        grep -i "warn\|error\|exception" "$LOG_FILE" | tail -n "$LINES"
        ;;
    
    "search")
        if [ -z "$PATTERN" ]; then
            echo -e "${RED}❌ Search pattern required${NC}"
            exit 1
        fi
        echo -e "${BLUE}🔍 Searching for '$PATTERN' (last $LINES matches):${NC}"
        echo ""
        grep -i "$PATTERN" "$LOG_FILE" | tail -n "$LINES"
        ;;
    
    "websocket")
        echo -e "${BLUE}🌐 WebSocket messages (last $LINES):${NC}"
        echo ""
        grep -i "websocket\|ws\|message_service\|flow.*message" "$LOG_FILE" | tail -n "$LINES"
        ;;
    
    "follow")
        echo -e "${GREEN}👀 Following log (Ctrl+C to stop):${NC}"
        echo ""
        tail -f "$LOG_FILE"
        ;;
    
    *)
        echo -e "${RED}❌ Invalid mode: $MODE${NC}"
        echo "Valid modes: tail, errors, warnings, search, websocket, follow"
        exit 1
        ;;
esac