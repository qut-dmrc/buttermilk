#!/bin/bash

# Buttermilk Server Management Tool
# Start, stop, and manage the Buttermilk API server

ACTION="$1"
DEBUG="${2:-false}"
FLOWS="${3:-trans,zot,osb}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Buttermilk Server Manager${NC}"
echo "============================"
echo ""

# Convert comma-separated flows to array format
FLOWS_ARRAY="[$(echo "$FLOWS" | sed 's/,/","/g' | sed 's/^/"/;s/$/"/')]"

case "$ACTION" in
    "start")
        echo -e "${GREEN}Starting Buttermilk server...${NC}"
        
        # Check if already running
        if pgrep -f "buttermilk.runner.cli" > /dev/null; then
            echo -e "${YELLOW}⚠️  Server already running${NC}"
            echo "PID: $(pgrep -f "buttermilk.runner.cli")"
            exit 0
        fi
        
        if [ "$DEBUG" = "true" ]; then
            echo "Mode: Debug"
            echo "Flows: $FLOWS"
            echo "Log location: /tmp/buttermilk_*.log"
            echo ""
            
            # Start in debug mode
            nohup uv run python -m buttermilk.runner.cli "+flows=$FLOWS_ARRAY" +run=api llms=full verbose=true > /dev/null 2>&1 &
            
            echo -e "${GREEN}✅ Server starting in debug mode...${NC}"
            echo "Check logs with: ls -la /tmp/buttermilk_*.log | tail -1"
        else
            echo "Mode: Normal"
            echo "Flows: $FLOWS"
            echo ""
            
            # Start in normal mode
            nohup uv run python -m buttermilk.runner.cli "+flows=$FLOWS_ARRAY" +run=api llms=full > /dev/null 2>&1 &
            
            echo -e "${GREEN}✅ Server starting...${NC}"
        fi
        
        # Wait a moment and check if it started
        sleep 2
        if pgrep -f "buttermilk.runner.cli" > /dev/null; then
            echo -e "${GREEN}✅ Server started successfully${NC}"
            echo "PID: $(pgrep -f "buttermilk.runner.cli")"
        else
            echo -e "${RED}❌ Failed to start server${NC}"
            exit 1
        fi
        ;;
    
    "stop")
        echo -e "${YELLOW}Stopping Buttermilk server...${NC}"
        
        if ! pgrep -f "buttermilk.runner.cli" > /dev/null; then
            echo -e "${YELLOW}⚠️  Server not running${NC}"
            exit 0
        fi
        
        # Kill the process
        pkill -f "buttermilk.runner.cli"
        
        # Wait for it to stop
        sleep 2
        
        if pgrep -f "buttermilk.runner.cli" > /dev/null; then
            echo -e "${RED}❌ Failed to stop server, trying force kill...${NC}"
            pkill -9 -f "buttermilk.runner.cli"
            sleep 1
        fi
        
        if ! pgrep -f "buttermilk.runner.cli" > /dev/null; then
            echo -e "${GREEN}✅ Server stopped${NC}"
        else
            echo -e "${RED}❌ Failed to stop server${NC}"
            exit 1
        fi
        ;;
    
    "status")
        echo -e "${BLUE}Server Status:${NC}"
        echo ""
        
        if pgrep -f "buttermilk.runner.cli" > /dev/null; then
            PID=$(pgrep -f "buttermilk.runner.cli")
            echo -e "Status: ${GREEN}Running${NC}"
            echo "PID: $PID"
            
            # Show process info
            ps -p "$PID" -o pid,vsz,rss,pmem,pcpu,etime,comm
            
            # Check if port is listening
            if netstat -tlnp 2>/dev/null | grep -q ":8000"; then
                echo -e "Port 8000: ${GREEN}Listening${NC}"
            else
                echo -e "Port 8000: ${YELLOW}Not listening${NC}"
            fi
        else
            echo -e "Status: ${RED}Not running${NC}"
        fi
        ;;
    
    "health")
        echo -e "${BLUE}Health Check:${NC}"
        echo ""
        
        # Try health endpoint
        if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "API Health: ${GREEN}OK${NC}"
            curl -s http://localhost:8000/health | jq . 2>/dev/null || curl -s http://localhost:8000/health
        else
            echo -e "API Health: ${RED}Failed${NC}"
            echo "Server may not be running or still starting up"
        fi
        ;;
    
    "flows")
        echo -e "${BLUE}Available Flows:${NC}"
        echo ""
        
        # Try to get flows from API
        if curl -s -f http://localhost:8000/api/flows > /dev/null 2>&1; then
            curl -s http://localhost:8000/api/flows | jq . 2>/dev/null || curl -s http://localhost:8000/api/flows
        else
            echo -e "${RED}❌ Cannot retrieve flows${NC}"
            echo "Server may not be running"
        fi
        ;;
    
    *)
        echo -e "${RED}❌ Invalid action: $ACTION${NC}"
        echo "Valid actions: start, stop, status, health, flows"
        echo ""
        echo "Examples:"
        echo "  buttermilk-server start"
        echo "  buttermilk-server start true"
        echo "  buttermilk-server start true trans,zot"
        echo "  buttermilk-server stop"
        echo "  buttermilk-server status"
        exit 1
        ;;
esac