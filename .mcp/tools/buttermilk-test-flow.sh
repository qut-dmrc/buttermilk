#!/bin/bash

# Buttermilk Flow Testing Tool
# Test specific Buttermilk flows with test data

FLOW="$1"
PROMPT="$2"
MODE="${3:-http}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🧪 Buttermilk Flow Tester${NC}"
echo "========================="
echo ""

if [ -z "$FLOW" ] || [ -z "$PROMPT" ]; then
    echo -e "${RED}❌ Missing required parameters${NC}"
    echo "Usage: buttermilk-test-flow <flow> <prompt> [mode]"
    echo "  flow: Flow name to test (e.g., zot, trans, osb)"
    echo "  prompt: Test prompt/query"
    echo "  mode: http or websocket (default: http)"
    exit 1
fi

echo -e "Flow: ${BLUE}$FLOW${NC}"
echo -e "Prompt: ${YELLOW}$PROMPT${NC}"
echo -e "Mode: ${BLUE}$MODE${NC}"
echo ""

# Check if server is running
if ! curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${RED}❌ Server not running or not healthy${NC}"
    echo "Start the server with: buttermilk-server start"
    exit 1
fi

case "$MODE" in
    "http")
        echo -e "${BLUE}📡 Testing via HTTP...${NC}"
        echo ""
        
        # Create JSON payload
        PAYLOAD=$(cat <<EOF
{
    "prompt": "$PROMPT",
    "parameters": {
        "test_mode": true
    }
}
EOF
)
        
        # Make the request
        RESPONSE=$(curl -s -X POST "http://localhost:8000/flow/$FLOW" \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD" 2>&1)
        
        # Check if successful
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Request successful${NC}"
            echo ""
            echo "Response:"
            echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"
            
            # Extract key information if possible
            if command -v jq > /dev/null 2>&1; then
                echo ""
                echo -e "${BLUE}Key Information:${NC}"
                
                # Try to extract session ID
                SESSION_ID=$(echo "$RESPONSE" | jq -r '.session_id // empty' 2>/dev/null)
                [ -n "$SESSION_ID" ] && echo "Session ID: $SESSION_ID"
                
                # Try to extract status
                STATUS=$(echo "$RESPONSE" | jq -r '.status // empty' 2>/dev/null)
                [ -n "$STATUS" ] && echo "Status: $STATUS"
                
                # Try to extract results
                RESULTS=$(echo "$RESPONSE" | jq -r '.results // empty' 2>/dev/null)
                [ -n "$RESULTS" ] && echo "Results: Available"
            fi
        else
            echo -e "${RED}❌ Request failed${NC}"
            echo "Error: $RESPONSE"
        fi
        ;;
    
    "websocket")
        echo -e "${BLUE}🌐 Testing via WebSocket...${NC}"
        echo ""
        
        # First get a session
        echo "Getting session..."
        SESSION_RESPONSE=$(curl -s http://localhost:8000/api/session)
        SESSION_ID=$(echo "$SESSION_RESPONSE" | jq -r '.session_id // empty' 2>/dev/null)
        
        if [ -z "$SESSION_ID" ]; then
            echo -e "${RED}❌ Failed to get session ID${NC}"
            echo "Response: $SESSION_RESPONSE"
            exit 1
        fi
        
        echo "Session ID: $SESSION_ID"
        echo ""
        
        # Create WebSocket test script
        TEMP_FILE=$(mktemp /tmp/ws_test.XXXXXX.py)
        cat > "$TEMP_FILE" << 'EOF'
import asyncio
import websockets
import json
import sys

async def test_flow(session_id, flow, prompt):
    uri = f"ws://localhost:8000/ws/{session_id}"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to WebSocket")
            
            # Send flow start message
            start_msg = {
                "type": "start_flow",
                "flow": flow,
                "inputs": {
                    "prompt": prompt
                }
            }
            
            await websocket.send(json.dumps(start_msg))
            print(f"Sent: {start_msg}")
            print("")
            
            # Receive messages
            print("Receiving messages:")
            timeout = 30  # 30 second timeout
            start_time = asyncio.get_event_loop().time()
            
            while True:
                try:
                    if asyncio.get_event_loop().time() - start_time > timeout:
                        print("\nTimeout reached")
                        break
                        
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    msg_type = data.get("type", "unknown")
                    print(f"[{msg_type}] {json.dumps(data, indent=2)}")
                    
                    if msg_type == "flow_complete" or msg_type == "error":
                        break
                        
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("\nConnection closed")
                    break
                    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    session_id = sys.argv[1]
    flow = sys.argv[2] 
    prompt = sys.argv[3]
    
    exit_code = asyncio.run(test_flow(session_id, flow, prompt))
    sys.exit(exit_code)
EOF
        
        # Run the WebSocket test
        if command -v python3 > /dev/null 2>&1; then
            python3 /tmp/ws_test.py "$SESSION_ID" "$FLOW" "$PROMPT"
        else
            echo -e "${RED}❌ Python not available for WebSocket testing${NC}"
            echo "WebSocket testing requires Python with websockets library"
        fi
        
        # Cleanup
        rm -f /tmp/ws_test.py
        ;;
    
    *)
        echo -e "${RED}❌ Invalid mode: $MODE${NC}"
        echo "Valid modes: http, websocket"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ Flow test completed${NC}"