#!/bin/bash

# Test script for Buttermilk MCP tools
# Verifies that all tools are working correctly

echo "🧪 Testing Buttermilk MCP Tools"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Track results
PASSED=0
FAILED=0

# Test function
test_tool() {
    local tool_name="$1"
    local args="$2"
    local expected_exit="$3"
    
    echo -n "Testing $tool_name... "
    
    if .mcp/tools/"$tool_name".sh $args > /dev/null 2>&1; then
        ACTUAL_EXIT=0
    else
        ACTUAL_EXIT=1
    fi
    
    if [ "$ACTUAL_EXIT" -eq "$expected_exit" ]; then
        echo -e "${GREEN}✅ PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}❌ FAILED${NC} (expected exit $expected_exit, got $ACTUAL_EXIT)"
        ((FAILED++))
    fi
}

# Test each tool
echo "1. Testing buttermilk-workflow-check:"
test_tool "buttermilk-workflow-check" "STOP" 0
test_tool "buttermilk-workflow-check" "INVALID" 1

echo ""
echo "2. Testing buttermilk-logs:"
test_tool "buttermilk-logs" "tail 10" 0
test_tool "buttermilk-logs" "invalid_mode" 1

echo ""
echo "3. Testing buttermilk-server:"
test_tool "buttermilk-server" "status" 0
test_tool "buttermilk-server" "invalid_action" 1

echo ""
echo "4. Testing buttermilk-test-flow:"
test_tool "buttermilk-test-flow" "" 1  # Should fail without args

echo ""
echo "5. Testing buttermilk-config-validate:"
test_tool "buttermilk-config-validate" "" 1  # Should fail without args

echo ""
echo "6. Testing buttermilk-github-issue:"
test_tool "buttermilk-github-issue" "search test" 0

echo ""
echo "7. Testing MCP server:"
echo -n "Testing MCP server list command... "
if python3 .mcp/buttermilk-mcp-server.py list > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ FAILED${NC}"
    ((FAILED++))
fi

# Summary
echo ""
echo "================================"
echo "Test Summary:"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi