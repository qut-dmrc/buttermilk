#!/bin/bash

# Buttermilk Workflow Validation Tool
# Enforces the 9-step workflow from INSTRUCTIONS.md

STEP="$1"
TASK_DESCRIPTION="$2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔍 Buttermilk Workflow Checker${NC}"
echo "================================="
echo ""

# Define the workflow steps
WORKFLOW_STEPS=("STOP" "ANALYZE" "PLAN" "TEST" "IMPLEMENT" "DOCUMENT" "VALIDATE" "COMMIT" "REFLECT")

# Check if step is valid
if [[ ! " ${WORKFLOW_STEPS[@]} " =~ " ${STEP} " ]]; then
    echo -e "${RED}❌ Invalid step: $STEP${NC}"
    echo "Valid steps: ${WORKFLOW_STEPS[*]}"
    exit 1
fi

echo -e "Current Step: ${YELLOW}$STEP${NC}"
[[ -n "$TASK_DESCRIPTION" ]] && echo "Task: $TASK_DESCRIPTION"
echo ""

# Step-specific checks
case "$STEP" in
    "STOP")
        echo "✅ STOP: Understanding the problem scope"
        echo ""
        echo "Checklist:"
        echo "□ Have you read the relevant documentation?"
        echo "□ Do you understand the project goals and architecture?"
        echo "□ Have you checked GitHub issues for relevant past work?"
        ;;
    
    "ANALYZE")
        echo "✅ ANALYZE: Mapping system architecture"
        echo ""
        echo "Checklist:"
        echo "□ Have you identified the components involved?"
        echo "□ Have you found the root cause of the issue?"
        echo "□ Do you understand the data flow?"
        
        # Check for recent GitHub issue searches
        if command -v gh &> /dev/null; then
            echo ""
            echo "Recent issues in buttermilk repo:"
            gh issue list --repo qut-dmrc/buttermilk --limit 5 --state all
        fi
        ;;
    
    "PLAN")
        echo "✅ PLAN: Creating implementation plan"
        echo ""
        echo "Checklist:"
        echo "□ Have you created or found a GitHub issue?"
        echo "□ Is your plan documented with clear phases?"
        echo "□ Have you defined validation criteria?"
        
        # Check if there's an open issue
        if command -v gh &> /dev/null; then
            echo ""
            echo "Open issues assigned to you:"
            gh issue list --repo qut-dmrc/buttermilk --assignee @me --state open
        fi
        ;;
    
    "TEST")
        echo "✅ TEST: Writing failing tests first"
        echo ""
        echo "Checklist:"
        echo "□ Have you written tests that capture expected behavior?"
        echo "□ Do the tests currently fail?"
        echo "□ Are the tests comprehensive?"
        
        # Check for test files
        echo ""
        echo "Recent test files:"
        if [[ -z "$BUTTERMILK_DIR" ]]; then
            echo -e "${RED}❌ Environment variable BUTTERMILK_DIR is not set.${NC}"
            exit 1
        fi
        find "$BUTTERMILK_DIR" -name "*test*.py" -type f -mmin -60 2>/dev/null | head -5
        ;;
    
    "IMPLEMENT")
        echo "✅ IMPLEMENT: Making minimal changes"
        echo ""
        echo "Checklist:"
        echo "□ Are you making minimal changes that solve the root cause?"
        echo "□ Are you following existing code patterns?"
        echo "□ Have you checked that tests exist?"
        
        # Check if tests exist
        if [ -d "/src/buttermilk/tests" ]; then
            echo ""
            echo "Test coverage areas:"
            ls -la /src/buttermilk/tests/ | grep -E "test_.*\.py" | head -5
        fi
        ;;
    
    "DOCUMENT")
        echo "✅ DOCUMENT: Adding documentation"
        echo ""
        echo "Checklist:"
        echo "□ Have you added docstrings to new functions?"
        echo "□ Have you added inline comments where needed?"
        echo "□ Is the documentation clear and helpful?"
        
        # Check for undocumented Python files
        echo ""
        echo "Recently modified Python files:"
        find /src/buttermilk -name "*.py" -type f -mmin -60 2>/dev/null | head -5
        ;;
    
    "VALIDATE")
        echo "✅ VALIDATE: Running validation tools"
        echo ""
        echo "Checklist:"
        echo "□ Have you run the tests?"
        echo "□ Have you run linting (ruff)?"
        echo "□ Have you tested end-to-end?"
        
        # Suggest validation commands
        echo ""
        echo "Validation commands:"
        echo "  uv run pytest"
        echo "  uv run ruff check buttermilk"
        echo "  make test"
        ;;
    
    "COMMIT")
        echo "✅ COMMIT: Committing changes"
        echo ""
        echo "Checklist:"
        echo "□ Are changes committed in logical chunks?"
        echo "□ Are commit messages clear and descriptive?"
        echo "□ Have you updated the GitHub issue?"
        
        # Show git status
        echo ""
        echo "Current git status:"
        cd /src/buttermilk && git status --short
        ;;
    
    "REFLECT")
        echo "✅ REFLECT: Reviewing and updating docs"
        echo ""
        echo "Checklist:"
        echo "□ Have you reviewed your performance?"
        echo "□ Should docs/bots be updated?"
        echo "□ Are there lessons learned to document?"
        
        # Check docs/bots
        echo ""
        echo "Bot documentation files:"
        ls -la /src/buttermilk/docs/bots/ 2>/dev/null | grep -E "\.md$"
        ;;
esac

echo ""
echo -e "${GREEN}✨ Remember: NO EXCEPTIONS to the workflow!${NC}"