#!/bin/bash

# Buttermilk GitHub Issue Management Tool
# Search and manage GitHub issues for the Buttermilk project

ACTION="$1"
QUERY="$2"
BODY="$3"
LABELS="$4"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}📋 Buttermilk GitHub Issue Manager${NC}"
echo "=================================="
echo ""

# Check if gh CLI is available
if ! command -v gh &> /dev/null; then
    echo -e "${RED}❌ GitHub CLI (gh) not installed${NC}"
    echo "Install with: https://cli.github.com/"
    exit 1
fi

# Check if we're in a git repo and set the right one
cd /src/buttermilk 2>/dev/null || true

case "$ACTION" in
    "search")
        if [ -z "$QUERY" ]; then
            echo -e "${RED}❌ Search query required${NC}"
            exit 1
        fi
        
        echo -e "${BLUE}🔍 Searching for: $QUERY${NC}"
        echo ""
        
        # Search in issues and PRs
        echo "Open Issues:"
        gh issue list --repo qut-dmrc/buttermilk --search "$QUERY" --state open --limit 10
        
        echo ""
        echo "Closed Issues:"
        gh issue list --repo qut-dmrc/buttermilk --search "$QUERY" --state closed --limit 5
        
        echo ""
        echo "Related PRs:"
        gh pr list --repo qut-dmrc/buttermilk --search "$QUERY" --state all --limit 5
        ;;
    
    "create")
        if [ -z "$QUERY" ]; then
            echo -e "${RED}❌ Issue title required${NC}"
            exit 1
        fi
        
        echo -e "${BLUE}📝 Creating new issue...${NC}"
        echo "Title: $QUERY"
        
        # Prepare body with template
        if [ -z "$BODY" ]; then
            BODY="## Description
[Describe the issue or feature request]

## Context
- Working on: [current task]
- Following workflow step: [STOP/ANALYZE/PLAN/TEST/etc]

## Expected Behavior
[What should happen]

## Current Behavior
[What currently happens]

## Possible Solution
[Optional: Suggest a fix or implementation]

---
Created by Buttermilk MCP tools"
        fi
        
        # Prepare labels
        if [ -n "$LABELS" ]; then
            LABEL_ARGS="--label $LABELS"
        else
            LABEL_ARGS=""
        fi
        
        # Create the issue
        if gh issue create --repo qut-dmrc/buttermilk --title "$QUERY" --body "$BODY" $LABEL_ARGS; then
            echo -e "${GREEN}✅ Issue created successfully${NC}"
            
            # Show the created issue
            echo ""
            echo "Latest issue:"
            gh issue list --repo qut-dmrc/buttermilk --limit 1
        else
            echo -e "${RED}❌ Failed to create issue${NC}"
            exit 1
        fi
        ;;
    
    "link")
        if [ -z "$QUERY" ]; then
            echo -e "${RED}❌ Issue number or commit message required${NC}"
            exit 1
        fi
        
        echo -e "${BLUE}🔗 Linking to issue...${NC}"
        
        # If it's a number, show how to reference in commit
        if [[ "$QUERY" =~ ^[0-9]+$ ]]; then
            echo "To link in your commit message, use:"
            echo "  - References: #$QUERY"
            echo "  - Closes: closes #$QUERY"
            echo "  - Fixes: fixes #$QUERY"
            echo ""
            echo "Example commit:"
            echo "  git commit -m \"Add feature X (refs #$QUERY)\""
            echo ""
            
            # Show the issue details
            echo "Issue details:"
            gh issue view "$QUERY" --repo qut-dmrc/buttermilk
        else
            # It's a commit message, suggest how to add issue reference
            echo "Add issue reference to your commit message:"
            echo "  $QUERY (refs #ISSUE_NUMBER)"
            echo ""
            echo "Recent issues you might want to reference:"
            gh issue list --repo qut-dmrc/buttermilk --assignee @me --limit 5
        fi
        ;;
    
    *)
        echo -e "${RED}❌ Invalid action: $ACTION${NC}"
        echo "Valid actions: search, create, link"
        echo ""
        echo "Examples:"
        echo "  buttermilk-github-issue search \"workflow validation\""
        echo "  buttermilk-github-issue create \"Add MCP tools\" \"Need to create MCP tools for the project\""
        echo "  buttermilk-github-issue link 123"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ GitHub operation completed${NC}"