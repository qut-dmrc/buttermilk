#!/usr/bin/env python3
"""Buttermilk GitHub Issue Management Tool

Search and manage GitHub issues for the Buttermilk project.
This script is used by MCP tools to interact with GitHub issues.
"""

import sys
import subprocess
import json
from typing import List, Optional, Dict, Any


class GitHubIssueManager:
    def __init__(self, repo: str = "qut-dmrc/buttermilk"):
        self.repo = repo
        
    def _run_gh_command(self, args: List[str]) -> Optional[str]:
        """Run a gh CLI command and return the output."""
        try:
            result = subprocess.run(
                ["gh"] + args,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"❌ GitHub CLI error: {e.stderr}")
            return None
        except FileNotFoundError:
            print("❌ GitHub CLI (gh) not installed")
            print("Install with: https://cli.github.com/")
            return None
            
    def search(self, query: str) -> bool:
        """Search for issues and PRs."""
        print(f"🔍 Searching for: {query}")
        print("")
        
        # Search open issues
        print("Open Issues:")
        result = self._run_gh_command([
            "issue", "list", "--repo", self.repo,
            "--search", query, "--state", "open", "--limit", "10"
        ])
        if result:
            print(result)
        else:
            print("No open issues found or error occurred")
            
        print("\nClosed Issues:")
        result = self._run_gh_command([
            "issue", "list", "--repo", self.repo,
            "--search", query, "--state", "closed", "--limit", "5"
        ])
        if result:
            print(result)
        else:
            print("No closed issues found or error occurred")
            
        print("\nRelated PRs:")
        result = self._run_gh_command([
            "pr", "list", "--repo", self.repo,
            "--search", query, "--state", "all", "--limit", "5"
        ])
        if result:
            print(result)
        else:
            print("No PRs found or error occurred")
            
        return True
        
    def create(self, title: str, body: Optional[str] = None, labels: Optional[str] = None) -> bool:
        """Create a new issue."""
        print(f"📝 Creating new issue...")
        print(f"Title: {title}")
        
        # Default body if not provided
        if not body:
            body = """## Description
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
Created by Buttermilk MCP tools"""
        
        # Build command
        cmd = ["issue", "create", "--repo", self.repo, "--title", title, "--body", body]
        
        if labels:
            cmd.extend(["--label", labels])
            
        # Create the issue
        result = self._run_gh_command(cmd)
        
        if result:
            print("✅ Issue created successfully")
            print(result)
            
            # Show the latest issue
            print("\nLatest issue:")
            latest = self._run_gh_command([
                "issue", "list", "--repo", self.repo, "--limit", "1"
            ])
            if latest:
                print(latest)
                
            return True
        else:
            print("❌ Failed to create issue")
            return False
            
    def link(self, issue_or_commit: str) -> bool:
        """Show how to link to an issue in commits."""
        print(f"🔗 Linking to issue...")
        
        # Check if it's a number (issue reference)
        if issue_or_commit.isdigit():
            issue_num = issue_or_commit
            
            print(f"To link in your commit message, use:")
            print(f"  - References: #{issue_num}")
            print(f"  - Closes: closes #{issue_num}")
            print(f"  - Fixes: fixes #{issue_num}")
            print("")
            print("Example commit:")
            print(f'  git commit -m "Add feature X (refs #{issue_num})"')
            print("")
            
            # Show issue details
            print("Issue details:")
            result = self._run_gh_command([
                "issue", "view", issue_num, "--repo", self.repo
            ])
            if result:
                print(result)
            else:
                print(f"Could not retrieve issue #{issue_num}")
                
        else:
            # It's a commit message
            print(f"Add issue reference to your commit message:")
            print(f'  {issue_or_commit} (refs #ISSUE_NUMBER)')
            print("")
            print("Recent issues you might want to reference:")
            
            result = self._run_gh_command([
                "issue", "list", "--repo", self.repo,
                "--assignee", "@me", "--limit", "5"
            ])
            if result:
                print(result)
            else:
                print("Could not retrieve issues")
                
        return True


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: github_issue.py <action> [args...]")
        print("Actions:")
        print("  search <query> - Search for issues and PRs")
        print("  create <title> [body] [labels] - Create a new issue")
        print("  link <issue_number|commit_message> - Show how to link issues")
        print("")
        print("Examples:")
        print('  github_issue.py search "workflow validation"')
        print('  github_issue.py create "Add MCP tools" "Need to create MCP tools"')
        print('  github_issue.py link 123')
        sys.exit(1)
        
    action = sys.argv[1]
    manager = GitHubIssueManager()
    
    if action == "search":
        if len(sys.argv) < 3:
            print("❌ Search query required")
            sys.exit(1)
        query = sys.argv[2]
        success = manager.search(query)
        
    elif action == "create":
        if len(sys.argv) < 3:
            print("❌ Issue title required")
            sys.exit(1)
        title = sys.argv[2]
        body = sys.argv[3] if len(sys.argv) > 3 else None
        labels = sys.argv[4] if len(sys.argv) > 4 else None
        success = manager.create(title, body, labels)
        
    elif action == "link":
        if len(sys.argv) < 3:
            print("❌ Issue number or commit message required")
            sys.exit(1)
        ref = sys.argv[2]
        success = manager.link(ref)
        
    else:
        print(f"❌ Invalid action: {action}")
        print("Valid actions: search, create, link")
        success = False
        
    print("")
    print("✅ GitHub operation completed" if success else "❌ GitHub operation failed")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()