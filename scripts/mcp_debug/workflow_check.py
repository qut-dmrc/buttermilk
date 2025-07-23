#!/usr/bin/env python3
"""Buttermilk Workflow Validation Tool

Enforces the 9-step workflow from INSTRUCTIONS.md.
This script is used by MCP tools to validate workflow compliance.
"""

import sys
import subprocess
import os
from typing import List, Dict, Any
from pathlib import Path


class WorkflowChecker:
    def __init__(self):
        self.workflow_steps = [
            "STOP", "ANALYZE", "PLAN", "TEST", "IMPLEMENT", 
            "DOCUMENT", "VALIDATE", "COMMIT", "REFLECT"
        ]
        
        self.step_info = {
            "STOP": {
                "description": "Understanding the problem scope",
                "checklist": [
                    "Have you read the relevant documentation?",
                    "Do you understand the project goals and architecture?",
                    "Have you checked GitHub issues for relevant past work?"
                ]
            },
            "ANALYZE": {
                "description": "Mapping system architecture",
                "checklist": [
                    "Have you identified the components involved?",
                    "Have you found the root cause of the issue?",
                    "Do you understand the data flow?"
                ]
            },
            "PLAN": {
                "description": "Creating implementation plan",
                "checklist": [
                    "Have you created or found a GitHub issue?",
                    "Is your plan documented with clear phases?",
                    "Have you defined validation criteria?"
                ]
            },
            "TEST": {
                "description": "Writing failing tests first",
                "checklist": [
                    "Have you written tests that capture expected behavior?",
                    "Do the tests currently fail?",
                    "Are the tests comprehensive?"
                ]
            },
            "IMPLEMENT": {
                "description": "Making minimal changes",
                "checklist": [
                    "Are you making minimal changes that solve the root cause?",
                    "Are you following existing code patterns?",
                    "Have you checked that tests exist?"
                ]
            },
            "DOCUMENT": {
                "description": "Adding documentation",
                "checklist": [
                    "Have you added docstrings to new functions?",
                    "Have you added inline comments where needed?",
                    "Is the documentation clear and helpful?"
                ]
            },
            "VALIDATE": {
                "description": "Running validation tools",
                "checklist": [
                    "Have you run the tests?",
                    "Have you run linting (ruff)?",
                    "Have you tested end-to-end?"
                ]
            },
            "COMMIT": {
                "description": "Committing changes",
                "checklist": [
                    "Are changes committed in logical chunks?",
                    "Are commit messages clear and descriptive?",
                    "Have you updated the GitHub issue?"
                ]
            },
            "REFLECT": {
                "description": "Reviewing and updating docs",
                "checklist": [
                    "Have you reviewed your performance?",
                    "Should docs/bots be updated?",
                    "Are there lessons learned to document?"
                ]
            }
        }
        
    def _run_command(self, cmd: List[str]) -> Optional[str]:
        """Run a command and return output."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except Exception:
            return None
            
    def check_step(self, step: str, task_description: Optional[str] = None) -> bool:
        """Check a specific workflow step."""
        if step not in self.workflow_steps:
            print(f"❌ Invalid step: {step}")
            print(f"Valid steps: {', '.join(self.workflow_steps)}")
            return False
            
        print(f"🔍 Buttermilk Workflow Checker")
        print("=" * 33)
        print("")
        print(f"Current Step: {step}")
        if task_description:
            print(f"Task: {task_description}")
        print("")
        
        info = self.step_info[step]
        print(f"✅ {step}: {info['description']}")
        print("")
        print("Checklist:")
        for item in info['checklist']:
            print(f"□ {item}")
            
        # Step-specific additional information
        if step == "ANALYZE":
            self._show_recent_issues()
            
        elif step == "PLAN":
            self._show_assigned_issues()
            
        elif step == "TEST":
            self._show_recent_test_files()
            
        elif step == "IMPLEMENT":
            self._show_test_coverage()
            
        elif step == "DOCUMENT":
            self._show_recent_python_files()
            
        elif step == "VALIDATE":
            self._show_validation_commands()
            
        elif step == "COMMIT":
            self._show_git_status()
            
        elif step == "REFLECT":
            self._show_docs_files()
            
        print("")
        print("✨ Remember: NO EXCEPTIONS to the workflow!")
        return True
        
    def _show_recent_issues(self):
        """Show recent GitHub issues."""
        if self._command_exists("gh"):
            print("")
            print("Recent issues in buttermilk repo:")
            output = self._run_command([
                "gh", "issue", "list", "--repo", "qut-dmrc/buttermilk",
                "--limit", "5", "--state", "all"
            ])
            if output:
                print(output)
                
    def _show_assigned_issues(self):
        """Show issues assigned to current user."""
        if self._command_exists("gh"):
            print("")
            print("Open issues assigned to you:")
            output = self._run_command([
                "gh", "issue", "list", "--repo", "qut-dmrc/buttermilk",
                "--assignee", "@me", "--state", "open"
            ])
            if output:
                print(output)
                
    def _show_recent_test_files(self):
        """Show recently modified test files."""
        print("")
        print("Recent test files:")
        # Find test files modified in the last hour
        import glob
        import time
        
        test_files = []
        for pattern in ["**/test*.py", "**/*test.py"]:
            test_files.extend(glob.glob(pattern, recursive=True))
            
        # Filter by modification time
        recent_files = []
        current_time = time.time()
        for file in test_files[:5]:
            try:
                if os.path.getmtime(file) > current_time - 3600:
                    recent_files.append(file)
            except:
                pass
                
        if recent_files:
            for file in recent_files:
                print(f"  {file}")
        else:
            print("  No recently modified test files")
            
    def _show_test_coverage(self):
        """Show test coverage areas."""
        test_dir = Path("tests")
        if test_dir.exists():
            print("")
            print("Test coverage areas:")
            test_files = list(test_dir.glob("test_*.py"))
            for file in test_files[:5]:
                print(f"  {file.name}")
                
    def _show_recent_python_files(self):
        """Show recently modified Python files."""
        print("")
        print("Recently modified Python files:")
        import glob
        import time
        
        py_files = glob.glob("**/*.py", recursive=True)
        recent_files = []
        current_time = time.time()
        
        for file in py_files:
            try:
                if os.path.getmtime(file) > current_time - 3600:
                    recent_files.append(file)
            except:
                pass
                
        for file in recent_files[:5]:
            print(f"  {file}")
            
    def _show_validation_commands(self):
        """Show validation commands."""
        print("")
        print("Validation commands:")
        print("  uv run pytest")
        print("  uv run ruff check buttermilk")
        print("  make test")
        
    def _show_git_status(self):
        """Show current git status."""
        print("")
        print("Current git status:")
        output = self._run_command(["git", "status", "--short"])
        if output:
            print(output)
        else:
            print("  (clean working directory)")
            
    def _show_docs_files(self):
        """Show documentation files."""
        docs_dir = Path("docs/bots")
        if docs_dir.exists():
            print("")
            print("Bot documentation files:")
            for file in docs_dir.glob("*.md"):
                print(f"  {file.name}")
                
    def _command_exists(self, cmd: str) -> bool:
        """Check if a command exists."""
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=False)
            return True
        except FileNotFoundError:
            return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: workflow_check.py <step> [task_description]")
        print(f"Steps: {', '.join(WorkflowChecker().workflow_steps)}")
        print("")
        print("Examples:")
        print('  workflow_check.py STOP')
        print('  workflow_check.py ANALYZE "Fix WebSocket connection issue"')
        sys.exit(1)
        
    step = sys.argv[1].upper()
    task_description = sys.argv[2] if len(sys.argv) > 2 else None
    
    checker = WorkflowChecker()
    success = checker.check_step(step, task_description)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()