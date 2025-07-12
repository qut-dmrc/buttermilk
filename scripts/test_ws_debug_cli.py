#!/usr/bin/env python3
"""Test script for the WebSocket Debug CLI.

This script demonstrates the capabilities of the standalone WebSocket debug CLI.
"""

import subprocess
import sys
import time


def test_cli_help():
    """Test that the CLI shows help."""
    print("Testing CLI help command...")
    result = subprocess.run(
        ["uv", "run", "python", "-m", "buttermilk.debug.cli", "websocket", "--help"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Help command works")
        print(result.stdout)
    else:
        print("✗ Help command failed")
        print(result.stderr)
        return False
    
    return True


def test_cli_standalone():
    """Test running the standalone CLI script."""
    print("\nTesting standalone CLI script...")
    result = subprocess.run(
        ["uv", "run", "python", "buttermilk/debug/ws_debug_cli.py", "--help"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Standalone script works")
    else:
        print("✗ Standalone script failed")
        print(result.stderr)
        return False
    
    return True


def main():
    """Run all tests."""
    print("🧪 Testing WebSocket Debug CLI")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1: CLI help
    if not test_cli_help():
        all_passed = False
    
    # Test 2: Standalone script
    if not test_cli_standalone():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed!")
        print("\nTo use the WebSocket Debug CLI interactively:")
        print("  1. Start the Buttermilk API: make debug")
        print("  2. Run: uv run python -m buttermilk.debug.cli websocket")
        print("     OR: uv run python buttermilk/debug/ws_debug_cli.py")
        print("\nAvailable commands:")
        print("  - help: Show all commands")
        print("  - start <flow> [query]: Start a flow")
        print("  - response <text>: Send a response")
        print("  - messages [n]: Show messages")
        print("  - agents: Show active agents")
        print("  - logs [n]: Show log lines")
        print("  - status: Show session status")
        print("  - quit: Exit the debugger")
    else:
        print("❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()