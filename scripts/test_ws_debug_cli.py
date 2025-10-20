#!/usr/bin/env python3
"""Test script for the WebSocket Debug CLI infrastructure commands.

This script demonstrates the capabilities of the ws_debug_cli infrastructure commands.
For flow debugging, use DebugAgent's puppet mode instead.
"""

import subprocess
import sys


def test_cli_help():
    """Test that the CLI shows help."""
    print("Testing CLI help command...")
    result = subprocess.run(
        ["uv", "run", "python", "-m", "buttermilk.debug.ws_debug_cli", "--help"],
        check=False, capture_output=True,
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


def test_infrastructure_commands():
    """Test infrastructure commands."""
    print("\nTesting infrastructure commands...")
    
    commands = [
        (["logs", "--help"], "logs command"),
        (["list-logs", "--help"], "list-logs command"),
        (["test-connection", "--help"], "test-connection command"),
    ]
    
    all_passed = True
    for cmd_args, name in commands:
        result = subprocess.run(
            ["uv", "run", "python", "-m", "buttermilk.debug.ws_debug_cli"] + cmd_args,
            check=False, capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ {name} works")
        else:
            print(f"✗ {name} failed")
            print(result.stderr)
            all_passed = False
    
    return all_passed


def main():
    """Run all tests."""
    print("🧪 Testing WebSocket Debug CLI Infrastructure Commands")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1: CLI help
    if not test_cli_help():
        all_passed = False
    
    # Test 2: Infrastructure commands
    if not test_infrastructure_commands():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed!")
        print("\nTo use the WebSocket Debug CLI infrastructure commands:")
        print("  1. Start the Buttermilk API: make debug")
        print("  2. Test connection: uv run python -m buttermilk.debug.ws_debug_cli test-connection")
        print("  3. View logs: uv run python -m buttermilk.debug.ws_debug_cli logs -n 20")
        print("  4. List log files: uv run python -m buttermilk.debug.ws_debug_cli list-logs")
        print("\nFor flow debugging, use DebugAgent's puppet mode:")
        print("  - See docs/agents/debugging.md for details")
        print("  - Puppet mode provides complete message access")
        print("  - Legacy flow commands (start, send, etc.) were deprecated in Issue #274")
    else:
        print("❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
