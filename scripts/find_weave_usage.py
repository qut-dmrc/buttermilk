#!/usr/bin/env python
"""Find all weave usage patterns to help with lazy loading refactoring.

This script scans the codebase to identify:
1. Where weave is imported
2. Where weave decorators are used (@weave.op)
3. Where weave functions are called (weave.init, etc)

Run with: uv run python scripts/find_weave_usage.py
"""

import re
from pathlib import Path
from collections import defaultdict


def scan_file_for_weave(filepath: Path) -> dict[str, list[tuple[int, str]]]:
    """Scan a Python file for weave usage patterns.

    Returns dict with keys: 'imports', 'decorators', 'calls', 'type_hints'
    """
    results = {
        'imports': [],
        'decorators': [],
        'calls': [],
        'type_hints': [],
    }

    try:
        with open(filepath) as f:
            lines = f.readlines()

        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith('#'):
                continue

            # Imports
            if re.search(r'^\s*import weave|^\s*from weave', line):
                results['imports'].append((i, line.strip()))

            # Decorators
            if re.search(r'@weave\.|@.*weave\.op', line):
                results['decorators'].append((i, line.strip()))

            # Function calls
            if re.search(r'weave\.(init|op|client|finish)', line):
                results['calls'].append((i, line.strip()))

            # Type hints
            if re.search(r':\s*weave\.|-> weave\.', line):
                results['type_hints'].append((i, line.strip()))

    except Exception as e:
        print(f"Error scanning {filepath}: {e}")

    return results


def main():
    """Scan buttermilk codebase for weave usage."""
    print("="*80)
    print("🔍 Weave Usage Analysis")
    print("="*80)
    print("\nScanning buttermilk/ directory for weave usage patterns...\n")

    # Scan all Python files
    root = Path("buttermilk")
    all_results = defaultdict(lambda: defaultdict(list))

    for py_file in root.rglob("*.py"):
        # Skip __pycache__ and .venv
        if '__pycache__' in str(py_file) or '.venv' in str(py_file):
            continue

        results = scan_file_for_weave(py_file)

        # Only record files that have weave usage
        if any(results.values()):
            for category, items in results.items():
                if items:
                    all_results[category][str(py_file)] = items

    # Display results
    categories = [
        ('imports', '📦 Module-Level Imports', 'CRITICAL - These cause eager loading'),
        ('decorators', '🎨 Decorators', 'MEDIUM - Evaluated at import time'),
        ('calls', '📞 Function Calls', 'LOW - Can be lazy loaded'),
        ('type_hints', '📝 Type Hints', 'LOW - Use TYPE_CHECKING'),
    ]

    for category, title, priority in categories:
        if category in all_results:
            print(f"\n{title} ({priority})")
            print("-" * 80)

            files = all_results[category]
            print(f"Found in {len(files)} file(s):\n")

            for filepath, occurrences in sorted(files.items()):
                print(f"\n📄 {filepath}")
                for line_num, line_content in occurrences:
                    print(f"   Line {line_num:4d}: {line_content}")

    # Summary
    print("\n" + "="*80)
    print("📊 Summary")
    print("="*80)

    print(f"""
Files with weave imports:    {len(all_results.get('imports', {}))}
Files with weave decorators: {len(all_results.get('decorators', {}))}
Files with weave calls:      {len(all_results.get('calls', {}))}
Files with weave type hints: {len(all_results.get('type_hints', {}))}

REFACTORING STRATEGY:

1. IMPORTS (Highest Priority)
   - Move 'import weave' statements to function level
   - Or make them conditional based on config
   - Use TYPE_CHECKING for type hints only

2. DECORATORS (Medium Priority)
   - Create conditional decorator factory:
     def optional_weave_op(func):
         if config.weave_enabled:
             import weave
             return weave.op(func)
         return func

3. FUNCTION CALLS (Lowest Priority)
   - These are fine - weave will be imported when needed
   - Already lazy by nature (only loads when called)

4. TYPE HINTS (Lowest Priority)
   - Use TYPE_CHECKING pattern:
     from typing import TYPE_CHECKING
     if TYPE_CHECKING:
         import weave
""")


if __name__ == "__main__":
    main()
