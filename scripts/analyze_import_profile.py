#!/usr/bin/env python
"""Analyze import profiling data to identify optimization opportunities.

Run with: uv run python scripts/analyze_import_profile.py
"""

import re
from collections import defaultdict
from pathlib import Path


def parse_import_profile(filename: str) -> list[tuple[int, str]]:
    """Parse -X importtime output file.

    Returns list of (cumulative_time_us, module_name) tuples.
    """
    imports = []
    with open(filename) as f:
        for line in f:
            if "import time:" in line:
                # Format: import time: self [us] | cumulative | imported package
                match = re.search(r"import time:\s+\d+\s+\|\s+(\d+)\s+\|\s+(.+)", line)
                if match:
                    cumulative_us = int(match.group(1))
                    module_name = match.group(2).strip()
                    imports.append((cumulative_us, module_name))
    return imports


def group_by_top_level(imports: list[tuple[int, str]]) -> dict[str, int]:
    """Group imports by top-level package and sum their times."""
    grouped = defaultdict(int)

    for cumulative_us, module_name in imports:
        # Extract top-level package
        top_level = module_name.split(".")[0]
        grouped[top_level] += cumulative_us

    return dict(grouped)


def format_time(us: int) -> str:
    """Format microseconds as human-readable time."""
    ms = us / 1000
    if ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms / 1000:.2f}s"


def analyze_profile(profile_file: str, title: str):
    """Analyze and display import profile."""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}\n")

    imports = parse_import_profile(profile_file)

    if not imports:
        print(f"❌ No import data found in {profile_file}")
        return

    # Sort by time and show top slowest individual imports
    imports_sorted = sorted(imports, key=lambda x: x[0], reverse=True)

    print("📊 Top 20 Slowest Individual Imports:")
    print(f"{'Time':<12} {'Module':<60}")
    print("-" * 80)

    for cumulative_us, module_name in imports_sorted[:20]:
        print(f"{format_time(cumulative_us):<12} {module_name:<60}")

    # Group by top-level package
    grouped = group_by_top_level(imports)
    grouped_sorted = sorted(grouped.items(), key=lambda x: x[1], reverse=True)

    total_time_us = sum(grouped.values())

    print("\n📦 Top Packages by Total Import Time:")
    print(f"{'Package':<30} {'Time':<12} {'% of Total':<12}")
    print("-" * 80)

    for package, time_us in grouped_sorted[:20]:
        percentage = (time_us / total_time_us) * 100
        print(f"{package:<30} {format_time(time_us):<12} {percentage:>6.1f}%")

    print(f"\n{'Total import time:':<30} {format_time(total_time_us):<12} 100.0%")


def find_lazy_loading_opportunities():
    """Identify modules that could benefit from lazy loading."""
    print(f"\n{'=' * 80}")
    print("💡 Lazy Loading Opportunities")
    print(f"{'=' * 80}\n")

    opportunities = [
        {
            "package": "wandb",
            "time": "2.5s",
            "usage": "Weights & Biases tracking - only needed if observability.weave.enabled=true",
            "where": "buttermilk._core or imports",
            "priority": "🔴 CRITICAL",
        },
        {
            "package": "litellm",
            "time": "1.0s",
            "usage": "LLM proxy - only needed when making LLM calls",
            "where": "Likely imported eagerly in _core",
            "priority": "🟡 HIGH",
        },
        {
            "package": "chromadb",
            "time": "~500ms",
            "usage": "Vector database - only needed if using RAG/vector storage",
            "where": "Imported in storage modules",
            "priority": "🟡 HIGH",
        },
        {
            "package": "google.cloud.aiplatform",
            "time": "~500ms",
            "usage": "GCP AI Platform - only needed for GCP deployments",
            "where": "Cloud infrastructure setup",
            "priority": "🟡 HIGH",
        },
        {
            "package": "fastapi",
            "time": "~200ms",
            "usage": "Web framework - only needed in API mode",
            "where": "buttermilk.api modules",
            "priority": "🟢 MEDIUM",
        },
        {
            "package": "autogen_core",
            "time": "Variable",
            "usage": "Multi-agent framework - only for agent flows",
            "where": "Agent modules",
            "priority": "🟢 MEDIUM",
        },
    ]

    print("The following packages take significant import time but may not be needed")
    print("for all operations. Consider lazy loading them:\n")

    for opp in opportunities:
        print(f"{opp['priority']} {opp['package']} ({opp['time']})")
        print(f"   Usage: {opp['usage']}")
        print(f"   Location: {opp['where']}")
        print()


def main():
    """Main analysis function."""
    print("\n" + "=" * 80)
    print("🔍 Buttermilk Import Profile Analysis")
    print("=" * 80)

    # Analyze both profiles
    profiles = [
        ("/tmp/import_core.txt", "buttermilk._core Import Profile"),
        ("/tmp/import_cli.txt", "buttermilk.runner.cli Import Profile"),
    ]

    for profile_file, title in profiles:
        if Path(profile_file).exists():
            analyze_profile(profile_file, title)
        else:
            print(f"\n⚠️  Profile not found: {profile_file}")
            print(
                f"   Run: uv run python -X importtime -c 'from buttermilk._core import config_bootstrap' 2> {profile_file}"
            )

    # Show recommendations
    find_lazy_loading_opportunities()

    print("\n" + "=" * 80)
    print("🎯 Key Recommendations")
    print("=" * 80)
    print("""
1. WANDB (2.5s) - The biggest offender
   - Check if wandb is imported eagerly in buttermilk._core
   - Only import when weave/wandb features are actually used
   - Add lazy import: 'import wandb' -> 'if config.weave.enabled: import wandb'

2. LITELLM (1.0s) - Second biggest
   - Defer import until LLM call is actually made
   - Move from module-level import to function-level import in llm client

3. CHROMADB (~500ms)
   - Only import when vector storage is configured
   - Check buttermilk.storage modules for eager imports

4. CLOUD LIBRARIES (GCP, Azure) - Combined ~1s
   - Only import cloud clients when actually configured
   - Check ExecutionContext setup for eager cloud client imports

5. FASTAPI (~200ms)
   - Already should only be imported in API mode
   - Verify it's not imported in buttermilk.runner.cli module level

Next step: Find WHERE these are being imported
Run: grep -r "^import wandb" buttermilk/
Run: grep -r "^from wandb" buttermilk/
""")


if __name__ == "__main__":
    main()
