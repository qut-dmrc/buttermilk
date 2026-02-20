"""Unified CLI entry point for Buttermilk.

Routes commands to appropriate handlers:
- `bm batch ...` -> click-based batch commands
- `bm ...` (anything else) -> Hydra-based runner CLI
"""

import sys


def main() -> None:
    """Unified CLI entry point.

    Dispatches to batch commands (click) or runner (Hydra) based on first argument.
    """
    # Check if first arg is "batch" - route to click-based batch CLI
    if len(sys.argv) > 1 and sys.argv[1] not in ("--help", "-h"):
        if sys.argv[1] == "batch":
            # Remove "batch" from argv so click sees the subcommand
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            from buttermilk.cli.batch import batch

            batch()
            return

    # Otherwise route to Hydra-based runner
    from buttermilk.runner.cli import main as hydra_main

    hydra_main()


if __name__ == "__main__":
    main()
