#!/usr/bin/env python3
"""Buttermilk Log Viewer Tool

View and analyze Buttermilk server logs.
This script is used by MCP tools to access log information.
"""

import argparse
import glob
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from rich import print
except ImportError:
    pass


class LogViewer:
    def __init__(self, min_level: str = "DEBUG"):
        self.log_pattern = "/tmp/buttermilk_*.log"
        self.latest_log = self.find_latest_log()
        self.levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
        self.min_level_num = self.levels.get(min_level.upper(), 0)

    def find_latest_log(self) -> str | None:
        """Find the most recent debug log file. If not found, falls back to the most recent info log file."""
        all_logs = sorted(glob.glob(self.log_pattern), key=os.path.getmtime, reverse=True)
        if not all_logs:
            return None

        # Prefer the debug log if it's one of the two most recent files.
        # This is based on the user hint that info and debug logs are created together.
        for log_file in all_logs[:2]:
            if log_file.endswith("_debug.log"):
                return log_file

        # Otherwise, return the most recent log file.
        return all_logs[0]

    def list_logs(self) -> list[dict]:
        """List all log files with metadata."""
        log_files = glob.glob(self.log_pattern)
        if not log_files:
            return []

        logs_info = []
        for log_file in sorted(log_files, key=lambda x: Path(x).stat().st_mtime, reverse=True):
            stat = os.stat(log_file)
            size_mb = stat.st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(stat.st_mtime)

            logs_info.append(
                {
                    "path": log_file,
                    "size_mb": round(size_mb, 2),
                    "modified": mod_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "lines": self._count_lines(log_file),
                },
            )

        return logs_info

    def _count_lines(self, file_path: str) -> int:
        """Count lines in a file."""
        try:
            with open(file_path) as f:
                return sum(1 for _ in f)
        except:
            return 0

    def _get_line_level(self, line: str) -> int:
        """Extracts log level from a line and returns its numeric value."""
        match = re.search(r" - (DEBUG|INFO|WARNING|ERROR|CRITICAL) - ", line)
        if match:
            level_name = match.group(1)
            return self.levels.get(level_name, -1)
        return 99  # Lines without a level are always included

    def _filter_by_level(self, lines: list[str]) -> list[str]:
        """Filter lines by the minimum log level."""
        if self.min_level_num == 0:  # if DEBUG, no filtering needed
            return lines

        filtered_lines = []
        for line in lines:
            if self._get_line_level(line) >= self.min_level_num:
                filtered_lines.append(line)
        return filtered_lines

    def tail(self, lines: int = 50) -> bool:
        """Show the last N lines of the log."""
        if not self.latest_log:
            print("❌ No Buttermilk log files found")
            print("\nTo create logs:")
            print("1. Start server in debug mode: make debug")
            print("2. Or run with verbose: uv run python -m buttermilk.runner.cli +run=api verbose=true")
            return False

        level_name = list(self.levels.keys())[self.min_level_num]
        print(f"📋 Log file: {self.latest_log}")
        print(f"Last {lines} lines (level: {level_name}+):")
        print("")

        try:
            with open(self.latest_log) as f:
                all_lines = f.readlines()
                display_lines = self._filter_by_level(all_lines)
                for line in display_lines[-lines:]:
                    print(line.rstrip())
            return True
        except Exception as e:
            print(f"❌ Error reading log: {e}")
            return False

    def errors(self, lines: int = 50) -> bool:
        """Show recent errors from the log."""
        if not self.latest_log:
            print("❌ No log files found")
            return False

        print(f"❌ Errors from: {self.latest_log}")
        print(f"(last {lines} matches)")
        print("")

        error_pattern = re.compile(r"error|exception|traceback|failed", re.IGNORECASE)

        try:
            with open(self.latest_log) as f:
                matching_lines = []
                for line in f:
                    if error_pattern.search(line):
                        matching_lines.append(line.rstrip())

                for line in matching_lines[-lines:]:
                    print(line)

            if not matching_lines:
                print("No errors found")

            return True
        except Exception as e:
            print(f"❌ Error reading log: {e}")
            return False

    def warnings(self, lines: int = 50) -> bool:
        """Show warnings and errors from the log."""
        if not self.latest_log:
            print("❌ No log files found")
            return False

        print(f"⚠️ Warnings and Errors from: {self.latest_log}")
        print(f"(last {lines} matches)")
        print("")

        warn_pattern = re.compile(r"warn|error|exception", re.IGNORECASE)

        try:
            with open(self.latest_log) as f:
                matching_lines = []
                for line in f:
                    if warn_pattern.search(line):
                        matching_lines.append(line.rstrip())

                for line in matching_lines[-lines:]:
                    print(line)

            if not matching_lines:
                print("No warnings or errors found")

            return True
        except Exception as e:
            print(f"❌ Error reading log: {e}")
            return False

    def search(self, pattern: str, lines: int = 50) -> bool:
        """Search for a pattern in the log."""
        if not self.latest_log:
            print("❌ No log files found")
            return False

        print(f"🔍 Searching for '{pattern}' in: {self.latest_log}")
        print(f"(last {lines} matches)")
        print("")

        search_pattern = re.compile(pattern, re.IGNORECASE)

        try:
            with open(self.latest_log) as f:
                matching_lines = []
                for line in f:
                    if search_pattern.search(line):
                        matching_lines.append(line.rstrip())

                for line in matching_lines[-lines:]:
                    print(line)

            if not matching_lines:
                print("No matches found")

            return True
        except Exception as e:
            print(f"❌ Error reading log: {e}")
            return False

    def websocket(self, lines: int = 50) -> bool:
        """Show WebSocket-related messages."""
        if not self.latest_log:
            print("❌ No log files found")
            return False

        print(f"🌐 WebSocket messages from: {self.latest_log}")
        print(f"(last {lines} matches)")
        print("")

        ws_pattern = re.compile(r"websocket|ws|message_service|flow.*message", re.IGNORECASE)

        try:
            with open(self.latest_log) as f:
                matching_lines = []
                for line in f:
                    if ws_pattern.search(line):
                        matching_lines.append(line.rstrip())

                for line in matching_lines[-lines:]:
                    print(line)

            if not matching_lines:
                print("No WebSocket messages found")

            return True
        except Exception as e:
            print(f"❌ Error reading log: {e}")
            return False

    def follow(self) -> bool:
        """Follow the log file (like tail -f)."""
        if not self.latest_log:
            print("❌ No log files found")
            return False

        level_name = list(self.levels.keys())[self.min_level_num]
        print(f"👀 Following log: {self.latest_log} (level: {level_name}+)")
        print("(Ctrl+C to stop)")
        print("")

        try:
            with open(self.latest_log) as f:
                f.seek(0, 2)
                while True:
                    line = f.readline()
                    if line:
                        if self._get_line_level(line.rstrip()) >= self.min_level_num:
                            print(line.rstrip())
                    else:
                        time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nStopped following log")
            return True
        except Exception as e:
            print(f"❌ Error following log: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Buttermilk Log Viewer Tool",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  buttermilk_logs.py tail --lines 30
  buttermilk_logs.py tail --level INFO
  buttermilk_logs.py errors
  buttermilk_logs.py search 'pattern' --lines 20
""",
    )

    parser.add_argument(
        "mode",
        nargs="?",
        default="latest",
        choices=["tail", "errors", "warnings", "search", "websocket", "follow", "list", "latest"],
        help="The mode to run in. If no mode is provided, it prints the path of the latest log.",
    )
    parser.add_argument("pattern", nargs="?", default=None, help="The pattern to search for (in search mode).")
    parser.add_argument("-n", "--lines", type=int, default=20, help="Number of lines to show.")
    parser.add_argument(
        "-l",
        "--level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Minimum log level to display for tail and follow.",
    )

    args = parser.parse_args()

    viewer = LogViewer(min_level=args.level)
    success = False

    if args.mode == "latest":
        if viewer.latest_log:
            print(viewer.latest_log)
            success = True
        else:
            print("No logs found", file=sys.stderr)
            success = False

    elif args.mode == "tail":
        success = viewer.tail(args.lines)

    elif args.mode == "errors":
        success = viewer.errors(args.lines)

    elif args.mode == "warnings":
        success = viewer.warnings(args.lines)

    elif args.mode == "search":
        if not args.pattern:
            print("❌ Search pattern required for search mode", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        success = viewer.search(args.pattern, args.lines)

    elif args.mode == "websocket":
        success = viewer.websocket(args.lines)

    elif args.mode == "follow":
        success = viewer.follow()

    elif args.mode == "list":
        logs = viewer.list_logs()
        if logs:
            print("📋 Buttermilk Log Files:")
            print("")
            for log in logs:
                print(f"File: {log['path']}")
                print(f"  Size: {log['size_mb']} MB")
                print(f"  Lines: {log['lines']}")
                print(f"  Modified: {log['modified']}")
                print("")
            success = True
        else:
            print("❌ No log files found")
            success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
