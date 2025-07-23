#!/usr/bin/env python3
"""Buttermilk Log Viewer Tool

View and analyze Buttermilk server logs.
This script is used by MCP tools to access log information.
"""

import sys
import os
import glob
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional


class LogViewer:
    def __init__(self):
        self.log_pattern = "/tmp/buttermilk_*.log"
        
    def find_latest_log(self) -> Optional[str]:
        """Find the most recent log file."""
        log_files = glob.glob(self.log_pattern)
        if not log_files:
            return None
            
        # Sort by modification time
        log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return log_files[0]
        
    def list_logs(self) -> List[str]:
        """List all log files with metadata."""
        log_files = glob.glob(self.log_pattern)
        if not log_files:
            return []
            
        logs_info = []
        for log_file in sorted(log_files, key=lambda x: os.path.getmtime(x), reverse=True):
            stat = os.stat(log_file)
            size_mb = stat.st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(stat.st_mtime)
            
            logs_info.append({
                "path": log_file,
                "size_mb": round(size_mb, 2),
                "modified": mod_time.strftime("%Y-%m-%d %H:%M:%S"),
                "lines": self._count_lines(log_file)
            })
            
        return logs_info
        
    def _count_lines(self, file_path: str) -> int:
        """Count lines in a file."""
        try:
            with open(file_path, 'r') as f:
                return sum(1 for _ in f)
        except:
            return 0
            
    def tail(self, lines: int = 50) -> bool:
        """Show the last N lines of the log."""
        log_file = self.find_latest_log()
        if not log_file:
            print("❌ No Buttermilk log files found")
            print("\nTo create logs:")
            print("1. Start server in debug mode: make debug")
            print("2. Or run with verbose: uv run python -m buttermilk.runner.cli +run=api verbose=true")
            return False
            
        print(f"📋 Log file: {log_file}")
        print(f"Last {lines} lines:")
        print("")
        
        try:
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                for line in all_lines[-lines:]:
                    print(line.rstrip())
            return True
        except Exception as e:
            print(f"❌ Error reading log: {e}")
            return False
            
    def errors(self, lines: int = 50) -> bool:
        """Show recent errors from the log."""
        log_file = self.find_latest_log()
        if not log_file:
            print("❌ No log files found")
            return False
            
        print(f"❌ Errors from: {log_file}")
        print(f"(last {lines} matches)")
        print("")
        
        error_pattern = re.compile(r'error|exception|traceback|failed', re.IGNORECASE)
        
        try:
            with open(log_file, 'r') as f:
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
        log_file = self.find_latest_log()
        if not log_file:
            print("❌ No log files found")
            return False
            
        print(f"⚠️ Warnings and Errors from: {log_file}")
        print(f"(last {lines} matches)")
        print("")
        
        warn_pattern = re.compile(r'warn|error|exception', re.IGNORECASE)
        
        try:
            with open(log_file, 'r') as f:
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
        log_file = self.find_latest_log()
        if not log_file:
            print("❌ No log files found")
            return False
            
        print(f"🔍 Searching for '{pattern}' in: {log_file}")
        print(f"(last {lines} matches)")
        print("")
        
        search_pattern = re.compile(pattern, re.IGNORECASE)
        
        try:
            with open(log_file, 'r') as f:
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
        log_file = self.find_latest_log()
        if not log_file:
            print("❌ No log files found")
            return False
            
        print(f"🌐 WebSocket messages from: {log_file}")
        print(f"(last {lines} matches)")
        print("")
        
        ws_pattern = re.compile(r'websocket|ws|message_service|flow.*message', re.IGNORECASE)
        
        try:
            with open(log_file, 'r') as f:
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
        log_file = self.find_latest_log()
        if not log_file:
            print("❌ No log files found")
            return False
            
        print(f"👀 Following log: {log_file}")
        print("(Ctrl+C to stop)")
        print("")
        
        try:
            # Simple tail -f implementation
            with open(log_file, 'r') as f:
                # Go to end of file
                f.seek(0, 2)
                
                while True:
                    line = f.readline()
                    if line:
                        print(line.rstrip())
                    else:
                        import time
                        time.sleep(0.1)
                        
        except KeyboardInterrupt:
            print("\nStopped following log")
            return True
        except Exception as e:
            print(f"❌ Error following log: {e}")
            return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: buttermilk_logs.py <mode> [lines] [pattern]")
        print("Modes: tail, errors, warnings, search, websocket, follow, list")
        print("Examples:")
        print("  buttermilk_logs.py tail 100")
        print("  buttermilk_logs.py errors")
        print("  buttermilk_logs.py search 'pattern' 50")
        sys.exit(1)
        
    mode = sys.argv[1]
    lines = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    viewer = LogViewer()
    
    if mode == "tail":
        success = viewer.tail(lines)
        
    elif mode == "errors":
        success = viewer.errors(lines)
        
    elif mode == "warnings":
        success = viewer.warnings(lines)
        
    elif mode == "search":
        if len(sys.argv) < 3:
            print("❌ Search pattern required")
            sys.exit(1)
        pattern = sys.argv[2]
        lines = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        success = viewer.search(pattern, lines)
        
    elif mode == "websocket":
        success = viewer.websocket(lines)
        
    elif mode == "follow":
        success = viewer.follow()
        
    elif mode == "list":
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
            
    else:
        print(f"❌ Invalid mode: {mode}")
        print("Valid modes: tail, errors, warnings, search, websocket, follow, list")
        success = False
        
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()