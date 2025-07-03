"""
GCP Log Analysis for Runtime Debugging

Flow-agnostic log analysis using gcloud CLI for real-time debugging of buttermilk flows.
Provides structured analysis of daemon startup, agent initialization, and flow execution.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from buttermilk._core.log import logger


class LogEntry(BaseModel):
    """Structured representation of a GCP log entry."""
    
    timestamp: str = Field(description="Log entry timestamp")
    severity: str = Field(description="Log severity level")
    message: str = Field(description="Log message content")
    labels: Dict[str, Any] = Field(default_factory=dict, description="Log labels")
    source_location: Optional[Dict[str, Any]] = Field(default=None, description="Source location info")
    trace: Optional[str] = Field(default=None, description="Trace ID if available")
    span_id: Optional[str] = Field(default=None, description="Span ID if available")
    
    
class LogAnalysis(BaseModel):
    """Analysis results for a set of log entries."""
    
    total_entries: int = Field(description="Total number of log entries analyzed")
    error_count: int = Field(description="Number of error-level entries")
    warning_count: int = Field(description="Number of warning-level entries")
    time_range: tuple[str, str] = Field(description="Time range of analyzed logs")
    key_errors: List[str] = Field(default_factory=list, description="Key error messages")
    agent_errors: List[str] = Field(default_factory=list, description="Agent-specific errors")
    flow_performance: Dict[str, Any] = Field(default_factory=dict, description="Flow performance metrics")
    startup_issues: List[str] = Field(default_factory=list, description="Daemon startup issues")


class GCPLogAnalyzer:
    """
    Flow-agnostic GCP log analyzer for buttermilk runtime debugging.
    
    Uses gcloud CLI to fetch and analyze logs for daemon startup, agent initialization,
    and flow execution issues. Designed to be modular and extensible for different flows.
    """
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize GCP log analyzer.
        
        Args:
            project_id: GCP project ID (auto-detected if not provided)
        """
        self.project_id = project_id or self._detect_project_id()
        
    def _detect_project_id(self) -> str:
        """Auto-detect GCP project ID from environment or gcloud config."""
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not detect GCP project ID: {e}")
            return "unknown-project"
    
    def fetch_recent_logs(
        self,
        minutes_back: int = 30,
        filter_expression: Optional[str] = None,
        max_entries: int = 1000
    ) -> List[LogEntry]:
        """
        Fetch recent logs from GCP using gcloud CLI.
        
        Args:
            minutes_back: How many minutes back to fetch logs
            filter_expression: Additional GCP logging filter
            max_entries: Maximum number of log entries to return
            
        Returns:
            List of structured log entries
        """
        start_time = datetime.utcnow() - timedelta(minutes=minutes_back)
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Build gcloud logging command
        cmd = [
            "gcloud", "logging", "read",
            f"timestamp >= \"{start_time_str}\"",
            "--format=json",
            f"--limit={max_entries}",
            f"--project={self.project_id}"
        ]
        
        # Add additional filters
        if filter_expression:
            cmd[2] = f"({cmd[2]}) AND ({filter_expression})"
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            raw_logs = json.loads(result.stdout) if result.stdout.strip() else []
            
            # Convert to structured LogEntry objects
            log_entries = []
            for raw_log in raw_logs:
                try:
                    entry = LogEntry(
                        timestamp=raw_log.get("timestamp", ""),
                        severity=raw_log.get("severity", "INFO"),
                        message=raw_log.get("textPayload") or str(raw_log.get("jsonPayload", "")),
                        labels=raw_log.get("labels", {}),
                        source_location=raw_log.get("sourceLocation"),
                        trace=raw_log.get("trace"),
                        span_id=raw_log.get("spanId")
                    )
                    log_entries.append(entry)
                except Exception as e:
                    logger.warning(f"Failed to parse log entry: {e}")
                    
            return log_entries
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to fetch GCP logs: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GCP log JSON: {e}")
            return []
    
    def analyze_daemon_startup(self, minutes_back: int = 10) -> LogAnalysis:
        """
        Analyze daemon startup logs for configuration and initialization issues.
        
        Args:
            minutes_back: How far back to look for startup logs
            
        Returns:
            Analysis of startup process
        """
        # Fetch logs with startup-related filters
        filter_expr = 'resource.type="cloud_run_revision" OR resource.type="gce_instance"'
        logs = self.fetch_recent_logs(minutes_back, filter_expr)
        
        analysis = self._analyze_logs(logs)
        
        # Look for specific startup issues
        startup_keywords = [
            "Enhanced RAG agent error",
            "Subscripted generics",
            "validation error",
            "ImportError",
            "ModuleNotFoundError",
            "Configuration error"
        ]
        
        for entry in logs:
            for keyword in startup_keywords:
                if keyword.lower() in entry.message.lower():
                    analysis.startup_issues.append(f"[{entry.timestamp}] {entry.message}")
        
        return analysis
    
    def analyze_flow_execution(
        self, 
        flow_name: str, 
        session_id: Optional[str] = None,
        minutes_back: int = 60
    ) -> LogAnalysis:
        """
        Analyze logs for a specific flow execution.
        
        Args:
            flow_name: Name of the flow to analyze
            session_id: Specific session ID to track (optional)
            minutes_back: How far back to look for flow logs
            
        Returns:
            Analysis of flow execution
        """
        # Build filter for flow-specific logs
        filter_parts = [f'textPayload:"{flow_name}"']
        if session_id:
            filter_parts.append(f'textPayload:"{session_id}"')
        
        filter_expr = " AND ".join(filter_parts)
        logs = self.fetch_recent_logs(minutes_back, filter_expr)
        
        analysis = self._analyze_logs(logs)
        
        # Analyze flow-specific patterns
        agent_keywords = ["agent", "researcher", "policy_analyst", "fact_checker", "explorer"]
        timing_keywords = ["timeout", "duration", "elapsed", "latency"]
        
        for entry in logs:
            # Track agent-specific errors
            for keyword in agent_keywords:
                if keyword.lower() in entry.message.lower() and entry.severity in ["ERROR", "WARNING"]:
                    analysis.agent_errors.append(f"[{entry.timestamp}] {entry.message}")
            
            # Track performance issues
            for keyword in timing_keywords:
                if keyword.lower() in entry.message.lower():
                    if "performance" not in analysis.flow_performance:
                        analysis.flow_performance["performance"] = []
                    analysis.flow_performance["performance"].append({
                        "timestamp": entry.timestamp,
                        "message": entry.message
                    })
        
        return analysis
    
    def analyze_agent_errors(self, agent_name: str, minutes_back: int = 30) -> LogAnalysis:
        """
        Analyze logs for specific agent errors and issues.
        
        Args:
            agent_name: Name of the agent to analyze
            minutes_back: How far back to look for agent logs
            
        Returns:
            Analysis of agent-specific issues
        """
        filter_expr = f'textPayload:"{agent_name}"'
        logs = self.fetch_recent_logs(minutes_back, filter_expr)
        
        analysis = self._analyze_logs(logs)
        
        # Look for agent-specific error patterns
        error_patterns = [
            "type.*error",
            "validation.*error", 
            "import.*error",
            "attribute.*error",
            "key.*error",
            "value.*error"
        ]
        
        for entry in logs:
            if entry.severity in ["ERROR", "WARNING"]:
                for pattern in error_patterns:
                    if any(p in entry.message.lower() for p in pattern.split(".*")):
                        analysis.agent_errors.append(f"[{entry.timestamp}] {entry.message}")
        
        return analysis
    
    def stream_logs_realtime(
        self,
        filter_expression: Optional[str] = None,
        callback: Optional[callable] = None
    ):
        """
        Stream logs in real-time for live debugging.
        
        Args:
            filter_expression: GCP logging filter to apply
            callback: Function to call for each log entry
        """
        cmd = [
            "gcloud", "logging", "tail",
            "--format=json",
            f"--project={self.project_id}"
        ]
        
        if filter_expression:
            cmd.append(filter_expression)
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            print(f"🔍 Streaming GCP logs for project: {self.project_id}")
            if filter_expression:
                print(f"   Filter: {filter_expression}")
            print("   Press Ctrl+C to stop...")
            print("-" * 60)
            
            for line in process.stdout:
                if line.strip():
                    try:
                        log_data = json.loads(line)
                        entry = LogEntry(
                            timestamp=log_data.get("timestamp", ""),
                            severity=log_data.get("severity", "INFO"),
                            message=log_data.get("textPayload") or str(log_data.get("jsonPayload", "")),
                            labels=log_data.get("labels", {})
                        )
                        
                        # Default display
                        severity_emoji = {
                            "ERROR": "❌",
                            "WARNING": "⚠️", 
                            "INFO": "ℹ️",
                            "DEBUG": "🐛"
                        }.get(entry.severity, "📝")
                        
                        timestamp = entry.timestamp[:19].replace("T", " ")
                        print(f"{severity_emoji} [{timestamp}] {entry.message}")
                        
                        # Call custom callback if provided
                        if callback:
                            callback(entry)
                            
                    except json.JSONDecodeError:
                        continue
                        
        except KeyboardInterrupt:
            print("\n⏹️  Log streaming stopped")
            process.terminate()
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stream logs: {e}")
    
    def _analyze_logs(self, logs: List[LogEntry]) -> LogAnalysis:
        """Perform basic analysis on a list of log entries."""
        if not logs:
            return LogAnalysis(
                total_entries=0,
                error_count=0,
                warning_count=0,
                time_range=("", ""),
                key_errors=[],
                agent_errors=[],
                flow_performance={},
                startup_issues=[]
            )
        
        error_count = sum(1 for log in logs if log.severity == "ERROR")
        warning_count = sum(1 for log in logs if log.severity == "WARNING")
        
        # Extract key errors
        key_errors = []
        for log in logs:
            if log.severity == "ERROR":
                key_errors.append(f"[{log.timestamp}] {log.message}")
        
        # Time range
        timestamps = [log.timestamp for log in logs if log.timestamp]
        time_range = (min(timestamps), max(timestamps)) if timestamps else ("", "")
        
        return LogAnalysis(
            total_entries=len(logs),
            error_count=error_count,
            warning_count=warning_count,
            time_range=time_range,
            key_errors=key_errors[:10],  # Limit to top 10
            agent_errors=[],
            flow_performance={},
            startup_issues=[]
        )


def analyze_current_issue() -> None:
    """Quick analysis of the current Enhanced RAG agent issue."""
    analyzer = GCPLogAnalyzer()
    
    print("🔍 ANALYZING ENHANCED RAG AGENT ISSUE")
    print("=" * 50)
    
    # Look for recent startup issues
    startup_analysis = analyzer.analyze_daemon_startup(minutes_back=30)
    
    print(f"📊 Startup Analysis (last 30 minutes):")
    print(f"   Total entries: {startup_analysis.total_entries}")
    print(f"   Errors: {startup_analysis.error_count}")
    print(f"   Warnings: {startup_analysis.warning_count}")
    
    if startup_analysis.startup_issues:
        print(f"\n🚨 Startup Issues Found:")
        for issue in startup_analysis.startup_issues[:5]:
            print(f"   {issue}")
    
    if startup_analysis.key_errors:
        print(f"\n❌ Key Errors:")
        for error in startup_analysis.key_errors[:3]:
            print(f"   {error}")
    
    # Look for Enhanced RAG specific errors
    print(f"\n🎯 Enhanced RAG Agent Analysis:")
    rag_analysis = analyzer.analyze_agent_errors("enhanced", minutes_back=60)
    
    if rag_analysis.agent_errors:
        print(f"   Found {len(rag_analysis.agent_errors)} agent-related errors")
        for error in rag_analysis.agent_errors[:3]:
            print(f"   {error}")
    else:
        print("   No agent-specific errors found in recent logs")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_current_issue()
    else:
        # Default: stream logs real-time
        analyzer = GCPLogAnalyzer()
        analyzer.stream_logs_realtime()