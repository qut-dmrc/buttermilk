#!/usr/bin/env python3
"""
Launch script for Buttermilk Streamlit dashboard.

Usage:
    uv run python buttermilk/frontend/streamlit/run_dashboard.py
    
    Or directly:
    uv run streamlit run buttermilk/frontend/streamlit/app.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit dashboard."""
    # Get the path to the app.py file
    app_path = Path(__file__).parent / "app.py"
    
    # Run streamlit with the app
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    
    print("🧈 Starting Buttermilk Chat Analysis Dashboard...")
    print(f"Command: {' '.join(cmd)}")
    print("Dashboard will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")

if __name__ == "__main__":
    main()