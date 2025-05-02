#!/usr/bin/env python
"""Development server script for running the Streamlit dashboard.
This is a standalone script that runs the dashboard as a separate application.
"""
import os
import sys
import pandas as pd
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import our dashboard app module
from buttermilk.web.streamlit_frontend.app import create_dashboard_app


class MockFlowRunner:
    """A mock FlowRunner class for development/testing purposes"""

    def __init__(self):
        """Initialize with some sample flows"""
        self.flows = {
            "sample_flow": MockFlow("Sample Flow"),
            "judge_flow": MockFlow("Judge Flow"),
            "analytics_flow": MockFlow("Analytics Flow"),
        }
        self.bm = MockBM()

    async def run_flow(self, flow_name=None, run_request=None):
        """Mock method to simulate running a flow"""
        import streamlit as st
        st.write(f"Mock running flow: {flow_name}")

        # Create a simple callback function that just logs messages
        async def mock_callback(message):
            """Mock callback for handling messages from the UI"""
            print(f"Received message: {message}")
            # We could add more sophisticated mock responses here if needed
            # But for now, we'll keep it simple

        return mock_callback


class MockFlow:
    """A mock Flow class for development/testing purposes"""

    def __init__(self, name):
        """Initialize with a name and sample parameters"""
        self.name = name
        self.parameters = {
            "criteria": ["accuracy", "clarity", "relevance"],
        }
        # Create mock data that can be accessed by prepare_step_df
        self.data = {
            "mock_data": {
                "record_ids": ["record1", "record2", "record3", "record4"],
                "index": ["record1", "record2", "record3", "record4"]
            }
        }
        
    def get_record_ids(self):
        """Mock method to get record IDs"""
        df = pd.DataFrame({
            "content": ["Sample content 1", "Sample content 2", "Sample content 3", "Sample content 4"]
        }, index=["record1", "record2", "record3", "record4"])
        return df


class MockBM:
    """A mock BM class for development/testing purposes"""

    def __init__(self):
        """Initialize with minimal requirements"""
        pass

    def run_query(self, sql):
        """Mock method to return sample data"""
        return pd.DataFrame({
            "flow_name": ["sample_flow", "judge_flow"],
            "record_id": ["record1", "record2"],
            "criteria": ["accuracy", "clarity"],
            "agent_name": ["judge", "scorer"],
            "score": [0.85, 0.72],
            "created_at": ["2025-04-29", "2025-04-28"],
        })


def main():
    """Create and run a standalone Streamlit application for development/testing"""
    # Create a mock flow runner
    flows = MockFlowRunner()

    # Create the dashboard app
    app = create_dashboard_app(flows)
    app.run()


if __name__ == "__main__":
    main()
