import uuid

import pandas as pd
import streamlit as st

from buttermilk._core.agent import (
    ManagerRequest,
)
from buttermilk._core.contract import (
    ErrorEvent,
    ManagerResponse,
    RunRequest,
    TaskProcessingComplete,
    TaskProgressUpdate,
)
from buttermilk.bm import logger
from buttermilk.runner.helpers import prepare_step_df
from buttermilk.web.messages import _format_message_for_client


class StreamlitDashboardApp:
    """Qualitative Data Science Dashboard with LLM Chatbots - Streamlit version"""

    def __init__(self, flows):
        """Initialize the dashboard application
        
        Args:
            flows: FlowRunner or MockFlowRunner instance with workflow configurations

        """
        self.flows = flows
        self.session_callbacks = {}

    def _initialize_session_state(self):
        """Initialize the session state variables if they don't exist"""
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        # Initialize flow state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "progress" not in st.session_state:
            st.session_state.progress = {"current_step": 0, "total_steps": 100, "status": "waiting"}
        if "requires_confirmation" not in st.session_state:
            st.session_state.requires_confirmation = False
        if "flow_running" not in st.session_state:
            st.session_state.flow_running = False
        if "error" not in st.session_state:
            st.session_state.error = None

    def _get_flow_choices(self):
        """Get available flow choices from the flow runner"""
        return list(self.flows.flows.keys())

    def _get_criteria_options(self, flow_name):
        """Get criteria options for a specific flow"""
        criteria = []
        if flow_name:
            try:
                criteria = self.flows.flows[flow_name].parameters.get("criteria", [])
            except:
                pass
        return criteria

    def _get_record_ids(self, flow_name):
        """Get record IDs for a specific flow"""
        record_ids = []

        try:
            if flow_name in self.flows.flows:
                flow_obj = self.flows.flows.get(flow_name)
                if flow_obj:
                    # Try to use get_record_ids method if it exists
                    if hasattr(flow_obj, "get_record_ids") and callable(flow_obj.get_record_ids):
                        df = flow_obj.get_record_ids()
                        if hasattr(df, "index"):
                            record_ids = df.index.tolist()
                        else:
                            logger.error("get_record_ids did not return an object with an index attribute")
                    # Otherwise try to use the data directly
                    elif hasattr(flow_obj, "data") and flow_obj.data:
                        # Check if we have mock data first (simpler approach)
                        if "mock_data" in flow_obj.data and "record_ids" in flow_obj.data["mock_data"]:
                            record_ids = flow_obj.data["mock_data"]["record_ids"]
                        # Otherwise, try prepare_step_df as non-async
                        else:
                            try:
                                df_dict = prepare_step_df(flow_obj.data)
                                if df_dict and isinstance(df_dict, dict) and len(df_dict) > 0:
                                    df = list(df_dict.values())[-1]
                                    record_ids = df.index.values.tolist()
                            except Exception as e:
                                logger.error(f"Error preparing data: {e}")
        except Exception as e:
            logger.error(f"Error loading data for flow '{flow_name}': {e}")

        return record_ids

    def _get_run_history(self, flow_name, criteria, record_id):
        """Get run history for a specific flow, criteria, and record"""
        try:
            # Format of the SQL query to get judge and synth runs
            sql = """
            SELECT
                *
            FROM
                `prosocial-443205.testing.flow_score_results`
            """
            # Execute the query
            results_df = self.flows.bm.run_query(sql)

            # If the results are empty
            if not isinstance(results_df, pd.DataFrame) or results_df.empty:
                return []

            return results_df.to_dict("records")
        except Exception as e:
            logger.error(f"Error loading run history: {e}")
            return []

    def callback_to_ui(self, session_id):
        """Create a callback function for the flow to send messages back to the UI
        
        Args:
            session_id: The session ID
            
        Returns:
            Callable: The callback function

        """
        async def callback(message):
            """Handle messages from the flow
            
            Args:
                message: The message from the flow

            """
            # Format the message for the client
            content = _format_message_for_client(message)

            # If we have a formatted message, add it to the session state
            if content:
                st.session_state.messages.append({
                    "content": content,
                    "type": type(message).__name__,
                })

            # Handle progress updates
            if isinstance(message, TaskProgressUpdate):
                progress_data = {
                    "role": message.role,
                    "step_name": message.step_name,
                    "status": message.status,
                    "message": message.message,
                    "total_steps": message.total_steps,
                    "current_step": message.current_step,
                }

                # Update session state
                st.session_state.progress = progress_data

            # Handle completion and interaction states
            if isinstance(message, TaskProcessingComplete):
                st.session_state.flow_running = False
                st.session_state.progress["status"] = "completed"

            elif isinstance(message, ErrorEvent):
                st.session_state.flow_running = False
                st.session_state.progress["status"] = "error"
                st.session_state.error = str(message)

            elif isinstance(message, ManagerRequest):
                st.session_state.requires_confirmation = True

        return callback

    async def start_flow(self, flow_name, record_id, criteria=None):
        """Start a flow run
        
        Args:
            flow_name: The name of the flow to run
            record_id: The record ID to use
            criteria: The criteria to use (optional)

        """
        if not flow_name or not record_id:
            st.error("Missing required fields: flow and record_id")
            return None

        # Create run request
        run_request = RunRequest(
            flow=flow_name,
            record_id=record_id,
            parameters=dict(criteria=criteria) if criteria else {},
            client_callback=self.callback_to_ui(st.session_state.session_id),
            session_id=st.session_state.session_id,
        )

        # Start the flow
        try:
            # Reset session state
            st.session_state.messages = []
            st.session_state.progress = {"current_step": 0, "total_steps": 100, "status": "started"}
            st.session_state.requires_confirmation = False
            st.session_state.flow_running = True
            st.session_state.error = None

            # Store the callback
            callback = await self.flows.run_flow(
                flow_name=flow_name,
                run_request=run_request,
            )

            # Store the callback in a dictionary keyed by session ID
            self.session_callbacks[st.session_state.session_id] = callback

            return True
        except Exception as e:
            logger.error(f"Error starting flow: {e}")
            st.error(f"Error starting flow: {e!s}")
            st.session_state.flow_running = False
            return False

    async def send_user_input(self, user_input):
        """Send user input to the flow
        
        Args:
            user_input: The user input to send

        """
        if not user_input.strip():
            return None

        callback = self.session_callbacks.get(st.session_state.session_id)

        if callback:
            await callback(user_input)
            return True
        st.error("No active flow to send message to")
        return False

    async def confirm_flow(self):
        """Send confirmation to the flow"""
        callback = self.session_callbacks.get(st.session_state.session_id)

        if callback:
            message = ManagerResponse(confirm=True)
            await callback(message)

            # Update session state
            st.session_state.requires_confirmation = False
            return True
        st.error("No active flow to confirm")
        return False

    def run(self):
        """Run the Streamlit application"""
        # Initialize session state
        self._initialize_session_state()

        # Set up the page
        st.set_page_config(
            page_title="Buttermilk Dashboard",
            page_icon="🥛",
            layout="wide",
        )

        # Title
        st.title("Qualitative Data Science Dashboard")

        # Create sidebar and main content in a 2-column layout
        col1, col2 = st.columns([1, 3])

        # Sidebar controls
        with col1:
            st.header("Configuration")

            # Flow selection
            flow_choices = self._get_flow_choices()
            selected_flow = st.selectbox(
                "Choose Flow",
                [""] + flow_choices,
                index=0,
                key="flow_selectbox",
            )

            # Criteria selection (only available if flow is selected)
            criteria_options = self._get_criteria_options(selected_flow) if selected_flow else []
            selected_criteria = st.selectbox(
                "Choose Criteria",
                [""] + criteria_options,
                index=0,
                disabled=not selected_flow,
                key="criteria_selectbox",
            )

            # Record ID selection (only available if flow is selected)
            record_ids = self._get_record_ids(selected_flow) if selected_flow else []
            selected_record_id = st.selectbox(
                "Record ID",
                [""] + record_ids,
                index=0,
                disabled=not selected_flow,
                key="record_id_selectbox",
            )

            # Start flow button
            start_disabled = not selected_flow or not selected_record_id or st.session_state.flow_running
            if st.button("Start Flow", disabled=start_disabled, key="start_flow_button"):
                # Use asyncio to run the async function
                import asyncio
                asyncio.run(self.start_flow(selected_flow, selected_record_id, selected_criteria))
                st.rerun()

            # Confirmation button (only shown when confirmation is required)
            if st.session_state.requires_confirmation:
                if st.button("Confirm", key="confirm_button"):
                    # Use asyncio to run the async function
                    import asyncio
                    asyncio.run(self.confirm_flow())
                    st.rerun()

            # Load history button
            history_disabled = not selected_flow or not selected_criteria or not selected_record_id
            if st.button("Load Run History", disabled=history_disabled, key="load_history_button"):
                st.session_state.history = self._get_run_history(selected_flow, selected_criteria, selected_record_id)
                st.rerun()

        # Main content area
        with col2:
            # Chat interface
            st.header("Agent Chat")

            # Display messages
            chat_container = st.container()
            with chat_container:
                if not st.session_state.messages:
                    st.info("No messages yet. Start a flow to begin the conversation.")
                else:
                    for msg in st.session_state.messages:
                        # The content is already HTML formatted by _format_message_for_client
                        # We'll need to display it as markdown to preserve formatting
                        st.markdown(msg["content"], unsafe_allow_html=True)

            # Progress tracker
            st.header("Workflow Progress")

            # Display progress bar
            progress_status = st.session_state.progress["status"]
            progress_color = "blue"
            if progress_status == "completed":
                progress_color = "green"
            elif progress_status == "error":
                progress_color = "red"

            progress_value = (st.session_state.progress["current_step"] /
                             st.session_state.progress["total_steps"])

            st.progress(progress_value)

            # Display progress details
            if progress_status == "waiting":
                st.text("Waiting to start")
            else:
                progress_text = f"Status: {progress_status}"
                if st.session_state.progress.get("step_name"):
                    progress_text += f" - {st.session_state.progress['step_name']}"
                if st.session_state.progress.get("role"):
                    progress_text += f" ({st.session_state.progress['role']})"
                st.text(progress_text)

            # User input
            st.text_input(
                "Type your message here...",
                key="user_input",
                disabled=not st.session_state.flow_running,
                on_change=self._handle_user_input,
            )

            # Run history
            st.header("Run History")

            if "history" in st.session_state and st.session_state.history:
                # Display history as a table
                st.dataframe(st.session_state.history)
            else:
                st.info("No history loaded. Use the 'Load Run History' button to view previous runs.")

    def _handle_user_input(self):
        """Handle user input submission"""
        user_input = st.session_state.user_input
        if user_input and st.session_state.flow_running:
            # Use asyncio to run the async function
            import asyncio
            asyncio.run(self.send_user_input(user_input))

            # Clear the input field
            st.session_state.user_input = ""

            # Force a rerun to update the UI
            st.rerun()


def create_dashboard_app(flows):
    """Create and configure the dashboard application
    
    Args:
        flows: FlowRunner or mock runner instance with workflow configurations
        
    Returns:
        StreamlitDashboardApp: The Streamlit dashboard application

    """
    dashboard = StreamlitDashboardApp(flows)
    return dashboard
