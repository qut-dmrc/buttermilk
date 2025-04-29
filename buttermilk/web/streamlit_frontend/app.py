import asyncio
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
    """Qualitative Data Science Dashboard with LLM Chatbots - Streamlit version (Async)"""

    def __init__(self, flows):
        """Initialize the dashboard application
        
        Args:
            flows: FlowRunner or MockFlowRunner instance with workflow configurations

        """
        self.flows = flows
        # Use st.session_state to store callbacks per session
        if "session_callbacks" not in st.session_state:
            st.session_state.session_callbacks = {}

    def _initialize_session_state(self):
        """Initialize the session state variables if they don't exist"""
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
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
        if "history" not in st.session_state:
            st.session_state.history = None  # Initialize history

    def _get_flow_choices(self):
        """Get available flow choices from the flow runner"""
        return list(self.flows.flows.keys())

    def _get_criteria_options(self, flow_name):
        """Get criteria options for a specific flow"""
        criteria = []
        if flow_name:
            try:
                criteria = self.flows.flows[flow_name].parameters.get("criteria", [])
            except Exception as e:
                logger.warning(f"Could not get criteria for {flow_name}: {e}")
                # Keep criteria empty if error
        return criteria

    async def _get_record_ids(self, flow_name):
        """Get record IDs for a specific flow (async version)"""
        record_ids = []
        # Ensure flow_name is valid before proceeding
        if not flow_name or flow_name not in self.flows.flows:
            return record_ids

        try:
            flow_obj = self.flows.flows.get(flow_name)
            if flow_obj:
                # Try to use get_record_ids method if it exists and is async
                if hasattr(flow_obj, "get_record_ids") and callable(flow_obj.get_record_ids):
                    # Assuming get_record_ids is potentially async
                    df = await asyncio.to_thread(flow_obj.get_record_ids)  # Use to_thread if get_record_ids is sync but blocking
                    if hasattr(df, "index"):
                        record_ids = df.index.tolist()
                    else:
                        logger.error("get_record_ids did not return an object with an index attribute")
                # Otherwise try to use the data directly
                elif hasattr(flow_obj, "data") and flow_obj.data:
                    # Check if we have mock data first (simpler approach)
                    if "mock_data" in flow_obj.data and "record_ids" in flow_obj.data["mock_data"]:
                        record_ids = flow_obj.data["mock_data"]["record_ids"]
                    # Otherwise, try prepare_step_df potentially in a thread
                    else:
                        try:
                            # Run synchronous prepare_step_df in a thread to avoid blocking
                            df_dict = await asyncio.to_thread(prepare_step_df, flow_obj.data)
                            if df_dict and isinstance(df_dict, dict) and len(df_dict) > 0:
                                # Assuming the last df in the dict is the relevant one
                                df = list(df_dict.values())[-1]
                                if hasattr(df, "index"):
                                    record_ids = df.index.values.tolist()
                                else:
                                    logger.warning("Prepared DataFrame lacks an index.")
                        except Exception as e:
                            logger.error(f"Error preparing data asynchronously: {e}")
        except Exception as e:
            logger.error(f"Error loading data for flow '{flow_name}': {e}")

        return record_ids

    async def _get_run_history(self, flow_name, criteria, record_id):
        """Get run history for a specific flow, criteria, and record (async)"""
        # Ensure required parameters are present
        if not flow_name or not criteria or not record_id:
            logger.warning("Attempted to load history with missing parameters.")
            return []
        try:
            # Format of the SQL query to get judge and synth runs
            sql = f"""
            SELECT
                *
            FROM
                `prosocial-443205.testing.flow_score_results`
            WHERE flow = '{flow_name}' 
              AND criteria = '{criteria}' 
              AND record_id = '{record_id}' 
            ORDER BY timestamp DESC 
            LIMIT 100 
            """  # Added filtering and limit
            # Execute the query asynchronously using to_thread
            results_df = await asyncio.to_thread(self.flows.bm.run_query, sql)

            # If the results are empty or not a DataFrame
            if not isinstance(results_df, pd.DataFrame) or results_df.empty:
                logger.info(f"No run history found for {flow_name}/{criteria}/{record_id}")
                return []

            return results_df.to_dict("records")
        except Exception as e:
            logger.error(f"Error loading run history: {e}")
            st.error(f"Failed to load run history: {e}")  # Show error in UI
            return []

    def callback_to_ui(self, session_id):
        """Create a callback function for the flow to send messages back to the UI.
        This callback itself remains async, but triggers Streamlit updates.
        """
        async def callback(message):
            """Handle messages from the flow (async)"""
            logger.debug(f"Callback received message: {type(message).__name__} for session {session_id}")
            content = _format_message_for_client(message)
            needs_rerun = False

            if content:
                # Ensure messages list exists for the session
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                st.session_state.messages.append({
                    "content": content,
                    "type": type(message).__name__,
                })
                needs_rerun = True

            if isinstance(message, TaskProgressUpdate):
                st.session_state.progress = {
                    "role": message.role,
                    "step_name": message.step_name,
                    "status": message.status,
                    "message": message.message,
                    "total_steps": message.total_steps,
                    "current_step": message.current_step,
                }
                needs_rerun = True

            elif isinstance(message, TaskProcessingComplete):
                st.session_state.flow_running = False
                st.session_state.progress["status"] = "completed"
                needs_rerun = True

            elif isinstance(message, ErrorEvent):
                st.session_state.flow_running = False
                st.session_state.progress["status"] = "error"
                st.session_state.error = str(message)
                needs_rerun = True

            elif isinstance(message, ManagerRequest):
                st.session_state.requires_confirmation = True
                needs_rerun = True

            # If any state changed that requires UI update, trigger a rerun
            if needs_rerun:
                st.rerun()

        return callback

    async def start_flow(self, flow_name, record_id, criteria=None):
        """Start a flow run (async)"""
        if not flow_name or not record_id:
            st.error("Missing required fields: flow and record_id")
            return False  # Indicate failure

        session_id = st.session_state.session_id
        # Create run request with the session-specific callback
        run_request = RunRequest(
            flow=flow_name,
            record_id=record_id,
            parameters=dict(criteria=criteria) if criteria else {},
            client_callback=self.callback_to_ui(session_id),
            session_id=session_id,
        )

        try:
            # Reset session state for the new run
            st.session_state.messages = []
            st.session_state.progress = {"current_step": 0, "total_steps": 100, "status": "started"}
            st.session_state.requires_confirmation = False
            st.session_state.flow_running = True
            st.session_state.error = None
            st.session_state.history = None  # Clear previous history

            logger.info(f"Starting flow '{flow_name}' for record '{record_id}' with criteria '{criteria}'")
            # Start the flow asynchronously
            callback_from_flow = await self.flows.run_flow(
                flow_name=flow_name,
                run_request=run_request,
            )

            # Store the callback returned by run_flow (for sending user input)
            st.session_state.session_callbacks[session_id] = callback_from_flow
            logger.info(f"Flow '{flow_name}' started successfully for session {session_id}.")
            return True  # Indicate success
        except Exception as e:
            logger.error(f"Error starting flow '{flow_name}': {e}", exc_info=True)
            st.error(f"Error starting flow: {e!s}")
            st.session_state.flow_running = False
            st.session_state.error = f"Failed to start flow: {e!s}"
            return False  # Indicate failure

    async def send_user_input(self, user_input):
        """Send user input to the flow (async)"""
        if not user_input or not user_input.strip():
            logger.warning("Attempted to send empty user input.")
            return False

        session_id = st.session_state.session_id
        callback = st.session_state.session_callbacks.get(session_id)

        if callback and st.session_state.flow_running:
            try:
                logger.info(f"Sending user input to flow for session {session_id}")
                # Add user message to chat immediately for responsiveness
                st.session_state.messages.append({"content": f"**You:** {user_input}", "type": "UserInput"})
                # Send the input via the stored callback
                await callback(user_input)
                logger.info(f"User input sent successfully for session {session_id}")
                return True
            except Exception as e:
                logger.error(f"Error sending user input for session {session_id}: {e}", exc_info=True)
                st.error(f"Error sending message: {e!s}")
                return False
        else:
            logger.warning(f"No active flow or callback found for session {session_id} to send message.")
            st.error("No active flow to send message to, or flow is not running.")
            return False

    async def confirm_flow(self):
        """Send confirmation to the flow (async)"""
        session_id = st.session_state.session_id
        callback = st.session_state.session_callbacks.get(session_id)

        if callback and st.session_state.flow_running:
            try:
                logger.info(f"Sending confirmation to flow for session {session_id}")
                message = ManagerResponse(confirm=True)
                await callback(message)
                # Update session state immediately
                st.session_state.requires_confirmation = False
                logger.info(f"Confirmation sent successfully for session {session_id}")
                return True
            except Exception as e:
                logger.error(f"Error sending confirmation for session {session_id}: {e}", exc_info=True)
                st.error(f"Error sending confirmation: {e!s}")
                return False
        else:
            logger.warning(f"No active flow or callback found for session {session_id} to confirm.")
            st.error("No active flow to confirm, or flow is not running.")
            return False

    async def run(self):
        """Run the Streamlit application (async)"""
        # Initialize session state
        self._initialize_session_state()

        # Title
        st.title("Qualitative Data Science Dashboard")

        # Create sidebar and main content
        col1, col2 = st.columns([1, 3])

        # Sidebar controls
        with col1:
            st.header("Configuration")

            # Flow selection
            flow_choices = self._get_flow_choices()
            selected_flow = st.selectbox(
                "Choose Flow",
                [""] + flow_choices,
                index=flow_choices.index(st.session_state.get("selected_flow", "")) + 1 if st.session_state.get("selected_flow") in flow_choices else 0,
                key="selected_flow",
            )

            # Criteria selection
            criteria_options = self._get_criteria_options(selected_flow) if selected_flow else []
            selected_criteria = st.selectbox(
                "Choose Criteria",
                [""] + criteria_options,
                index=criteria_options.index(st.session_state.get("selected_criteria", "")) + 1 if st.session_state.get("selected_criteria") in criteria_options else 0,
                disabled=not selected_flow,
                key="selected_criteria",
            )

            # Record ID selection
            record_ids = await self._get_record_ids(selected_flow) if selected_flow else []
            selected_record_id = st.selectbox(
                "Record ID",
                [""] + record_ids,
                index=record_ids.index(st.session_state.get("selected_record_id", "")) + 1 if st.session_state.get("selected_record_id") in record_ids else 0,
                disabled=not selected_flow,
                key="selected_record_id",
            )

            # Start flow button
            start_button_pressed = st.button(
                "Start Flow",
                disabled=not selected_flow or not selected_record_id or st.session_state.flow_running,
                key="start_flow_button",
            )
            if start_button_pressed:
                success = await self.start_flow(selected_flow, selected_record_id, selected_criteria)
                if success:
                    st.rerun()

            # Confirmation button
            if st.session_state.requires_confirmation:
                confirm_button_pressed = st.button("Confirm", key="confirm_button")
                if confirm_button_pressed:
                    success = await self.confirm_flow()
                    if success:
                        st.rerun()

            # Load history button
            history_button_pressed = st.button(
                "Load Run History",
                disabled=not selected_flow or not selected_criteria or not selected_record_id,
                key="load_history_button",
            )
            if history_button_pressed:
                history_data = await self._get_run_history(selected_flow, selected_criteria, selected_record_id)
                st.session_state.history = history_data
                st.rerun()

        # Main content area
        with col2:
            # Chat interface
            st.header("Agent Chat")
            chat_container = st.container()
            with chat_container:
                if not st.session_state.messages:
                    st.info("No messages yet. Start a flow to begin the conversation.")
                else:
                    for msg in st.session_state.messages:
                        st.markdown(msg["content"], unsafe_allow_html=True)

            # Progress tracker
            st.header("Workflow Progress")
            progress_status = st.session_state.progress.get("status", "waiting")
            current_step = st.session_state.progress.get("current_step", 0)
            total_steps = st.session_state.progress.get("total_steps", 1)
            progress_value = min(1.0, current_step / total_steps if total_steps > 0 else 0)

            st.progress(progress_value)

            # Display progress details
            progress_text = f"Status: {progress_status}"
            if st.session_state.progress.get("step_name"):
                progress_text += f" - Step: {st.session_state.progress['step_name']}"
            if st.session_state.progress.get("role"):
                progress_text += f" ({st.session_state.progress['role']})"
            if st.session_state.progress.get("message"):
                progress_text += f" - {st.session_state.progress['message']}"
            st.text(progress_text)

            # Display error if any
            if st.session_state.error:
                st.error(f"Error: {st.session_state.error}")

            # User input text area
            user_input = st.text_area(
                "Your message:",
                key="user_input_area",
                disabled=not st.session_state.flow_running or st.session_state.requires_confirmation,
                height=100,
            )
            send_button_pressed = st.button(
                "Send Message",
                key="send_message_button",
                disabled=not st.session_state.flow_running or st.session_state.requires_confirmation or not user_input.strip(),
            )

            if send_button_pressed and user_input.strip():
                success = await self.send_user_input(user_input)
                if success:
                    st.rerun()

            # Run history display
            st.header("Run History")
            if st.session_state.history is not None:
                if st.session_state.history:
                    st.dataframe(st.session_state.history)
                else:
                    st.info("No run history found for the selected parameters.")
            else:
                st.info("Use the 'Load Run History' button to view previous runs.")


def create_dashboard_app(flows):
    """Create and configure the dashboard application"""
    dashboard = StreamlitDashboardApp(flows)
    return dashboard
