import json
import uuid
from typing import Union

import pandas as pd
from shiny import App, Session, reactive, render, ui

from buttermilk._core.agent import AgentInput, AgentOutput, ManagerMessage, ManagerRequest, StepRequest, ToolOutput
from buttermilk._core.contract import (
    ConductorRequest,
    ConductorResponse,
    ErrorEvent,
    HeartBeat,
    ManagerResponse,
    ProceedToNextTaskSignal,
    RunRequest,
    TaskProcessingComplete,
    TaskProcessingStarted,
    TaskProgressUpdate,
)
from buttermilk._core.types import Record
from buttermilk.bm import logger
from buttermilk.runner.flowrunner import FlowRunner

# Assuming prepare_step_df is potentially async
from buttermilk.runner.helpers import prepare_step_df
from buttermilk.web.messages import _format_message_for_client

ALL_MESSAGES = Union[
    ManagerMessage,
    ManagerRequest,
    ManagerResponse,
    TaskProcessingComplete,
    TaskProcessingStarted,
    ConductorResponse,
    ConductorRequest,
    ErrorEvent, StepRequest, ProceedToNextTaskSignal, HeartBeat,
    AgentOutput,
    ToolOutput,
    AgentInput,
    Record]


def get_shiny_app(flows: FlowRunner):

    flow_choices = list(flows.flows.keys())
    # Data loading moved to server function

    app_ui = ui.page_sidebar(
        ui.sidebar(
            ui.input_select(
                "flow",
                "Choose flow",
                choices=flow_choices,
                selected=flow_choices[0] if flow_choices else None,
            ),
            ui.input_select(
                "criteria",
                "Choose criteria",
                choices=[],
                selected=None,

            ),
            ui.input_select(
                "record_id",
                "Record ID",
                choices=[],  # Start empty, populated by server effect
                selected=None,
            ),
            ui.input_action_button("go", "Rate"),
            ui.output_ui("confirm_button_ui"),
            ui.hr(),
            ui.input_action_button("load_history", "Load Run History", class_="btn btn-info"),
        ),
        ui.layout_columns(
            ui.column(
                8,
                ui.card(
                    ui.card_header("autogen chat"),
                    ui.chat_ui("chat", width="min(1400px, 100%)"),
                    style="width:min(1400px, 100%)",
                ),
            height="800px",
            fill=True,
            ),
            ui.column(
                4,
                ui.card(
                    ui.card_header("Judge & Synth Run History"),
                    ui.output_data_frame("run_history_table"),
                ),
                ui.card(
                    ui.card_header("Workflow Progress"),
                    ui.output_ui("progress_tracker_ui"),
                ),
            ),
            fill=False,
            height="800px",
        ),
    )

    def server(input, output, session: Session):
        chat = ui.Chat(id="chat")
        flow_runner = flows
        # Cache for loaded dataframes within the server function scope
        loaded_flow_data = {}
        current_session_id = reactive.Value(None)
        callback_to_chat = reactive.Value(None)
        confirm_button_state = reactive.Value("neutral")
        # New reactive value to store run history from BigQuery
        run_history_df = reactive.Value(pd.DataFrame())

        # Progress tracking
        current_progress = reactive.Value({})  # type: reactive.Value[dict[str, object]]

        # This effect loads data when the flow selection changes and updates the record_id dropdown and criteria dropdown
        @reactive.Effect
        @reactive.event(input.flow)  # Trigger when flow selection changes
        async def _load_and_update_record_ids():
            selected_flow = input.flow()
            record_ids = []
            criteria = []
            df = None
            try:
                criteria = flows.flows[selected_flow].parameters["criteria"]
                ui.update_select(
                    "criteria",
                    choices=criteria,
                    selected=criteria[0] if criteria else None,
                )
            except:
                pass
            if selected_flow:
                if selected_flow not in loaded_flow_data:
                    try:
                        if flow_obj := flows.flows.get(selected_flow):
                            print(f"Loading data for flow: {selected_flow}")  # Debugging
                            # Await the async data loading method
                            df = await prepare_step_df(flow_obj.data)
                            df = list(df.values())[-1]
                            loaded_flow_data[selected_flow] = df
                        else:
                             logger.warning(f"Warning: Could not find flow object for '{selected_flow}'")
                             # Store empty df to prevent repeated load attempts for invalid flows
                             loaded_flow_data[selected_flow] = pd.DataFrame({"record_id": []})
                             df = loaded_flow_data[selected_flow]

                    except Exception as e:
                        print(f"Error loading data for flow '{selected_flow}': {e}")
                        await chat.append_message(f"*Error loading data for flow {selected_flow}. See logs.*\n")
                        # Store empty df on error to prevent repeated load attempts
                        loaded_flow_data[selected_flow] = pd.DataFrame({"record_id": []})
                        df = loaded_flow_data[selected_flow]
                else:
                    # Data already loaded, retrieve from cache
                    df = loaded_flow_data[selected_flow]
                    print(f"Using cached data for flow: {selected_flow}")  # Debugging

                # Extract record_ids from the dataframe (either newly loaded or cached)
                record_ids = df.index.values.tolist()

            # Update the select input's choices and selection
            ui.update_select(
                "record_id",
                choices=record_ids,
                selected=record_ids[0] if record_ids else None,
            )

        # The @reactive.event(input.flow) on _load_and_update_record_ids handles
        # the initial load automatically when input.flow() gets its initial value.

        @reactive.effect
        @reactive.event(input.go)
        async def run_flow():
            session_id = str(uuid.uuid4())
            current_session_id.set(session_id)
            confirm_button_state.set("neutral")  # Reset confirm button on new run

            selected_record_id = input.record_id()
            selected_flow = input.flow()

            # Ensure flow and record_id are selected before proceeding
            if not selected_flow:
                 await chat.append_message("*Please select a Flow.*\n")
                 return
            if not selected_record_id:
                 await chat.append_message("*Please select a Record ID.*\n")
                 return

            run_request = RunRequest(flow=selected_flow, record_id=selected_record_id, parameters=dict(criteria=input.criteria()), client_callback=callback_to_ui, session_id=session_id)

            # Disable button while running
            ui.update_action_button("go", label="Running...", disabled=True)

            try:
                # Assuming run_flow should use the selected flow name, not hardcoded 'batch'
                # If 'batch' is correct, keep it. Otherwise, use selected_flow.
                # Using selected_flow based on context:
                callback_to_chat.set(await flow_runner.run_flow(flow_name=selected_flow, run_request=run_request))
                await chat.append_message(f"*Flow '{selected_flow}' started...*\n")
            except Exception as e:
                 await chat.append_message(f"*Error starting flow '{selected_flow}': {e}*\n")
                 # Re-enable button on error
                 ui.update_action_button("go", label="Rate", disabled=False)

        @chat.on_user_submit
        async def handle_user_input(user_input: str):
            cb = callback_to_chat.get()
            if cb:
                await cb(user_input)

        async def callback_to_ui(message):
            # Log all received messages for debugging
            logger.debug(f"Received message in callback_to_ui: Type={type(message)}, Content={message}")

            content = _format_message_for_client(message)
            if content:
                # Raw HTML content should work with default behavior
                await chat.append_message(content)

            if isinstance(message, ManagerRequest):
                confirm_button_state.set("ready")
            # Re-enable the 'Go' button when the flow completes or needs confirmation or errors out
            # Handle task progress updates without sending to chat
            if isinstance(message, TaskProgressUpdate):
                # Update the progress tracking
                # Exclude timestamp just in case it causes issues with reactivity/rendering
                progress_data = {
                    "role": message.role,
                    "step_name": message.step_name,
                    "status": message.status,
                    "message": message.message,
                    "total_steps": message.total_steps,
                    "current_step": message.current_step,
                    # "timestamp": message.timestamp # Excluded
                }

                # Log the specific progress data being set
                logger.info(f"Updating progress tracker with data: {progress_data}")

                # Update progress tracker
                current_progress.set(progress_data)

                # Don't continue processing this message for the chat UI
                return

            if isinstance(message, (TaskProcessingComplete, ManagerRequest, ErrorEvent)):
                # Log when the flow signals completion or requires interaction
                logger.debug(f"Received flow end/interaction signal: {type(message)}")
                ui.update_action_button("go", label="Rate", disabled=False)

        @reactive.effect
        @reactive.event(input.confirm)
        async def handle_confirm():
            message = ManagerResponse(confirm=True)
            cb = callback_to_chat.get()
            if cb:
                await cb(message)
                confirm_button_state.set("confirmed")

        @reactive.effect
        @reactive.event(input.load_history)
        async def load_run_history():
            # Get the currently selected values from inputs
            selected_flow = input.flow()
            selected_criteria = input.criteria()
            selected_record_id = input.record_id()

            # Ensure required inputs are selected
            if not all([selected_flow, selected_criteria, selected_record_id]):
                await chat.append_message("*Please select a Flow, Criteria, and Record ID before loading history.*\n")
                return

            try:
                # Format of the SQL query to get judge and synth runs
                # Get the dataset from the config in local.yaml
                sql = """
                SELECT
                    *
                FROM
                    `prosocial-443205.testing.flow_score_results`
                """
                    # --`{flows.save.dataset}`

                # Execute the query using bm.run_query
                results_df = flows.bm.run_query(sql)

                # If we got results, update the reactive value
                if isinstance(results_df, pd.DataFrame) and not results_df.empty:
                    run_history_df.set(results_df)
                    await chat.append_message("*Run history loaded successfully.*\n")
                else:
                    await chat.append_message(f"*No judge or synth runs found for Flow: {selected_flow}, Criteria: {selected_criteria}, Record ID: {selected_record_id}*\n")
            except Exception as e:
                await chat.append_message(f"*Error loading run history: {e}*\n")
                logger.error(f"Error loading run history: {e}")

        @output
        @render.ui
        def confirm_button_ui():
            state = confirm_button_state.get()
            if state == "ready":
                return ui.input_action_button("confirm", "Confirm", class_="btn btn-success", disabled=False)
            if state == "confirmed":
                return ui.input_action_button("confirm", "Confirmed", class_="btn btn-success", disabled=False)
            # neutral state
            return ui.input_action_button("confirm", "Confirm", class_="btn btn-secondary", disabled=False)

        @output
        @render.data_frame
        def run_history_table():
            # Get the current dataframe from the reactive value
            df = run_history_df.get()
            if df.empty:
                return {}

            # Format the dataframe for display
            # - Convert timestamps to readable format
            # - Truncate long text fields
            display_df = df.copy()
            if "created_at" in display_df.columns:
                display_df["created_at"] = pd.to_datetime(display_df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")

            # For JSON columns that might contain complex nested structures, show a summary
            for col in ["parameters", "result"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(
                        lambda x: (json.dumps(x)[:100] + "...") if isinstance(x, (dict, list)) and len(json.dumps(x)) > 100 else x,
                    )

            return display_df

        @output
        @render.ui
        def progress_tracker_ui():
            """Render a simple progress bar indicating overall workflow progress."""
            progress_data = current_progress.get()
            # Log when the UI render function is called and what data it sees
            logger.debug(f"Rendering progress_tracker_ui with data: {progress_data}")

            if not progress_data or "total_steps" not in progress_data or "current_step" not in progress_data:
                # Default state or before the first progress update
                return ui.div(
                    ui.tags.h5("Workflow Progress"),
                    ui.div(
                        ui.tags.div(
                            class_="progress-bar",
                            role="progressbar",
                            style="width: 0%;",
                            aria_valuenow="0",
                            aria_valuemin="0",
                            aria_valuemax="100",
                        ),
                        class_="progress mb-2",
                        style="height: 20px;",  # Make the bar a bit thicker
                    ),
                    ui.div("Waiting for workflow to start..."),
                )

            current_step = int(progress_data.get("current_step", 1))
            status = str(progress_data.get("status", "running"))

            # Ensure total_steps is at least 1
            total_steps = 101
            # Ensure current_step doesn't exceed total_steps for calculation
            current_step = min(current_step, total_steps)

            # Ensure integer division works correctly
            progress_percent = int((float(current_step) / float(total_steps)) * 100) if total_steps > 0 else 1

            # Determine progress bar color based on status
            bar_class = "progress-bar"
            if status == "completed":
                bar_class += " bg-success"
                progress_percent = 99  # Ensure completed shows 100%
            elif status == "error":
                bar_class += " bg-danger"
            elif status == "started" or status == "running":
                 bar_class += " progress-bar-striped progress-bar-animated"  # Animate while running

            return ui.div(
                ui.tags.h5("Workflow Progress"),
                ui.div(
                    ui.tags.div(
                        class_=bar_class,
                        role="progressbar",
                        style=f"width: {int(progress_percent)}%;",
                    ),
                    class_="progress mb-2",
                    style="height: 20px;",  # Make the bar a bit thicker
                ),
            )

    app = App(app_ui, server)
    return app
