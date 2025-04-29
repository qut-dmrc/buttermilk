import asyncio
import os
from typing import Callable, Optional, Union
import uuid
import json

from shiny import App, reactive, render, ui, Session, req
import pandas as pd

from buttermilk._core.agent import AgentInput, AgentOutput, ManagerMessage, ManagerRequest, StepRequest, ToolOutput
from buttermilk._core.contract import ConductorRequest, ConductorResponse, ErrorEvent, HeartBeat, ProceedToNextTaskSignal, RunRequest, ManagerResponse, TaskProcessingComplete, TaskProcessingStarted
from buttermilk._core.types import Record
from buttermilk.runner.flowrunner import FlowRunner

# Assuming prepare_step_df is potentially async
from buttermilk.runner.helpers import prepare_step_df
from buttermilk.web.messages import _format_message_for_client
from buttermilk.bm import bm, logger

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
                selected=flow_choices[0] if flow_choices else None
            ),
            ui.input_select(
                "criteria",
                "Choose criteria",
                choices=[],
                selected=None

            ),
            ui.input_select(
                "record_id",
                "Record ID",
                choices=[], # Start empty, populated by server effect
                selected=None
            ),
            ui.input_action_button("go", "Rate"),
            ui.output_ui("confirm_button_ui"),
            ui.hr(),
            ui.input_action_button("load_history", "Load Run History", class_="btn btn-info")
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("autogen chat"),
                ui.chat_ui("chat", width="100%"),
                style="width:100%",
            ),
            ui.card(
                ui.card_header("Judge & Synth Run History"),
                ui.output_data_frame("run_history_table"),
                style="width:100%",
            ),
            col_widths=[6, 6],  # Equal width columns
            fill=True,
            height="800px"
        )
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

        # This effect loads data when the flow selection changes and updates the record_id dropdown and criteria dropdown
        @reactive.Effect
        @reactive.event(input.flow) # Trigger when flow selection changes
        async def _load_and_update_record_ids():
            selected_flow = input.flow()
            record_ids = []
            criteria = []
            df = None
            try:
                criteria = flows.flows[selected_flow].parameters['criteria']
                ui.update_select(
                    "criteria",
                    choices=criteria,
                    selected=criteria[0] if criteria else None
                )
            except:
                pass
            if selected_flow:
                if selected_flow not in loaded_flow_data:
                    try:
                        if flow_obj:= flows.flows.get(selected_flow):
                            print(f"Loading data for flow: {selected_flow}") # Debugging
                            # Await the async data loading method
                            df = await prepare_step_df(flow_obj.data)
                            df = list(df.values())[-1]
                            loaded_flow_data[selected_flow] = df 
                        else:
                             logger.warning(f"Warning: Could not find flow object for '{selected_flow}'")
                             # Store empty df to prevent repeated load attempts for invalid flows
                             loaded_flow_data[selected_flow] = pd.DataFrame({'record_id': []})
                             df = loaded_flow_data[selected_flow]

                    except Exception as e:
                        print(f"Error loading data for flow '{selected_flow}': {e}")
                        await chat.append_message(f"*Error loading data for flow {selected_flow}. See logs.*\n")
                        # Store empty df on error to prevent repeated load attempts
                        loaded_flow_data[selected_flow] = pd.DataFrame({'record_id': []})
                        df = loaded_flow_data[selected_flow]
                else:
                    # Data already loaded, retrieve from cache
                    df = loaded_flow_data[selected_flow]
                    print(f"Using cached data for flow: {selected_flow}") # Debugging

                # Extract record_ids from the dataframe (either newly loaded or cached)
                record_ids = df.index.values.tolist()

            # Update the select input's choices and selection
            ui.update_select(
                "record_id",
                choices=record_ids,
                selected=record_ids[0] if record_ids else None
            )

        # The @reactive.event(input.flow) on _load_and_update_record_ids handles
        # the initial load automatically when input.flow() gets its initial value.

        @reactive.effect
        @reactive.event(input.go)
        async def run_flow():
            session_id = str(uuid.uuid4())
            current_session_id.set(session_id)
            confirm_button_state.set("neutral") # Reset confirm button on new run

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
            content = _format_message_for_client(message)
            if content:
                await chat.append_message(str(content))

            if isinstance(message, ManagerRequest):
                confirm_button_state.set("ready")
            # Re-enable the 'Go' button when the flow completes or needs confirmation or errors out
            if isinstance(message, (TaskProcessingComplete, ManagerRequest, ErrorEvent)):
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
                sql = f"""
                SELECT
                    *
                FROM
                    `prosocial-443205.testing.flow_score_results`
                """
                    # --`{flows.save.dataset}`
                
                # Execute the query using bm.run_query
                results_df = bm.run_query(sql)
                
                # If we got results, update the reactive value
                if isinstance(results_df, pd.DataFrame) and not results_df.empty:
                    run_history_df.set(results_df)
                    await chat.append_message(f"*Run history loaded successfully.*\n")
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
            elif state == "confirmed":
                return ui.input_action_button("confirm", "Confirmed", class_="btn btn-success", disabled=False)
            else: # neutral state
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
            if 'created_at' in display_df.columns:
                display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # For JSON columns that might contain complex nested structures, show a summary
            for col in ['parameters', 'result']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(
                        lambda x: (json.dumps(x)[:100] + '...') if isinstance(x, (dict, list)) and len(json.dumps(x)) > 100 else x
                    )
            
            return display_df


    app = App(app_ui, server)
    return app
