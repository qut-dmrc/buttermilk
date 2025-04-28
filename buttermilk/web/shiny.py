import asyncio
import os
from typing import Callable, Optional, Union
import uuid
import json

from shiny import App, reactive, render, ui, Session
import pandas as pd

from buttermilk._core.agent import AgentInput, AgentOutput, ManagerMessage, ManagerRequest, StepRequest, ToolOutput, logger
from buttermilk._core.contract import ConductorRequest, ConductorResponse, ErrorEvent, HeartBeat, ProceedToNextTaskSignal, RunRequest, ManagerResponse, TaskProcessingComplete, TaskProcessingStarted
from buttermilk._core.types import Record
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
                selected=flow_choices[0] if flow_choices else None
            ),
            ui.input_select(
                "criteria",
                "Choose criteria",
                choices=["cte", "tja"], # Keep this static for now, or update if needed
            ),
            ui.input_select(
                "record_id",
                "Record ID",
                choices=[], # Start empty, populated by server effect
                selected=None
            ),
            ui.input_action_button("go", "Rate"),
            ui.output_ui("confirm_button_ui"),
        ),
        ui.card(
            ui.card_header("autogen chat"),
            ui.chat_ui("chat", width="min(1400px, 100%)"),
            style="width:min(1400px, 100%)",
            class_="mx-auto",
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

        # This effect loads data when the flow selection changes and updates the record_id dropdown
        @reactive.Effect
        @reactive.event(input.flow) # Trigger when flow selection changes
        async def _load_and_update_record_ids():
            selected_flow = input.flow()
            record_ids = []
            df = None

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


    app = App(app_ui, server)
    return app
