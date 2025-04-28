import asyncio
import os
from typing import Callable, Optional, Union
import uuid
import json

from shiny import App, reactive, ui, Session

import asyncio
import os
from typing import Callable
import uuid

from shiny import App, reactive, ui

from buttermilk._core.agent import AgentInput, AgentOutput, ManagerMessage, ManagerRequest, StepRequest, ToolOutput
from buttermilk._core.contract import ConductorRequest, ConductorResponse, ErrorEvent, HeartBeat, ProceedToNextTaskSignal, RunRequest, ManagerResponse, TaskProcessingComplete, TaskProcessingStarted
from buttermilk._core.types import Record
from buttermilk.runner.flowrunner import FlowRunner
 
from buttermilk._core.contract import ManagerResponse
from buttermilk.runner.flowrunner import FlowRunner
from buttermilk.web.messages import _format_message_for_client

ALL_MESSAGES = Union[
    ManagerMessage,
    ManagerRequest,
    ManagerResponse,
    TaskProcessingComplete,
    TaskProcessingStarted,
    ConductorResponse,
    ConductorRequest,
    ErrorEvent, StepRequest,ProceedToNextTaskSignal, HeartBeat,
    AgentOutput,
    ToolOutput,
    AgentInput,
    Record]

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select(
            "flow",
            "Choose flow",
            choices=["trans"],
        ),
        ui.input_select(
            "criteria",
            "Choose criteria",
            choices=["cte", "tja"],
        ),
        ui.input_text(
            "record_id",
            "Record ID",
            value="jenner_criticises_khalif_dailymail"
        ),
        ui.input_action_button("go", "Rate"),
        # Initialize confirm button as disabled
        ui.input_action_button("confirm", "Confirm", disabled=True),
    ),
    ui.card(
        ui.card_header("autogen chat"),
        ui.chat_ui("chat"),
        style="width:min(1400px, 100%)",
        class_="mx-auto",
    ),
)


def get_shiny_app(flows: FlowRunner):
     
    def server(input, output, session: Session): # Add session parameter
        # stream = ui.MarkdownStream(id="chat")
        chat = ui.Chat(id="chat")
        flow_runner = flows
        current_session_id = reactive.Value(None)
        callback_to_chat = reactive.Value(None)
        # Reactive value to control confirm button state
        confirm_enabled = reactive.Value(False)

        @reactive.effect
        @reactive.event(input.go)
        async def run_flow():
            session_id = str(uuid.uuid4())
            current_session_id.set(session_id)
            # Reset confirm button state on new run
            confirm_enabled.set(False)
            run_request=RunRequest(flow=input.flow(), record_id=input.record_id(), parameters=dict(criteria=input.criteria()), client_callback=callback_to_ui, session_id=session_id)
            
            callback_to_chat.set(await flow_runner.run_flow(flow_name="batch", run_request=run_request))

            await chat.append_message("*Flow started...*\n")

        # Define a callback to run when the user submits a message
        @chat.on_user_submit
        async def handle_user_input(user_input: str):
            # Create a response message stream
                await callback_to_chat.get()(user_input)

        async def callback_to_ui(message):
            content = _format_message_for_client(message)
            if content:
                 await chat.append_message(str(content))
            
            # Update the reactive value based on message type
            if isinstance(message,ManagerRequest ):
                confirm_enabled.set(True)

        # Add reactive effect for the new confirm button
        @reactive.effect
        @reactive.event(input.confirm)
        async def handle_confirm():
            """Handles the confirm button click."""
            # Only proceed if the button should be enabled (redundant check, but safe)
            if confirm_enabled.get():
                message = ManagerResponse(confirm=True)
                if callback_to_chat.get():
                    await callback_to_chat.get()(message)
                    # Disable button after confirming
                    confirm_enabled.set(False)
                else:
                    # Handle case where callback is not yet set (should not happen if button is enabled)
                    await chat.append_message("*Error: Cannot confirm, flow not ready.*")


            # await stream.write(f"\n\n*Confirm button pressed. Message prepared: `{message}`*\n\n")

        # Effect to update the button's disabled state based on the reactive value
        @reactive.effect
        def _update_confirm_button_state():
            ui.update_action_button("confirm", disabled=not confirm_enabled.get())


    app = App(app_ui, server)
    return app
