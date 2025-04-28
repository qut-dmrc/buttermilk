import asyncio
import os
from typing import Callable, Optional, Union
import uuid
import json

from shiny import App, reactive, render, ui, Session

from buttermilk._core.agent import AgentInput, AgentOutput, ManagerMessage, ManagerRequest, StepRequest, ToolOutput
from buttermilk._core.contract import ConductorRequest, ConductorResponse, ErrorEvent, HeartBeat, ProceedToNextTaskSignal, RunRequest, ManagerResponse, TaskProcessingComplete, TaskProcessingStarted
from buttermilk._core.types import Record
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
    ErrorEvent, StepRequest, ProceedToNextTaskSignal, HeartBeat,
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
        ui.output_ui("confirm_button_ui"),
    ),
    ui.card(
        ui.card_header("autogen chat"),
        ui.chat_ui("chat"),
        style="width:min(1400px, 100%)",
        class_="mx-auto",
    ),
)


def get_shiny_app(flows: FlowRunner):

    def server(input, output, session: Session):
        chat = ui.Chat(id="chat")
        flow_runner = flows
        current_session_id = reactive.Value(None)
        callback_to_chat = reactive.Value(None)
        confirm_button_state = reactive.Value("neutral")

        @reactive.effect
        @reactive.event(input.go)
        async def run_flow():
            session_id = str(uuid.uuid4())
            current_session_id.set(session_id)
            confirm_button_state.set("neutral")
            run_request = RunRequest(flow=input.flow(), record_id=input.record_id(), parameters=dict(criteria=input.criteria()), client_callback=callback_to_ui, session_id=session_id)

            callback_to_chat.set(await flow_runner.run_flow(flow_name="batch", run_request=run_request))

            await chat.append_message("*Flow started...*\n")

        @chat.on_user_submit
        async def handle_user_input(user_input: str):
            if callback_to_chat.get():
                await callback_to_chat.get()(user_input)

        async def callback_to_ui(message):
            content = _format_message_for_client(message)
            if content:
                await chat.append_message(str(content))

            if isinstance(message, ManagerRequest):
                confirm_button_state.set("ready")

        @reactive.effect
        @reactive.event(input.confirm)
        async def handle_confirm():
            message = ManagerResponse(confirm=True)
            if callback_to_chat.get():
                await callback_to_chat.get()(message)
                confirm_button_state.set("confirmed")

        @output
        @render.ui
        def confirm_button_ui():
            state = confirm_button_state.get()
            if state == "ready":
                return ui.input_action_button("confirm", "Confirm", class_="btn btn-success", disabled=False)
            elif state == "confirmed":
                return ui.input_action_button("confirm", "Confirmed", class_="btn btn-danger", disabled=False)
            else:
                return ui.input_action_button("confirm", "Confirm", class_="btn btn-secondary", disabled=False)

    app = App(app_ui, server)
    return app
