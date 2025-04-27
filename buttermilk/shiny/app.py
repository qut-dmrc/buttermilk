
import asyncio
import os
from typing import Callable
import uuid

from shiny import App, reactive, ui

from buttermilk._core.contract import RunRequest, ManagerResponse
from buttermilk.runner.flowrunner import FlowRunner
 
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select(
            "criteria",
            "Choose criteria",
            choices=["cte", "tja"],
        ),
        ui.input_text(
            "record_id",
            "Record ID",value="jenner_criticises_khalif_dailymail"
        ),
        ui.input_action_button("go", "Rate"),
        ui.input_action_button("confirm", "Confirm"),
    ),
    ui.card(
        ui.card_header("Hello"),
        ui.chat_ui("chat"),
        style="width:min(680px, 100%)",
        class_="mx-auto",
    ),
)


def create_app(flows: FlowRunner):
     
    def server(input):
        # stream = ui.MarkdownStream(id="chat")
        chat = ui.Chat(id="chat")
        flow_runner = flows
        current_session_id = reactive.Value(None)
        callback = None

        @reactive.effect
        @reactive.event(input.go)
        async def run_flow():
            global callback
            session_id = uuid.uuid4()
            current_session_id.set(session_id)
            run_request=RunRequest(record_id=input.record_id(), websocket=chat, session_id=session_id)
            
            callback = await flow_runner.run_flow(flow_name="batch", run_request=run_request)
            await chat.append_message("*Flow started...*\n")

        # Define a callback to run when the user submits a message
        @chat.on_user_submit
        async def handle_user_input(user_input: str):
            # Create a response message stream
            if callback:
                await callback(user_input)

        # Add reactive effect for the new confirm button
        @reactive.effect
        @reactive.event(input.confirm)
        async def handle_confirm():
            """Handles the confirm button click."""
            message = ManagerResponse(confirm=True)
            if callback:
                await callback(message)

            # await stream.write(f"\n\n*Confirm button pressed. Message prepared: `{message}`*\n\n")


    app = App(app_ui, server)
    return app
