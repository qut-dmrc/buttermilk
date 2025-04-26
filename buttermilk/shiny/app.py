
import os
import uuid

from shiny import App, reactive, ui

from buttermilk._core.contract import RunRequest
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
    ),
    ui.card(
        ui.card_header("Hello"),
        ui.chat_ui("chat", messages=["Hello! How can I help you today?"], width="100%"),
        style="width:min(680px, 100%)",
        class_="mx-auto",
    ),
    ui.output_markdown_stream("my_stream"),
)

## THIS CODE DOESNT WORK YET


def create_app(flows: FlowRunner):
     
    def server(input):
        stream = ui.MarkdownStream(id="chat")
        # chat = ui.Chat(id="chat")
        flow_runner = flows

        @reactive.effect
        @reactive.event(input.go)
        async def run_flow():
            run_request=RunRequest(record_id=input.record_id(), websocket=stream, session_id=uuid.uuid4())
            await flow_runner.run_flow(flow_name="batch", run_request=run_request)

        # # Generate a response when the user submits a message
        # @chat.on_user_submit
        # async def handle_user_input(user_input: str):
        #     response = await stream.stream([user_input])
        #     await chat.append_message_stream(response)


    app = App(app_ui, server)
    return app

# app_ui = ui.page_fluid(
#     ui.tags.link(rel="stylesheet", href="style.css"),
#     ui.tags.script(src="script.js"),
#     ui.chat_ui("chat"),
# )

# # Create a chat instance, with an initial message
# chat = ui.Chat(
#     id="chat",
#     messages=["Hello! How can I help you today?"],
# )
# chat.ui()

# # Store chat state in the url when an "assistant" response occurs
# chat.enable_bookmarking(chat_client, bookmark_store="url")

