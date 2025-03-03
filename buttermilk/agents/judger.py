import os
from dataclasses import dataclass

# !pip install google-genai
from typing import Any, AsyncGenerator, Awaitable, Callable, List, Sequence

from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import (
    CancellationToken,
    Component,
    RoutedAgent,
)
from autogen_core.memory import Memory
from autogen_core.model_context import (
    ChatCompletionContext,
    UnboundedChatCompletionContext,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    RequestUsage,
    UserMessage,
)
from autogen_core.tools import BaseTool
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing_extensions import Self

from buttermilk import BM
from buttermilk.bm import BM
from buttermilk.utils.templating import (
    _parse_prompty,
    load_template_vars,
)


@dataclass
class WorkerTask:
    task: str
    previous_results: List[str]


@dataclass
class WorkerTaskResult:
    result: str


@dataclass
class UserTask:
    task: str


@dataclass
class FinalResult:
    result: str


class ChatAgent(RoutedAgent):
    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient | str,
        *,
        tools: List[
            BaseTool[Any, Any] | Callable[..., Any] | Callable[..., Awaitable[Any]]
        ]
        | None = None,
        handoffs: List[str] | None = None,
        model_context: ChatCompletionContext | None = None,
        description: str = "An agent that provides assistance with ability to use tools.",
        model_client_stream: bool = False,
        reflect_on_tool_use: bool = False,
        tool_call_summary_format: str = "{result}",
        memory: Sequence[Memory] | None = None,
        parameters: dict = {},
    ):
        if isinstance(model_client, str):
            bm = BM()
            model_client = bm.llms.get_autogen_client(model_client)

        # Construct list of messages from the templates
        rendered_template, remaining_inputs = load_template_vars(**parameters)

        # Interpret the template as a Prompty; split it into separate messages with
        # role and content keys

        # Parse messages using Prompty format
        # First we strip the header information from the markdown
        prompty = _parse_prompty(rendered_template)

        # Next we use Prompty's format to set roles within the template
        from promptflow.core._prompty_utils import parse_chat

        messages = parse_chat(
            prompty,
            valid_roles=["system", "user", "developer", "human", "placeholder"],
        )
        system_message = None
        if not model_context:
            model_context = []
        if messages[0]["role"] in ("system", "developer"):
            system_message = messages[0]["content"]
            model_context.extend(messages[1:])

        super().__init__(
            name,
            model_client,
            tools=tools,
            handoffs=handoffs,
            model_context=model_context,
            description=description,
            system_message=system_message,
            model_client_stream=model_client_stream,
            reflect_on_tool_use=reflect_on_tool_use,
            tool_call_summary_format=tool_call_summary_format,
            memory=memory,
        )


# class GeminiAssistantAgentConfig(BaseModel):
#     name: str
#     description: str = "An agent that provides assistance with ability to use tools."
#     model: str = "gemini-1.5-flash-002"
#     system_message: str | None = None


# class GeminiAssistantAgent(BaseChatAgent, Component[GeminiAssistantAgentConfig]):  # type: ignore[no-redef]
#     component_config_schema = GeminiAssistantAgentConfig
#     # component_provider_override = "mypackage.agents.GeminiAssistantAgent"

#     def __init__(
#         self,
#         name: str,
#         description: str = "An agent that provides assistance with ability to use tools.",
#         model: str = "gemini-1.5-flash-002",
#         api_key: str = os.environ["GEMINI_API_KEY"],
#         system_message: str
#         | None = "You are a helpful assistant that can respond to messages. Reply with TERMINATE when the task has been completed.",
#     ):
#         super().__init__(name=name, description=description)
#         self._model_context = UnboundedChatCompletionContext()
#         self._model_client = genai.Client(api_key=api_key)
#         self._system_message = system_message
#         self._model = model

#     @property
#     def produced_message_types(self) -> Sequence[type[ChatMessage]]:
#         return (TextMessage,)

#     async def on_messages(
#         self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
#     ) -> Response:
#         final_response = None
#         async for message in self.on_messages_stream(messages, cancellation_token):
#             if isinstance(message, Response):
#                 final_response = message

#         if final_response is None:
#             raise AssertionError("The stream should have returned the final result.")

#         return final_response

#     async def on_messages_stream(
#         self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
#     ) -> AsyncGenerator[AgentEvent | ChatMessage | Response, None]:
#         # Add messages to the model context
#         for msg in messages:
#             await self._model_context.add_message(
#                 UserMessage(content=msg.content, source=msg.source)
#             )

#         # Get conversation history
#         history = [
#             (msg.source if hasattr(msg, "source") else "system")
#             + ": "
#             + (msg.content if isinstance(msg.content, str) else "")
#             + "\n"
#             for msg in await self._model_context.get_messages()
#         ]

#         # Generate response using Gemini
#         response = self._model_client.models.generate_content(
#             model=self._model,
#             contents=f"History: {history}\nGiven the history, please provide a response",
#             config=types.GenerateContentConfig(
#                 system_instruction=self._system_message,
#                 temperature=0.3,
#             ),
#         )

#         # Create usage metadata
#         usage = RequestUsage(
#             prompt_tokens=response.usage_metadata.prompt_token_count,
#             completion_tokens=response.usage_metadata.candidates_token_count,
#         )

#         # Add response to model context
#         await self._model_context.add_message(
#             AssistantMessage(content=response.text, source=self.name)
#         )

#         # Yield the final response
#         yield Response(
#             chat_message=TextMessage(
#                 content=response.text, source=self.name, models_usage=usage
#             ),
#             inner_messages=[],
#         )

#     async def on_reset(self, cancellation_token: CancellationToken) -> None:
#         """Reset the assistant by clearing the model context."""
#         await self._model_context.clear()

#     @classmethod
#     def _from_config(cls, config: GeminiAssistantAgentConfig) -> Self:
#         return cls(
#             name=config.name,
#             description=config.description,
#             model=config.model,
#             system_message=config.system_message,
#         )

#     def _to_config(self) -> GeminiAssistantAgentConfig:
#         return GeminiAssistantAgentConfig(
#             name=self.name,
#             description=self.description,
#             model=self._model,
#             system_message=self._system_message,
#         )


async def run():
    # Create the primary agent.
    primary_agent = AssistantAgent(
        "primary",
        model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
        system_message="You are a helpful AI assistant.",
    )

    # Create a critic agent based on our new GeminiAssistantAgent.
    gemini_critic_agent = GeminiAssistantAgent(
        "gemini_critic",
        system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedbacks are addressed.",
    )

    # Define a termination condition that stops the task if the critic approves or after 10 messages.
    termination = TextMentionTermination("APPROVE") | MaxMessageTermination(10)

    # Create a team with the primary and critic agents.
    team = RoundRobinGroupChat(
        [primary_agent, gemini_critic_agent], termination_condition=termination
    )

    await Console(
        team.run_stream(task="Write a Haiku poem with 4 lines about the fall season.")
    )
