import asyncio
import string
from dataclasses import dataclass
from typing import List

import hydra
from autogen_core import (
    AgentId,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TypeSubscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown

from buttermilk.bm import BM


class WorkerTask(BaseModel):
    task: str
    previous_results: List[str]


class WorkerTaskResult(BaseModel):
    result: str


class UserTask(BaseModel):
    task: str


@dataclass
class FinalResult(BaseModel):
    result: str


class GroupChatMessage(BaseModel):
    body: UserMessage


class RequestToSpeak(BaseModel):
    pass


class BaseGroupChatAgent(RoutedAgent):
    """A group chat participant using an LLM."""

    def __init__(
        self,
        description: str,
        group_chat_topic_type: str,
        model_client: ChatCompletionClient,
        system_message: str,
    ) -> None:
        super().__init__(description=description)
        self._group_chat_topic_type = group_chat_topic_type
        self._model_client = model_client
        self._system_message = SystemMessage(content=system_message)
        self._chat_history: List[LLMMessage] = []

    @message_handler
    async def handle_message(
        self, message: GroupChatMessage, ctx: MessageContext
    ) -> None:
        self._chat_history.extend(
            [
                UserMessage(
                    content=f"Transferred to {message.body.source}", source="system"
                ),
                message.body,
            ]
        )

    @message_handler
    async def handle_request_to_speak(
        self, message: RequestToSpeak, ctx: MessageContext
    ) -> None:
        # print(f"\n{'-'*80}\n{self.id.type}:", flush=True)
        Console().print(Markdown(f"### {self.id.type}: "))
        self._chat_history.append(
            UserMessage(
                content=f"Transferred to {self.id.type}, adopt the persona immediately.",
                source="system",
            )
        )
        completion = await self._model_client.create(
            [self._system_message] + self._chat_history
        )
        assert isinstance(completion.content, str)
        self._chat_history.append(
            AssistantMessage(content=completion.content, source=self.id.type)
        )
        Console().print(Markdown(completion.content))
        # print(completion.content, flush=True)
        await self.publish_message(
            GroupChatMessage(
                body=UserMessage(content=completion.content, source=self.id.type)
            ),
            topic_id=DefaultTopicId(type=self._group_chat_topic_type),
        )


class WriterAgent(BaseGroupChatAgent):
    def __init__(
        self,
        description: str,
        group_chat_topic_type: str,
        model_client: ChatCompletionClient,
    ) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="You are a Writer. You produce good work.",
        )


class EditorAgent(BaseGroupChatAgent):
    def __init__(
        self,
        description: str,
        group_chat_topic_type: str,
        model_client: ChatCompletionClient,
    ) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="You are an Editor. Plan and guide the task given by the user. Provide critical feedbacks to the draft and illustration produced by Writer and Illustrator. "
            "Approve if the task is completed and the draft and illustration meets user's requirements.",
        )


class WorkerAgent(RoutedAgent):
    def __init__(
        self,
        model_client: ChatCompletionClient,
    ) -> None:
        super().__init__(description="Worker Agent")
        self._model_client = model_client

    @message_handler
    async def handle_task(
        self, message: WorkerTask, ctx: MessageContext
    ) -> WorkerTaskResult:
        if message.previous_results:
            # If previous results are provided, we need to synthesize them to create a single prompt.
            system_prompt = "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n\nResponses from models:"
            system_prompt += "\n" + "\n\n".join(
                [f"{i + 1}. {r}" for i, r in enumerate(message.previous_results)]
            )
            model_result = await self._model_client.create(
                [
                    SystemMessage(content=system_prompt),
                    UserMessage(content=message.task, source="user"),
                ]
            )
        else:
            # If no previous results are provided, we can simply pass the user query to the model.
            model_result = await self._model_client.create(
                [UserMessage(content=message.task, source="user")]
            )
        assert isinstance(model_result.content, str)
        Console().print(f"{'-' * 80}\nWorker-{self.id}:\n{model_result.content}")
        return WorkerTaskResult(result=model_result.content)


class OrchestratorAgent(RoutedAgent):
    def __init__(
        self,
        model_client: ChatCompletionClient,
        worker_agent_types: List[str],
        num_layers: int,
    ) -> None:
        super().__init__(description="Aggregator Agent")
        self._model_client = model_client
        self._worker_agent_types = worker_agent_types
        self._num_layers = num_layers

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> FinalResult:
        Console().print(
            f"{'-' * 80}\nOrchestrator-{self.id}:\nReceived task: {message.task}"
        )
        # Create task for the first layer.
        worker_task = WorkerTask(task=message.task, previous_results=[])
        # Iterate over layers.
        for i in range(self._num_layers - 1):
            # Assign workers for this layer.
            worker_ids = [
                AgentId(worker_type, f"{self.id.key}/layer_{i}/worker_{j}")
                for j, worker_type in enumerate(self._worker_agent_types)
            ]
            # Dispatch tasks to workers.
            Console().print(
                f"{'-' * 80}\nOrchestrator-{self.id}:\nDispatch to workers at layer {i}"
            )
            results = await asyncio.gather(
                *[self.send_message(worker_task, worker_id) for worker_id in worker_ids]
            )
            Console().print(
                f"{'-' * 80}\nOrchestrator-{self.id}:\nReceived results from workers at layer {i}"
            )
            # Prepare task for the next layer.
            worker_task = WorkerTask(
                task=message.task, previous_results=[r.result for r in results]
            )
        # Perform final aggregation.
        Console().print(
            f"{'-' * 80}\nOrchestrator-{self.id}:\nPerforming final aggregation"
        )
        system_prompt = "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n\nResponses from models:"
        system_prompt += "\n" + "\n\n".join(
            [f"{i + 1}. {r}" for i, r in enumerate(worker_task.previous_results)]
        )
        model_result = await self._model_client.create(
            [
                SystemMessage(content=system_prompt),
                UserMessage(content=message.task, source="user"),
            ]
        )
        assert isinstance(model_result.content, str)
        return FinalResult(result=model_result.content)


class UserAgent(RoutedAgent):
    def __init__(self, description: str, group_chat_topic_type: str) -> None:
        super().__init__(description=description)
        self._group_chat_topic_type = group_chat_topic_type

    @message_handler
    async def handle_message(
        self, message: GroupChatMessage, ctx: MessageContext
    ) -> None:
        # When integrating with a frontend, this is where group chat message would be sent to the frontend.
        pass

    @message_handler
    async def handle_request_to_speak(
        self, message: RequestToSpeak, ctx: MessageContext
    ) -> None:
        user_input = input("Enter your message, type 'APPROVE' to conclude the task: ")
        Console().print(Markdown(f"### User: \n{user_input}"))
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=user_input, source=self.id.type)),
            DefaultTopicId(type=self._group_chat_topic_type),
        )


class GroupChatManager(RoutedAgent):
    def __init__(
        self,
        participant_topic_types: List[str],
        model_client: ChatCompletionClient,
        participant_descriptions: List[str],
    ) -> None:
        super().__init__("Group chat manager")
        self._participant_topic_types = participant_topic_types
        self._model_client = model_client
        self._chat_history: List[UserMessage] = []
        self._participant_descriptions = participant_descriptions
        self._previous_participant_topic_type: str | None = None

    @message_handler
    async def handle_message(
        self, message: GroupChatMessage, ctx: MessageContext
    ) -> None:
        assert isinstance(message.body, UserMessage)
        self._chat_history.append(message.body)
        # If the message is an approval message from the user, stop the chat.
        if message.body.source == "User":
            assert isinstance(message.body.content, str)
            if (
                message.body.content.lower()
                .strip(string.punctuation)
                .endswith("approve")
            ):
                return
        # Format message history.
        messages: List[str] = []
        for msg in self._chat_history:
            if isinstance(msg.content, str):
                messages.append(f"{msg.source}: {msg.content}")
            elif isinstance(msg.content, list):
                line: List[str] = []
                for item in msg.content:
                    if isinstance(item, str):
                        line.append(item)
                    else:
                        line.append("[Image]")
                messages.append(f"{msg.source}: {', '.join(line)}")
        history = "\n".join(messages)
        # Format roles.
        roles = "\n".join(
            [
                f"{topic_type}: {description}".strip()
                for topic_type, description in zip(
                    self._participant_topic_types,
                    self._participant_descriptions,
                    strict=True,
                )
                if topic_type != self._previous_participant_topic_type
            ]
        )
        selector_prompt = """You are in a role play game. The following roles are available:
{roles}.
Read the following conversation. Then select the next role from {participants} to play. Only return the role.

{history}

Read the above conversation. Then select the next role from {participants} to play. Only return the role.
"""
        system_message = SystemMessage(
            content=selector_prompt.format(
                roles=roles,
                history=history,
                participants=str(
                    [
                        topic_type
                        for topic_type in self._participant_topic_types
                        if topic_type != self._previous_participant_topic_type
                    ]
                ),
            )
        )
        completion = await self._model_client.create(
            [system_message], cancellation_token=ctx.cancellation_token
        )
        assert isinstance(completion.content, str)
        selected_topic_type: str
        for topic_type in self._participant_topic_types:
            if topic_type.lower() in completion.content.lower():
                selected_topic_type = topic_type
                self._previous_participant_topic_type = selected_topic_type
                await self.publish_message(
                    RequestToSpeak(), DefaultTopicId(type=selected_topic_type)
                )
                return
        raise ValueError(f"Invalid role selected: {completion.content}")


# def start():
#     runtime = SingleThreadedAgentRuntime()
#     await WorkerAgent.register(
#         runtime,
#         "worker",
#         lambda: WorkerAgent(
#             model_client=OpenAIChatCompletionClient(model="gpt-4o-mini")
#         ),
#     )
#     await OrchestratorAgent.register(
#         runtime,
#         "orchestrator",
#         lambda: OrchestratorAgent(
#             model_client=OpenAIChatCompletionClient(model="gpt-4o"),
#             worker_agent_types=["worker"] * 3,
#             num_layers=3,
#         ),
#     )

#     runtime.start()
#     result = await runtime.send_message(
#         UserTask(task=task), AgentId("orchestrator", "default")
#     )
#     await runtime.stop_when_idle()
#     Console().print(f"{'-' * 80}\nFinal result:\n{result.result}")


async def start():
    runtime = SingleThreadedAgentRuntime()
    bm = BM()
    editor_topic_type = "Editor"
    writer_topic_type = "Writer"
    illustrator_topic_type = "Illustrator"
    user_topic_type = "User"
    group_chat_topic_type = "group_chat"

    writer_description = "Writer for creating any text content."
    user_description = "User for providing final approval."

    model_client = bm.llms.get_autogen_client("o3-mini-high")
    writer_agent_type = await WriterAgent.register(
        runtime,
        writer_topic_type,  # Using topic type as the agent type.
        lambda: WriterAgent(
            description=writer_description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            # api_key="YOUR_API_KEY",
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(
            topic_type=writer_topic_type, agent_type=writer_agent_type.type
        )
    )
    await runtime.add_subscription(
        TypeSubscription(
            topic_type=group_chat_topic_type, agent_type=writer_agent_type.type
        )
    )

    user_agent_type = await UserAgent.register(
        runtime,
        user_topic_type,
        lambda: UserAgent(
            description=user_description, group_chat_topic_type=group_chat_topic_type
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=user_topic_type, agent_type=user_agent_type.type)
    )
    await runtime.add_subscription(
        TypeSubscription(
            topic_type=group_chat_topic_type, agent_type=user_agent_type.type
        )
    )

    group_chat_manager_type = await GroupChatManager.register(
        runtime,
        "group_chat_manager",
        lambda: GroupChatManager(
            participant_topic_types=[
                writer_topic_type,
                user_topic_type,
            ],
            model_client=model_client,
            participant_descriptions=[
                writer_description,
                user_description,
            ],
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(
            topic_type=group_chat_topic_type, agent_type=group_chat_manager_type.type
        )
    )
    runtime.start()

    # Start the conversation
    await runtime.publish_message(
        RequestToSpeak(), DefaultTopicId(type=user_topic_type)
    )

    while True:
        try:
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            break


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg) -> None:
    # Hydra will automatically instantiate the objects
    objs = hydra.utils.instantiate(cfg)
    bm = objs.bm
    bm = BM()
    asyncio.run(start())


if __name__ == "__main__":
    main()
