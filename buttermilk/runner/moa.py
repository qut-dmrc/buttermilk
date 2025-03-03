import asyncio
import string
from dataclasses import dataclass
from typing import Any, List

import autogen
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
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
import shortuuid

from buttermilk._core.config import DataSource, SaveInfo
from buttermilk.agents.judger import ChatAgent
from buttermilk.bm import BM
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

class WorkerTask(BaseModel):
    task: str
    previous_results: List[str]

class WorkerTaskResult(BaseModel):
    result: str

class UserTask(BaseModel):
    task: str

class FinalResult(BaseModel):
    result: str

class GroupChatMessage(BaseModel):
    body: UserMessage|AssistantMessage|SystemMessage

class RequestToSpeak(BaseModel):
    pass

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
        print(f"{'-'*80}\nOrchestrator-{self.id}:\nReceived task: {message.task}")
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
            print(f"{'-'*80}\nOrchestrator-{self.id}:\nDispatch to workers at layer {i}")
            results = await asyncio.gather(*[self.send_message(worker_task, worker_id) for worker_id in worker_ids])
            print(f"{'-'*80}\nOrchestrator-{self.id}:\nReceived results from workers at layer {i}")
            # Prepare task for the next layer.
            worker_task = WorkerTask(task=message.task, previous_results=[r.result for r in results])
        # Perform final aggregation.
        print(f"{'-'*80}\nOrchestrator-{self.id}:\nPerforming final aggregation")
        system_prompt = "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n\nResponses from models:"
        system_prompt += "\n" + "\n\n".join([f"{i+1}. {r}" for i, r in enumerate(worker_task.previous_results)])
        model_result = await self._model_client.create(
            [SystemMessage(content=system_prompt), UserMessage(content=message.task, source="user")]
        )
        assert isinstance(model_result.content, str)
        return FinalResult(result=model_result.content)
    
# Create a UserAgent instead of UserProxyAgent
class UserAgent(RoutedAgent):
    def __init__(self, topic: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "User"
        self.topic = topic

    @message_handler
    async def handle_request_to_speak(
        self, message: RequestToSpeak, ctx: MessageContext
    ) -> None:
        user_input = input("Enter your message, type 'APPROVE' to conclude the task: ")
        Console().print(Markdown(f"### User: \n{user_input}"))
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=user_input, source=self.id.type)),
            DefaultTopicId(type=self.topic),
        )

    @message_handler
    async def handle_user_task(self, message: UserMessage, ctx: MessageContext) -> UserMessage:
        # Handle user input here
        return UserMessage(content=message.content, source=self.name)

class JudgeAgent(RoutedAgent):
    def __init__(self, name: str, topic: str, llm_client: ChatCompletionClient, *args, **kwargs):
        super().__init__(*args, description="Applies rules to content.", **kwargs)
        self.name = name
        self.llm_client = llm_client
        self.topic = topic

    @message_handler
    async def handle_group_chat(self, message: GroupChatMessage, ctx: MessageContext) -> AssistantMessage:
        # Process the message using the LLM client
        response = await self.llm_client.create(
            messages=[
                SystemMessage(content="You are a helpful judge."),
                message.body
            ]
        )
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=response.choices[0].message.content, source=self.name)),
            DefaultTopicId(type=self.topic),
        )
        return AssistantMessage(content=response.choices[0].message.content, source=self.name)
    
class MoA(BaseModel):
    save: SaveInfo
    source: str
    steps: list[Any]
    data: list[DataSource] | None = Field(default_factory=list)
    llms: list[str] = Field(default_factory=list)

    async def moa_chat(self) -> str:
        """Execute AutoGen group chat"""
        runtime = SingleThreadedAgentRuntime()
        bm = BM()
        group_chat_topic_type = "groupchat"
        judger_topic_type = "Judge"
        user_topic_type = "User"

        # Register the UserAgent
        await UserAgent.register(runtime, user_topic_type, lambda: UserAgent(description="User input", topic=user_topic_type))

        await runtime.add_subscription(
            TypeSubscription(topic_type=user_topic_type, agent_type=user_topic_type)
        )
        await runtime.add_subscription(
            TypeSubscription(
                topic_type=group_chat_topic_type, agent_type=user_topic_type
            )
        )
        
        for llm_name in self.llms:
            llm_client = bm.llms.get_autogen_client(llm_name)
            judge = JudgeAgent(
                    name=f"Judge-{shortuuid.uuid()[4]}",
                    llm_client=llm_client, topic=judger_topic_type
                )
                
            await judge.register(runtime, judger_topic_type, lambda: judge)

        await runtime.add_subscription(
            TypeSubscription(
                topic_type=judger_topic_type, agent_type=judger_topic_type
            )
        )

        await runtime.add_subscription(
            TypeSubscription(
                topic_type=group_chat_topic_type, agent_type=judger_topic_type
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

        pass 
        # return "\n".join(response_messages)
    

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg) -> None:
    # Hydra will automatically instantiate the objects
    objs = hydra.utils.instantiate(cfg)
    bm = objs.bm
    bm = BM()
    moa = objs.flows.moa
    asyncio.run(moa.moa_chat())


if __name__ == "__main__":
    main()
