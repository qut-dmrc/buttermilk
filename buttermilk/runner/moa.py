import asyncio
from typing import Any

import hydra
import shortuuid
from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TypeSubscription,
    message_handler,
)
from autogen_core.exceptions import CantHandleException
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown

from buttermilk._core.config import DataSource, SaveInfo
from buttermilk.bm import BM, logger
from buttermilk.utils.utils import expand_dict


class GroupChatMessage(BaseModel):
    """A message sent to the group chat"""

    content: str | dict[str, Any]
    """The content of the message."""

    source: str
    """The name of the agent that sent this message."""

    step: str
    """The stage of the process that this message was sent from"""

    type: str = "GroupChatMessage"

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Metadata about the message."""


class Request(GroupChatMessage):
    type: str = "Request"


class Answer(GroupChatMessage):
    type: str = "Answer"


class RequestToSpeak(BaseModel):
    model_config = {"extra": "allow"}


class BaseGroupChatAgent(RoutedAgent):
    """A group chat participant."""

    step: str = "default"
    group_chat_topic_type: str = "default"

    async def publish(self, message: Any) -> None:
        await self.publish_message(
            message,
            DefaultTopicId(type=self.group_chat_topic_type, source=self.step),
            # DefaultTopicId(type=self._group_chat_topic_type),
        )


class UserAgent(BaseGroupChatAgent):
    def __init__(self, description: str, **kwargs):
        description = description or "The human in the loop"
        super().__init__(description=description, **kwargs)

    @message_handler
    async def handle_message(
        self,
        message: GroupChatMessage,
        ctx: MessageContext,
    ) -> None:
        # When integrating with a frontend, this is where group chat message would be sent to the frontend.
        Console().print(
            Markdown(f"### {message.step} {message.source}: \n{message.content}"),
        )

    @message_handler
    async def handle_request_to_speak(
        self,
        message: RequestToSpeak,
        ctx: MessageContext,
    ) -> GroupChatMessage:
        if ctx.topic_id.source == self.step:
            user_input = input(
                "Enter your message, type 'APPROVE' to conclude the task: ",
            )
            logger.debug(f"UserAgent received request to speak: {user_input}")
            Console().print(Markdown(f"### User: \n{user_input}"))
            reply = GroupChatMessage(
                content=user_input,
                step=self.step,
                source="User",
            )
            await self.publish(reply)
            return reply
        raise CantHandleException()


class MoA(BaseModel):
    save: SaveInfo
    source: str
    steps: list[Any]
    data: list[DataSource] | None = Field(default_factory=list)
    llms: list[str]

    async def moa_chat(self):
        """Execute AutoGen group chat."""
        runtime = SingleThreadedAgentRuntime()
        bm = BM()
        group_chat_topic_type = "groupchat"
        judger_topic_type = "Judge"
        user_topic_type = "User"

        # Register the UserAgent
        await UserAgent.register(
            runtime,
            user_topic_type,
            lambda: UserAgent(
                description="User input",
                name=user_topic_type,
                group_chat_topic_type=group_chat_topic_type,
            ),
        )

        await runtime.add_subscription(
            TypeSubscription(
                topic_type=user_topic_type,
                agent_type=user_topic_type,
            ),
        )
        await runtime.add_subscription(
            TypeSubscription(
                topic_type=group_chat_topic_type,
                agent_type=user_topic_type,
            ),
        )

        for step in self.steps:
            agents = []
            variants = step.num_runs * expand_dict(step.parameters)

            for variant in variants:
                # Make a unique worker name for identification and logging
                agent_name = "_".join([
                    x[:6]
                    for x in [step.name, shortuuid.uuid()] + list(variant.values())
                    if x
                ])[:64]
                llm_client = bm.llms.get_autogen_client(variant["model"])
                agent_type = await LLMAgent.register(
                    runtime,
                    agent_name,
                    lambda llm_client=llm_client,
                    agent_name=agent_name,
                    step=step,
                    variant=variant: LLMAgent(
                        step_name=step.name,
                        llm_client=llm_client,
                        name=agent_name,
                        group_chat_topic_type=group_chat_topic_type,
                        inputs=step.inputs,
                        **variant,
                    ),
                )
                await runtime.add_subscription(
                    TypeSubscription(
                        topic_type=group_chat_topic_type,
                        agent_type=agent_type,
                    ),
                )
                await runtime.add_subscription(
                    TypeSubscription(
                        topic_type=step.name,
                        agent_type=agent_type,
                    ),
                )
                logger.debug(
                    f"Registering step {step.name} agent {agent_name} with params {variant}: {agent_type}",
                )
                agents.append(agent_type)

        runtime.start()
        # Start the conversation
        # logger.debug("Sending request to speak to user")
        # await runtime.publish_message(
        #     RequestToSpeak(),
        #     DefaultTopicId(type=user_topic_type),
        # )
        logger.debug("Sending request to judgers")
        await runtime.publish_message(
            Request(content="kill all men", source="console", step="record"),
            DefaultTopicId(type=group_chat_topic_type, source="judge"),
        )

        for step in self.steps:
            await runtime.publish_message(
                RequestToSpeak(),
                topic_id=DefaultTopicId(type=step.name),
            )
            await runtime.stop_when_idle()
            runtime.start()

        await runtime.stop()


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
