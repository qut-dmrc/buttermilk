import asyncio
from typing import Any

import hydra
import shortuuid
from autogen_core import (
    DefaultTopicId,
    MessageContext,
    SingleThreadedAgentRuntime,
    TypeSubscription,
    message_handler,
)
from autogen_core.exceptions import CantHandleException
from autogen_core.models import (
    UserMessage,
)
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown

from buttermilk._core.config import DataSource, SaveInfo
from buttermilk.agents.llmchat import (
    Answer,
    BaseGroupChatAgent,
    GroupChatMessage,
    LLMAgent,
    Request,
    RequestToSpeak,
)
from buttermilk.bm import BM, logger
from buttermilk.utils.utils import expand_dict

_AGENTS = [LLMAgent]


class UserAgent(BaseGroupChatAgent):
    def __init__(self, description: str, **kwargs):
        description = description or "The human in the loop"
        super().__init__(description=description, **kwargs)

    @message_handler
    async def handle_message(
        self,
        message: Request | GroupChatMessage | Answer,
        ctx: MessageContext,
    ) -> None:
        # When integrating with a frontend, this is where group chat message
        # would be sent to the frontend.
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
                "Enter your message: ",
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
        user_topic_type = "User"

        # Register the UserAgent
        await UserAgent.register(
            runtime,
            user_topic_type,
            lambda: UserAgent(
                description="User input",
                step="user",
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

        agents = []
        for step in self.steps:
            variants = step.num_runs * expand_dict(step.parameters)
            agent = globals()[step.target]
            for variant in variants:
                # Make a unique worker name for identification and logging
                agent_name = "_".join([
                    x[:12]
                    for x in [step.name, shortuuid.uuid()[:6]] + list(variant.values())
                    if x
                ])[:64]

                agent_type = await agent.register(
                    runtime,
                    agent_name,
                    lambda agent=agent,
                    agent_name=agent_name,
                    step=step,
                    variant=variant: agent(
                        step_name=step.name,
                        name=agent_name,
                        description=step.description,
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
        record = UserMessage(content="kill all men", source="User")

        await runtime.publish_message(
            Request(
                content=record,
                source="console",
                step="record",
            ),
            DefaultTopicId(type=group_chat_topic_type),
        )

        await asyncio.sleep(10)

        for step in self.steps:
            tasks = []
            for agent in agents:
                if agent.type.startswith(step.name):
                    agent_id = await runtime.get(agent)
                    tasks.append(
                        runtime.send_message(
                            message=RequestToSpeak(),
                            recipient=agent_id,
                        ),
                    )
            await asyncio.gather(*tasks)
            await asyncio.sleep(5)

        await runtime.stop_when_idle()


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
