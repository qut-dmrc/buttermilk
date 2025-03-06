import asyncio
from functools import cached_property

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
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.markdown import Markdown

from buttermilk._core.config import DataSource, SaveInfo
from buttermilk.agents.llmchat import (
    Answer,
    BaseGroupChatAgent,
    GroupChatMessage,
    LLMAgent,
    Payload,
    RequestToSpeak,
)
from buttermilk.bm import BM, logger
from buttermilk.utils.utils import expand_dict
from buttermilk.utils.validators import make_list_validator

_AGENTS = [LLMAgent]


class UserAgent(BaseGroupChatAgent):
    def __init__(self, description: str, **kwargs):
        description = description or "The human in the loop"
        super().__init__(description=description, **kwargs)

    @message_handler
    async def handle_message(
        self,
        message: Payload | GroupChatMessage | Answer,
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


class VariantParameters(BaseModel):
    template: list[str]
    model: list[str]

    _ensure_list = field_validator("template", "model", mode="before")(
        make_list_validator(),
    )

    model_config = {"extra": "allow"}


class MoAAgentFactory(BaseModel):
    """A factory for creating LLMAgent instance variants for a single
    step of a workflow.

    Creates a new agent for every combination of parameters in a given
    step of the workflow to run. Agents have a variants mapping;
    each permutation of these is multiplied by num_runs. Agents also
    have an inputs mapping that does not get multiplied.
    """

    description: str
    name: str = Field(
        ...,
        description="The step in the workflow that this agent can perform.",
    )
    agent: str = Field(..., description="The agent object to use for this step.")
    variants: VariantParameters = Field(
        ...,
        description="A set of initialisation parameters that will be multiplied together to create individual variant agents.",
    )
    num_runs: int = Field(
        default=1,
        description="The number of times to run each variant.",
    )

    inputs: list[str] = Field(
        default_factory=list,
        description="The inputs that this agent will receive from the workflow.",
    )
    outputs: list[str] = Field(
        default_factory=list,
        description="The outputs that this agent will produce to the workflow.",
    )

    async def register_variants(self, runtime, group_chat_topic_type):
        # Get the object that provides the agent template for all variants
        agent_cls = globals()[self.agent]

        # Create variants (permutations of vars multiplied by num_runs)
        variant_configs = self.num_runs * expand_dict(self.variants.model_dump())
        registered_agents = []

        for variant in variant_configs:
            # Make a unique worker name for identification and logging
            agent_name = "_".join([
                x[:12]
                for x in [self.name, shortuuid.uuid()[:6]] + list(variant.values())
                if x
            ])[:64]

            agent_type = await agent_cls.register(
                runtime,
                agent_name,
                lambda agent_name=agent_name, variant=variant: agent_cls(
                    step_name=self.name,
                    name=agent_name,
                    description=self.description,
                    group_chat_topic_type=group_chat_topic_type,
                    inputs=self.inputs,
                    **variant,
                ),
            )

            # Subscribe the agent to relevant topics
            await runtime.add_subscription(
                TypeSubscription(
                    topic_type=group_chat_topic_type,
                    agent_type=agent_type,
                ),
            )
            await runtime.add_subscription(
                TypeSubscription(
                    topic_type=self.name,
                    agent_type=agent_type,
                ),
            )
            logger.debug(
                f"Registered step {self.name} agent {agent_name} with params {', '.join(variant.keys())}: {agent_type}",
            )
            registered_agents.append(agent_type)
        return registered_agents


class MoA(BaseModel):
    save: SaveInfo
    source: str
    steps: list[MoAAgentFactory]
    data: list[DataSource] | None = Field(default_factory=list)

    @cached_property
    def group_chat_topic_type(self) -> str:
        bm = BM()

        """The group chat topic type (common to all agents in the chat)."""
        topic = f"groupchat-{bm.run_info.name}-{bm.run_info.job}"

        # remove punctuation
        topic = "".join([x for x in topic if x.isalnum() or x == "-"])

        return topic

    async def moa_chat(self):
        """Execute AutoGen group chat."""
        runtime = SingleThreadedAgentRuntime()

        user_topic_type = "User"

        # Register the UserAgent
        user_agent_type = await UserAgent.register(
            runtime,
            user_topic_type,
            lambda: UserAgent(
                description="User input",
                group_chat_topic_type=self.group_chat_topic_type,
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
                topic_type=self.group_chat_topic_type,
                agent_type=user_topic_type,
            ),
        )

        # Dictionary to track agents by step
        step_agents = {}

        # Register all agent variants for each step
        for step_factory in self.steps:
            # Register variants and collect the agent types
            agents_for_step = await step_factory.register_variants(
                runtime,
                group_chat_topic_type=self.group_chat_topic_type,
            )
            step_agents[step_factory.name] = agents_for_step

        # Start the runtime
        runtime.start()

        # Initial message to start the workflow
        logger.debug("Sending initial request to the group chat")
        record = "kill all men"
        await runtime.publish_message(
            Payload(
                content=record,
                source="console",
                step="record",
                data=dict(record=record),
            ),
            DefaultTopicId(type=self.group_chat_topic_type),
        )

        # Start the conversation
        # logger.debug("Sending request to speak to user")
        # await runtime.publish_message(
        #     RequestToSpeak(),
        #     DefaultTopicId(type=user_topic_type),
        # )
        logger.debug("Sending record to agents")

        # Allow some time for initialization
        await asyncio.sleep(5)

        # Process each step in sequence
        for step_factory in self.steps:
            logger.debug(f"Processing step: {step_factory.name}")
            tasks = []

            # Request each agent variant for this step to speak
            for agent_type in step_agents.get(step_factory.name, []):
                agent_id = await runtime.get(agent_type)
                tasks.append(
                    runtime.send_message(
                        message=RequestToSpeak(),
                        recipient=agent_id,
                    ),
                )

            # Wait for all agents in this step to complete
            if tasks:
                await asyncio.gather(*tasks)
                # Give agents time to process and respond
                await asyncio.sleep(5)

        # Wait for all processing to complete
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
