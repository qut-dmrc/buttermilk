import asyncio
from functools import cached_property

import shortuuid
from autogen_core import (
    CancellationToken,
    DefaultTopicId,
    MessageContext,
    SingleThreadedAgentRuntime,
    TypeSubscription,
    message_handler,
)
from autogen_core.model_context import (
    UnboundedChatCompletionContext,
)
from autogen_core.models import (
    UserMessage,
)
from pydantic import BaseModel

from buttermilk._core.agent import AgentVariants
from buttermilk._core.config import DataSource, SaveInfo
from buttermilk.agents.llmchat import (
    LLMAgent,
)
from buttermilk.bm import BM, logger
from buttermilk.runner.chat import (
    Answer,
    BaseGroupChatAgent,
    GroupChatMessage,
    InputRecord,
    IOInterface,
    MessagesCollector,
    RequestToSpeak,
)
from buttermilk.runner.varmap import FlowVariableRouter
from buttermilk.utils.media import download_and_convert
from buttermilk.utils.templating import KeyValueCollector

_AGENTS = [LLMAgent]
USER_AGENT_TYPE = "User"


class MoAAgentFactory(AgentVariants):
    """A factory for creating LLMAgent instance variants for a single
    step of a workflow.

    Creates a new agent for every combination of parameters in a given
    step of the workflow to run. Agents have a variants mapping;
    each permutation of these is multiplied by num_runs. Agents also
    have an inputs mapping that does not get multiplied.
    """

    async def register_variants(self, runtime, group_chat_topic_type: str):
        # Get the object that provides the agent template for all variants
        agent_cls: LLMAgent = globals()[self.agent]

        registered_agents = []
        for variant in self.get_variant_configs():
            unique_id = f"{self.name}-{shortuuid.uuid()[:6]}"

            agent_type = await agent_cls.register(
                runtime,
                unique_id,
                lambda variant=variant: agent_cls(
                    config=variant,
                    group_chat_topic_type=group_chat_topic_type,
                ),  # type: ignore
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
                    topic_type=self.name,  # shared topic is step name
                    agent_type=agent_type,
                ),
            )
            logger.debug(
                f"Registered step {self.name} with params {', '.join(variant.model_fields_set)}: {agent_type}",
            )
            registered_agents.append(agent_type)
        return registered_agents


class Conductor(BaseGroupChatAgent):
    def __init__(self, description: str, group_chat_topic_type: str, steps):
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
        )

        # Generally, in a group chat, you want most of the information to be publicly
        # visible. These collectors provide storage for agents, usually by reading from
        # the group chat log of messages. Useful for small objects predominantly.
        self._placeholders: KeyValueCollector = MessagesCollector()
        self._flow_data: FlowVariableRouter = FlowVariableRouter()
        self._context = UnboundedChatCompletionContext()
        self.running = False
        self.steps = steps

    @message_handler
    async def start_signal(
        self,
        message: RequestToSpeak,
        ctx: MessageContext,
    ) -> None:
        if not self.running:
            self.running = True
            # asyncio.get_event_loop().create_task(self.run())
            logger.info("Conductor started.")
            await self.run()
        else:
            logger.error("Conductor already running.")

    @message_handler
    async def handle_answer(
        self,
        message: Answer,
        ctx: MessageContext,
    ) -> None:
        # store messages
        # draft = dict(agent_id=message.agent_id, body=message.body)

        self._flow_data.add(message.step, message)

    @message_handler
    async def handle_inputrecords(
        self,
        message: InputRecord,
        ctx: MessageContext,
    ) -> None:
        try:
            src = ctx.sender.type
        except:
            src = self.id.type
        msg = UserMessage(content=message.payload.fulltext, source=src)
        self._placeholders.add("record", msg)

    @message_handler
    async def handle_other(
        self,
        message: GroupChatMessage,
        ctx: MessageContext,
    ) -> None:
        msg = UserMessage(
            content=message.content,
            source=ctx.sender.type if ctx.sender else self.id.type,
        )
        await self._context.add_message(msg)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the assistant by clearing the model context."""
        self._placeholders = MessagesCollector()
        await self._context.clear()
        self._flow_data = FlowVariableRouter()

    async def run(self) -> None:
        # Dictionary to track agents by step
        step_agents = {}

        # Register all agent variants for each step
        for step_factory in self.steps:
            # Register variants and collect the agent types
            agents_for_step = await step_factory.register_variants(
                self.runtime,
                group_chat_topic_type=self._group_chat_topic_type,
            )
            step_agents[step_factory.name] = agents_for_step

        # Start the conversation
        logger.debug("Sending request to speak to user")
        user_id = await self.runtime.get(USER_AGENT_TYPE)
        result = await self.runtime.send_message(
            RequestToSpeak(content="Over to you..."),
            recipient=user_id,
        )
        record = await download_and_convert(result.content)

        # Start the group chat
        await self.runtime.publish_message(
            InputRecord(
                content=record.fulltext,
                step="record",
                payload=record,
            ),
            DefaultTopicId(type=self._group_chat_topic_type),
        )

        # Allow some time for initialization
        await asyncio.sleep(1)

        # Process each step in sequence
        for step_factory in self.steps:
            logger.debug(f"Processing step: {step_factory.name}")
            tasks = []

            step_data = self._flow_data._resolve_mappings(step_factory.inputs)

            # Request each agent variant for this step to speak
            for agent_type in step_agents.get(step_factory.name, []):
                agent_id = await self.runtime.get(agent_type)
                tasks.append(
                    self.runtime.send_message(
                        message=RequestToSpeak(
                            inputs=step_data,
                            placeholders=self._placeholders.get_dict(),
                        ),
                        recipient=agent_id,
                    ),
                )

            # Wait for all agents in this step to complete
            if tasks:
                await asyncio.gather(*tasks)


class MoA(BaseModel):
    save: SaveInfo
    source: str
    steps: list[MoAAgentFactory]
    data: list[DataSource] | None = list()

    @cached_property
    def group_chat_topic_type(self) -> str:
        bm = BM()

        """The group chat topic type (common to all agents in this chat)."""
        topic = f"groupchat-{bm.run_info.name}-{bm.run_info.job}-{shortuuid.uuid()[:4]}"

        # remove punctuation
        topic = "".join([x for x in topic if x.isalnum() or x == "-"])

        return topic

    async def moa_chat(self, io_interface: IOInterface):
        """Execute AutoGen group chat."""
        runtime = SingleThreadedAgentRuntime()

        conductor_topic_type = "conductor"
        # Register the conductor
        conductor = await Conductor.register(
            runtime,
            conductor_topic_type,
            lambda: Conductor(
                description="The conductor",
                group_chat_topic_type=self.group_chat_topic_type,
                steps=self.steps,
            ),
        )
        conductor_id = await runtime.get(conductor)

        await runtime.add_subscription(
            TypeSubscription(
                topic_type=self.group_chat_topic_type,
                agent_type=conductor_topic_type,
            ),
        )

        # Register the UserAgent with the provided IO interface
        user_agent_type = await io_interface.register(
            runtime,
            USER_AGENT_TYPE,
            lambda: io_interface(
                description="User input",
                group_chat_topic_type=self.group_chat_topic_type,
            ),
        )

        await runtime.add_subscription(
            TypeSubscription(
                topic_type=self.group_chat_topic_type,
                agent_type=USER_AGENT_TYPE,
            ),
        )
        # Start the runtime
        runtime.start()

        # Get the conductor started:
        logger.debug("Sending start signal to Conductor")
        await runtime.send_message(
            RequestToSpeak(),
            recipient=conductor_id,
        )

        # wait for the conversation to spin up
        await asyncio.sleep(5)

        # Wait for all processing to complete
        await runtime.stop_when_idle()

        await asyncio.sleep(5)
