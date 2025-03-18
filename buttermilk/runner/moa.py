import asyncio
from functools import cached_property
from distutils.util import strtobool
from typing import Type
import shortuuid
import weave
from autogen_core import (
    CancellationToken,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TypeSubscription,
    message_handler,
)
from autogen_core.model_context import (
    UnboundedChatCompletionContext,
)
from autogen_core.models import AssistantMessage, UserMessage
from pydantic import BaseModel

from buttermilk._core.config import SaveInfo
from buttermilk.agents import Fetch, LLMAgent
from buttermilk.bm import bm, logger
from buttermilk.runner.chat import (
    Answer,
    BaseGroupChatAgent,
    FlowMessage,
    GroupChatMessageType,
    InputRecord,
    IOInterface,
    MessagesCollector,
    NullAnswer,
    RequestToSpeak,
)
from buttermilk.utils.templating import KeyValueCollector

_AGENTS = [LLMAgent, Fetch]
USER_AGENT_TYPE = "User"


class MoAAgentFactory:
    """A factory for creating LLMAgent instance variants for a single
    step of a workflow.

    Creates a new agent for every combination of parameters in a given
    step of the workflow to run. Agents have a variants mapping;
    each permutation of these is multiplied by num_runs. Agents also
    have an inputs mapping that does not get multiplied.
    """

    async def register_variants(self, runtime, group_chat_topic_type: str):
        # Get the object that provides the agent template for all variants
        agent_cls: BaseGroupChatAgent = globals()[self.agent]

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


class Conductor(RoutedAgent):
    def __init__(self, description: str, group_chat_topic_type: str, steps):
        super().__init__(
            description=description,
        )
        self._group_chat_topic_type = group_chat_topic_type

        # Generally, in a group chat, you want most of the information to be publicly
        # visible. These collectors provide storage for agents, usually by reading from
        # the group chat log of messages. Useful for small objects predominantly.
        self._placeholders: KeyValueCollector = MessagesCollector()
        self._flow_data: FlowVariableRouter = FlowVariableRouter()
        self._context = UnboundedChatCompletionContext()
        self.running = False
        self.steps = steps

    @message_handler
    async def handle_request_to_speak(
        self,
        message: RequestToSpeak,
        ctx: MessageContext,
    ) -> GroupChatMessageType:
        if not self.running:
            self.running = True
            logger.info("Conductor started.")
            await self.run()
        else:
            logger.error("Conductor already running.")

        return NullAnswer(content="Conductor started.", step=self.id.type)

    @message_handler
    async def handle_messages(
        self,
        message: GroupChatMessageType,
        ctx: MessageContext,
    ) -> None:
        """
        Special cases:
        - 'record': List of InputRecord objects
        - 'content': List of fulltext values from InputRecord objects
        - 'context': List of all received messages
        - 'history': List of text from history messages
        """
        source=ctx.sender.type if ctx.sender else self.id.type
        
        if isinstance(message, InputRecord):
            msg = UserMessage(
                content=message.content,
                source=ctx.sender.type if ctx.sender else self.id.type,
            )
            
            self._placeholders.add('record', UserMessage(content=message.payload.fulltext, 
                source=source))
            self._flow_data.add("content", message.content)

        elif message.step == "User":
            if isinstance(message, NullAnswer):
                # No content, just the value -- handled elsewhere, we don't need to log it.
                return
            msg = UserMessage(
                content=message.content,
                source=source,
            )

        else:
            msg = AssistantMessage(
                content=message.content,
                source=source,
            )

        self._flow_data.add(message.step, message)

        
        # Add to our history as text
        self._flow_data.add("history", message.content)

        # also add to generic autogen collector
        await self._context.add_message(msg)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the assistant by clearing the model context."""
        self._placeholders = MessagesCollector()
        await self._context.clear()
        self._flow_data = FlowVariableRouter()

    async def run(self, init_text: str = None) -> None:
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

        # Start the group chat with the user's first message
        if not await self.query_user(content="OK, group chat started, go ahead. Enter a prompt, a URL, or a record ID (format: `!Record_ID`)"):
            logger.info("User did not confirm, exiting.")
            return
        

        # Allow some time for initialization
        await asyncio.sleep(1)

        # Process each step in sequence
        for step_factory in self.steps:
            logger.debug(f"Processing step: {step_factory.name}")

            if not await self.confirm_user():
                logger.info("User did not confirm, exiting.")
                return
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
                            context=await self._context.get_messages(),
                        ),
                        recipient=agent_id,
                    ),
                )

            # Wait for all agents in this step to complete
            if tasks:
                await asyncio.gather(*tasks)
            await asyncio.sleep(1)

    async def confirm_user(self, prompt: str = '') -> bool:
        """Ask the user to confirm the next step."""
        user_id = await self.runtime.get(USER_AGENT_TYPE)
        
        result = await self.runtime.send_message(
            RequestToSpeak(
                prompt=prompt or "Ready to proceed? (y/n)",
            ),
            recipient=user_id,
        )
        try:
            return result.value
        except AttributeError:
            try:
                return bool(strtobool(result.content))
            except ValueError:
                logger.error(f"Invalid input in confirm_user: {result.content}")
                return False
        
    async def query_user(self, content: str) -> FlowMessage:
        """Ask the user for input."""
        user_id = await self.runtime.get(USER_AGENT_TYPE)
        
        result = await self.runtime.send_message(
            RequestToSpeak(content=content
            ),
            recipient=user_id,
        )
        return result

class MoA(BaseModel):
    save: SaveInfo | None = None
    source: str
    steps: list[MoAAgentFactory]
    conductor: str = ""

    @cached_property
    def group_chat_topic_type(self) -> str:

        """The group chat topic type (common to all agents in this chat)."""
        topic = f"groupchat-{bm.run_info.name}-{bm.run_info.job}-{shortuuid.uuid()[:4]}"

        # remove punctuation
        topic = "".join([x for x in topic if x.isalnum() or x == "-"])

        return topic

    @weave.op
    async def moa_chat(self, io_interface: IOInterface, conductor: Type[Conductor | RoutedAgent], init_text: str = None):
        """Execute AutoGen group chat."""
        runtime = SingleThreadedAgentRuntime()

        conductor_topic_type = "conductor"
        # Register the conductor
        conductor_type = await conductor.register(
            runtime,
            conductor_topic_type,
            lambda: conductor(
                description="The conductor",
                group_chat_topic_type=self.group_chat_topic_type,
                steps=self.steps,
            ),
        )
        conductor_id = await runtime.get(conductor_type)
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
            RequestToSpeak(content=init_text),
            recipient=conductor_id,
        )

        # wait for the conversation to spin up
        await asyncio.sleep(5)

        # Wait for all processing to complete
        await runtime.stop_when_idle()

        await asyncio.sleep(5)


class FFA(Conductor):

    async def run(self, init_text: str = None) -> None:
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

        # Start the group chat with the user's first message
        prompt = await self.query_user(content="OK, group chat started, go ahead. Enter a prompt, a URL, or a record ID (format: `!Record_ID`)")
        if not prompt:
            logger.info("User did not confirm, exiting.")
            return
        
        tasks = []

        # First step, fetch records
        for agent_type in step_agents.get("fetch", []):
            agent_id = await self.runtime.get(agent_type)
            tasks.append(
                self.runtime.send_message(
                    message=RequestToSpeak(
                        inputs={"prompt": prompt},
                    ),
                    recipient=agent_id,
                ),
            )

        await asyncio.gather(*tasks)
        tasks = []

        while q := await self.query_user(content="Enter your query"):
            # Add history to query
            context = await self._context.get_messages()
            
            # Request each agent variant to speak
            for agent_type in step_agents.get("general", []):
                agent_id = await self.runtime.get(agent_type)
                tasks.append(
                    self.runtime.send_message(
                        message=RequestToSpeak(
                            inputs={"prompt": q},
                            placeholders=self._placeholders.get_dict(),
                            context=context,
                        ),
                        recipient=agent_id,
                    ),
                )

                # Wait for all agents in this step to complete
                if tasks:
                    await asyncio.gather(*tasks)
                await asyncio.sleep(1)
                

