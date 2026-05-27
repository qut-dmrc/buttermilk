"""Native orchestrator for managing multi-agent interactions.

Direct-dispatch orchestrator for managing multi-agent interactions.
Holds agent instances directly, dispatches messages by topic subscription,
and checks termination inline.
"""

import asyncio
import itertools
from collections.abc import Awaitable, Callable
from typing import Any

import shortuuid
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from buttermilk import (
    AllMessages,
    StepRequest,
    bm,
    logger,
)
from buttermilk._core.agent import Agent
from buttermilk._core.constants import MANAGER
from buttermilk._core.contract import (
    ConductorRequest,
    FlowEvent,
    FlowMessage,
    TaskProcessingComplete,
    UserResponseMessage,
)
from buttermilk._core.exceptions import FatalError, ProcessingError
from buttermilk._core.orchestrator import Orchestrator
from buttermilk._core.runtime_types import AgentIdentity, DefaultTopicId, MessageContext, TopicId
from buttermilk._core.types import RunRequest
from buttermilk.agents.spy import SpyAgent
from buttermilk.api.services.session_storage import SessionStorageService


class InterruptHandler(BaseModel):
    """Manages flow pause/resume via user interrupt messages."""

    interrupt: asyncio.Event = Field(default_factory=asyncio.Event)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def on_publish(self, message: Any, *, message_context: MessageContext | None = None) -> Any:
        if isinstance(message, UserResponseMessage):
            if message.interrupt:
                logger.info(f"Manager interrupt message received: {message}")
                self.interrupt.set()
            elif self.interrupt.is_set():
                logger.info(f"Manager resume message received: {message}")
                self.interrupt.clear()
        return message


class TerminationHandler:
    def __init__(self) -> None:
        self._termination_value: StepRequest | None = None

    def check(self, message: Any) -> None:
        if isinstance(message, StepRequest) and message.role == "END":
            logger.info(f"Termination message received: {message}")
            self._termination_value = message

    def request_termination(self):
        self._termination_value = StepRequest(role="END", content="Termination requested")

    @property
    def termination_value(self) -> StepRequest | None:
        return self._termination_value

    @property
    def has_terminated(self) -> bool:
        return self._termination_value is not None


class AutogenOrchestrator(Orchestrator):
    """Native direct-dispatch orchestrator for multi-agent workflows.

    Manages agent instances directly, dispatches messages by topic
    subscription, and checks termination inline.
    """

    _agents: dict[str, Any] = PrivateAttr(default_factory=dict)
    _subscriptions: dict[str, list[Any]] = PrivateAttr(default_factory=dict)
    _termination_handler: TerminationHandler | None = PrivateAttr(default=None)
    _interrupt_handler: InterruptHandler | None = PrivateAttr(default=None)
    _ui_callback: Callable[..., Awaitable[None]] | None = PrivateAttr(default=None)
    _session_collector: Callable[..., Awaitable[None]] | None = PrivateAttr(default=None)
    _pending_messages: list[tuple[FlowMessage, str]] = PrivateAttr(default_factory=list)
    _is_initialized: bool = PrivateAttr(default=False)
    _storage_service: SessionStorageService | None = PrivateAttr(default=None)
    _session_id: str | None = PrivateAttr(default=None)
    _topic: str = PrivateAttr(default="")
    _dispatch_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    def _subscribe(self, agent: Any, topic: str) -> None:
        """Subscribe an agent to a topic."""
        if topic not in self._subscriptions:
            self._subscriptions[topic] = []
        self._subscriptions[topic].append(agent)

    async def _dispatch_message(self, message: Any, topic: str) -> None:
        """Dispatch a message to all agents subscribed to the given topic."""
        self._termination_handler.check(message)
        if self._interrupt_handler is not None:
            await self._interrupt_handler.on_publish(message, message_context=None)

        ctx = MessageContext(
            topic_id=topic,
            sender=self._get_sender_identity(message),
        )

        # Dispatch to subscribed agents
        subscribers = self._subscriptions.get(topic, [])
        for agent in subscribers:
            try:
                if hasattr(agent, "dispatch"):
                    await agent.dispatch(message, ctx)
                elif callable(agent):
                    await agent(message, ctx)
            except Exception as e:
                agent_name = getattr(agent, "agent_name", getattr(agent, "__name__", str(agent)))
                logger.error(f"Error dispatching to {agent_name}: {e}", exc_info=True)

    def _get_sender_identity(self, message: Any) -> AgentIdentity | None:
        """Extract sender identity from a message if available."""
        agent_id = getattr(message, "agent_id", None) or getattr(message, "source", None)
        if agent_id:
            return AgentIdentity(key=str(agent_id))
        return None

    async def _publish(self, message: Any, topic: str) -> None:
        """Publish a message — entry point used by agents via their _publish_fn callback."""
        async with self._dispatch_lock:
            await self._dispatch_message(message, topic)

    def _make_agent_publish_fn(self) -> Callable[..., Awaitable[None]]:
        """Create a publish callback for agents to use."""
        async def publish_fn(message: Any, topic: TopicId | str) -> None:
            topic_str = str(topic)
            await self._publish(message, topic_str)
        return publish_fn

    async def _setup(self, request: RunRequest) -> tuple[TerminationHandler, InterruptHandler]:
        """Initialize orchestrator and register all configured agents."""
        if not self._topic:
            from buttermilk._core.execution_context import get_execution_context
            exec_ctx = get_execution_context()
            suffix = shortuuid.uuid()[:4]
            self._topic = f"{bm.session_info.project_name}-{exec_ctx.slug}-{bm.session_info.slug}-{suffix}"

        msg = f"Setting up orchestrator for topic: {self._topic}"
        logger.info(f"[AutogenOrchestrator._setup] {msg} (callback_to_ui: {'set' if request.callback_to_ui else 'not set'})")

        self._termination_handler = TerminationHandler()
        self._interrupt_handler = InterruptHandler()

        await self._register_agents(params=request)
        await self._register_ui(callback_to_ui=request.callback_to_ui)
        await self._register_session_collector()

        # Broadcast initialization event
        logger.info(f"Broadcasting initialization message to topic '{self._topic}' to wake up all agents")
        await self._publish(
            FlowEvent(source="orchestrator", content="Initializing group chat participants"),
            self._topic,
        )

        # Welcome message to UI
        flow_event = FlowEvent(source="orchestrator", content=msg)
        logger.debug("[AutogenOrchestrator._setup] Publishing welcome message to MANAGER topic")
        await self._publish(flow_event, MANAGER)

        # Start up the host agent with participants and their tools
        logger.info(
            f"Sending ConductorRequest to topic '{self._topic}' with {len(self.agents)} agents: {list(self.agents.keys())} and {len(self.observers)} observers: {list(self.observers.keys())}",
        )
        conductor_request = ConductorRequest(
            inputs=request.inputs,
            participants={v.role: v.description for k, v in self.agents.items()},
        )
        logger.debug(
            f"ConductorRequest details - participants: {list(conductor_request.participants.keys())}",
            agents=list(self.agents.keys()),
            observers=list(self.observers.keys()),
            input_keys=list(conductor_request.inputs.keys()) if conductor_request.inputs else [],
        )
        await self._publish(conductor_request, self._topic)

        # Mark as initialized and process any pending messages
        self._is_initialized = True

        if self._pending_messages:
            logger.debug(f"[AutogenOrchestrator._setup] Processing {len(self._pending_messages)} pending messages")
            for pending_message, topic_id in self._pending_messages:
                await self._publish(pending_message, topic_id)
        self._pending_messages.clear()

        return self._termination_handler, self._interrupt_handler

    async def _register_agents(self, params: RunRequest) -> None:
        """Register all configured agents."""
        logger.debug("Registering agents with native orchestrator...")

        publish_fn = self._make_agent_publish_fn()

        for role_name, step_config in itertools.chain(self.agents.items(), self.observers.items()):
            for agent_cls, variant_config in step_config.get_configs(params=params, flow_default_params=self.parameters):
                actual_role = step_config.role.upper()
                try:
                    agent_instance = self._create_agent(
                        agent_cls=agent_cls,
                        variant_config=variant_config,
                        params=params,
                        publish_fn=publish_fn,
                    )

                    agent_key = variant_config.agent_id
                    self._agents[agent_key] = agent_instance

                    # Subscribe to main topic and role topic
                    self._subscribe(agent_instance, self._topic)
                    self._subscribe(agent_instance, actual_role)

                    logger.debug(
                        f"Registered agent {agent_cls.__name__}: ID='{variant_config.agent_name}', "
                        f"Role='{actual_role}'. Subscribed to topics: '{self._topic}', '{actual_role}'",
                    )
                except Exception as e:
                    error_msg = f"FATAL: Agent registration failed for role '{role_name}': {e}"
                    logger.critical(error_msg)
                    raise FatalError(error_msg) from e

    def _create_agent(
        self,
        agent_cls: type,
        variant_config: Any,
        params: RunRequest,
        publish_fn: Callable[..., Awaitable[None]],
    ) -> Any:
        """Create a single agent instance."""
        if issubclass(agent_cls, Agent):
            dumped_config = variant_config.model_dump()
            logger.debug(f"Agent {variant_config.role}: model_dump required field = {dumped_config.get('required')!r}")
            config_with_session = {
                **dumped_config,
                "session_id": params.session_id,
                "topic_id": self._topic,
                "publish_fn": publish_fn,
            }
            if hasattr(self, "get_effective_bm"):
                config_with_session["bm"] = self.get_effective_bm()

            return agent_cls(**config_with_session)
        elif issubclass(agent_cls, SpyAgent):
            return agent_cls(
                **variant_config.parameters,
                publish_fn=publish_fn,
            )
        else:
            return agent_cls(**variant_config.parameters)

    async def _register_ui(self, callback_to_ui: Callable[..., Awaitable[None]]) -> None:
        """Register the UI callback as a subscriber."""
        if not callback_to_ui:
            logger.warning("No UI callback provided. Messages will not be sent to the UI.")
            return

        self._ui_callback = callback_to_ui

        async def ui_handler(message: Any, ctx: MessageContext) -> None:
            try:
                await callback_to_ui(message)
            except Exception as e:
                logger.error(f"[MANAGER handler] Error calling callback_to_ui: {e}", exc_info=True)

        # Subscribe UI handler to MANAGER and main topic
        self._subscribe(ui_handler, MANAGER)
        self._subscribe(ui_handler, self._topic)
        logger.debug(f"[AutogenOrchestrator._register_ui] UI handler registered for topics: {MANAGER}, {self._topic}")

    async def _register_session_collector(self) -> None:
        """Register a message collector for session persistence."""
        from buttermilk.api.services.message_service import MessageService

        async def persist_message(message: Any, ctx: MessageContext) -> None:
            if self._storage_service is None or self._session_id is None:
                return
            try:
                formatted = MessageService.format_message_for_client(message)
                if formatted and self._storage_service.should_persist_message(formatted):
                    self._storage_service.save_message(self._session_id, formatted)
            except Exception as e:
                logger.warning(f"Failed to persist session message: {e}")

        self._session_collector = persist_message
        self._subscribe(persist_message, self._topic)

    async def _run(self, request: RunRequest, flow_name: str = "") -> None:
        """Main execution loop."""
        self._session_id = request.session_id
        self._storage_service = SessionStorageService()

        try:
            try:
                logger.debug(f"[AutogenOrchestrator._run] Calling _setup with request.callback_to_ui: {request.callback_to_ui is not None}")
                termination_handler, interrupt_handler = await self._setup(request)
            except Exception as e:
                logger.error(f"Error during setup: {e}")
                raise FatalError(f"Orchestrator setup failed: {e}") from e

            while True:
                try:
                    if termination_handler.has_terminated:
                        logger.info("Termination message received.")
                        await self._publish(
                            FlowEvent(source="orchestrator", content="flow_completed"),
                            MANAGER,
                        )
                        logger.debug("[AutogenOrchestrator._run] Publishing TaskProcessingComplete message.")
                        await self._publish(
                            TaskProcessingComplete(
                                agent_id="orchestrator",
                                role="orchestrator",
                            ),
                            MANAGER,
                        )
                        logger.debug("[AutogenOrchestrator._run] TaskProcessingComplete message published.")
                        break
                    if interrupt_handler.interrupt.is_set():
                        logger.info("Flow is paused. Waiting for resume...")
                        while interrupt_handler.interrupt.is_set():
                            await asyncio.sleep(0.5)
                        logger.info("Flow resumed.")
                    await asyncio.sleep(0.1)

                except ProcessingError as e:
                    logger.error(f"Error in execution: {e}")
                except (StopAsyncIteration, KeyboardInterrupt):
                    raise
                except FatalError:
                    raise
                except Exception as e:
                    raise FatalError from e

        except KeyboardInterrupt:
            logger.info("Flow terminated by user.")
            raise
        except (FatalError, Exception) as e:
            logger.exception(f"Unexpected and unhandled fatal error: {e}", exc_info=True)
            raise
        finally:
            # Close all agents
            for agent_key, agent in self._agents.items():
                try:
                    if hasattr(agent, "close"):
                        await agent.close()
                except Exception as e:
                    logger.warning(f"Failed to close agent {agent_key}: {e}")

            if self._storage_service and self._session_id:
                try:
                    self._storage_service.finalize_session(self._session_id, "completed")
                except Exception as e:
                    logger.warning(f"Failed to finalize session: {e}")

    def make_publish_callback(self) -> Callable[[FlowMessage], Awaitable[None]]:
        """Creates an asynchronous callback function for the UI to use."""
        async def publish_callback(message: FlowMessage) -> None:
            logger.debug(f"[AutogenOrchestrator.make_publish_callback] Publishing message to runtime: {message}")

            if not self._is_initialized:
                logger.info(f"[AutogenOrchestrator] Runtime not initialized yet, queueing message: {message}")
                self._pending_messages.append((message, self._topic))
                return

            await self._publish(message, self._topic)

        return publish_callback
