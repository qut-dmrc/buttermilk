from abc import ABC, abstractmethod
import asyncio
from collections.abc import Awaitable, Callable
import time
from typing import Annotated, Any, AsyncGenerator, Callable, Self, Sequence, Union

from autogen_core.model_context import UnboundedChatCompletionContext, ChatCompletionContext
import jmespath
import pydantic
from shortuuid import ShortUUID, uuid
import weave
from autogen_core import CancellationToken, MessageContext
from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
)

from autogen_core.tools import BaseTool, FunctionTool
from buttermilk._core.log import logger
from buttermilk._core.config import DataSourceConfig
from buttermilk._core.contract import (
    COMMAND_SYMBOL,
    AgentInput,
    AgentOutput,
    AllMessages,
    AssistantMessage,
    ConductorRequest,
    ConductorResponse,
    ErrorEvent,
    FlowMessage,
    GroupchatMessageTypes,
    ManagerResponse,
    OOBMessages,
    TaskProcessingComplete,
    TaskProcessingStarted,
    ToolOutput,
    UserInstructions,
    UserMessage,
)
from buttermilk._core.exceptions import FatalError, ProcessingError
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, AsyncGenerator, Self

import pydantic
import weave
from autogen_core import CancellationToken
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
)

from buttermilk import logger
from buttermilk._core.config import DataSourceConfig, ToolConfig
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    AllMessages, # Keep for type hinting if needed elsewhere, but not used in Agent interface directly
    UserInstructions,
)
from buttermilk._core.exceptions import FatalError, ProcessingError
from buttermilk._core.flow import KeyValueCollector
from buttermilk._core.types import Record
from buttermilk.utils.validators import convert_omegaconf_objects, lowercase_validator


#########
# Agent
#
# A simple class with a function that process input.
#
#
##########


class AgentConfig(BaseModel):
    id: str = Field(
        default_factory=uuid,
        description="A unique identifier for this agent.",
    )
    agent_obj: str = Field(
        default="",
        description="The object name to instantiate",
        exclude=True,
    )
    role: Annotated[str, AfterValidator(lowercase_validator)] = Field(
        ...,
        description="The role type that this agent fulfils.",
    )

    name: str = Field(description="The human friendly name of this agent type.")

    description: str = Field(
        ...,
        description="Short explanation of what this agent type does",
    )
    tools: list[ToolConfig] = Field(
        default=[],
        description="Tools the agent can invoke",
    )
    data: list[DataSourceConfig] = Field(
        default=[],
        description="Specifications for data that the Agent should load",
    )
    num_runs: int = Field(
        default=1,
        description="Number of times to replicate each parallel variant agent instance.",
        exclude=True,
    )
    variants: dict = Field(
        default={},
        description="Parameters to create parallel agent instances via cross-multiplication.",
        exclude=True,
    )
    tasks: dict = Field(
        default={},
        description="Parameters defining sequential tasks for each agent instance via cross-multiplication.",
        exclude=True,
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Initialisation parameters to pass to the agent",
    )
    sequential_tasks: list[dict[str, Any]] = Field(
        default_factory=lambda: [{}], # Default to one task with empty parameters
        description="List of tasks for each agent computed from .tasks.",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="A mapping of data to agent inputs",
    )
    outputs: dict[str, Any] = {}
    model_config = {
        "extra": "allow",
        "arbitrary_types_allowed": False,
        "populate_by_name": True,
    }

    _validate_variants = field_validator(
        "variants", "tasks", "parameters", mode="before"
    )(convert_omegaconf_objects())


class Agent(AgentConfig):
    """Base Agent interface for all processing units.

    Agents are stateful. Context is stored internally by the agent and merged
    with data passed via AgentInput. _reset() clears state.
    """

    _trace_this = True

    _records: list[Record] = PrivateAttr(default_factory=list)
    _model_context: ChatCompletionContext = PrivateAttr(
        default_factory=UnboundedChatCompletionContext,
    )
    _message_types_handled: type[FlowMessage] = PrivateAttr(default=AgentInput)
    _heartbeat: asyncio.Queue = PrivateAttr(default_factory=lambda: asyncio.Queue(maxsize=1))
    _data: KeyValueCollector = PrivateAttr(default_factory=KeyValueCollector)

    async def _check_heartbeat(self, timeout=60) -> bool:
        """Check if the heartbeat queue is empty."""
        try:
            await asyncio.wait_for(self._heartbeat.get(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _add_state_to_input(self, inputs: AgentInput) -> AgentInput:
        """Add local agent state to inputs"""

        # Fill inputs based on input map
        inputs.inputs.update(self._data._resolve_mappings(self.inputs))

        # add additional placeholders
        inputs.context.extend(await self._model_context.get_messages())
        inputs.records.extend(self._records)

        return inputs

    async def _run_fn(
        self,
        *,
        message: AgentInput | ConductorRequest,
        cancellation_token: CancellationToken|None=None,
        public_callback: Callable = None,
        message_callback: Callable = None,
        **kwargs,
    ) -> AgentOutput | ToolOutput | TaskProcessingComplete | None:
        """Run all tasks within a variant configuration."""
        # Agents come in variants, and each variant has a list of tasks that it runs asynchronously.
        tasks = []
        for n, task_params in enumerate(self.sequential_tasks):
            task_inputs = message.model_copy(deep=True)
            task_inputs.parameters = dict(task_params)
            task_inputs.parameters.update(self.parameters)
            task_inputs = await self._add_state_to_input(task_inputs)
            tasks.append(asyncio.create_task(self._run_task(task_index=n, task_inputs=task_inputs, cancellation_token=cancellation_token, public_callback=public_callback)))

        for t in asyncio.as_completed(tasks):
            try:
                result = await t
                if result:
                    outputs.append(result)
            except ProcessingError as e:
                logger.error(
                    f"Agent {self.role} {self.name} hit processing error: {e} {e.args=}.",
                )
            except FatalError as e:
                logger.error(f"Agent {self.role} {self.name} hit fatal error: {e}", exc_info=True)
                raise e
            except Exception as e:
                logger.error(f"Agent {self.role} {self.name} hit unexpected error: {e}", exc_info=True)
                raise e

    async def _run_task(self, *, task_index: int, task_inputs: AgentInput, cancellation_token: CancellationToken, public_callback: Callable
    ) -> AgentOutput | ToolOutput | TaskProcessingComplete | None:
        """Run a single task, trace it, and evaluate it."""
        # Signal that we have started
        await public_callback(TaskProcessingStarted(agent_id=self.id, role=self.role, task_index=task_index))

        # Start a trace
        with weave.attributes(dict(task_inputs.parameters)):
            _traced = weave.op(
                self._process,
                call_display_name=f"{self.name} {self.id}",
            )
            from weave.trace.weave_client import Call

            # Run task
            result, call = await _traced.call(inputs=task_inputs, cancellation_token=cancellation_token)

            # Score task
            # call.apply_scorer()
        return result

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        cancellation_token: CancellationToken = None,
        public_callback: Callable = None,
        message_callback: Callable = None,
        source: str = "unknown",
        **kwargs,
    ) -> None:
        """Save incoming messages for later use."""
        # Look for matching roles in our inputs mapping
        if isinstance(message, (AgentOutput, ConductorResponse)):
            for key, mapping in self.inputs.items():
                if mapping and isinstance(mapping, str) and mapping.startswith(message.role):
                    # Possible direct match, let's try to extract data
                    if key == "records":
                        # records are stored separately in our memory cache and we know where to find them
                        # this helps us avoid turning the record into a dict below.
                        if message.records:
                            self._records.extend(message.records)
                            continue

                    if mapping == message.role:
                        # no dot delineated field path
                        # so add the whole object
                        self._data.add(key, message.model_dump())
                        continue

                # otherwise, try to find the value in the outputs dict
                search_dict = {message.role: message.model_dump()}
                if mapping and (value := jmespath.search(mapping, search_dict)):
                    self._data.add(key, value)

        if isinstance(message, (AgentOutput, ConductorResponse)):
            if message.content:
                await self._model_context.add_message(AssistantMessage(content=str(message.content), source=source))
        elif isinstance(message, (ToolOutput, UserInstructions)):
            if not message.content.startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=str(message.content), source=source))
        else:
            # don't log other types of messages
            pass

    async def _process(self, *, inputs: AgentInput, cancellation_token: CancellationToken = None, **kwargs) -> AgentOutput | ToolOutput | None:
        """Internal process function. Replace this in subclasses.

        Process input data and publish any output(s).

        Outputs:
            None
        """
        raise NotImplementedError

    async def _handle_control_message(
        self, message: OOBMessages, cancellation_token: CancellationToken = None, public_callback: Callable = None, message_callback: Callable= None,   **kwargs
    ) -> OOBMessages | None:
        """Handle non-standard messages if needed (e.g., from orchestrator)."""
        logger.debug(f"Agent {self.role} {self.name} dropping control message: {message}")
        return None

    async def __call__(
        self,
        message: AgentInput,
        cancellation_token: CancellationToken,
        public_callback: Callable = None,
        message_callback: Callable = None,
        **kwargs,
    ) -> AgentOutput | ToolOutput | TaskProcessingComplete | None:
        """Allow agents to be called directly as functions by the orchestrator."""
        output = await self._run_fn(
            message=message, cancellation_token=cancellation_token, public_callback=public_callback, message_callback=message_callback, **kwargs
        )
        return output

    async def initialize(self, input_callback: Callable[..., Awaitable[None]] | None = None, **kwargs) -> None:
        """Initialize the agent (e.g., load resources)."""
        pass # Default implementation

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Reset the agent's internal state."""
        pass # Default implementation
