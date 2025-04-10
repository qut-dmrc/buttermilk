from abc import ABC, abstractmethod
import asyncio
from collections.abc import Awaitable, Callable
import time
from typing import Any, AsyncGenerator, Callable, Self, Union

from autogen_core.model_context import UnboundedChatCompletionContext, ChatCompletionContext
import pydantic
import weave
from autogen_core import CancellationToken, MessageContext
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
)

from autogen_core.tools import BaseTool, FunctionTool
from buttermilk import logger
from buttermilk._core.config import DataSourceConfig
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    AllMessages,
    ConductorRequest,
    ConductorResponse,
    ErrorEvent,
    FlowMessage,
    GroupchatMessageTypes,
    OOBMessages,
    TaskProcessingComplete,
    ToolOutput,
    UserInstructions,
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
from buttermilk._core.config import DataSourceConfig
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    AllMessages, # Keep for type hinting if needed elsewhere, but not used in Agent interface directly
    UserInstructions,
)
from buttermilk._core.exceptions import FatalError, ProcessingError
from buttermilk._core.runner_types import Record
from buttermilk.utils.validators import convert_omegaconf_objects


class ToolConfig(BaseModel):
    role: str= Field(
        default="")
    description: str= Field(
        default="")
    tool_obj: str = Field(
        default="")


    data: list[DataSourceConfig] = Field(
        default=[],
        description="Specifications for data that the Agent should load",
    )

    def get_functions(self) -> list[Any]:
        """Create function definitions for this tool."""
        raise NotImplementedError()


    async def _run(
        self, **kwargs
    ) -> list[ToolOutput] | None:
        raise NotImplementedError()

#########
# Agent
#
# A simple class with a function that process input.
#
#
##########


class AgentConfig(BaseModel):
    agent_obj: str = Field(
        default="",
        description="The object name to instantiate",
        exclude=True,
    )
    id: str = Field(
        ...,
        description="The unique name of this agent.",
    )
    role: str = Field(
        ...,
        description="The role that this agent fulfils.",
    )
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

    Agents are stateful. Context is stored internally by the agent
    or passed via AgentInput. _reset() clears state.
    """
    _trace_this = True

    _records: list[Record] = PrivateAttr(default_factory=list)
    _model_context: ChatCompletionContext = PrivateAttr(
        default_factory=UnboundedChatCompletionContext,
    )
    _message_types_handled: type[FlowMessage] = PrivateAttr(default=AgentInput)
    _heartbeat: asyncio.Queue = PrivateAttr(default_factory=lambda: asyncio.Queue(maxsize=1))

    async def _check_heartbeat(self, timeout=60) -> bool:
        """Check if the heartbeat queue is empty."""
        t0 = time.time()
        while True:
            try:
                return self._heartbeat.get_nowait()
            except asyncio.QueueEmpty:
                if time.time() - t0 > timeout:
                    return False
                await asyncio.sleep(1)

    async def _run_fn(
        self, *, message: AgentInput | ConductorRequest, cancellation_token=None, publish_callback=None, **kwargs
    ) -> AsyncGenerator[AgentOutput | ToolOutput | None | TaskProcessingComplete, None]:
        # Agents come in variants, and each variant has a list of tasks that it iterates through.
        # And are supplemented by placeholders for records and contextual history
        n = 0
        for task_params in self.sequential_tasks:
            try:
                inputs = AgentInput(**message.model_dump())
                # add data from our local state
                inputs.records.extend(self._records)
                inputs.params.update(task_params)
                inputs.params.update(self.parameters)
                inputs.context.extend(await self._model_context.get_messages())

                _traced = weave.op(
                    self._process,
                    call_display_name=f"{self.id} ({self.role})",
                )

                async for result in _traced(inputs=inputs, cancellation_token=cancellation_token, publish_callback=publish_callback):
                    await asyncio.sleep(0.1)
                    yield result
                    yield TaskProcessingComplete(source=self.id, task_index=n, more_tasks_remain=True)
                    # if not await self._check_heartbeat():
                    #     logger.info(
                    #         f"Agent {self.id} {self.role} did not receive heartbeat; canceling.",
                    #     )
                    #     raise StopAsyncIteration
            except ProcessingError as e:
                logger.error(
                    f"Agent {self.id} {self.role} hit processing error: {e} {e.args=}.",
                )
                # yield ErrorEvent(
                #     source=self.id,
                #     role=self.role,
                #     content=f"Processing Error: {e}",
                #     error=[str(e)],
                #     records=input_data.records,
                # )
                continue
            except FatalError as e:
                logger.error(f"Agent {self.id} {self.role} hit fatal error: {e}", exc_info=True)
                raise e
            except Exception as e:
                logger.error(f"Agent {self.id} {self.role} hit unexpected error: {e}", exc_info=True)
                raise e
            finally:
                yield TaskProcessingComplete(source=self.id, task_index=n, more_tasks_remain=False)

        # if self._trace_this:
        #     self._run_fn = weave.op(traced_process, call_display_name=f"{self.id} ({self.role})")
        # self._run_fn = traced_process

        # return self

    async def _listen(
        self, message: GroupchatMessageTypes, cancellation_token: CancellationToken = None, publish_callback: Callable = None, **kwargs
    ) -> AsyncGenerator[GroupchatMessageTypes | None, None]:
        """Save incoming messages for later use."""
        # Not implemented generically. Discard input.
        yield None

    async def _process(
        self, inputs: AgentInput, cancellation_token: CancellationToken = None, publish_callback: Callable = None, **kwargs
    ) -> AsyncGenerator[AgentOutput | ToolOutput | None, None]:
        """Internal process function. Replace this in subclasses.

        Process input data and yield any output(s).

        Outputs:
            Yields AgentOutput messages.
        """
        raise NotImplementedError
        yield # Required for async generator typing

    async def _handle_control_message(
        self, message: OOBMessages, cancellation_token: CancellationToken = None, publish_callback: Callable = None, **kwargs
    ) -> AsyncGenerator[OOBMessages | None, None]:
        """Handle non-standard messages if needed (e.g., from orchestrator)."""
        logger.debug(f"Agent {self.id} {self.role} dropping control message: {message}")
        yield None

    async def __call__(
        self, message: AgentInput, cancellation_token: CancellationToken, publish_callback: Callable, **kwargs
    ) -> AsyncGenerator[AgentOutput | ToolOutput | TaskProcessingComplete | None, None]:
        """Allow agents to be called directly as functions by the orchestrator."""
        async for result in self._run_fn(message=message, cancellation_token=cancellation_token, publish_callback=publish_callback, **kwargs):
            yield result

    async def invoke_privately(
        self,
        message: ConductorRequest,
        cancellation_token: Any,
        publish_callback: Callable,
        **kwargs,
    ) -> AsyncGenerator[FlowMessage, None]:
        """Respond directly to the orchestrator"""
        response = []
        outputs = None
        async for output in self._run_fn(message=message, cancellation_token=cancellation_token, publish_callback=publish_callback, **kwargs):
            if isinstance(response, (AgentOutput, ToolOutput)):
                response.append(output)
            yield output

        if response:
            # convert to ConductorResponse
            outputs = ConductorResponse(**response[-1].model_dump())
            if len(response)>1:
                outputs.internal_messages = response[:-1]

            yield outputs

    @weave.op
    async def invoke(
        self,
        message: AgentInput,
        cancellation_token: Any,
        publish_callback: Callable,
        **kwargs,
    ) -> AsyncGenerator[AgentOutput | ToolOutput | TaskProcessingComplete | None, None]:
        """Run the main function."""
        # Check if we need to exit out before invoking the decorated tracing function self._run_fn
        if not isinstance(message, self._message_types_handled):
            logger.debug(f"Agent {self.id} received non-supported message type {type(message)} in _process. Ignoring.")
            return

        async for output in self._run_fn(message=message, cancellation_token=cancellation_token, publish_callback=publish_callback, **kwargs):
            if isinstance(output, (AgentOutput, ToolOutput)):
                yield output

    async def initialize(self, input_callback: Callable[..., Awaitable[None]] | None = None, **kwargs) -> None:
        """Initialize the agent (e.g., load resources)."""
        pass # Default implementation

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Reset the agent's internal state."""
        pass # Default implementation
