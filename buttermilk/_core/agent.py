from abc import ABC, abstractmethod
import asyncio
from collections.abc import Awaitable, Callable
import time
from typing import Annotated, Any, AsyncGenerator, Callable, Self, Union

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
from buttermilk import logger
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
    OOBMessages,
    TaskProcessingComplete,
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
from buttermilk._core.config import DataSourceConfig
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
    _data: KeyValueCollector = PrivateAttr(default_factory=KeyValueCollector)

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
        self,
        *,
        message: AgentInput | ConductorRequest,
        cancellation_token=None,
        public_callback: Callable = None,
        message_callback: Callable = None,
        **kwargs,
    ) -> list[AgentOutput | ToolOutput | TaskProcessingComplete] | None:
        # Agents come in variants, and each variant has a list of tasks that it iterates through.
        # And are supplemented by placeholders for records and contextual history
        n = 0
        tasks = []
        outputs = []
        for task_params in self.sequential_tasks:
            inputs = message.model_copy(deep=True)
            # add data from our local state

            inputs.context.extend(await self._model_context.get_messages())

            _local_state = self._data.get_dict()

            inputs.records.extend(self._records)
            inputs.inputs.update(_local_state)
            inputs.params.update(task_params)
            inputs.params.update(self.parameters)

            # make sure data does not have records or context placeholders in inputs fields
            inputs.inputs.pop("records", None)
            inputs.inputs.pop("context", None)

            _traced = weave.op(
                self._process,
                call_display_name=self.name,
            )
            t = asyncio.create_task(_traced(inputs=inputs, cancellation_token=cancellation_token))
            tasks.append(t)

        for t in asyncio.as_completed(tasks):
            n += 1
            try:
                result = await t
                if result:
                    outputs.append(result)
                    await public_callback(
                        TaskProcessingComplete(role=self.role, task_index=n, more_tasks_remain=True, is_error=result.is_error, source=self.id)
                    )
            except ProcessingError as e:
                logger.error(
                    f"Agent {self.role} {self.name} hit processing error: {e} {e.args=}.",
                )
                await public_callback(TaskProcessingComplete(role=self.role, task_index=n, more_tasks_remain=True, is_error=True, source=self.id))
                continue
            except FatalError as e:
                logger.error(f"Agent {self.role} {self.name} hit fatal error: {e}", exc_info=True)
                raise e
            except Exception as e:
                logger.error(f"Agent {self.role} {self.name} hit unexpected error: {e}", exc_info=True)
                raise e
            finally:
                await public_callback(TaskProcessingComplete(role=self.role, task_index=n, more_tasks_remain=False, is_error=False, source=self.id))
        return outputs

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        cancellation_token: CancellationToken = None,
        public_callback: Callable = None,
        message_callback: Callable = None,
        **kwargs,
    ) -> None:
        """Save incoming messages for later use."""
        # Look for matching roles in our inputs mapping
        if message.role in [x.split(".")[0] for x in self.inputs.values() if x and isinstance(x, str)]:
            # Possible match, let's extract data
            result_dict = {message.role: message.model_dump()}
            for var_name, field_path in self.inputs.items():
                value = jmespath.search(field_path, result_dict)
                if value:
                    if var_name == "records":
                        self._records.extend(message.records)
                    else:
                        self._data.add(var_name, value)
        else:
            if isinstance(message, (AgentOutput, ConductorResponse)):
                await self._model_context.add_message(AssistantMessage(content=str(message.content), source=message.source))
            elif isinstance(message, (ToolOutput, UserInstructions)):
                if not message.content.startswith(COMMAND_SYMBOL):
                    await self._model_context.add_message(UserMessage(content=str(message.content), source=message.source))
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
    ) -> list[AgentOutput | ToolOutput | TaskProcessingComplete] | None:
        """Allow agents to be called directly as functions by the orchestrator."""
        output = await self._run_fn(
            message=message, cancellation_token=cancellation_token, public_callback=public_callback, message_callback=message_callback, **kwargs
        )
        return output

    async def invoke_privately(
        self,
        message: ConductorRequest,
        cancellation_token: Any,
        public_callback: Callable= None, 
        message_callback: Callable= None, 
        **kwargs,
    ) -> FlowMessage|None:
        """Respond directly to the orchestrator"""
        outputs = None
        response = await self._run_fn(message=message, cancellation_token=cancellation_token, public_callback=public_callback, message_callback=message_callback, **kwargs)

        if response:
            # convert to ConductorResponse
            outputs = ConductorResponse(**response[-1].model_dump())
            if len(response)>1:
                outputs.internal_messages = response[:-1]
            await message_callback(outputs)
            return outputs

    # def custom_attribute_name(call):
    #     model = call.attributes["model"]
    #     return f"{model}"

    #     @weave.op(call_display_name=custom_attribute_name)

    async def invoke(
        self,
        message: AgentInput,
        cancellation_token: Any,
        public_callback: Callable = None,
        message_callback: Callable = None,
        **kwargs,
    ) -> list[AgentOutput | ToolOutput | TaskProcessingComplete] | None:
        """Run the main function."""
        # Check if we need to exit out before invoking the decorated tracing function self._run_fn
        if not isinstance(message, self._message_types_handled):
            logger.debug(f"Agent {self.role} received non-supported message type {type(message)} in _process. Ignoring.")
            return

        outputs = await self._run_fn(
            message=message, cancellation_token=cancellation_token, public_callback=public_callback, message_callback=message_callback, **kwargs
        )
        await public_callback(outputs)
        return outputs

    async def initialize(self, input_callback: Callable[..., Awaitable[None]] | None = None, **kwargs) -> None:
        """Initialize the agent (e.g., load resources)."""
        pass # Default implementation

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Reset the agent's internal state."""
        pass # Default implementation
