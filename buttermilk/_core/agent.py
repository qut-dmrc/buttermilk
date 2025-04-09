from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, AsyncGenerator, Self, Union

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
    _run_fn: AsyncGenerator = PrivateAttr()

    _records: list[Record] = PrivateAttr(default_factory=list)
    _model_context: ChatCompletionContext = PrivateAttr(
        default_factory=UnboundedChatCompletionContext,
    )
    _message_types_handled: type[FlowMessage] = PrivateAttr(default=AgentInput)

    @pydantic.model_validator(mode="after")
    def _get_process_func(self) -> Self:
        """Returns the appropriate processing function based on tracing setting."""

        def _process_fn():
            if self._trace_this:
                # # Ensure weave.op wraps the async generator correctly
                # # This might require adjustments depending on weave's async support
                # async def traced_process(*args, **kwargs):
                #      async for item in self._process(*args, **kwargs):
                #          yield item
                # return weave.op(traced_process, call_display_name=f"{self.id}")
                return weave.op(self._process, call_display_name=f"{self.id} ({self.role})")
            return self._process

        self._run_fn = _process_fn()
        return self

    async def _ready_to_execute(self) -> bool:
        """Check if the agent is ready to execute."""
        return True
    
    async def listen(self, message: GroupchatMessageTypes, 
        ctx: MessageContext = None,
        **kwargs):
        """Save incoming messages for later use."""
        # Not implemented generically. Discard input.
        pass

    async def _process(
        self,
        message: AgentInput,
        cancellation_token: CancellationToken | None = None,
        **kwargs,
    ) -> AsyncGenerator[AgentOutput | TaskProcessingComplete | None, None]:
        """Process input data and (optionally) yield output(s).

        Inputs:
            input_data: AgentInput containing content, context/history, and other relevant data.
            cancellation_token: Token to signal cancellation.

        Outputs:
            Yields AgentOutput messages.
        """
        raise NotImplementedError
        yield # Required for async generator typing

    async def handle_control_message(
        self,
        message: OOBMessages,
        ctx: MessageContext = None,
        **kwargs
    ) -> OOBMessages | None:
        """Handle non-standard messages if needed (e.g., from orchestrator)."""
        logger.debug(f"Agent {self.id} {self.role} dropping control message: {message}")
        return None

    async def __call__(
        self,
        input_data: AgentInput,
        cancellation_token: CancellationToken | None = None,
        **kwargs,
    ) -> AsyncGenerator[AgentOutput  | UserInstructions | TaskProcessingComplete | None, None]:
        """Allow agents to be called directly as functions by the orchestrator."""
        # Check if we need to exit out before invoking the decorated tracing function self._run_fn
        if not isinstance(input_data, self._message_types_handled):
            logger.debug(f"Agent {self.id} received non-AgentInput message type {type(input_data)} in _process. Ignoring.")
            return
        
        try:
            async for output in self._run_fn(input_data, cancellation_token=cancellation_token, **kwargs):
                yield output
        except ProcessingError as e:
            logger.error(
                f"Agent {self.id} {self.role} hit processing error: {e}. Task content: {input_data.content[:100]}",
            )
            # yield ErrorEvent(
            #     source=self.id,
            #     role=self.role,
            #     content=f"Processing Error: {e}",
            #     error=[str(e)],
            #     records=input_data.records,
            # )
        except FatalError as e:
            logger.error(f"Agent {self.id} {self.role} hit fatal error: {e}", exc_info=True)
            raise e
        except Exception as e:
             logger.error(f"Agent {self.id} {self.role} hit unexpected error: {e}", exc_info=True)
            #  yield AgentOutput(
            #      source=self.id,
            #      role=self.role,
            #      content=f"Unexpected Error: {e}",
            #      error=[str(e)],
            #      records=input_data.records,
            #  )

    async def initialize(self, input_callback: Callable[..., Awaitable[None]] | None = None, **kwargs) -> None:
        """Initialize the agent (e.g., load resources)."""
        pass # Default implementation

    async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
        """Reset the agent's internal state."""
        pass # Default implementation
