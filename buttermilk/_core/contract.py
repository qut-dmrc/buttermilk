
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, Union

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from buttermilk._core.config import DataSource, SaveInfo
from buttermilk._core.runner_types import Record
from buttermilk.utils.validators import make_list_validator

BASE_DIR = Path(__file__).absolute().parent


class FlowProtocol(Protocol):
    save: SaveInfo
    data: Sequence[DataSource]
    steps: Sequence["Agent"]

    async def run(self, job: "Job") -> "Job": ...

    async def __call__(self, job: "Job") -> "Job": ...


class OrchestratorProtocol(Protocol):
    bm: "BM"
    flows: Mapping[str, FlowProtocol]


######
# Communication between Agents
#


# A base class
class FlowMessage(BaseModel):
    type: str = "AgentMessage"

    content: str = Field(
        default="",
        description="The human-readable digest representation of the message.",
    )

    records: list[Record] = Field(
        default=[],
        description="Records relevant to this message",
    )

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Metadata about the message."""

    @model_validator(mode="before")
    @classmethod
    def coerce_content_to_string(cls, values):
        if "content" in values:
            content = values["content"]
            if isinstance(content, str):
                pass  # Already a string
            elif hasattr(content, "content"):  # Handle LLMMessage case
                values["content"] = str(content.content)
            else:
                values["content"] = str(content)
        return values


class ManagerMessage(FlowMessage):
    """OOB message to manage the flow.

    Usually involves an automated
    conductor or a human in the loop (or both).
    """

    type: str = "ManagerMessage"
    stop: bool = Field(default=False, description="Whether to stop the flow")


class AgentInput(FlowMessage):
    type: str = "InputRequest"
    """Base class for agent inputs with built-in validation"""
    prompt: str = Field(default="", description="A question or prompt from a user")
    context: list[Any] = Field(
        default=[],
        description="History or context to provide to the agent.",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary of inputs to the agent",
    )

    _ensure_list = field_validator("records", "context", mode="before")(
        make_list_validator(),
    )


class AgentOutput(FlowMessage):
    """Base class for agent outputs with built-in validation"""

    type: str = "Answer"
    agent: str

    error: list[str] = Field(
        default_factory=list,
        description="A list of errors that occurred during the agent's execution",
    )

    response: Any = Field(
        default=None,
        description="The response from the agent",
    )

    _ensure_list = field_validator("error", "records", mode="before")(
        make_list_validator(),
    )


AgentMessages = Union[FlowMessage, AgentInput, AgentOutput]
