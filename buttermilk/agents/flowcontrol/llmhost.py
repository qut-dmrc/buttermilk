
from collections.abc import AsyncGenerator

from pydantic import BaseModel, Field, field_validator

from buttermilk._core import AgentInput, StepRequest
from buttermilk.agents.flowcontrol.host import HostAgent
from buttermilk.agents.llm import LLMAgent


class CallOnAgent(BaseModel):
    """A request from a flow control agent for action from another agent.
    """

    role: str = Field(
        ...,
        description="The role of the agent to call on. This should be a valid role name.",
    )
    prompt: str = Field(
        ...,
        description="The prompt to send to the agent.",
    )

    @field_validator("role")
    @classmethod
    def _role_must_be_uppercase(cls, v: str) -> str:
        """Ensures the role field is always uppercase for consistency."""
        if v:
            return v.upper()
        return v


class LLMHostAgent(LLMAgent, HostAgent):
    """An agent that can call on other agents to perform actions.
    """

    _output_model: type[BaseModel] | None = CallOnAgent

    async def _sequence(self) -> AsyncGenerator[StepRequest, None]:
        while True:
            result = await self._process(message=AgentInput())
            next_step = StepRequest(
                role=result.outputs.role,
                prompt=result.outputs.prompt,
            )
            yield next_step
